from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from app.business_rules.confidence import ConfidenceScoringService
from app.business_rules.mapping import FieldMappingService
from app.business_rules.temporal import TemporalMappingService
from app.business_rules.validation import ValidationService
from app.config.field_registry import FieldRegistry, get_default_registry
from app.config.settings import Settings
from app.etl.export import EcrfExportService
from app.extraction.langextract_extractor import LangExtractExtractor
from app.extraction.llama_extractor import LlamaExtractor
from app.extraction.service import ExtractionService
from app.indexing import build_default_vector_service
from app.indexing.vector_service import VectorIndexService
from app.ingestion.service import DocumentIngestionService, LocalFileIngestionService
from app.parsing.chunking import ChunkingService, DefaultChunkingService
from app.parsing.service import ParsingService
from app.parsing.smart_service import SmartParsingService
from app.retrieval.workflow_orchestrator import (
    BaseWorkflowOrchestrator,
    LocalWorkflowOrchestrator,
    VectorStoreWorkflowOrchestrator,
)
from app.routing.router import DocumentRouter, KeywordHeuristicDocumentRouter
from app.schemas.enums import FieldFamily
from app.schemas.models import AuditRecord, EcrfCellUpdate, PipelineResult
from app.utils.audit import AuditTrailService


class PipelineOrchestrator:
    """Assemble les services pour exécuter la chaîne eCRF Autofill V1."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        registry: FieldRegistry | None = None,
        ingestion: DocumentIngestionService | None = None,
        parsing: ParsingService | None = None,
        chunking: ChunkingService | None = None,
        router: DocumentRouter | None = None,
        workflow: BaseWorkflowOrchestrator | None = None,
        vector_index: VectorIndexService | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.registry = registry or get_default_registry()
        self.ingestion = ingestion or LocalFileIngestionService()
        self.parsing = parsing or SmartParsingService(self.settings)
        self.chunking = chunking or DefaultChunkingService(self.settings)
        self.router = router or KeywordHeuristicDocumentRouter()
        self.vector_index = vector_index or build_default_vector_service(self.settings)
        if workflow is not None:
            self.workflow = workflow
        elif self.settings.vector_backend == "qdrant":
            self.workflow = VectorStoreWorkflowOrchestrator(self.vector_index)
        else:
            self.workflow = LocalWorkflowOrchestrator()
        self.extraction = ExtractionService(
            langextract=LangExtractExtractor(
                model_name=self.settings.mock_langextract_version,
                schema_version=self.settings.mock_langextract_version,
            ),
            llama=LlamaExtractor(
                model_name=self.settings.mock_llama_model_name,
                schema_version="lab-ft-mock",
            ),
        )
        self.temporal = TemporalMappingService()
        self.mapping = FieldMappingService(self.registry)
        self.validation = ValidationService(self.registry)
        self.confidence = ConfidenceScoringService()
        self.export_service = EcrfExportService()
        self.audit = AuditTrailService(self.settings.pipeline_version)

    def run(self, document_path: str, patient_id: str, study_id: str) -> PipelineResult:
        audit: list[AuditRecord] = []

        raw = self.ingestion.ingest(document_path, patient_id, study_id)
        audit.append(
            self.audit.record_step(
                document_id=raw.document_id,
                patient_id=patient_id,
                study_id=study_id,
                step="ingestion",
                details={"mime": raw.mime_type, "path": raw.source_path},
            )
        )

        parsed = self.parsing.parse(raw)
        audit.append(
            self.audit.record_step(
                document_id=raw.document_id,
                patient_id=patient_id,
                study_id=study_id,
                step="parsing",
                details={
                    "chars": len(parsed.full_text),
                    "pdf_parser": parsed.metadata.get("pdf_parser"),
                    "structured_sections": len(parsed.structured_sections),
                    "structured_lab_lines": len(parsed.structured_lab_lines),
                },
            )
        )

        chunks = self.chunking.chunk(parsed)
        audit.append(
            self.audit.record_step(
                document_id=raw.document_id,
                patient_id=patient_id,
                study_id=study_id,
                step="chunking",
                details={"chunks": len(chunks)},
            )
        )

        tenant_id = study_id  # Option C : tenant par défaut = study_id
        if isinstance(self.workflow, VectorStoreWorkflowOrchestrator):
            self.workflow._tenant_id = tenant_id  # noqa: SLF001 — paramétrage runtime
        self.vector_index.upsert_chunks(
            chunks,
            tenant_id=tenant_id,
            patient_id=patient_id,
            study_id=study_id,
            document_type=parsed.document_type_hint.value,
        )
        audit.append(
            self.audit.record_step(
                document_id=raw.document_id,
                patient_id=patient_id,
                study_id=study_id,
                step="indexing",
                details={
                    "backend": type(self.vector_index).__name__,
                    "tenant_id": tenant_id,
                    "chunks_indexed": len(chunks),
                },
            )
        )

        doc_type = self.router.route(parsed)
        audit.append(
            self.audit.record_step(
                document_id=raw.document_id,
                patient_id=patient_id,
                study_id=study_id,
                step="routing",
                details={"document_type": doc_type.value},
            )
        )

        field_defs = self.registry.for_document_type(doc_type)
        families = sorted({fd.field_family for fd in field_defs}, key=lambda f: f.value)
        family_hits = self.workflow.retrieve_for_families(
            document_id=raw.document_id,
            chunks=chunks,
            families=families,
            top_k_per_family=5,
        )
        best_retrieval: dict[FieldFamily, float | None] = {
            fam: max((h.score for h in hits), default=None) for fam, hits in family_hits.items()
        }
        audit.append(
            self.audit.record_step(
                document_id=raw.document_id,
                patient_id=patient_id,
                study_id=study_id,
                step="retrieval",
                details={
                    "families": [f.value for f in families],
                    "hits_per_family": {k.value: len(v) for k, v in family_hits.items()},
                },
            )
        )

        observations_by_family: dict[FieldFamily, list] = defaultdict(list)
        for fam in families:
            hits = family_hits.get(fam, [])
            obs = self.extraction.extract_for_family(
                doc_type=doc_type,
                field_family=fam,
                hits=hits,
                field_defs=field_defs,
            )
            observations_by_family[fam].extend(obs)
            for ob in obs:
                audit.append(
                    self.audit.from_observation(
                        document_id=raw.document_id,
                        patient_id=patient_id,
                        study_id=study_id,
                        step="extraction",
                        observation=ob,
                    )
                )

        observations: list = []
        seen_obs: set[str] = set()
        for fam in families:
            for o in observations_by_family[fam]:
                if o.observation_id in seen_obs:
                    continue
                seen_obs.add(o.observation_id)
                observations.append(o)

        candidates = self.mapping.build_candidates(doc_type=doc_type, observations=observations)
        enriched_candidates = []
        for cand in candidates:
            fd = self.registry.get(cand.field_name)
            if fd is None:
                continue
            obs_temp = self.temporal.apply_for_field(parsed, cand.observation, fd)
            anchor = self.temporal.build_anchor(fd, parsed)
            enriched_candidates.append(
                cand.model_copy(
                    update={
                        "observation": obs_temp,
                        "temporal_anchor": anchor,
                    }
                )
            )

        audit.append(
            self.audit.record_step(
                document_id=raw.document_id,
                patient_id=patient_id,
                study_id=study_id,
                step="temporal_mapping",
                details={"candidates": len(enriched_candidates)},
            )
        )

        temporal_by_id = {c.observation.observation_id: c.observation for c in enriched_candidates}
        observations = [temporal_by_id.get(o.observation_id, o) for o in observations]

        cell_updates: list[EcrfCellUpdate] = []
        for cand in enriched_candidates:
            fd = self.registry.get(cand.field_name)
            if fd is None:
                continue
            blended, ok = self.confidence.score_candidate(
                cand,
                retrieval_score=best_retrieval.get(cand.observation.field_family),
                autofill_threshold=fd.autofill_threshold,
            )
            obs_scored = cand.observation.model_copy(update={"confidence": blended})
            cand = cand.model_copy(update={"observation": obs_scored, "confidence": blended})
            if not ok:
                continue
            update = EcrfCellUpdate(
                patient_id=patient_id,
                study_id=study_id,
                column_key=fd.target_column,
                value=cand.observation.normalized_value,
                provenance=cand,
                validated=False,
            )
            cell_updates.append(update)

        audit.append(
            self.audit.record_step(
                document_id=raw.document_id,
                patient_id=patient_id,
                study_id=study_id,
                step="field_mapping",
                details={"cell_updates": len(cell_updates)},
            )
        )

        validated_updates: list[EcrfCellUpdate] = []
        for upd in cell_updates:
            valid, errors = self.validation.validate_update(upd)
            validated_updates.append(upd.model_copy(update={"validated": valid}))
            audit.append(
                self.audit.record_step(
                    document_id=raw.document_id,
                    patient_id=patient_id,
                    study_id=study_id,
                    step="validation",
                    details={"column": upd.column_key, "ok": valid, "errors": errors},
                )
            )

        output_dir = Path("outputs") / raw.document_id
        result = PipelineResult(
            document_id=raw.document_id,
            patient_id=patient_id,
            study_id=study_id,
            document_type=doc_type,
            observations=observations,
            cell_updates=validated_updates,
            audit_trail=audit,
            metadata={"document_path": str(Path(document_path).resolve())},
        )
        paths = self.export_service.export(result, output_dir)
        audit.append(
            self.audit.record_step(
                document_id=raw.document_id,
                patient_id=patient_id,
                study_id=study_id,
                step="export",
                details=paths,
            )
        )
        return result.model_copy(update={"export_paths": paths, "audit_trail": audit})


def run_pipeline(document_path: str, patient_id: str, study_id: str) -> PipelineResult:
    """Point d'entrée fonctionnel V1."""
    return PipelineOrchestrator().run(document_path, patient_id, study_id)
