from __future__ import annotations

import logging
from pathlib import Path

from app.business_rules.confidence import ConfidenceScoringService
from app.business_rules.mapping import FieldMappingService
from app.business_rules.normalization import NormalizationService
from app.business_rules.temporal import TemporalMappingService
from app.business_rules.validation import ValidationService
from app.config.field_registry import FieldRegistry
from app.config.settings import Settings
from app.config.study_schema_provider import resolve_study_schema
from app.etl.export import EcrfExportService
from app.extraction.llama_extractor import LlamaExtractor
from app.extraction.planner import field_families_from_jobs, plan_extraction_jobs
from app.extraction.service import ExtractionService
from app.extraction.strategy_extractors import (
    ExtractorRegistry,
    ImagingLangextractStrategy,
    LabDeterministicStrategy,
    NarrativeKeywordsStrategy,
)
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
from app.schemas.study_schema import StudySchema
from app.utils.audit import AuditTrailService

_LOG = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Assemble les services pour exécuter la chaîne eCRF Autofill V1."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        registry: FieldRegistry | None = None,
        study_schema: StudySchema | None = None,
        ingestion: DocumentIngestionService | None = None,
        parsing: ParsingService | None = None,
        chunking: ChunkingService | None = None,
        router: DocumentRouter | None = None,
        workflow: BaseWorkflowOrchestrator | None = None,
        vector_index: VectorIndexService | None = None,
    ) -> None:
        self.settings = settings or Settings()
        schema_path = self.settings.study_schema_path
        self.study_schema = study_schema or resolve_study_schema(schema_path)
        self.registry = registry or FieldRegistry.from_study_schema(self.study_schema)
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
        llama = LlamaExtractor(
            model_name=self.settings.mock_llama_model_name,
            schema_version=self.study_schema.schema_version,
        )
        extractor_registry = ExtractorRegistry(
            imaging=ImagingLangextractStrategy(
                study_schema=self.study_schema,
                model_id=self.settings.ollama_model_id,
                model_url=self.settings.ollama_url,
                timeout=self.settings.ollama_timeout_s,
                schema_version=self.settings.langextract_schema_version,
                langextract_enabled=self.settings.langextract_enabled,
            ),
            lab=LabDeterministicStrategy(
                study_schema=self.study_schema,
                schema_version=self.study_schema.schema_version,
            ),
            narrative=NarrativeKeywordsStrategy(
                study_schema=self.study_schema,
                schema_version=self.study_schema.schema_version,
            ),
            llama=llama,
        )
        self.extraction = ExtractionService(
            extractor_registry=extractor_registry,
            study_schema=self.study_schema,
            imaging_extraction_max_workers=self.settings.imaging_extraction_max_workers,
        )
        self.temporal = TemporalMappingService()
        self.normalization = NormalizationService()
        self.mapping = FieldMappingService(self.registry)
        self.validation = ValidationService(self.registry)
        self.confidence = ConfidenceScoringService()
        self.export_service = EcrfExportService()
        self.audit = AuditTrailService(self.settings.pipeline_version)

    def run(self, document_path: str, patient_id: str, study_id: str) -> PipelineResult:
        audit: list[AuditRecord] = []
        if self.study_schema.study_id != study_id:
            _LOG.warning(
                "study_id argument %r ≠ study_schema.study_id %r — tenant/index utilisent l'argument.",
                study_id,
                self.study_schema.study_id,
            )

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

        tenant_id = study_id  # Option C : tenant par défaut = study_id
        self.vector_index.upsert_chunks(
            chunks,
            tenant_id=tenant_id,
            patient_id=patient_id,
            study_id=study_id,
            document_type=doc_type.value,
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

        extraction_jobs = plan_extraction_jobs(self.study_schema, doc_type)
        families = field_families_from_jobs(extraction_jobs)
        family_queries: dict = {}
        for fam in families:
            q = self.study_schema.retrieval_query_for_family(fam)
            if q:
                family_queries[fam] = q
        family_hits = self.workflow.retrieve_for_families(
            document_id=raw.document_id,
            chunks=chunks,
            families=families,
            top_k_per_family=5,
            tenant_id=tenant_id,
            family_queries=family_queries or None,
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

        observations = self.extraction.extract_all_for_document(
            doc_type=doc_type,
            family_hits=family_hits,
        )
        for ob in observations:
            audit.append(
                self.audit.from_observation(
                    document_id=raw.document_id,
                    patient_id=patient_id,
                    study_id=study_id,
                    step="extraction",
                    observation=ob,
                )
            )

        candidates = self.mapping.build_candidates(
            doc_type=doc_type,
            observations=observations,
            parsed=parsed,
            normalize_fn=self.normalization.apply,
        )
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
        best_by_column: dict[str, tuple[float, EcrfCellUpdate]] = {}
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
            prev = best_by_column.get(update.column_key)
            if prev is None or blended > prev[0]:
                best_by_column[update.column_key] = (blended, update)
        cell_updates = [pair[1] for pair in best_by_column.values()]

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
            metadata={
                "document_path": str(Path(document_path).resolve()),
                "study_schema_version": self.study_schema.schema_version,
                "study_id_config": self.study_schema.study_id,
            },
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
