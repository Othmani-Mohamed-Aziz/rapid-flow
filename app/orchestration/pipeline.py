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
from app.etl.column_aliases import resolve_column_aliases
from app.etl.ecrf_export_policy import (
    columns_to_skip_for_policy,
    partition_cell_updates_for_ecrf,
)
from app.etl.export import EcrfExportService
from app.etl.imaging_template_export import export_imaging_template
from app.etl.lab_template_export import export_lab_template
from app.etl.overwrite_policy import (
    EcrfTemplateSnapshot,
    XlsOverwritePolicy,
    target_columns_for_document,
)
from app.etl.patient_resolver import (
    ensure_ma_patient_in_template,
    resolve_ma_patient_key,
)
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
from app.schemas.enums import DocumentType, FieldFamily
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
        self.lab_extraction = LabDeterministicStrategy(
            study_schema=self.study_schema,
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
            lab=self.lab_extraction,
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
        self.export_service = EcrfExportService.from_settings(self.settings)
        self.audit = AuditTrailService(self.settings.pipeline_version)

    def run(
        self,
        document_path: str,
        patient_id: str,
        study_id: str,
        *,
        ma_patient_key: str | None = None,
    ) -> PipelineResult:
        audit: list[AuditRecord] = []
        if self.study_schema.study_id != study_id:
            _LOG.warning(
                "study_id argument %r ≠ study_schema.study_id %r — tenant/index utilisent l'argument.",
                study_id,
                self.study_schema.study_id,
            )

        ma_resolution = resolve_ma_patient_key(
            patient_id,
            self.settings,
            ma_patient_key=ma_patient_key,
        )
        ensure_ma_patient_in_template(ma_resolution, self.settings)
        ma_key = ma_resolution.ma_patient_key

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

        skip_columns: set[str] = set()
        overwrite_policy = XlsOverwritePolicy(self.settings.export_xls_overwrite_policy)
        snapshot = EcrfTemplateSnapshot.from_settings(self.settings, ma_patient_key=ma_key)
        target_cols = target_columns_for_document(self.study_schema, doc_type)
        skip_columns = columns_to_skip_for_policy(overwrite_policy, snapshot, target_cols)
        column_aliases = resolve_column_aliases(self.settings)

        if skip_columns and overwrite_policy == XlsOverwritePolicy.EMPTY_ONLY:
            _LOG.info(
                "eCRF prefilter (empty_only) : %d colonne(s) déjà remplies, extraction ignorée : %s",
                len(skip_columns),
                sorted(skip_columns),
            )
        elif overwrite_policy == XlsOverwritePolicy.NEVER and skip_columns:
            _LOG.info(
                "eCRF prefilter (never) : extraction eCRF ignorée pour %d colonne(s)",
                len(skip_columns),
            )

        audit.append(
            self.audit.record_step(
                document_id=raw.document_id,
                patient_id=patient_id,
                study_id=study_id,
                step="ecrf_prefilter",
                details={
                    "policy": overwrite_policy.value,
                    "skipped_columns": sorted(skip_columns),
                    "snapshot_available": snapshot.available,
                    "empty_sentinels": sorted(self.settings.parsed_empty_sentinels()),
                    "duplicate_headers": snapshot.duplicate_headers,
                    **ma_resolution.to_metadata(),
                },
            )
        )

        extraction_jobs = plan_extraction_jobs(
            self.study_schema,
            doc_type,
            exclude_target_columns=skip_columns,
        )
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
            extraction_jobs=extraction_jobs,
        )
        if doc_type == DocumentType.LAB_BLOOD_PANEL and parsed.structured_lab_lines:
            observations = [
                observation
                for observation in observations
                if observation.extraction_method != "lab_deterministic"
            ]
            observations.extend(self.lab_extraction.extract_document_lab_rows(parsed, chunks))
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
            retrieval_score = (
                None
                if cand.observation.extraction_method == "lab_document_rows"
                else best_retrieval.get(cand.observation.field_family)
            )
            blended, ok = self.confidence.score_candidate(
                cand,
                retrieval_score=retrieval_score,
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

        ecrf_updates, suppressed_updates = partition_cell_updates_for_ecrf(
            validated_updates,
            policy=overwrite_policy,
            snapshot=snapshot,
            column_aliases=column_aliases,
        )
        audit.append(
            self.audit.record_step(
                document_id=raw.document_id,
                patient_id=patient_id,
                study_id=study_id,
                step="ecrf_export_policy",
                details={
                    "policy": overwrite_policy.value,
                    "cell_updates_raw": len(validated_updates),
                    "cell_updates_ecrf": len(ecrf_updates),
                    "suppressed": suppressed_updates,
                },
            )
        )

        output_dir = Path("outputs") / raw.document_id
        result = PipelineResult(
            document_id=raw.document_id,
            patient_id=patient_id,
            study_id=study_id,
            document_type=doc_type,
            observations=observations,
            cell_updates=ecrf_updates,
            audit_trail=audit,
            ma_patient_key=ma_key,
            metadata={
                "document_path": str(Path(document_path).resolve()),
                "study_schema_version": self.study_schema.schema_version,
                "study_id_config": self.study_schema.study_id,
                "export_xls_overwrite_policy": overwrite_policy.value,
                "ecrf_skipped_columns": sorted(skip_columns),
                "cell_updates_raw_count": len(validated_updates),
                "cell_updates_suppressed": suppressed_updates,
                **ma_resolution.to_metadata(),
            },
        )
        paths = self.export_service.export(result, output_dir)
        if (
            doc_type == DocumentType.LAB_BLOOD_PANEL
            and self.settings.export_lab_template_json_enabled
        ):
            lab_template_path = export_lab_template(
                observations,
                output_dir,
                source_path=document_path,
            )
            paths["lab_template_json"] = str(lab_template_path.resolve())
        if (
            doc_type == DocumentType.IMAGING_REPORT
            and self.settings.export_imaging_template_json_enabled
        ):
            imaging_template_path = export_imaging_template(
                ecrf_updates,
                output_dir,
                source_path=document_path,
            )
            paths["imaging_template_json"] = str(imaging_template_path.resolve())
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


def run_pipeline(
    document_path: str,
    patient_id: str,
    study_id: str,
    *,
    ma_patient_key: str | None = None,
) -> PipelineResult:
    """Point d'entrée fonctionnel V1."""
    return PipelineOrchestrator().run(
        document_path,
        patient_id,
        study_id,
        ma_patient_key=ma_patient_key,
    )
