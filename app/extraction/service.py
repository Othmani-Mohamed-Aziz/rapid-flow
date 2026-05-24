from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.extraction.imaging_langextract import LangExtractNotInstalledError
from app.extraction.planner import plan_extraction_jobs
from app.extraction.strategy_extractors import ExtractorRegistry
from app.schemas.enums import DocumentType, FieldFamily
from app.schemas.models import ExtractedObservation, FieldDefinition, RetrievalHit
from app.schemas.study_schema import ExtractionJob, ExtractionStrategy, StudySchema

_LOG = logging.getLogger(__name__)


class ExtractionService:
    """
    Orchestre l'extraction selon le `StudySchema` (jobs planifiés par stratégie).

    Chaque hit RAG est traité comme un chunk distinct (extraction + `source_chunk_id`).
    """

    def __init__(
        self,
        *,
        extractor_registry: ExtractorRegistry,
        study_schema: StudySchema,
        imaging_extraction_max_workers: int = 8,
    ) -> None:
        self._registry = extractor_registry
        self._study_schema = study_schema
        self._imaging_workers = max(1, int(imaging_extraction_max_workers))

    @property
    def study_schema(self) -> StudySchema:
        return self._study_schema

    def plan_jobs(self, doc_type: DocumentType) -> list[ExtractionJob]:
        return plan_extraction_jobs(self._study_schema, doc_type)

    def extract_for_job(
        self,
        job: ExtractionJob,
        hits: list[RetrievalHit],
    ) -> list[ExtractedObservation]:
        """Exécute un job d'extraction sur les hits de sa `field_family`."""
        strategy = job.extraction_strategy
        if strategy == ExtractionStrategy.NONE:
            return []

        extractor = self._registry.get(strategy)
        tasks = [h for h in hits if (h.chunk.text or "").strip()]
        if not tasks:
            return []

        def run_one(hit: RetrievalHit) -> list[ExtractedObservation]:
            return extractor.extract_chunk(
                (hit.chunk.text or "").strip(),
                job,
                source_chunk_id=hit.chunk.chunk_id,
                retrieval_hit_score=hit.score,
            )

        parallel = (
            strategy == ExtractionStrategy.IMAGING_LANGEXTRACT
            and self._imaging_workers > 1
            and len(tasks) > 1
        )
        observations: list[ExtractedObservation] = []
        if not parallel:
            for hit in tasks:
                observations.extend(run_one(hit))
            return self._dedupe(job.field_family, observations)

        workers = min(self._imaging_workers, len(tasks))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(run_one, h): h for h in tasks}
            for fut in as_completed(future_map):
                hit = future_map[fut]
                try:
                    observations.extend(fut.result())
                except LangExtractNotInstalledError:
                    raise
                except Exception as exc:  # pragma: no cover
                    _LOG.warning(
                        "Extraction %s échouée chunk %s : %s",
                        strategy.value,
                        hit.chunk.chunk_id,
                        exc,
                    )
        return self._dedupe(job.field_family, observations)

    def extract_all_for_document(
        self,
        *,
        doc_type: DocumentType,
        family_hits: dict[FieldFamily, list[RetrievalHit]],
    ) -> list[ExtractedObservation]:
        """Planifie et exécute tous les jobs pour un type documentaire."""
        all_obs: list[ExtractedObservation] = []
        for job in self.plan_jobs(doc_type):
            hits = family_hits.get(job.field_family, [])
            all_obs.extend(self.extract_for_job(job, hits))
        return all_obs

    def extract_for_family(
        self,
        *,
        doc_type: DocumentType,
        field_family: FieldFamily,
        hits: list[RetrievalHit],
        field_defs: list[FieldDefinition],
    ) -> list[ExtractedObservation]:
        """Rétrocompat : extraction pour une seule famille."""
        _ = field_defs
        jobs = [j for j in self.plan_jobs(doc_type) if j.field_family == field_family]
        observations: list[ExtractedObservation] = []
        for job in jobs:
            observations.extend(self.extract_for_job(job, hits))
        return observations

    @staticmethod
    def _dedupe(
        field_family: FieldFamily,
        observations: list[ExtractedObservation],
    ) -> list[ExtractedObservation]:
        seen: set[tuple[str, str | None, str | None]] = set()
        deduped: list[ExtractedObservation] = []
        for o in observations:
            if o.field_family != field_family:
                continue
            canon = (o.extra or {}).get("canonical_imaging_key") or (o.extra or {}).get(
                "canonical_lab_key"
            )
            key = (o.field_family.value, str(canon), o.evidence_text)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(o)
        return deduped
