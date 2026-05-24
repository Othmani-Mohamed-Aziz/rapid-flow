"""Extracteurs par stratégie (`ExtractionJob` + schéma d'étude)."""

from __future__ import annotations

import logging
from typing import Protocol

from app.extraction.imaging_config import ImagingExtractionConfig
from app.extraction.imaging_langextract import LangExtractNotInstalledError, run_imaging_langextract
from app.extraction.lab_heuristics import (
    extract_lab_observations,
    extract_narrative_keyword_observations,
)
from app.extraction.llama_extractor import LlamaExtractor
from app.schemas.enums import FieldFamily
from app.schemas.models import ExtractedObservation
from app.schemas.study_schema import ExtractionJob, ExtractionStrategy, StudySchema

_LOG = logging.getLogger(__name__)


class StrategyExtractor(Protocol):
    def extract_chunk(
        self,
        chunk_text: str,
        job: ExtractionJob,
        *,
        source_chunk_id: str | None,
        retrieval_hit_score: float | None,
    ) -> list[ExtractedObservation]: ...


class ImagingLangextractStrategy:
    def __init__(
        self,
        *,
        study_schema: StudySchema,
        model_id: str,
        model_url: str,
        timeout: int,
        schema_version: str,
        langextract_enabled: bool,
    ) -> None:
        self._study_schema = study_schema
        self.model_id = model_id
        self.model_url = model_url
        self.timeout = timeout
        self.schema_version = schema_version
        self.langextract_enabled = langextract_enabled

    def extract_chunk(
        self,
        chunk_text: str,
        job: ExtractionJob,
        *,
        source_chunk_id: str | None,
        retrieval_hit_score: float | None,
    ) -> list[ExtractedObservation]:
        if not self.langextract_enabled:
            _LOG.debug("LangExtract désactivé (ECRF_LANGEXTRACT_ENABLED=false)")
            return []
        config = ImagingExtractionConfig.from_job(job, self._study_schema)
        try:
            return run_imaging_langextract(
                chunk_text,
                model_id=self.model_id,
                model_url=self.model_url,
                timeout=self.timeout,
                schema_version=self.schema_version,
                source_chunk_id=source_chunk_id,
                retrieval_hit_score=retrieval_hit_score,
                imaging_config=config,
            )
        except LangExtractNotInstalledError:
            raise
        except Exception as exc:
            _LOG.warning("LangExtract/Ollama chunk %s : %s", source_chunk_id, exc)
            return []


class LabDeterministicStrategy:
    def __init__(self, *, study_schema: StudySchema, schema_version: str | None) -> None:
        self._study_schema = study_schema
        self.schema_version = schema_version

    def extract_chunk(
        self,
        chunk_text: str,
        job: ExtractionJob,
        *,
        source_chunk_id: str | None,
        retrieval_hit_score: float | None,
    ) -> list[ExtractedObservation]:
        _ = retrieval_hit_score
        allowed = job.canonical_key_set
        specs = self._study_schema.lab_analytes_for_keys(allowed)
        obs = extract_lab_observations(
            chunk_text,
            extraction_method="lab_deterministic",
            model_name=None,
            schema_version=self.schema_version,
            source_chunk_id=source_chunk_id,
            allowed_canonical_keys=allowed if allowed else None,
            analyte_specs=specs,
        )
        if job.field_family != FieldFamily.COMORBIDITIES:
            obs = [o for o in obs if o.field_family == job.field_family]
        return obs


class NarrativeKeywordsStrategy:
    def __init__(self, *, study_schema: StudySchema, schema_version: str | None) -> None:
        self._study_schema = study_schema
        self.schema_version = schema_version

    def extract_chunk(
        self,
        chunk_text: str,
        job: ExtractionJob,
        *,
        source_chunk_id: str | None,
        retrieval_hit_score: float | None,
    ) -> list[ExtractedObservation]:
        _ = retrieval_hit_score
        allowed = job.canonical_key_set
        specs = self._study_schema.narrative_specs_for_keys(allowed)
        return extract_narrative_keyword_observations(
            chunk_text,
            extraction_method="narrative_keywords",
            model_name=None,
            schema_version=self.schema_version,
            source_chunk_id=source_chunk_id,
            specs=specs,
            allowed_canonical_keys=allowed if allowed else None,
            field_family=job.field_family,
        )


class ExtractorRegistry:
    """Registre strategy → extracteur."""

    def __init__(
        self,
        *,
        imaging: ImagingLangextractStrategy,
        lab: LabDeterministicStrategy,
        narrative: NarrativeKeywordsStrategy,
        llama: LlamaExtractor | None = None,
    ) -> None:
        self._imaging = imaging
        self._lab = lab
        self._narrative = narrative
        self._llama = llama

    def get(self, strategy: ExtractionStrategy) -> StrategyExtractor:
        if strategy == ExtractionStrategy.IMAGING_LANGEXTRACT:
            return self._imaging
        if strategy == ExtractionStrategy.LAB_DETERMINISTIC:
            return self._lab
        if strategy == ExtractionStrategy.NARRATIVE_KEYWORDS:
            return self._narrative
        raise ValueError(f"Stratégie d'extraction non supportée : {strategy}")
