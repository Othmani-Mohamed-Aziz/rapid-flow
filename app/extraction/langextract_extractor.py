from __future__ import annotations

import logging
from typing import Any

from app.extraction.imaging_langextract import (
    SCHEMA_VERSION,
    LangExtractNotInstalledError,
    run_imaging_langextract,
)
from app.extraction.lab_heuristics import build_langextract_schema_stub, extract_lab_observations
from app.schemas.enums import ExtractionFamily, FieldFamily
from app.schemas.models import ExtractedObservation

_LOG = logging.getLogger(__name__)


class LangExtractExtractor:
    """
    Extracteur structuré par chunk.

    - **Labo** : heuristiques locales (V1, hors scope LLM).
    - **Imagerie RECIST** : LangExtract + Ollama si `langextract_enabled`.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        schema_version: str | None = None,
        langextract_enabled: bool = True,
        ollama_url: str = "http://localhost:11434",
        ollama_timeout_s: int = 120,
    ) -> None:
        self.model_name = model_name or "gemma2:2b"
        self.schema_version = schema_version or SCHEMA_VERSION
        self.langextract_enabled = langextract_enabled
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_timeout_s = ollama_timeout_s

    def extract(
        self,
        text: str,
        output_schema: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        source_chunk_id: str | None = None,
        retrieval_hit_score: float | None = None,
    ) -> list[ExtractedObservation]:
        """Extraction sur un **sous-texte** (typiquement un chunk RAG)."""
        _ = context
        family_token = output_schema.get("x-field-family")
        extraction_family_token = output_schema.get("x-extraction-family")
        if family_token:
            family = FieldFamily(family_token)
        else:
            family = FieldFamily.IMAGING_RECIST

        if extraction_family_token == ExtractionFamily.LAB_VALUES.value or family in (
            FieldFamily.HEMATOLOGY,
            FieldFamily.HEPATIC_BIOCHEMISTRY,
            FieldFamily.INFLAMMATION_BIOMARKERS,
            FieldFamily.COAGULATION,
        ):
            return extract_lab_observations(
                text,
                extraction_method="lab_heuristic",
                model_name=self.model_name,
                schema_version=self.schema_version,
                source_chunk_id=source_chunk_id,
            )

        if family != FieldFamily.IMAGING_RECIST:
            return []

        if not self.langextract_enabled:
            _LOG.debug("LangExtract désactivé (ECRF_LANGEXTRACT_ENABLED=false)")
            return []

        try:
            return run_imaging_langextract(
                text,
                model_id=self.model_name,
                model_url=self.ollama_url,
                timeout=self.ollama_timeout_s,
                schema_version=self.schema_version,
                source_chunk_id=source_chunk_id,
                retrieval_hit_score=retrieval_hit_score,
            )
        except LangExtractNotInstalledError:
            raise
        except Exception as exc:
            _LOG.warning(
                "LangExtract/Ollama a échoué sur chunk %s : %s",
                source_chunk_id,
                exc,
            )
            return []

    @staticmethod
    def default_schema_for_family(family: FieldFamily) -> dict[str, Any]:
        if family == FieldFamily.IMAGING_RECIST:
            return {
                "type": "object",
                "x-field-family": FieldFamily.IMAGING_RECIST.value,
                "x-extraction-family": ExtractionFamily.IMAGING_RECIST.value,
                "description": "Extraction imagerie RECIST via LangExtract",
            }
        return build_langextract_schema_stub(family)
