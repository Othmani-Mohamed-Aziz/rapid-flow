"""Fixtures et builders pour les tests de la phase extraction."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.config.field_registry import FieldRegistry
from app.config.settings import Settings
from app.config.study_schema_provider import load_study_schema_from_json, resolve_study_schema
from app.extraction.service import ExtractionService
from app.extraction.strategy_extractors import (
    ExtractorRegistry,
    ImagingLangextractStrategy,
    LabDeterministicStrategy,
    NarrativeKeywordsStrategy,
)
from app.schemas.enums import (
    DocumentType,
    ExtractionFamily,
    FieldFamily,
    TemporalScope,
)
from app.schemas.extraction_catalog import ExtractionCatalog, LabAnalyteSpec
from app.schemas.models import DocumentChunk, RetrievalHit
from app.schemas.study_schema import StudyFieldDefinition, StudySchema

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "data" / "study_schema_default.json"
MOCK_BLOOD_PANEL_TXT = PROJECT_ROOT / "scripts" / "sample_data" / "mock_blood_panel.txt"


def fake_langextract_extraction(
    *,
    extraction_class: str,
    extraction_text: str,
    attributes: dict[str, Any] | None = None,
    grounded: bool = True,
) -> SimpleNamespace:
    char_iv = SimpleNamespace(start_pos=0, end_pos=len(extraction_text)) if grounded else None
    return SimpleNamespace(
        extraction_class=extraction_class,
        extraction_text=extraction_text,
        attributes=attributes or {},
        char_interval=char_iv,
    )


def make_retrieval_hit(
    *,
    chunk_id: str,
    text: str,
    document_id: str = "doc-1",
    score: float = 0.9,
    family: FieldFamily = FieldFamily.IMAGING_RECIST,
    rank: int = 1,
) -> RetrievalHit:
    return RetrievalHit(
        chunk=DocumentChunk(chunk_id=chunk_id, document_id=document_id, text=text),
        score=score,
        query_family=family,
        rank=rank,
    )


def build_dual_ast_schema() -> StudySchema:
    """Deux champs eCRF partageant canonical_key AST (scopes temporels différents)."""
    base_field = dict(
        field_type="numeric",
        document_types_allowed=[DocumentType.LAB_BLOOD_PANEL],
        extraction_family=ExtractionFamily.LAB_VALUES,
        field_family=FieldFamily.HEPATIC_BIOCHEMISTRY,
        extraction_strategy="lab_deterministic",
        canonical_key="AST",
        langextract_class=None,
        retrieval_query=None,
        autofill_threshold=0.65,
        normalization_rule="numeric_u_l",
    )
    return StudySchema(
        study_id="DUAL_AST",
        schema_version="test-v1",
        fields=[
            StudyFieldDefinition(
                field_name="AST_baseline",
                temporal_scope=TemporalScope.BASELINE,
                target_column="AST_baseline",
                **base_field,
            ),
            StudyFieldDefinition(
                field_name="AST_followup",
                temporal_scope=TemporalScope.FOLLOW_UP,
                target_column="AST_followup",
                **base_field,
            ),
        ],
    )


def build_custom_lab_schema() -> StudySchema:
    """Schéma avec un analyte GGT défini uniquement dans le catalogue JSON."""
    return StudySchema(
        study_id="CUSTOM_LAB",
        schema_version="test-v1",
        fields=[
            StudyFieldDefinition(
                field_name="GGT_level",
                field_type="numeric",
                document_types_allowed=[DocumentType.LAB_BLOOD_PANEL],
                extraction_family=ExtractionFamily.LAB_VALUES,
                field_family=FieldFamily.HEPATIC_BIOCHEMISTRY,
                temporal_scope=TemporalScope.BASELINE,
                normalization_rule=None,
                target_column="GGT_level",
                autofill_threshold=0.6,
                canonical_key="GGT",
                extraction_strategy="lab_deterministic",
            )
        ],
        extraction_catalog=ExtractionCatalog(
            lab_analytes=[
                LabAnalyteSpec(
                    canonical_key="GGT",
                    pattern=r"\bGGT\b\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)",
                    unit="U/L",
                    field_family=FieldFamily.HEPATIC_BIOCHEMISTRY,
                )
            ],
        ),
    )


def build_extractor_registry(
    schema: StudySchema,
    *,
    langextract_enabled: bool = False,
) -> ExtractorRegistry:
    settings = Settings()
    return ExtractorRegistry(
        imaging=ImagingLangextractStrategy(
            study_schema=schema,
            model_id=settings.ollama_model_id,
            model_url=settings.ollama_url,
            timeout=settings.ollama_timeout_s,
            schema_version=settings.langextract_schema_version,
            langextract_enabled=langextract_enabled,
        ),
        lab=LabDeterministicStrategy(
            study_schema=schema,
            schema_version=schema.schema_version,
        ),
        narrative=NarrativeKeywordsStrategy(
            study_schema=schema,
            schema_version=schema.schema_version,
        ),
    )


def build_extraction_service(
    schema: StudySchema | None = None,
    *,
    langextract_enabled: bool = False,
    max_workers: int = 1,
) -> ExtractionService:
    schema = schema or resolve_study_schema(DEFAULT_SCHEMA_PATH)
    return ExtractionService(
        extractor_registry=build_extractor_registry(
            schema, langextract_enabled=langextract_enabled
        ),
        study_schema=schema,
        imaging_extraction_max_workers=max_workers,
    )


def write_temp_schema(tmp_path: Path, payload: dict[str, Any]) -> Path:
    p = tmp_path / "study_schema.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def registry_from_default_schema() -> FieldRegistry:
    return FieldRegistry.from_study_schema(load_study_schema_from_json(DEFAULT_SCHEMA_PATH))
