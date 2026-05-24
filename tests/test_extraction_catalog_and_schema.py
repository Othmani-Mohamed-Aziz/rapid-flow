"""Tests unitaires : catalogue d'extraction, StudySchema, validation, provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config.study_schema_provider import (
    load_study_schema_from_json,
    resolve_study_schema,
    study_schema_from_example_fields,
)
from app.schemas.enums import DocumentType, ExtractionFamily, FieldFamily, TemporalScope
from app.schemas.extraction_catalog import LabAnalyteSpec
from app.schemas.study_schema import (
    DEFAULT_LANGEXTRACT_CLASS_TO_CANONICAL,
    StudyFieldDefinition,
    StudySchema,
    validate_study_schema,
)
from tests.extraction_fixtures import DEFAULT_SCHEMA_PATH, write_temp_schema

_MINIMAL = Path(__file__).resolve().parent / "fixtures" / "study_schema_minimal.json"


def test_lab_analyte_spec_validates_regex() -> None:
    spec = LabAnalyteSpec(
        canonical_key="X",
        pattern=r"\bX\b\s*([0-9]+)",
        field_family=FieldFamily.HEMATOLOGY,
    )
    assert spec.compiled_pattern().search("X 12")


def test_lab_analyte_spec_invalid_regex_raises() -> None:
    with pytest.raises(Exception):
        LabAnalyteSpec(
            canonical_key="X",
            pattern=r"(?P<unclosed",
            field_family=FieldFamily.HEMATOLOGY,
        )


def test_effective_langextract_map_merges_catalog() -> None:
    schema = load_study_schema_from_json(_MINIMAL)
    merged = schema.effective_langextract_class_map()
    assert merged["lesion_size_mm"] == "Size_major_nodule_mm"
    assert merged["custom_size"] == "Size_major_nodule_mm"


def test_resolved_lab_analytes_uses_catalog_when_present() -> None:
    schema = load_study_schema_from_json(_MINIMAL)
    specs = schema.resolved_lab_analytes()
    keys = {s.canonical_key for s in specs}
    assert "GGT" in keys


def test_lab_analytes_for_keys_filters() -> None:
    schema = load_study_schema_from_json(_MINIMAL)
    subset = schema.lab_analytes_for_keys(frozenset({"GGT"}))
    assert len(subset) == 1
    assert subset[0].canonical_key == "GGT"


def test_load_minimal_json_catalog_and_validate() -> None:
    schema = load_study_schema_from_json(_MINIMAL)
    assert schema.study_id == "MINIMAL_TEST"
    assert schema.extraction_catalog.narrative_keywords
    validate_study_schema(schema)


def test_resolve_study_schema_from_temp_path(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    path = write_temp_schema(tmp_path, payload)
    schema = resolve_study_schema(path)
    assert len(schema.fields) >= 8


def test_validate_rejects_imaging_without_class_mapping() -> None:
    schema = StudySchema(
        study_id="BAD_IMG",
        fields=[
            StudyFieldDefinition(
                field_name="unknown_img",
                field_type="numeric",
                document_types_allowed=[DocumentType.IMAGING_REPORT],
                extraction_family=ExtractionFamily.IMAGING_RECIST,
                field_family=FieldFamily.IMAGING_RECIST,
                temporal_scope=TemporalScope.BASELINE,
                target_column="unknown_img",
                canonical_key="TotallyUnknownKey",
                extraction_strategy="imaging_langextract",
                langextract_class=None,
            )
        ],
    )
    with pytest.raises(ValueError, match="langextract_class"):
        validate_study_schema(schema)


def test_study_schema_from_example_fields_validates() -> None:
    schema = study_schema_from_example_fields()
    validate_study_schema(schema)
    assert schema.fields
    assert "imaging_recist" in schema.family_retrieval_queries


def test_to_field_definitions_serializes_strategy() -> None:
    schema = load_study_schema_from_json(DEFAULT_SCHEMA_PATH)
    defs = schema.to_field_definitions()
    ast = next(d for d in defs if d.field_name.startswith("AST"))
    assert ast.canonical_key == "AST"
    assert ast.extraction_strategy == "lab_deterministic"


def test_default_langextract_map_covers_imaging_keys() -> None:
    assert DEFAULT_LANGEXTRACT_CLASS_TO_CANONICAL["recist_response"] == "RECIST_response"
