from __future__ import annotations

from pathlib import Path

from app.business_rules.mapping import FieldMappingService
from app.config.field_registry import FieldRegistry
from app.config.study_schema_provider import load_study_schema_from_json, resolve_study_schema
from app.extraction.planner import plan_extraction_jobs
from app.schemas.enums import DocumentType
from app.schemas.study_schema import ExtractionStrategy

_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "study_schema_default.json"


def test_load_study_schema_default_json() -> None:
    schema = load_study_schema_from_json(_FIXTURE)
    assert schema.study_id
    assert len(schema.fields) >= 8
    imaging = schema.extractable_fields_for_document_type(DocumentType.IMAGING_REPORT)
    assert any(f.canonical_key == "RECIST_response" for f in imaging)


def test_plan_extraction_jobs_imaging_only_subset() -> None:
    schema = load_study_schema_from_json(_FIXTURE)
    schema.fields = [f for f in schema.fields if f.canonical_key != "RECIST_response"]
    jobs = plan_extraction_jobs(schema, DocumentType.IMAGING_REPORT)
    assert len(jobs) == 1
    job = jobs[0]
    assert job.extraction_strategy == ExtractionStrategy.IMAGING_LANGEXTRACT
    assert "RECIST_response" not in job.canonical_keys
    assert job.derive_recist_from_conclusion is False


def test_mapping_by_canonical_key() -> None:
    schema = resolve_study_schema(_FIXTURE)
    registry = FieldRegistry.from_study_schema(schema)
    mapping = FieldMappingService(registry)
    from types import SimpleNamespace

    from app.extraction.imaging_langextract import extractions_to_observations

    ext = SimpleNamespace(
        extraction_class="recist_response",
        extraction_text="SD",
        attributes={},
        char_interval=SimpleNamespace(start_pos=0, end_pos=2),
    )
    obs = extractions_to_observations(
        [ext], source_chunk_id="c", model_name="m", schema_version="v1"
    )
    names = {mapping.observation_to_field_name(o) for o in obs}
    assert "Response_at_first_imaging_RECIST" in names


def test_narrative_negation_skips_cirrhosis() -> None:
    from app.extraction.lab_heuristics import (
        builtin_narrative_specs,
        extract_narrative_keyword_observations,
    )

    obs = extract_narrative_keyword_observations(
        "Le patient n'a pas de cirrhose connue.",
        extraction_method="test",
        model_name=None,
        schema_version="v1",
        source_chunk_id="c",
        specs=builtin_narrative_specs(),
        allowed_canonical_keys=frozenset({"Cirrhosis"}),
    )
    assert obs == []


def test_validate_study_schema_rejects_missing_canonical_key() -> None:
    import pytest

    from app.schemas.enums import (
        DocumentType,
        ExtractionFamily,
        FieldFamily,
        TemporalScope,
    )
    from app.schemas.study_schema import StudyFieldDefinition, StudySchema, validate_study_schema

    bad = StudySchema(
        study_id="X",
        fields=[
            StudyFieldDefinition(
                field_name="bad_field",
                field_type="numeric",
                document_types_allowed=[DocumentType.LAB_BLOOD_PANEL],
                extraction_family=ExtractionFamily.LAB_VALUES,
                field_family=FieldFamily.HEMATOLOGY,
                temporal_scope=TemporalScope.BASELINE,
                target_column="bad_field",
                extraction_strategy="lab_deterministic",
                canonical_key=None,
            )
        ],
    )
    with pytest.raises(ValueError, match="canonical_key obligatoire"):
        validate_study_schema(bad)


def test_registry_from_study_schema_roundtrip() -> None:
    schema = resolve_study_schema(_FIXTURE)
    reg = FieldRegistry.from_study_schema(schema)
    fd = reg.get("AST_start_AtezoBev_D0")
    assert fd is not None
    assert fd.canonical_key == "AST"
    assert reg.by_canonical_key("AST")[0].field_name == "AST_start_AtezoBev_D0"
