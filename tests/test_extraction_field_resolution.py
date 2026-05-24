"""Tests unitaires : désambiguïsation temporelle des champs."""

from __future__ import annotations

from datetime import date

from app.business_rules.field_resolution import (
    preferred_temporal_scopes,
    select_fields_for_observation,
)
from app.schemas.enums import DocumentType, ExtractionFamily, FieldFamily, TemporalScope
from app.schemas.models import FieldDefinition, ParsedDocument
from tests.extraction_fixtures import build_dual_ast_schema


def test_preferred_scopes_imaging_includes_first_imaging() -> None:
    scopes = preferred_temporal_scopes(DocumentType.IMAGING_REPORT, None)
    assert TemporalScope.FIRST_IMAGING in scopes


def test_select_fields_prefers_baseline_on_lab_panel() -> None:
    schema = build_dual_ast_schema()
    defs = schema.to_field_definitions()
    matched = [d for d in defs if d.canonical_key == "AST"]
    selected = select_fields_for_observation(
        matched,
        doc_type=DocumentType.LAB_BLOOD_PANEL,
        parsed=None,
    )
    names = {f.field_name for f in selected}
    assert "AST_baseline" in names
    assert "AST_followup" not in names


def test_select_fields_single_match_unchanged() -> None:
    fd = FieldDefinition(
        field_name="only",
        field_type="numeric",
        document_types_allowed=[DocumentType.LAB_BLOOD_PANEL],
        extraction_family=ExtractionFamily.LAB_VALUES,
        field_family=FieldFamily.HEMATOLOGY,
        temporal_scope=TemporalScope.BASELINE,
        target_column="only",
        canonical_key="PLT",
    )
    out = select_fields_for_observation([fd], doc_type=DocumentType.LAB_BLOOD_PANEL)
    assert out == [fd]


def test_select_fields_uses_document_hint_for_imaging() -> None:
    schema = build_dual_ast_schema()
    defs = schema.to_field_definitions()
    parsed = ParsedDocument(
        document_id="d1",
        patient_id="p",
        study_id="s",
        full_text="",
        document_type_hint=DocumentType.IMAGING_REPORT,
        document_date=date(2026, 1, 1),
    )
    matched = defs
    selected = select_fields_for_observation(
        matched,
        doc_type=DocumentType.CLINICAL_LETTER,
        parsed=parsed,
    )
    assert selected
