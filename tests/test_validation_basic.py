from __future__ import annotations

from app.business_rules.validation import ValidationService
from app.config.field_registry import get_default_registry
from app.schemas.enums import FieldFamily
from app.schemas.models import EcrfCellUpdate, ExtractedObservation, FieldCandidate


def test_validation_accepts_numeric_lab_field() -> None:
    registry = get_default_registry()
    val = ValidationService(registry)
    obs = ExtractedObservation(
        observation_id="o1",
        field_family=FieldFamily.HEPATIC_BIOCHEMISTRY,
        normalized_value=42,
        confidence=0.9,
        evidence_text="AST 42",
        extraction_method="test",
    )
    cand = FieldCandidate(
        field_name="AST_start_AtezoBev_D0",
        observation=obs,
        target_column="AST_start_AtezoBev_D0",
        confidence=0.9,
    )
    upd = EcrfCellUpdate(
        patient_id="p1",
        study_id="s1",
        column_key="AST_start_AtezoBev_D0",
        value=42,
        provenance=cand,
        validated=False,
    )
    ok, errors = val.validate_update(upd)
    assert ok
    assert errors == []


def test_validation_rejects_non_numeric_for_numeric_field() -> None:
    registry = get_default_registry()
    val = ValidationService(registry)
    obs = ExtractedObservation(
        observation_id="o1",
        field_family=FieldFamily.HEPATIC_BIOCHEMISTRY,
        normalized_value="high",
        confidence=0.9,
        evidence_text="AST high",
        extraction_method="test",
    )
    cand = FieldCandidate(
        field_name="AST_start_AtezoBev_D0",
        observation=obs,
        target_column="AST_start_AtezoBev_D0",
        confidence=0.9,
    )
    upd = EcrfCellUpdate(
        patient_id="p1",
        study_id="s1",
        column_key="AST_start_AtezoBev_D0",
        value="high",
        provenance=cand,
        validated=False,
    )
    ok, errors = val.validate_update(upd)
    assert not ok
    assert errors
