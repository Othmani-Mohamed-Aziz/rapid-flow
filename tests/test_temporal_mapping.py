from __future__ import annotations

from datetime import date

from app.business_rules.temporal import TemporalMappingService
from app.config.field_registry import get_default_registry
from app.schemas.enums import FieldFamily, TemporalLabel, TemporalScope
from app.schemas.models import ExtractedObservation, ParsedDocument


def test_temporal_baseline_from_scope() -> None:
    registry = get_default_registry()
    svc = TemporalMappingService()
    fd = registry.get("AST_start_AtezoBev_D0")
    assert fd is not None
    parsed = ParsedDocument(
        document_id="d1",
        patient_id="p1",
        study_id="s1",
        full_text="",
        document_date=date(2026, 1, 10),
    )
    obs = ExtractedObservation(
        observation_id="o1",
        field_family=FieldFamily.HEPATIC_BIOCHEMISTRY,
        raw_value="40",
        normalized_value=40,
        confidence=0.9,
        evidence_text="AST 40",
        extraction_method="test",
    )
    updated = svc.apply_for_field(parsed, obs, fd)
    assert updated.temporal_label == TemporalLabel.BASELINE
    assert updated.event_date == date(2026, 1, 10)
    anchor = svc.build_anchor(fd, parsed)
    assert anchor.scope == TemporalScope.BASELINE
