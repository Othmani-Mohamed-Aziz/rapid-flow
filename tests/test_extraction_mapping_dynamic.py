"""Tests unitaires : FieldMappingService dynamique + normalisation par champ."""

from __future__ import annotations

from app.business_rules.mapping import FieldMappingService
from app.business_rules.normalization import NormalizationService
from app.config.field_registry import FieldRegistry
from app.schemas.enums import DocumentType, FieldFamily
from app.schemas.models import ExtractedObservation
from tests.extraction_fixtures import build_dual_ast_schema, registry_from_default_schema


def test_build_candidates_maps_recist_and_size() -> None:
    registry = registry_from_default_schema()
    mapper = FieldMappingService(registry)
    obs = [
        ExtractedObservation(
            observation_id="o1",
            field_family=FieldFamily.IMAGING_RECIST,
            normalized_value=28,
            unit="mm",
            confidence=0.8,
            evidence_text="28 mm",
            extraction_method="test",
            extra={"canonical_imaging_key": "Size_major_nodule_mm"},
        ),
        ExtractedObservation(
            observation_id="o2",
            field_family=FieldFamily.IMAGING_RECIST,
            normalized_value="SD",
            confidence=0.8,
            evidence_text="stable",
            extraction_method="test",
            extra={"canonical_imaging_key": "RECIST_response"},
        ),
    ]
    cands = mapper.build_candidates(doc_type=DocumentType.IMAGING_REPORT, observations=obs)
    names = {c.field_name for c in cands}
    assert "Size_major_nodule_mm_start_AtezoBev_D0" in names
    assert "Response_at_first_imaging_RECIST" in names


def test_build_candidates_dual_ast_selects_baseline_only() -> None:
    schema = build_dual_ast_schema()
    registry = FieldRegistry.from_study_schema(schema)
    mapper = FieldMappingService(registry)
    obs = [
        ExtractedObservation(
            observation_id="o",
            field_family=FieldFamily.HEPATIC_BIOCHEMISTRY,
            normalized_value=40,
            confidence=0.8,
            evidence_text="AST 40",
            extraction_method="test",
            extra={"canonical_lab_key": "AST"},
        )
    ]
    norm = NormalizationService()
    cands = mapper.build_candidates(
        doc_type=DocumentType.LAB_BLOOD_PANEL,
        observations=obs,
        normalize_fn=norm.apply,
    )
    assert len(cands) == 1
    assert cands[0].field_name == "AST_baseline"


def test_build_candidates_applies_normalize_fn_per_field() -> None:
    registry = registry_from_default_schema()
    mapper = FieldMappingService(registry)
    obs = [
        ExtractedObservation(
            observation_id="o",
            field_family=FieldFamily.IMAGING_RECIST,
            normalized_value="maladie stable",
            confidence=0.8,
            evidence_text="maladie stable",
            extraction_method="test",
            extra={"canonical_imaging_key": "RECIST_response"},
        )
    ]
    norm = NormalizationService()
    cands = mapper.build_candidates(
        doc_type=DocumentType.IMAGING_REPORT,
        observations=obs,
        normalize_fn=norm.apply,
    )
    assert cands[0].observation.normalized_value == "SD"
