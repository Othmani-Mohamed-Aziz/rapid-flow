"""Tests unitaires : NormalizationService."""

from __future__ import annotations

from app.business_rules.normalization import NormalizationService
from app.schemas.enums import DocumentType, ExtractionFamily, FieldFamily, TemporalScope
from app.schemas.models import ExtractedObservation, FieldDefinition


def _field(**kwargs) -> FieldDefinition:
    base = dict(
        field_name="Response_at_first_imaging_RECIST",
        field_type="categorical",
        document_types_allowed=[DocumentType.IMAGING_REPORT],
        extraction_family=ExtractionFamily.IMAGING_RECIST,
        field_family=FieldFamily.IMAGING_RECIST,
        temporal_scope=TemporalScope.FIRST_IMAGING,
        target_column="Response_at_first_imaging_RECIST",
        autofill_threshold=0.75,
        canonical_key="RECIST_response",
    )
    base.update(kwargs)
    return FieldDefinition(**base)


def test_recist_category_normalizes_stabilite() -> None:
    svc = NormalizationService()
    obs = ExtractedObservation(
        observation_id="o1",
        field_family=FieldFamily.IMAGING_RECIST,
        normalized_value="stabilité tumorale",
        confidence=0.8,
        evidence_text="x",
        extraction_method="test",
        extra={"canonical_imaging_key": "RECIST_response"},
    )
    fd = _field(normalization_rule="recist_category")
    out = svc.apply(obs, fd)
    assert out.normalized_value == "SD"


def test_recist_category_rejects_non_canon_free_text() -> None:
    svc = NormalizationService()
    obs = ExtractedObservation(
        observation_id="o2",
        field_family=FieldFamily.IMAGING_RECIST,
        normalized_value="texte libre sans code",
        confidence=0.8,
        evidence_text="x",
        extraction_method="test",
        extra={"canonical_imaging_key": "RECIST_response"},
    )
    fd = _field(normalization_rule="recist_category")
    out = svc.apply(obs, fd)
    assert out.normalized_value == "texte libre sans code"


def test_numeric_mm_normalization() -> None:
    svc = NormalizationService()
    obs = ExtractedObservation(
        observation_id="o3",
        field_family=FieldFamily.IMAGING_RECIST,
        normalized_value="45,5",
        confidence=0.8,
        evidence_text="45,5 mm",
        extraction_method="test",
        extra={"canonical_imaging_key": "Size_major_nodule_mm"},
    )
    fd = FieldDefinition(
        field_name="Size_major_nodule_mm_start_AtezoBev_D0",
        field_type="numeric",
        document_types_allowed=[DocumentType.IMAGING_REPORT],
        extraction_family=ExtractionFamily.IMAGING_RECIST,
        field_family=FieldFamily.IMAGING_RECIST,
        temporal_scope=TemporalScope.BASELINE,
        normalization_rule="numeric_mm",
        target_column="Size_major_nodule_mm_start_AtezoBev_D0",
        autofill_threshold=0.7,
        canonical_key="Size_major_nodule_mm",
    )
    out = svc.apply(obs, fd)
    assert out.normalized_value == 45.5
    assert out.unit == "mm"
