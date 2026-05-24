from __future__ import annotations

from app.business_rules.mapping import FieldMappingService
from app.config.field_registry import get_default_registry
from app.extraction.imaging_langextract import CANONICAL_LESION_SIZE_MM, CANONICAL_RECIST_RESPONSE
from app.extraction.lab_heuristics import extract_lab_observations
from app.schemas.enums import DocumentType, FieldFamily
from app.schemas.models import ExtractedObservation


def test_lab_mapping_to_registry_fields() -> None:
    registry = get_default_registry()
    mapper = FieldMappingService(registry)
    text = "AST 40 U/L, ALT 30 U/L, plaquettes 180 G/L, bilirubine totale 18 µmol/L, AFP 12 UI/mL"
    obs = extract_lab_observations(
        text,
        extraction_method="test",
        model_name=None,
        schema_version=None,
        source_chunk_id="c1",
    )
    candidates = mapper.build_candidates(doc_type=DocumentType.LAB_BLOOD_PANEL, observations=obs)
    names = {c.field_name for c in candidates}
    assert "AST_start_AtezoBev_D0" in names
    assert "ALT_start_AtezoBev_D0" in names
    assert "PLT_start_AtezoBev_D0" in names
    assert "Total_bilirubine_D0" in names
    assert "AFP_start_AtezoBev_D0" in names
    for c in candidates:
        assert c.observation.field_family in FieldFamily


def test_imaging_mapping_to_registry_fields() -> None:
    registry = get_default_registry()
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
            extra={"canonical_imaging_key": CANONICAL_LESION_SIZE_MM},
        ),
        ExtractedObservation(
            observation_id="o2",
            field_family=FieldFamily.IMAGING_RECIST,
            normalized_value="SD",
            confidence=0.8,
            evidence_text="maladie stable",
            extraction_method="test",
            extra={"canonical_imaging_key": CANONICAL_RECIST_RESPONSE},
        ),
    ]
    candidates = mapper.build_candidates(doc_type=DocumentType.IMAGING_REPORT, observations=obs)
    names = {c.field_name for c in candidates}
    assert "Size_major_nodule_mm_start_AtezoBev_D0" in names
    assert "Response_at_first_imaging_RECIST" in names
