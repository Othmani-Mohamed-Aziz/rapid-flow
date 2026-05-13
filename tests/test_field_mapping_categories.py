from __future__ import annotations

from app.business_rules.mapping import FieldMappingService
from app.config.field_registry import get_default_registry
from app.extraction.lab_heuristics import extract_lab_observations
from app.schemas.enums import DocumentType, FieldFamily


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
