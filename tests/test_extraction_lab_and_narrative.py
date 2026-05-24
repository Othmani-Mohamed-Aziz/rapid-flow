"""Tests unitaires : extracteurs lab déterministe et narratif."""

from __future__ import annotations

from app.extraction.lab_heuristics import (
    builtin_lab_analyte_specs,
    extract_comorbidity_observations,
    extract_lab_observations,
    extract_narrative_keyword_observations,
)
from app.schemas.extraction_catalog import NarrativeKeywordSpec
from tests.extraction_fixtures import build_custom_lab_schema


def test_lab_extract_ast_from_text() -> None:
    obs = extract_lab_observations(
        "AST 40 U/L",
        extraction_method="test",
        model_name=None,
        schema_version="v1",
        source_chunk_id="c1",
        allowed_canonical_keys=frozenset({"AST"}),
    )
    assert len(obs) == 1
    assert obs[0].extra["canonical_lab_key"] == "AST"
    assert obs[0].normalized_value == 40


def test_lab_respects_allowed_canonical_keys() -> None:
    text = "AST 40 U/L, AFP 12 UI/mL"
    obs = extract_lab_observations(
        text,
        extraction_method="test",
        model_name=None,
        schema_version="v1",
        source_chunk_id=None,
        allowed_canonical_keys=frozenset({"AFP"}),
    )
    assert len(obs) == 1
    assert obs[0].extra["canonical_lab_key"] == "AFP"


def test_lab_custom_ggt_from_schema_catalog() -> None:
    schema = build_custom_lab_schema()
    specs = schema.lab_analytes_for_keys(frozenset({"GGT"}))
    obs = extract_lab_observations(
        "GGT 55 U/L",
        extraction_method="test",
        model_name=None,
        schema_version=schema.schema_version,
        source_chunk_id="c",
        analyte_specs=specs,
    )
    assert obs[0].normalized_value == 55


def test_narrative_positive_cirrhosis() -> None:
    obs = extract_comorbidity_observations(
        "Antécédents : cirrhose connue.",
        extraction_method="test",
        model_name=None,
        schema_version="v1",
        source_chunk_id="c",
    )
    assert len(obs) == 1
    assert obs[0].extra.get("canonical_key") == "Cirrhosis"


def test_narrative_custom_keywords() -> None:
    spec = NarrativeKeywordSpec(canonical_key="Diabetes", keywords=["diabète", "diabetes"])
    obs = extract_narrative_keyword_observations(
        "Patient avec diabète de type 2.",
        extraction_method="test",
        model_name=None,
        schema_version="v1",
        source_chunk_id="c",
        specs=[spec],
        allowed_canonical_keys=frozenset({"Diabetes"}),
    )
    assert len(obs) == 1
    assert obs[0].extra["canonical_key"] == "Diabetes"


def test_builtin_lab_specs_cover_default_analytes() -> None:
    keys = {s.canonical_key for s in builtin_lab_analyte_specs()}
    assert {"AST", "ALT", "AFP", "PLT"}.issubset(keys)
