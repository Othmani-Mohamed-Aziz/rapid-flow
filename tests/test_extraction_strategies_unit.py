"""Tests unitaires : stratégies d'extraction (Lab / Narrative / Imaging mock)."""

from __future__ import annotations

from unittest.mock import patch

from app.extraction.planner import plan_extraction_jobs
from app.extraction.strategy_extractors import (
    ImagingLangextractStrategy,
    LabDeterministicStrategy,
    NarrativeKeywordsStrategy,
)
from app.schemas.enums import DocumentType, FieldFamily
from tests.extraction_fixtures import (
    DEFAULT_SCHEMA_PATH,
    build_custom_lab_schema,
    resolve_study_schema,
)


def test_lab_strategy_respects_job_family_filter() -> None:
    schema = resolve_study_schema(DEFAULT_SCHEMA_PATH)
    job = plan_extraction_jobs(schema, DocumentType.LAB_BLOOD_PANEL)[0]
    strat = LabDeterministicStrategy(study_schema=schema, schema_version="v1")
    obs = strat.extract_chunk(
        "AST 40 U/L",
        job,
        source_chunk_id="c",
        retrieval_hit_score=0.9,
    )
    assert all(o.field_family == job.field_family for o in obs)


def test_narrative_strategy_empty_when_key_not_allowed() -> None:
    schema = resolve_study_schema(DEFAULT_SCHEMA_PATH)
    jobs = [
        j
        for j in plan_extraction_jobs(schema, DocumentType.LAB_BLOOD_PANEL)
        if j.field_family == FieldFamily.COMORBIDITIES
    ]
    if not jobs:
        return
    job = jobs[0]
    job = job.model_copy(update={"canonical_keys": ["OtherKey"]})
    strat = NarrativeKeywordsStrategy(study_schema=schema, schema_version="v1")
    obs = strat.extract_chunk("cirrhose", job, source_chunk_id="c", retrieval_hit_score=None)
    assert obs == []


@patch("app.extraction.strategy_extractors.run_imaging_langextract", return_value=[])
def test_imaging_strategy_disabled_returns_empty(mock_lx) -> None:
    schema = resolve_study_schema(DEFAULT_SCHEMA_PATH)
    job = plan_extraction_jobs(schema, DocumentType.IMAGING_REPORT)[0]
    strat = ImagingLangextractStrategy(
        study_schema=schema,
        model_id="m",
        model_url="http://localhost:11434",
        timeout=30,
        schema_version="v1",
        langextract_enabled=False,
    )
    obs = strat.extract_chunk(
        "nodule 12 mm",
        job,
        source_chunk_id="c",
        retrieval_hit_score=0.8,
    )
    assert obs == []
    mock_lx.assert_not_called()


def test_lab_strategy_custom_catalog_ggt() -> None:
    schema = build_custom_lab_schema()
    job = plan_extraction_jobs(schema, DocumentType.LAB_BLOOD_PANEL)[0]
    strat = LabDeterministicStrategy(study_schema=schema, schema_version="v1")
    obs = strat.extract_chunk("GGT 33", job, source_chunk_id="c", retrieval_hit_score=None)
    assert obs[0].extra["canonical_lab_key"] == "GGT"
