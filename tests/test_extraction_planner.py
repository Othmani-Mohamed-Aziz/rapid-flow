"""Tests unitaires : ExtractionPlanner / jobs."""

from __future__ import annotations

from app.config.study_schema_provider import load_study_schema_from_json
from app.extraction.planner import field_families_from_jobs, plan_extraction_jobs
from app.schemas.enums import DocumentType, FieldFamily
from app.schemas.study_schema import ExtractionStrategy
from tests.extraction_fixtures import DEFAULT_SCHEMA_PATH, build_custom_lab_schema

_MINIMAL = (
    __import__("pathlib").Path(__file__).resolve().parent / "fixtures" / "study_schema_minimal.json"
)


def test_plan_lab_jobs_for_blood_panel() -> None:
    schema = load_study_schema_from_json(DEFAULT_SCHEMA_PATH)
    jobs = plan_extraction_jobs(schema, DocumentType.LAB_BLOOD_PANEL)
    strategies = {j.extraction_strategy for j in jobs}
    assert ExtractionStrategy.LAB_DETERMINISTIC in strategies
    families = field_families_from_jobs(jobs)
    assert FieldFamily.HEPATIC_BIOCHEMISTRY in families


def test_plan_imaging_job_includes_only_requested_classes() -> None:
    schema = load_study_schema_from_json(DEFAULT_SCHEMA_PATH)
    schema.fields = [
        f
        for f in schema.fields
        if f.canonical_key == "Size_major_nodule_mm"
        and DocumentType.IMAGING_REPORT in f.document_types_allowed
    ]
    jobs = plan_extraction_jobs(schema, DocumentType.IMAGING_REPORT)
    assert len(jobs) == 1
    job = jobs[0]
    assert "lesion_size_mm" in job.langextract_classes
    assert "imaging_conclusion" not in job.langextract_classes
    assert job.derive_recist_from_conclusion is False


def test_plan_imaging_recist_adds_conclusion_class_for_derivation() -> None:
    schema = load_study_schema_from_json(DEFAULT_SCHEMA_PATH)
    jobs = plan_extraction_jobs(schema, DocumentType.IMAGING_REPORT)
    recist_jobs = [j for j in jobs if "RECIST_response" in j.canonical_keys]
    assert recist_jobs
    assert recist_jobs[0].derive_recist_from_conclusion is True
    assert "imaging_conclusion" in recist_jobs[0].langextract_classes


def test_plan_no_jobs_for_pathology_when_not_in_schema() -> None:
    schema = load_study_schema_from_json(_MINIMAL)
    jobs = plan_extraction_jobs(schema, DocumentType.PATHOLOGY_REPORT)
    assert jobs == []


def test_plan_custom_ggt_lab_job() -> None:
    schema = build_custom_lab_schema()
    jobs = plan_extraction_jobs(schema, DocumentType.LAB_BLOOD_PANEL)
    assert len(jobs) == 1
    assert jobs[0].canonical_keys == ["GGT"]


def test_field_families_from_jobs_sorted() -> None:
    schema = load_study_schema_from_json(DEFAULT_SCHEMA_PATH)
    jobs = plan_extraction_jobs(schema, DocumentType.LAB_BLOOD_PANEL)
    fams = field_families_from_jobs(jobs)
    assert fams == sorted(fams, key=lambda f: f.value)
