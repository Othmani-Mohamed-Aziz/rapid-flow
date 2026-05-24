"""Tests unitaires : ImagingExtractionConfig."""

from __future__ import annotations

from app.config.study_schema_provider import load_study_schema_from_json
from app.extraction.imaging_config import ImagingExtractionConfig
from app.extraction.planner import plan_extraction_jobs
from app.schemas.enums import DocumentType


def test_from_job_uses_schema_class_map() -> None:
    schema = load_study_schema_from_json(
        __import__("pathlib").Path(__file__).resolve().parent
        / "fixtures"
        / "study_schema_minimal.json"
    )
    jobs = plan_extraction_jobs(schema, DocumentType.IMAGING_REPORT)
    cfg = ImagingExtractionConfig.from_job(jobs[0], schema)
    assert "lesion_size_mm" in cfg.allowed_langextract_classes
    assert cfg.class_to_canonical.get("lesion_size_mm") == "Size_major_nodule_mm"
    assert "custom_size" in cfg.class_descriptions or "lesion_size_mm" in cfg.class_descriptions


def test_legacy_default_includes_all_standard_classes() -> None:
    cfg = ImagingExtractionConfig.legacy_default()
    assert "recist_response" in cfg.allowed_langextract_classes
    assert cfg.derive_recist_from_conclusion is True
