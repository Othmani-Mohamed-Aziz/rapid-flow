"""Tests unitaires : ExtractionService (orchestration jobs / dédup)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.extraction.imaging_langextract import LangExtractNotInstalledError
from app.extraction.service import ExtractionService
from app.schemas.enums import DocumentType, FieldFamily
from app.schemas.models import ExtractedObservation
from app.schemas.study_schema import ExtractionStrategy
from tests.extraction_fixtures import (
    DEFAULT_SCHEMA_PATH,
    build_extraction_service,
    build_extractor_registry,
    make_retrieval_hit,
    resolve_study_schema,
)


def test_extract_all_for_document_lab_from_hits() -> None:
    schema = resolve_study_schema(DEFAULT_SCHEMA_PATH)
    svc = build_extraction_service(schema)
    hits = {
        FieldFamily.HEPATIC_BIOCHEMISTRY: [
            make_retrieval_hit(
                chunk_id="c1",
                text="AST 42 U/L, ALT 30 U/L",
                family=FieldFamily.HEPATIC_BIOCHEMISTRY,
            )
        ],
        FieldFamily.INFLAMMATION_BIOMARKERS: [
            make_retrieval_hit(
                chunk_id="c2",
                text="AFP 10 UI/mL",
                family=FieldFamily.INFLAMMATION_BIOMARKERS,
            )
        ],
        FieldFamily.HEMATOLOGY: [
            make_retrieval_hit(
                chunk_id="c3",
                text="Plaquettes 180 G/L",
                family=FieldFamily.HEMATOLOGY,
            )
        ],
        FieldFamily.COMORBIDITIES: [
            make_retrieval_hit(
                chunk_id="c4",
                text="cirrhose connue",
                family=FieldFamily.COMORBIDITIES,
            )
        ],
    }
    obs = svc.extract_all_for_document(doc_type=DocumentType.LAB_BLOOD_PANEL, family_hits=hits)
    lab_keys = {(o.extra or {}).get("canonical_lab_key") for o in obs}
    assert "AST" in lab_keys
    assert "AFP" in lab_keys


def test_dedupe_same_canon_and_evidence() -> None:
    schema = resolve_study_schema(DEFAULT_SCHEMA_PATH)
    svc = build_extraction_service(schema)
    dup = ExtractedObservation(
        observation_id="a",
        field_family=FieldFamily.HEPATIC_BIOCHEMISTRY,
        normalized_value=1,
        confidence=0.5,
        evidence_text="AST 1",
        extraction_method="t",
        extra={"canonical_lab_key": "AST"},
    )
    out = svc._dedupe(
        FieldFamily.HEPATIC_BIOCHEMISTRY, [dup, dup.model_copy(update={"observation_id": "b"})]
    )
    assert len(out) == 1


def test_parallel_imaging_aggregates_chunks() -> None:
    schema = resolve_study_schema(DEFAULT_SCHEMA_PATH)
    mock_imaging = MagicMock()
    mock_imaging.extract_chunk.side_effect = [
        [
            ExtractedObservation(
                observation_id="1",
                field_family=FieldFamily.IMAGING_RECIST,
                normalized_value=10,
                confidence=0.8,
                evidence_text="10 mm",
                extraction_method="t",
                extra={"canonical_imaging_key": "Size_major_nodule_mm"},
            )
        ],
        [
            ExtractedObservation(
                observation_id="2",
                field_family=FieldFamily.IMAGING_RECIST,
                normalized_value=20,
                confidence=0.8,
                evidence_text="20 mm",
                extraction_method="t",
                extra={"canonical_imaging_key": "Size_major_nodule_mm"},
            )
        ],
    ]
    registry = build_extractor_registry(schema)
    registry._imaging = mock_imaging  # noqa: SLF001
    svc = ExtractionService(
        extractor_registry=registry,
        study_schema=schema,
        imaging_extraction_max_workers=2,
    )
    hits = [
        make_retrieval_hit(chunk_id="a", text="chunk A"),
        make_retrieval_hit(chunk_id="b", text="chunk B", rank=2, score=0.8),
    ]
    jobs = svc.plan_jobs(DocumentType.IMAGING_REPORT)
    job = next(j for j in jobs if j.extraction_strategy == ExtractionStrategy.IMAGING_LANGEXTRACT)
    obs = svc.extract_for_job(job, hits)
    assert len(obs) == 2
    assert mock_imaging.extract_chunk.call_count == 2


def test_langextract_not_installed_propagates() -> None:
    schema = resolve_study_schema(DEFAULT_SCHEMA_PATH)
    mock_imaging = MagicMock()
    mock_imaging.extract_chunk.side_effect = LangExtractNotInstalledError("missing")
    registry = build_extractor_registry(schema)
    registry._imaging = mock_imaging  # noqa: SLF001
    svc = ExtractionService(
        extractor_registry=registry,
        study_schema=schema,
        imaging_extraction_max_workers=2,
    )
    hits = [
        make_retrieval_hit(chunk_id="a", text="t1"),
        make_retrieval_hit(chunk_id="b", text="t2", rank=2),
    ]
    jobs = svc.plan_jobs(DocumentType.IMAGING_REPORT)
    job = jobs[0]
    with pytest.raises(LangExtractNotInstalledError):
        svc.extract_for_job(job, hits)
