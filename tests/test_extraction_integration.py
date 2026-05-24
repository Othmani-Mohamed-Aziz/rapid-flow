"""Tests d'intégration : phase extraction (pipeline + schéma d'étude)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.config.field_registry import FieldRegistry
from app.config.settings import Settings
from app.config.study_schema_provider import load_study_schema_from_json
from app.extraction.planner import field_families_from_jobs, plan_extraction_jobs
from app.orchestration.pipeline import PipelineOrchestrator
from app.schemas.enums import DocumentType, FieldFamily
from tests.extraction_fixtures import (
    DEFAULT_SCHEMA_PATH,
    MOCK_BLOOD_PANEL_TXT,
    fake_langextract_extraction,
    resolve_study_schema,
)

pytestmark = pytest.mark.extraction_integration


def test_pipeline_lab_end_to_end_with_default_schema() -> None:
    if not MOCK_BLOOD_PANEL_TXT.is_file():
        pytest.skip(f"Fixture manquant : {MOCK_BLOOD_PANEL_TXT}")
    result = PipelineOrchestrator(settings=Settings(langextract_enabled=False)).run(
        str(MOCK_BLOOD_PANEL_TXT),
        patient_id="P-LAB",
        study_id="EXAMPLE",
    )
    assert result.document_type == DocumentType.LAB_BLOOD_PANEL
    assert result.metadata.get("study_schema_version")
    lab_keys = set()
    for o in result.observations:
        if ck := (o.extra or {}).get("canonical_lab_key"):
            lab_keys.add(ck)
    assert "AST" in lab_keys
    columns = {u.column_key for u in result.cell_updates}
    assert any("AST" in c for c in columns)


@patch("app.extraction.strategy_extractors.run_imaging_langextract")
def test_pipeline_imaging_mocked_langextract(mock_run_lx) -> None:
    from app.extraction.imaging_langextract import extractions_to_observations

    mock_run_lx.return_value = extractions_to_observations(
        [
            fake_langextract_extraction(
                extraction_class="lesion_size_mm",
                extraction_text="45 mm",
                attributes={"value_mm": 45},
            ),
            fake_langextract_extraction(
                extraction_class="recist_response",
                extraction_text="maladie stable",
            ),
        ],
        source_chunk_id="mock-chunk",
        model_name="test-model",
        schema_version="test-v1",
        retrieval_hit_score=0.9,
    )

    root = Path(__file__).resolve().parents[1]
    ct_pdf = root / "data" / "ct_scan_report_liver.pdf"
    if not ct_pdf.is_file():
        pytest.skip(f"PDF CT manquant : {ct_pdf}")

    schema = resolve_study_schema(DEFAULT_SCHEMA_PATH)
    registry = FieldRegistry.from_study_schema(schema)
    orch = PipelineOrchestrator(
        settings=Settings(langextract_enabled=True),
        registry=registry,
        study_schema=schema,
    )
    result = orch.run(str(ct_pdf), patient_id="P-IMG", study_id=schema.study_id)
    assert result.document_type == DocumentType.IMAGING_REPORT
    mock_run_lx.assert_called()
    imaging_keys = {(o.extra or {}).get("canonical_imaging_key") for o in result.observations}
    assert "Size_major_nodule_mm" in imaging_keys or "RECIST_response" in imaging_keys


def test_pipeline_retrieval_families_match_extraction_jobs_only() -> None:
    """Vérifie que seules les familles des jobs planifiés sont utilisées (pas de fallback registre)."""
    schema = load_study_schema_from_json(
        Path(__file__).resolve().parent / "fixtures" / "study_schema_minimal.json"
    )
    jobs = plan_extraction_jobs(schema, DocumentType.PATHOLOGY_REPORT)
    assert jobs == []
    fams = field_families_from_jobs(jobs)
    assert fams == []


def test_pipeline_study_id_mismatch_still_runs(caplog) -> None:
    if not MOCK_BLOOD_PANEL_TXT.is_file():
        pytest.skip(f"Fixture manquant : {MOCK_BLOOD_PANEL_TXT}")
    schema = resolve_study_schema(DEFAULT_SCHEMA_PATH)
    with caplog.at_level("WARNING"):
        result = PipelineOrchestrator(
            settings=Settings(langextract_enabled=False),
            study_schema=schema,
        ).run(str(MOCK_BLOOD_PANEL_TXT), patient_id="P1", study_id="OTHER_STUDY")
    assert result.observations
    assert any("study_schema.study_id" in r.message for r in caplog.records)


def test_workflow_retrieval_uses_schema_queries() -> None:
    from app.extraction.planner import plan_extraction_jobs
    from app.orchestration.pipeline import PipelineOrchestrator
    from app.retrieval.workflow_orchestrator import LocalWorkflowOrchestrator
    from app.schemas.models import DocumentChunk

    schema = resolve_study_schema(DEFAULT_SCHEMA_PATH)
    orch = PipelineOrchestrator(
        settings=Settings(langextract_enabled=False),
        workflow=LocalWorkflowOrchestrator(),
        study_schema=schema,
    )
    doc_type = DocumentType.LAB_BLOOD_PANEL
    jobs = plan_extraction_jobs(schema, doc_type)
    families = field_families_from_jobs(jobs)
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_id="d1",
            text="AST 50 U/L",
            field_family=FieldFamily.HEPATIC_BIOCHEMISTRY,
        )
    ]
    queries = {fam: schema.retrieval_query_for_family(fam) or "" for fam in families}
    hits = orch.workflow.retrieve_for_families(
        document_id="d1",
        chunks=chunks,
        families=families,
        top_k_per_family=2,
        family_queries=queries,
    )
    assert FieldFamily.HEPATIC_BIOCHEMISTRY in hits
    assert hits[FieldFamily.HEPATIC_BIOCHEMISTRY]
