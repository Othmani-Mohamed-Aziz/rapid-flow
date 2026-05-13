from __future__ import annotations

from pathlib import Path

from app.etl.export import EcrfExportService
from app.schemas.enums import DocumentType, FieldFamily
from app.schemas.models import EcrfCellUpdate, ExtractedObservation, FieldCandidate, PipelineResult


def test_export_writes_json_and_csv(tmp_path: Path) -> None:
    obs = ExtractedObservation(
        observation_id="o1",
        field_family=FieldFamily.HEMATOLOGY,
        normalized_value=150,
        confidence=0.9,
        evidence_text="PLT 150",
        extraction_method="test",
    )
    cand = FieldCandidate(
        field_name="PLT_start_AtezoBev_D0",
        observation=obs,
        target_column="PLT_start_AtezoBev_D0",
        confidence=0.9,
    )
    upd = EcrfCellUpdate(
        patient_id="p1",
        study_id="s1",
        column_key="PLT_start_AtezoBev_D0",
        value=150,
        provenance=cand,
        validated=True,
    )
    result = PipelineResult(
        document_id="d1",
        patient_id="p1",
        study_id="s1",
        document_type=DocumentType.LAB_BLOOD_PANEL,
        observations=[obs],
        cell_updates=[upd],
        audit_trail=[],
    )
    svc = EcrfExportService()
    paths = svc.export(result, tmp_path, basename="unit")
    assert Path(paths["json"]).is_file()
    assert Path(paths["csv"]).is_file()
    text = Path(paths["csv"]).read_text(encoding="utf-8")
    assert "PLT_start_AtezoBev_D0" in text
