from __future__ import annotations

from pathlib import Path

from app.orchestration.pipeline import run_pipeline


def test_pipeline_txt_still_runs_end_to_end() -> None:
    root = Path(__file__).resolve().parents[1]
    sample = root / "scripts" / "sample_data" / "mock_blood_panel.txt"
    result = run_pipeline(str(sample), patient_id="P1", study_id="S1")
    assert result.document_type.value == "lab_blood_panel"
    assert result.cell_updates
