from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.run_dataset_extraction import _run_one, discover_samples


def _labelled_sample(tmp_path, sample_id: str = "sample_0001"):
    root = tmp_path / "dataset"
    report = root / "reports" / f"{sample_id}.pdf"
    empty = root / "filtered_templates" / "empty" / f"{sample_id}_template_empty.json"
    filled = root / "filtered_templates" / "filled" / f"{sample_id}_template_filled.json"
    report.parent.mkdir(parents=True)
    empty.parent.mkdir(parents=True)
    filled.parent.mkdir(parents=True)
    report.write_bytes(b"%PDF-1.4")
    empty.write_text("{}", encoding="utf-8")
    filled.write_text("{}", encoding="utf-8")
    return root, report


def test_discover_samples_requires_complete_labels(tmp_path) -> None:
    root, report = _labelled_sample(tmp_path)
    incomplete = root / "reports" / "unlabelled.pdf"
    incomplete.write_bytes(b"%PDF-1.4")

    assert discover_samples(root) == [report]


def test_discover_samples_rejects_missing_reports_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="Missing reports directory"):
        discover_samples(tmp_path / "missing")


def test_run_one_copies_pipeline_prediction(tmp_path) -> None:
    root, report = _labelled_sample(tmp_path)
    generated = tmp_path / "pipeline-output.json"
    generated.write_text('{"Section": {}}', encoding="utf-8")

    def fake_pipeline(path: str, patient_id: str, study_id: str):
        assert path == str(report.resolve())
        assert patient_id == report.stem
        assert study_id == "EXAMPLE"
        return SimpleNamespace(export_paths={"lab_template_json": str(generated)})

    sample_id, status = _run_one(
        report,
        root=root,
        study_id="EXAMPLE",
        overwrite=False,
        run_pipeline=fake_pipeline,
    )

    destination = root / "extracted" / f"{report.stem}_template.json"
    assert (sample_id, status) == (report.stem, "written")
    assert destination.read_text(encoding="utf-8") == '{"Section": {}}'

    assert _run_one(
        report,
        root=root,
        study_id="EXAMPLE",
        overwrite=False,
        run_pipeline=fake_pipeline,
    ) == (report.stem, "skipped")
