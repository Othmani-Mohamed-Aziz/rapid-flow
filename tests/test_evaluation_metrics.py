from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.dataset import (
    EvaluationDatasetError,
    discover_dataset,
    flatten_template,
    load_json_object,
)
from app.evaluation.metrics import canonical_unit, evaluate_fields, values_match
from app.evaluation.models import TemplateField
from app.evaluation.retrieval import derive_silver_qrels
from app.schemas.models import DocumentChunk

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "evaluation"


def _field(path: tuple[str, ...], value, unit: str | None = None) -> TemplateField:
    return TemplateField(path=path, value=value, unit=unit, unit_declared=unit is not None)


def test_dataset_discovery_pairs_sample_names() -> None:
    discovery = discover_dataset(FIXTURE_ROOT)
    assert [sample.sample_id for sample in discovery.samples] == ["sample_a"]
    assert discovery.incomplete == []


def test_dataset_discovery_ignores_unlabeled_pdfs(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "sample_a.pdf").write_bytes(b"%PDF-1.4")
    (reports / "report_legacy.pdf").write_bytes(b"%PDF-1.4")
    for folder in ("filtered_templates/empty", "filtered_templates/filled", "extracted"):
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / "filtered_templates" / "empty" / "sample_a_template_empty.json").write_text(
        '{"Imaging_RECIST": {"Size": {"valeur": null}}}', encoding="utf-8"
    )
    (tmp_path / "filtered_templates" / "filled" / "sample_a_template_filled.json").write_text(
        '{"Imaging_RECIST": {"Size": {"valeur": 12}}}', encoding="utf-8"
    )
    (tmp_path / "extracted" / "sample_a_template.json").write_text(
        '{"Imaging_RECIST": {"Size": {"valeur": 12}}}', encoding="utf-8"
    )

    discovery = discover_dataset(tmp_path)
    assert [sample.sample_id for sample in discovery.samples] == ["sample_a"]
    assert discovery.incomplete == []


def test_flatten_template_uses_full_nested_path() -> None:
    flattened = flatten_template(
        {"Section": {"Subsection": {"Field": {"valeur": 3, "unité": "mg/L"}}}}
    )
    field = flattened[("Section", "Subsection", "Field")]
    assert field.value == 3
    assert field.unit == "mg/L"


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(EvaluationDatasetError, match="Duplicate JSON key"):
        load_json_object(path)


def test_presence_metrics_are_separate_from_value_accuracy() -> None:
    a = ("S", "A")
    b = ("S", "B")
    empty = {a: _field(a, None, "mg/L"), b: _field(b, None, "mg/L")}
    gold = {a: _field(a, 10.0, "mg/L"), b: _field(b, None, "mg/L")}
    predicted = {a: _field(a, 11.0, "mg/L"), b: _field(b, 5.0, "mg/L")}

    metrics, comparisons, extras = evaluate_fields(empty, gold, predicted)

    assert metrics.counts.true_positives == 1
    assert metrics.counts.false_positives == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 1.0
    assert metrics.value_accuracy == 0.0
    assert metrics.validation_pass_rate == 1.0
    assert {comparison.outcome for comparison in comparisons} == {
        "value_mismatch",
        "false_autofill",
    }
    assert extras == []


def test_zero_denominators_are_reported_as_none() -> None:
    path = ("S", "A")
    empty = {path: _field(path, None)}
    gold = {path: _field(path, None)}
    predicted = {path: _field(path, None)}
    metrics, _, _ = evaluate_fields(empty, gold, predicted)
    assert metrics.validation_pass_rate is None
    assert metrics.false_autofill_rate is None
    assert metrics.precision is None
    assert metrics.recall is None
    assert metrics.f1 is None
    assert metrics.value_accuracy is None


def test_numeric_tolerance_and_unit_aliases() -> None:
    assert values_match(1.001, 1.0, absolute_tolerance=0.01)
    assert not values_match(1.1, 1.0, absolute_tolerance=0.01)
    assert canonical_unit(" UI/L ") == canonical_unit("U/L")
    assert canonical_unit("μmol/L") == canonical_unit("umol/L")


def test_silver_qrels_require_one_exact_label_value_chunk() -> None:
    path = ("Biochimie", "Biochimie", "Créatinine")
    gold = _field(path, 88.0, "umol/L")
    chunks = [
        DocumentChunk(chunk_id="relevant", document_id="d", text="Créatinine 88.0 umol/L"),
        DocumentChunk(chunk_id="other", document_id="d", text="Créatinine 90 umol/L"),
    ]
    qrel = derive_silver_qrels(chunks, [gold])[0]
    assert qrel.status == "matched"
    assert qrel.relevant_chunk_ids == ["relevant"]

    ambiguous = derive_silver_qrels(
        [chunks[0], chunks[0].model_copy(update={"chunk_id": "dup"})], [gold]
    )
    assert ambiguous[0].status == "ambiguous"
    assert ambiguous[0].relevant_chunk_ids == []


def test_silver_qrels_support_recist_report_labels() -> None:
    size = _field(
        ("Imaging_RECIST", "Size_major_nodule_mm_start_AtezoBev_D0"),
        45,
        "mm",
    )
    response = _field(
        ("Imaging_RECIST", "Response_at_first_imaging_RECIST"),
        "SD",
    )
    chunks = [
        DocumentChunk(
            chunk_id="size",
            document_id="d",
            text="La lésion cible mesurait 45 mm de grand axe au bilan initial.",
        ),
        DocumentChunk(
            chunk_id="response",
            document_id="d",
            text="Maladie stable (SD) selon RECIST 1.1.",
        ),
    ]

    qrels = derive_silver_qrels(chunks, [size, response])

    assert [qrel.status for qrel in qrels] == ["matched", "matched"]
    assert qrels[0].relevant_chunk_ids == ["size"]
    assert qrels[1].relevant_chunk_ids == ["response"]
