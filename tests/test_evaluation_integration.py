from __future__ import annotations

from pathlib import Path

import pytest

from app.config.settings import Settings
from app.evaluation.retrieval import RetrievalEvaluator
from app.evaluation.runner import EvaluationRunner
from app.indexing.vector_service import InMemoryVectorIndexService
from app.schemas.enums import DocumentType, FieldFamily
from app.schemas.models import DocumentChunk, ParsedDocument, RawDocument
from scripts.run_evaluation import main

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "evaluation"


class _FixtureParser:
    def parse(self, raw: RawDocument) -> ParsedDocument:
        return ParsedDocument(
            document_id=raw.document_id,
            patient_id=raw.patient_id,
            study_id=raw.study_id,
            full_text=raw.content_bytes.decode("utf-8"),
            document_type_hint=DocumentType.LAB_BLOOD_PANEL,
        )


class _LineChunker:
    def chunk(self, parsed: ParsedDocument) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                chunk_id=f"chunk-{index}",
                document_id=parsed.document_id,
                text=line,
                field_family=FieldFamily.HEMATOLOGY,
            )
            for index, line in enumerate(parsed.full_text.splitlines(), start=1)
            if line.strip()
        ]


@pytest.mark.evaluation
def test_batch_runner_scores_saved_predictions_and_retrieval(tmp_path: Path) -> None:
    settings = Settings(
        vector_backend="memory",
        enable_sparse_bm25=False,
        enable_reranker=False,
    )
    retrieval = RetrievalEvaluator(
        settings,
        parser=_FixtureParser(),
        chunker=_LineChunker(),
        vector_index=InMemoryVectorIndexService(),
    )
    report, report_path = EvaluationRunner(
        settings,
        retrieval_evaluator=retrieval,
    ).run(FIXTURE_ROOT, output_root=tmp_path)

    assert report_path is not None and report_path.is_file()
    assert (report_path.parent / "summary.txt").is_file()
    metrics = report.aggregate.micro
    assert metrics.autofill_rate == 1.0
    assert metrics.false_autofill_rate == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 1.0
    assert metrics.value_accuracy == 1.0
    assert report.aggregate.retrieval_macro == {"1": 0.0, "3": 1.0, "5": 1.0}
    assert report.samples[0].out_of_template_paths == [
        ("Hematologie", "NumerationGlobulaire", "Hematocrite")
    ]


@pytest.mark.evaluation
def test_cli_returns_one_when_metric_threshold_fails(tmp_path: Path) -> None:
    code = main(
        [
            "--dataset-root",
            str(FIXTURE_ROOT),
            "--output-dir",
            str(tmp_path),
            "--skip-retrieval",
            "--min-precision",
            "0.9",
        ]
    )
    assert code == 1
