"""Batch orchestration and report writing for evaluation datasets."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from statistics import fmean

from app.config.settings import Settings
from app.evaluation.dataset import discover_dataset, load_flat_template
from app.evaluation.metrics import aggregate_field_metrics, evaluate_fields
from app.evaluation.models import (
    AggregateMetrics,
    EvaluationReport,
    RetrievalMetrics,
    SampleEvaluation,
)
from app.evaluation.retrieval import RetrievalEvaluator
from app.utils.safe_paths import redact_file_paths_in_jsonable


class EvaluationRunner:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        retrieval_evaluator: RetrievalEvaluator | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.retrieval_evaluator = retrieval_evaluator

    def run(
        self,
        dataset_root: str | Path = "data",
        *,
        output_root: str | Path = "outputs/evaluation",
        k_values: list[int] | None = None,
        absolute_tolerance: float = 1e-6,
        relative_tolerance: float = 1e-6,
        include_retrieval: bool = True,
        write_report: bool = True,
    ) -> tuple[EvaluationReport, Path | None]:
        ks = sorted(set(k_values or [1, 3, 5]))
        if not ks or any(k <= 0 for k in ks):
            raise ValueError("k values must be positive integers")
        if absolute_tolerance < 0 or relative_tolerance < 0:
            raise ValueError("numeric tolerances must be non-negative")

        root = Path(dataset_root)
        discovery = discover_dataset(root)
        if not discovery.samples:
            raise ValueError(f"No complete evaluation samples found under {root}")

        retrieval_evaluator = self.retrieval_evaluator
        if include_retrieval and retrieval_evaluator is None:
            retrieval_evaluator = RetrievalEvaluator(self.settings)

        samples: list[SampleEvaluation] = []
        for paths in discovery.samples:
            empty = load_flat_template(paths.empty_template_path)
            gold = load_flat_template(paths.filled_template_path)
            predicted = load_flat_template(paths.extracted_path)
            field_metrics, comparisons, extras = evaluate_fields(
                empty,
                gold,
                predicted,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            issues = [
                comparison
                for comparison in comparisons
                if comparison.outcome not in {"matched", "correctly_empty"}
                or (comparison.contract is not None and not comparison.contract.valid)
            ]
            retrieval: RetrievalMetrics | None = None
            if include_retrieval and retrieval_evaluator is not None:
                retrieval = retrieval_evaluator.evaluate(paths, gold, k_values=ks)
            samples.append(
                SampleEvaluation(
                    sample_id=paths.sample_id,
                    field_metrics=field_metrics,
                    comparisons=issues,
                    out_of_template_paths=extras,
                    retrieval=retrieval,
                )
            )

        micro, macro = aggregate_field_metrics(sample.field_metrics for sample in samples)
        retrieval_macro: dict[str, float | None] = {}
        for k in ks:
            values = [
                value
                for sample in samples
                if sample.retrieval is not None
                and (value := sample.retrieval.recall_at_k.get(str(k))) is not None
            ]
            retrieval_macro[str(k)] = fmean(values) if values else None

        run_id = uuid.uuid4().hex
        report = EvaluationReport(
            run_id=run_id,
            pipeline_version=self.settings.pipeline_version,
            dataset_root=root.name,
            k_values=ks,
            samples=samples,
            incomplete_samples=discovery.incomplete,
            aggregate=AggregateMetrics(
                micro=micro,
                macro=macro,
                retrieval_macro=retrieval_macro,
            ),
            settings={
                "absolute_tolerance": absolute_tolerance,
                "relative_tolerance": relative_tolerance,
                "retrieval_enabled": include_retrieval,
                "vector_backend": self.settings.vector_backend,
                "pdf_parser_backend": self.settings.pdf_parser_backend,
            },
        )
        if not write_report:
            return report, None

        run_dir = Path(output_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        payload = redact_file_paths_in_jsonable(report.model_dump(mode="json"))
        report_path = run_dir / "evaluation_report.json"
        report_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "summary.txt").write_text(_summary(report), encoding="utf-8")
        return report, report_path


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _summary(report: EvaluationReport) -> str:
    metrics = report.aggregate.micro
    lines = [
        f"Evaluation run: {report.run_id}",
        f"Samples: {len(report.samples)} complete, {len(report.incomplete_samples)} incomplete",
        f"Validation pass rate: {_fmt(metrics.validation_pass_rate)}",
        f"Autofill rate: {_fmt(metrics.autofill_rate)}",
        f"False-autofill rate: {_fmt(metrics.false_autofill_rate)}",
        f"Precision: {_fmt(metrics.precision)}",
        f"Recall: {_fmt(metrics.recall)}",
        f"F1: {_fmt(metrics.f1)}",
        f"Value accuracy: {_fmt(metrics.value_accuracy)}",
    ]
    for k, value in report.aggregate.retrieval_macro.items():
        lines.append(f"Silver Recall@{k}: {_fmt(value)}")
    return "\n".join(lines) + "\n"
