"""Evaluate paired reports, verified templates, and saved extraction results.

Examples:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --dataset-root data --skip-retrieval
    python scripts/run_evaluation.py --k 1 3 5 --min-f1 0.90 --min-value-accuracy 0.95
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, io.UnsupportedOperation):
        pass

from app.config.settings import Settings  # noqa: E402
from app.evaluation.runner import EvaluationRunner  # noqa: E402


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return parsed


def _silver_threshold(value: str) -> tuple[int, float]:
    try:
        raw_k, raw_threshold = value.split("=", 1)
        k = int(raw_k)
        threshold = _unit_interval(raw_threshold)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError("expected K=THRESHOLD, for example 3=0.80") from exc
    if k <= 0:
        raise argparse.ArgumentTypeError("K must be positive")
    return k, threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--min-validation-pass-rate", type=_unit_interval)
    parser.add_argument("--min-autofill-rate", type=_unit_interval)
    parser.add_argument("--max-false-autofill-rate", type=_unit_interval)
    parser.add_argument("--min-precision", type=_unit_interval)
    parser.add_argument("--min-recall", type=_unit_interval)
    parser.add_argument("--min-f1", type=_unit_interval)
    parser.add_argument("--min-value-accuracy", type=_unit_interval)
    parser.add_argument(
        "--min-silver-recall-at-k",
        action="append",
        default=[],
        type=_silver_threshold,
        metavar="K=THRESHOLD",
    )
    return parser


def _threshold_failures(args: argparse.Namespace, report) -> list[str]:
    metrics = report.aggregate.micro
    minimums = {
        "validation_pass_rate": args.min_validation_pass_rate,
        "autofill_rate": args.min_autofill_rate,
        "precision": args.min_precision,
        "recall": args.min_recall,
        "f1": args.min_f1,
        "value_accuracy": args.min_value_accuracy,
    }
    failures: list[str] = []
    for name, threshold in minimums.items():
        if threshold is None:
            continue
        actual = getattr(metrics, name)
        if actual is None or actual < threshold:
            failures.append(f"{name}: {actual!r} < {threshold}")
    maximum = args.max_false_autofill_rate
    if maximum is not None:
        actual = metrics.false_autofill_rate
        if actual is None or actual > maximum:
            failures.append(f"false_autofill_rate: {actual!r} > {maximum}")
    for k, threshold in args.min_silver_recall_at_k:
        actual = report.aggregate.retrieval_macro.get(str(k))
        if actual is None or actual < threshold:
            failures.append(f"silver_recall_at_{k}: {actual!r} < {threshold}")
    return failures


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report, report_path = EvaluationRunner(Settings()).run(
            args.dataset_root,
            output_root=args.output_dir,
            k_values=args.k,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
            include_retrieval=not args.skip_retrieval,
        )
    except (OSError, ValueError) as exc:
        print(f"Evaluation input error: {exc}", file=sys.stderr)
        return 2

    print(f"Evaluation report: {report_path}")
    print(f"Complete samples: {len(report.samples)}")
    if report.incomplete_samples:
        for sample in report.incomplete_samples:
            print(
                f"Incomplete sample {sample.sample_id}: {', '.join(sample.missing)}",
                file=sys.stderr,
            )
        return 2

    failures = _threshold_failures(args, report)
    if failures:
        print("Metric thresholds failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
