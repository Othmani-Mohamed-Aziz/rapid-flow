"""Run the complete pipeline on an evaluation dataset and save predictions.

The input dataset must contain:

    <root>/reports/<sample_id>.pdf
    <root>/filtered_templates/empty/<sample_id>_template_empty.json
    <root>/filtered_templates/filled/<sample_id>_template_filled.json

Predictions are written to:

    <root>/extracted/<sample_id>_template.json
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--study-id", default="EXAMPLE")
    parser.add_argument("--limit", type=int, default=0, help="0 processes every labelled sample")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--vector-backend", choices=("memory", "qdrant"), default="memory")
    parser.add_argument("--imaging-workers", type=int, default=1)
    return parser


def discover_samples(root: Path) -> list[Path]:
    reports = root / "reports"
    empty = root / "filtered_templates" / "empty"
    filled = root / "filtered_templates" / "filled"
    if not reports.is_dir():
        raise ValueError(f"Missing reports directory: {reports}")
    samples = [
        report
        for report in sorted(reports.glob("*.pdf"))
        if (empty / f"{report.stem}_template_empty.json").is_file()
        and (filled / f"{report.stem}_template_filled.json").is_file()
    ]
    if not samples:
        raise ValueError(f"No labelled PDF samples found under {root}")
    return samples


def _run_one(
    report: Path,
    *,
    root: Path,
    study_id: str,
    overwrite: bool,
    run_pipeline: Callable[..., Any],
) -> tuple[str, str]:
    destination = root / "extracted" / f"{report.stem}_template.json"
    if destination.is_file() and not overwrite:
        return report.stem, "skipped"

    result = run_pipeline(
        str(report.resolve()),
        patient_id=report.stem,
        study_id=study_id,
    )
    source_path = result.export_paths.get("lab_template_json") or result.export_paths.get(
        "imaging_template_json"
    )
    if not source_path:
        raise RuntimeError(
            f"{report.stem}: pipeline produced neither a lab nor an imaging prediction template"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(source_path), destination)
    return report.stem, "written"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.workers <= 0 or args.imaging_workers <= 0:
        raise ValueError("worker counts must be positive")

    os.environ["ECRF_VECTOR_BACKEND"] = args.vector_backend
    os.environ["ECRF_EXPORT_XLS_ENABLED"] = "false"
    os.environ["ECRF_EXPORT_XLS_REQUIRE_PATIENT_IN_MA"] = "false"
    os.environ["ECRF_IMAGING_EXTRACTION_MAX_WORKERS"] = str(args.imaging_workers)

    from app.orchestration.pipeline import run_pipeline

    root = args.dataset_root.resolve()
    samples = discover_samples(root)
    if args.limit:
        samples = samples[: args.limit]

    failures: list[str] = []
    written = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=min(args.workers, len(samples))) as pool:
        futures = {
            pool.submit(
                _run_one,
                report,
                root=root,
                study_id=args.study_id,
                overwrite=args.overwrite,
                run_pipeline=run_pipeline,
            ): report
            for report in samples
        }
        for future in as_completed(futures):
            report = futures[future]
            try:
                sample_id, status = future.result()
                if status == "written":
                    written += 1
                else:
                    skipped += 1
                print(f"{sample_id}: {status}", flush=True)
            except Exception as exc:
                failures.append(f"{report.stem}: {exc}")
                print(f"{report.stem}: failed: {exc}", file=sys.stderr, flush=True)

    print(
        f"Processed {len(samples)} sample(s): {written} written, "
        f"{skipped} skipped, {len(failures)} failed"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
