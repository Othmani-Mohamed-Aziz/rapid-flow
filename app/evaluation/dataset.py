"""Discovery and loading of the local paired evaluation dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.models import (
    DatasetDiscovery,
    EvaluationSamplePaths,
    IncompleteEvaluationSample,
    JsonScalar,
    TemplateField,
)


class EvaluationDatasetError(ValueError):
    """Raised when an evaluation fixture violates the dataset contract."""


def _sample_id(path: Path, suffix: str) -> str:
    name = path.stem
    return name[: -len(suffix)] if suffix and name.endswith(suffix) else name


def discover_dataset(dataset_root: str | Path) -> DatasetDiscovery:
    root = Path(dataset_root)
    reports = {p.stem: p for p in sorted((root / "reports").glob("*.pdf"))}
    empty = {
        _sample_id(p, "_template_empty"): p
        for p in sorted((root / "filtered_templates" / "empty").glob("*_template_empty.json"))
    }
    filled = {
        _sample_id(p, "_template_filled"): p
        for p in sorted((root / "filtered_templates" / "filled").glob("*_template_filled.json"))
    }
    extracted = {
        _sample_id(p, "_template"): p for p in sorted((root / "extracted").glob("*_template.json"))
    }

    samples: list[EvaluationSamplePaths] = []
    incomplete: list[IncompleteEvaluationSample] = []
    for sample_id in sorted(set(reports) | set(empty) | set(filled) | set(extracted)):
        # Unlabeled leftover PDFs are ignored; they are not evaluation samples.
        if sample_id not in empty and sample_id not in filled and sample_id not in extracted:
            continue
        missing = [
            label
            for label, collection in (
                ("report", reports),
                ("empty_template", empty),
                ("filled_template", filled),
                ("extracted", extracted),
            )
            if sample_id not in collection
        ]
        if missing:
            incomplete.append(IncompleteEvaluationSample(sample_id=sample_id, missing=missing))
            continue
        samples.append(
            EvaluationSamplePaths(
                sample_id=sample_id,
                report_path=reports[sample_id],
                empty_template_path=empty[sample_id],
                filled_template_path=filled[sample_id],
                extracted_path=extracted[sample_id],
            )
        )
    return DatasetDiscovery(samples=samples, incomplete=incomplete)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise EvaluationDatasetError(f"Duplicate JSON key: {key!r}")
        out[key] = value
    return out


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except json.JSONDecodeError as exc:
        raise EvaluationDatasetError(f"Invalid JSON in {source.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationDatasetError(f"{source.name} must contain a JSON object")
    return payload


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def flatten_template(payload: dict[str, Any]) -> dict[tuple[str, ...], TemplateField]:
    """Flatten nested template leaves identified by a ``valeur`` key."""

    fields: dict[tuple[str, ...], TemplateField] = {}

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if not isinstance(node, dict):
            raise EvaluationDatasetError(
                f"Expected an object at {' / '.join(path) or '<root>'}, got {type(node).__name__}"
            )
        if "valeur" in node:
            value = node["valeur"]
            if not _is_scalar(value):
                raise EvaluationDatasetError(
                    f"Non-scalar valeur at {' / '.join(path)}: {type(value).__name__}"
                )
            unit_key = "unité" if "unité" in node else "unite" if "unite" in node else None
            unit = node.get(unit_key) if unit_key else None
            if unit is not None and not isinstance(unit, str):
                raise EvaluationDatasetError(f"Non-string unit at {' / '.join(path)}")
            if not path:
                raise EvaluationDatasetError("A template field cannot be the root object")
            fields[path] = TemplateField(
                path=path,
                value=value,
                unit=unit,
                unit_declared=unit_key is not None,
            )
            return
        for key, child in node.items():
            if not isinstance(key, str):
                raise EvaluationDatasetError("Template keys must be strings")
            visit(child, (*path, key))

    visit(payload, ())
    if not fields:
        raise EvaluationDatasetError("Template contains no leaves with a 'valeur' key")
    return fields


def load_flat_template(path: str | Path) -> dict[tuple[str, ...], TemplateField]:
    return flatten_template(load_json_object(path))


def scalar_type(value: JsonScalar) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "string"
