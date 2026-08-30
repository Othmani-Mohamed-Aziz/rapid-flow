"""Pure field-level metric and normalization functions."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable
from statistics import fmean
from typing import TypeGuard

from app.evaluation.dataset import scalar_type
from app.evaluation.models import (
    ContractCheck,
    FieldComparison,
    FieldMetrics,
    MetricCounts,
    TemplateField,
)

_UNIT_ALIASES = {
    "ui/l": "u/l",
    "iu/l": "u/l",
    "µmol/l": "umol/l",
    "μmol/l": "umol/l",
    "µg/l": "ug/l",
    "μg/l": "ug/l",
}


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    text = text.casefold().replace(",", ".")
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    return " ".join(text.split())


def canonical_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    normalized = unicodedata.normalize("NFKC", unit).casefold().strip()
    normalized = normalized.replace("μ", "µ").replace(" ", "")
    return _UNIT_ALIASES.get(normalized, normalized)


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def values_match(
    predicted: object,
    expected: object,
    *,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-6,
) -> bool:
    if _is_number(predicted) and _is_number(expected):
        return math.isclose(
            float(predicted),
            float(expected),
            abs_tol=absolute_tolerance,
            rel_tol=relative_tolerance,
        )
    if isinstance(predicted, str) and isinstance(expected, str):
        return normalize_text(predicted) == normalize_text(expected)
    return predicted == expected


def validate_contract(
    predicted: TemplateField,
    template: TemplateField,
    gold: TemplateField,
) -> ContractCheck:
    checks = ["json_scalar", "template_path"]
    errors: list[str] = []

    if gold.value is not None:
        checks.append("gold_type")
        expected_type = scalar_type(gold.value)
        actual_type = scalar_type(predicted.value)
        if expected_type != actual_type:
            errors.append(f"type:{actual_type}!={expected_type}")

    if template.unit_declared:
        checks.append("unit")
        if canonical_unit(predicted.unit) != canonical_unit(template.unit):
            errors.append(
                f"unit:{canonical_unit(predicted.unit)!r}!={canonical_unit(template.unit)!r}"
            )

    return ContractCheck(valid=not errors, checks_applied=checks, errors=errors)


def metrics_from_counts(counts: MetricCounts) -> FieldMetrics:
    precision = safe_rate(counts.true_positives, counts.true_positives + counts.false_positives)
    recall = safe_rate(counts.true_positives, counts.true_positives + counts.false_negatives)
    f1 = None
    if precision is not None and recall is not None:
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return FieldMetrics(
        counts=counts,
        validation_pass_rate=safe_rate(counts.contract_valid, counts.predicted_fills),
        autofill_rate=safe_rate(counts.predicted_fills, counts.eligible_fields),
        false_autofill_rate=safe_rate(counts.false_positives, counts.predicted_fills),
        precision=precision,
        recall=recall,
        f1=f1,
        value_accuracy=safe_rate(counts.value_matches, counts.value_compared),
    )


def evaluate_fields(
    empty: dict[tuple[str, ...], TemplateField],
    gold: dict[tuple[str, ...], TemplateField],
    predicted: dict[tuple[str, ...], TemplateField],
    *,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-6,
) -> tuple[FieldMetrics, list[FieldComparison], list[tuple[str, ...]]]:
    missing_gold = sorted(set(empty) - set(gold))
    if missing_gold:
        raise ValueError(f"Filled template is missing {len(missing_gold)} eligible field(s)")

    counts = MetricCounts(eligible_fields=len(empty))
    comparisons: list[FieldComparison] = []
    for path, template_field in empty.items():
        gold_field = gold[path]
        predicted_field = predicted.get(path)
        expected_present = gold_field.value is not None
        predicted_present = predicted_field is not None and predicted_field.value is not None

        counts.gold_fills += int(expected_present)
        counts.predicted_fills += int(predicted_present)
        counts.true_positives += int(expected_present and predicted_present)
        counts.false_positives += int(not expected_present and predicted_present)
        counts.false_negatives += int(expected_present and not predicted_present)

        contract: ContractCheck | None = None
        value_matches: bool | None = None
        if predicted_present and predicted_field is not None:
            contract = validate_contract(predicted_field, template_field, gold_field)
            counts.contract_valid += int(contract.valid)

        if expected_present and predicted_present and predicted_field is not None:
            unit_matches = not template_field.unit_declared or (
                canonical_unit(predicted_field.unit) == canonical_unit(gold_field.unit)
            )
            value_matches = unit_matches and values_match(
                predicted_field.value,
                gold_field.value,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            counts.value_compared += 1
            counts.value_matches += int(value_matches)

        if expected_present and predicted_present:
            outcome = "matched" if value_matches else "value_mismatch"
        elif expected_present:
            outcome = "missing"
        elif predicted_present:
            outcome = "false_autofill"
        else:
            outcome = "correctly_empty"

        comparisons.append(
            FieldComparison(
                path=path,
                expected=gold_field.value,
                predicted=predicted_field.value if predicted_field else None,
                expected_unit=gold_field.unit,
                predicted_unit=predicted_field.unit if predicted_field else None,
                expected_present=expected_present,
                predicted_present=predicted_present,
                value_matches=value_matches,
                contract=contract,
                outcome=outcome,
            )
        )

    extras = sorted(set(predicted) - set(empty))
    return metrics_from_counts(counts), comparisons, extras


_METRIC_NAMES = (
    "validation_pass_rate",
    "autofill_rate",
    "false_autofill_rate",
    "precision",
    "recall",
    "f1",
    "value_accuracy",
)


def aggregate_field_metrics(
    metrics: Iterable[FieldMetrics],
) -> tuple[FieldMetrics, dict[str, float | None]]:
    values = list(metrics)
    total = MetricCounts()
    for item in values:
        for name in MetricCounts.model_fields:
            setattr(total, name, getattr(total, name) + getattr(item.counts, name))
    macro: dict[str, float | None] = {}
    for name in _METRIC_NAMES:
        present = [float(value) for item in values if (value := getattr(item, name)) is not None]
        macro[name] = fmean(present) if present else None
    return metrics_from_counts(total), macro
