"""Typed contracts for the offline evaluation pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

JsonScalar = str | int | float | bool | None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EvaluationSamplePaths(BaseModel):
    sample_id: str
    report_path: Path
    empty_template_path: Path
    filled_template_path: Path
    extracted_path: Path


class IncompleteEvaluationSample(BaseModel):
    sample_id: str
    missing: list[str]


class DatasetDiscovery(BaseModel):
    samples: list[EvaluationSamplePaths] = Field(default_factory=list)
    incomplete: list[IncompleteEvaluationSample] = Field(default_factory=list)


class TemplateField(BaseModel):
    path: tuple[str, ...]
    value: JsonScalar = None
    unit: str | None = None
    unit_declared: bool = False

    @property
    def path_key(self) -> str:
        return " / ".join(self.path)


class ContractCheck(BaseModel):
    valid: bool
    checks_applied: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class FieldComparison(BaseModel):
    path: tuple[str, ...]
    expected: JsonScalar = None
    predicted: JsonScalar = None
    expected_unit: str | None = None
    predicted_unit: str | None = None
    expected_present: bool
    predicted_present: bool
    value_matches: bool | None = None
    contract: ContractCheck | None = None
    outcome: str


class MetricCounts(BaseModel):
    eligible_fields: int = 0
    predicted_fills: int = 0
    gold_fills: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    contract_valid: int = 0
    value_matches: int = 0
    value_compared: int = 0


class FieldMetrics(BaseModel):
    counts: MetricCounts
    validation_pass_rate: float | None = None
    autofill_rate: float | None = None
    false_autofill_rate: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    value_accuracy: float | None = None


class RetrievalHitSummary(BaseModel):
    chunk_id: str
    rank: int
    score: float


class SilverQrel(BaseModel):
    field_path: tuple[str, ...]
    query: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    status: str


class RetrievalQueryEvaluation(BaseModel):
    field_path: tuple[str, ...]
    query: str
    qrel_status: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    hits: list[RetrievalHitSummary] = Field(default_factory=list)
    recall_at_k: dict[str, float | None] = Field(default_factory=dict)


class RetrievalMetrics(BaseModel):
    name: str = "silver_recall_at_k"
    recall_at_k: dict[str, float | None] = Field(default_factory=dict)
    evaluable_queries: int = 0
    unmatched_fields: int = 0
    ambiguous_fields: int = 0
    unsupported_fields: int = 0
    queries: list[RetrievalQueryEvaluation] = Field(default_factory=list)


class SampleEvaluation(BaseModel):
    sample_id: str
    field_metrics: FieldMetrics
    comparisons: list[FieldComparison] = Field(default_factory=list)
    out_of_template_paths: list[tuple[str, ...]] = Field(default_factory=list)
    retrieval: RetrievalMetrics | None = None


class AggregateMetrics(BaseModel):
    micro: FieldMetrics
    macro: dict[str, float | None] = Field(default_factory=dict)
    retrieval_macro: dict[str, float | None] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=utc_now)
    pipeline_version: str
    dataset_root: str
    k_values: list[int]
    samples: list[SampleEvaluation]
    incomplete_samples: list[IncompleteEvaluationSample] = Field(default_factory=list)
    aggregate: AggregateMetrics
    settings: dict[str, Any] = Field(default_factory=dict)
