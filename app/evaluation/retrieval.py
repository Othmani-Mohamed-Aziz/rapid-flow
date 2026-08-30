"""Silver-qrel generation and retrieval evaluation."""

from __future__ import annotations

from collections.abc import Iterable

from app.config.settings import Settings
from app.evaluation.metrics import normalize_text, safe_rate
from app.evaluation.models import (
    EvaluationSamplePaths,
    RetrievalHitSummary,
    RetrievalMetrics,
    RetrievalQueryEvaluation,
    SilverQrel,
    TemplateField,
)
from app.indexing import build_default_vector_service
from app.ingestion.service import LocalFileIngestionService
from app.parsing.chunking import DefaultChunkingService
from app.parsing.smart_service import SmartParsingService
from app.schemas.enums import FieldFamily
from app.schemas.models import DocumentChunk

_SECTION_FAMILIES = {
    "hematologie": FieldFamily.HEMATOLOGY,
    "hemostase": FieldFamily.COAGULATION,
    "biochimie": FieldFamily.HEPATIC_BIOCHEMISTRY,
    "imaging recist": FieldFamily.IMAGING_RECIST,
}
_PRODUCTION_LAB_QUERIES = {
    FieldFamily.HEMATOLOGY: "hématologie hémogramme plaquettes leucocytes",
    FieldFamily.COAGULATION: "coagulation INR TP TCA fibrinogène",
    FieldFamily.HEPATIC_BIOCHEMISTRY: "bilan hépatique AST ALT GGT bilirubine",
    FieldFamily.INFLAMMATION_BIOMARKERS: "inflammation CRP biomarqueurs AFP",
    FieldFamily.IMAGING_RECIST: (
        "RECIST réponse tumorale lésion cible mesure grand axe millimètres"
    ),
}
_FIELD_LABEL_ALIASES = {
    "size major nodule mm start atezobev d0": ("lesion cible", "grand axe"),
    "response at first imaging recist": ("recist", "reponse"),
}


def _value_needles(value: object) -> set[str]:
    raw = normalize_text(str(value))
    needles = {raw}
    if isinstance(value, float) and value.is_integer():
        needles.add(normalize_text(str(int(value))))
    return {needle for needle in needles if needle}


def _field_query(field: TemplateField) -> str:
    parts = [*field.path, str(field.value)]
    if field.unit:
        parts.append(field.unit)
    return " ".join(parts)


def derive_silver_qrels(
    chunks: Iterable[DocumentChunk],
    gold_fields: Iterable[TemplateField],
) -> list[SilverQrel]:
    """Match verified label + value to exactly one chunk, independently of ranking."""

    chunk_list = list(chunks)
    normalized_chunks = {chunk.chunk_id: normalize_text(chunk.text) for chunk in chunk_list}
    qrels: list[SilverQrel] = []
    for field in gold_fields:
        if field.value is None:
            continue
        label = normalize_text(field.path[-1])
        labels = {label, *_FIELD_LABEL_ALIASES.get(label, ())}
        value_needles = _value_needles(field.value)
        candidates = [
            chunk.chunk_id
            for chunk in chunk_list
            if any(
                candidate_label in normalized_chunks[chunk.chunk_id] for candidate_label in labels
            )
            and any(needle in normalized_chunks[chunk.chunk_id] for needle in value_needles)
        ]
        if len(candidates) > 1 and field.unit:
            unit = normalize_text(field.unit)
            with_unit = [
                chunk_id for chunk_id in candidates if unit and unit in normalized_chunks[chunk_id]
            ]
            if with_unit:
                candidates = with_unit
        status = (
            "matched" if len(candidates) == 1 else "unmatched" if not candidates else "ambiguous"
        )
        qrels.append(
            SilverQrel(
                field_path=field.path,
                query=_field_query(field),
                relevant_chunk_ids=candidates if status == "matched" else [],
                status=status,
            )
        )
    return qrels


def _family_for_path(path: tuple[str, ...]) -> FieldFamily | None:
    return _SECTION_FAMILIES.get(normalize_text(path[0])) if path else None


class RetrievalEvaluator:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        ingestion=None,
        parser=None,
        chunker=None,
        vector_index=None,
    ) -> None:
        self.settings = settings or Settings()
        self.ingestion = ingestion or LocalFileIngestionService()
        self.parser = parser or SmartParsingService(self.settings)
        self.chunker = chunker or DefaultChunkingService(self.settings)
        self.vector_index = vector_index or build_default_vector_service(self.settings)

    def evaluate(
        self,
        sample: EvaluationSamplePaths,
        gold: dict[tuple[str, ...], TemplateField],
        *,
        k_values: list[int],
    ) -> RetrievalMetrics:
        raw = self.ingestion.ingest(
            str(sample.report_path),
            patient_id=f"evaluation-{sample.sample_id}",
            study_id="evaluation",
        )
        parsed = self.parser.parse(raw)
        chunks = self.chunker.chunk(parsed)
        tenant_id = f"evaluation_{sample.sample_id}"
        self.vector_index.upsert_chunks(
            chunks,
            tenant_id=tenant_id,
            patient_id=raw.patient_id,
            study_id=raw.study_id,
            document_type=parsed.document_type_hint.value,
        )

        qrels = derive_silver_qrels(chunks, gold.values())
        query_results: list[RetrievalQueryEvaluation] = []
        top_k = max(k_values)
        ranking_cache: dict[FieldFamily, list[RetrievalHitSummary]] = {}
        try:
            for qrel in qrels:
                hits: list[RetrievalHitSummary] = []
                recall: dict[str, float | None] = {str(k): None for k in k_values}
                family = _family_for_path(qrel.field_path)
                status = qrel.status
                query = qrel.query
                if status == "matched" and family is None:
                    status = "unsupported_family"
                if status == "matched" and family is not None:
                    query = _PRODUCTION_LAB_QUERIES[family]
                    if family not in ranking_cache:
                        ranked = self.vector_index.search(
                            document_id=raw.document_id,
                            query_text=query,
                            field_family=family,
                            top_k=top_k,
                            tenant_id=tenant_id,
                        )
                        ranking_cache[family] = [
                            RetrievalHitSummary(
                                chunk_id=chunk.chunk_id,
                                rank=rank,
                                score=float(score),
                            )
                            for rank, (chunk, score) in enumerate(ranked, start=1)
                        ]
                    hits = ranking_cache[family]
                    relevant = set(qrel.relevant_chunk_ids)
                    for k in k_values:
                        retrieved = {hit.chunk_id for hit in hits[:k]}
                        recall[str(k)] = safe_rate(len(retrieved & relevant), len(relevant))
                query_results.append(
                    RetrievalQueryEvaluation(
                        field_path=qrel.field_path,
                        query=query,
                        qrel_status=status,
                        relevant_chunk_ids=qrel.relevant_chunk_ids,
                        hits=hits,
                        recall_at_k=recall,
                    )
                )
        finally:
            delete = getattr(self.vector_index, "delete_document", None)
            if callable(delete):
                delete(tenant_id=tenant_id, document_id=raw.document_id)

        aggregate: dict[str, float | None] = {}
        for k in k_values:
            values = [
                value
                for result in query_results
                if (value := result.recall_at_k[str(k)]) is not None
            ]
            aggregate[str(k)] = sum(values) / len(values) if values else None
        return RetrievalMetrics(
            recall_at_k=aggregate,
            evaluable_queries=sum(result.qrel_status == "matched" for result in query_results),
            unmatched_fields=sum(result.qrel_status == "unmatched" for result in query_results),
            ambiguous_fields=sum(result.qrel_status == "ambiguous" for result in query_results),
            unsupported_fields=sum(
                result.qrel_status == "unsupported_family" for result in query_results
            ),
            queries=query_results,
        )
