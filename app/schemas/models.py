from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.enums import (
    DocumentType,
    ExtractionFamily,
    FieldFamily,
    TemporalLabel,
    TemporalScope,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RawDocument(BaseModel):
    """Document brut après ingestion (bytes + métadonnées)."""

    document_id: str
    patient_id: str
    study_id: str
    source_path: str
    mime_type: str
    content_bytes: bytes
    ingested_at: datetime = Field(default_factory=_utc_now)


class DocumentSection(BaseModel):
    """Bloc texte (ex. section Markdown Docling ou segment logique)."""

    heading: str | None = None
    body: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class StructuredLabLine(BaseModel):
    """Ligne de laboratoire structurée (post-PDF / post-OCR), indépendante du pipeline eCRF."""

    name: str
    canonical_name: str | None = None
    value: str | float | int | None = None
    unit: str | None = None
    section: str | None = None
    raw_text: str
    confidence: float = Field(default=0.78, ge=0.0, le=1.0)


class ParsedDocument(BaseModel):
    """Document après parsing texte / structure légère."""

    document_id: str
    patient_id: str
    study_id: str
    full_text: str
    document_type_hint: DocumentType = DocumentType.UNKNOWN
    language: str | None = None
    document_date: date | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_path: str | None = None
    structured_sections: list[DocumentSection] = Field(default_factory=list)
    structured_lab_lines: list[StructuredLabLine] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    """Fragment indexable avec lien vers le document source."""

    chunk_id: str
    document_id: str
    text: str
    field_family: FieldFamily | None = None
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    """Résultat de retrieval (score + chunk)."""

    chunk: DocumentChunk
    score: float
    query_family: FieldFamily
    rank: int


class ExtractedObservation(BaseModel):
    """Observation clinique intermédiaire — pas une cellule eCRF finale."""

    observation_id: str
    field_family: FieldFamily
    raw_value: str | None = None
    normalized_value: str | float | int | bool | None = None
    unit: str | None = None
    document_date: date | None = None
    event_date: date | None = None
    temporal_label: TemporalLabel = TemporalLabel.UNKNOWN
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_text: str | None = None
    source_chunk_id: str | None = None
    extraction_method: str
    model_name: str | None = None
    schema_version: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class FieldCandidate(BaseModel):
    """Candidat pour un champ eCRF après mapping métier."""

    field_name: str
    observation: ExtractedObservation
    target_column: str
    temporal_anchor: TemporalAnchor | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class TemporalAnchor(BaseModel):
    """Ancrage temporel explicite pour règles métier."""

    label: TemporalLabel
    scope: TemporalScope
    anchor_date: date | None = None
    line_number: int | None = Field(
        default=None,
        description="Ligne thérapeutique (2,3,4) si applicable.",
    )


class EcrfCellUpdate(BaseModel):
    """Mise à jour d'une cellule eCRF (pré-ETL)."""

    patient_id: str
    study_id: str
    column_key: str
    value: str | float | int | bool | None
    provenance: FieldCandidate
    validated: bool = False


class AuditRecord(BaseModel):
    """Trace d'audit pour une étape ou un lot d'extractions."""

    record_id: str
    timestamp: datetime = Field(default_factory=_utc_now)
    pipeline_version: str
    document_id: str
    patient_id: str
    study_id: str
    step: str
    model_name: str | None = None
    source_chunk_id: str | None = None
    evidence_excerpt: str | None = None
    confidence: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class FieldDefinition(BaseModel):
    """Déclaration d'un champ eCRF (registry)."""

    field_name: str
    field_type: str
    document_types_allowed: list[DocumentType]
    extraction_family: ExtractionFamily
    field_family: FieldFamily
    temporal_scope: TemporalScope
    normalization_rule: str | None = None
    target_column: str
    autofill_threshold: float = Field(ge=0.0, le=1.0, default=0.65)
    canonical_key: str | None = None
    extraction_strategy: str | None = None
    langextract_class: str | None = None
    retrieval_query: str | None = None


class PipelineResult(BaseModel):
    """Sortie agrégée du pipeline V1."""

    document_id: str
    patient_id: str
    study_id: str
    document_type: DocumentType
    observations: list[ExtractedObservation]
    cell_updates: list[EcrfCellUpdate]
    audit_trail: list[AuditRecord]
    export_paths: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    #: Clé ligne gabarit MA (``ID_current_base``) ; peut différer de ``patient_id``.
    ma_patient_key: str | None = None
