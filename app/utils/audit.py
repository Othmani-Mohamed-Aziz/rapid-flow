from __future__ import annotations

import uuid
from typing import Any

from app.schemas.models import AuditRecord, ExtractedObservation


class AuditTrailService:
    """Construit des enregistrements d'audit homogènes."""

    def __init__(self, pipeline_version: str) -> None:
        self.pipeline_version = pipeline_version

    def record_step(
        self,
        *,
        document_id: str,
        patient_id: str,
        study_id: str,
        step: str,
        model_name: str | None = None,
        source_chunk_id: str | None = None,
        evidence_excerpt: str | None = None,
        confidence: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditRecord:
        return AuditRecord(
            record_id=str(uuid.uuid4()),
            pipeline_version=self.pipeline_version,
            document_id=document_id,
            patient_id=patient_id,
            study_id=study_id,
            step=step,
            model_name=model_name,
            source_chunk_id=source_chunk_id,
            evidence_excerpt=evidence_excerpt,
            confidence=confidence,
            details=details or {},
        )

    def from_observation(
        self,
        *,
        document_id: str,
        patient_id: str,
        study_id: str,
        step: str,
        observation: ExtractedObservation,
    ) -> AuditRecord:
        return self.record_step(
            document_id=document_id,
            patient_id=patient_id,
            study_id=study_id,
            step=step,
            model_name=observation.model_name,
            source_chunk_id=observation.source_chunk_id,
            evidence_excerpt=(observation.evidence_text or "")[:500],
            confidence=observation.confidence,
            details={
                "field_family": observation.field_family.value,
                "extraction_method": observation.extraction_method,
            },
        )
