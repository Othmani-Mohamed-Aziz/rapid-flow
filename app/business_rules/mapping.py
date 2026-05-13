from __future__ import annotations

from app.config.field_registry import FieldRegistry
from app.schemas.enums import DocumentType
from app.schemas.models import ExtractedObservation, FieldCandidate, FieldDefinition


class FieldMappingService:
    """Projette des `ExtractedObservation` vers des champs eCRF déclaratifs."""

    _LAB_KEY_TO_FIELD: dict[str, str] = {
        "AST": "AST_start_AtezoBev_D0",
        "ALT": "ALT_start_AtezoBev_D0",
        "Total_bilirubine": "Total_bilirubine_D0",
        "PLT": "PLT_start_AtezoBev_D0",
        "AFP": "AFP_start_AtezoBev_D0",
    }

    def __init__(self, registry: FieldRegistry) -> None:
        self._registry = registry

    def observation_to_field_name(self, observation: ExtractedObservation) -> str | None:
        extra = observation.extra or {}
        if "canonical_lab_key" in extra:
            mapped = self._LAB_KEY_TO_FIELD.get(str(extra["canonical_lab_key"]))
            if mapped and self._registry.get(mapped):
                return mapped
            return None
        if extra.get("comorbidity") == "Cirrhosis":
            return "Cirrhosis" if self._registry.get("Cirrhosis") else None
        return None

    def build_candidates(
        self,
        *,
        doc_type: DocumentType,
        observations: list[ExtractedObservation],
    ) -> list[FieldCandidate]:
        candidates: list[FieldCandidate] = []
        for obs in observations:
            fname = self.observation_to_field_name(obs)
            if fname is None:
                continue
            field_def = self._registry.get(fname)
            if field_def is None or doc_type not in field_def.document_types_allowed:
                continue
            candidates.append(
                FieldCandidate(
                    field_name=fname,
                    observation=obs,
                    target_column=field_def.target_column,
                    temporal_anchor=None,
                    confidence=obs.confidence,
                )
            )
        return candidates

    def attach_field_definitions(self, field_name: str) -> FieldDefinition | None:
        return self._registry.get(field_name)
