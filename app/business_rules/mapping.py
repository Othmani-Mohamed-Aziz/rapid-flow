from __future__ import annotations

from app.business_rules.field_resolution import select_fields_for_observation
from app.config.field_registry import FieldRegistry
from app.schemas.enums import DocumentType
from app.schemas.models import ExtractedObservation, FieldCandidate, FieldDefinition, ParsedDocument


class FieldMappingService:
    """Projette des `ExtractedObservation` vers des champs eCRF via `canonical_key` du schéma."""

    def __init__(self, registry: FieldRegistry) -> None:
        self._registry = registry

    def observation_canonical_keys(self, observation: ExtractedObservation) -> list[str]:
        extra = observation.extra or {}
        keys: list[str] = []
        if ck := extra.get("canonical_key"):
            keys.append(str(ck))
        if "canonical_lab_key" in extra:
            keys.append(str(extra["canonical_lab_key"]))
        if "canonical_imaging_key" in extra:
            keys.append(str(extra["canonical_imaging_key"]))
        if extra.get("comorbidity"):
            keys.append(str(extra["comorbidity"]))
        return list(dict.fromkeys(keys))

    def fields_for_observation(
        self,
        observation: ExtractedObservation,
        *,
        doc_type: DocumentType | None = None,
        parsed: ParsedDocument | None = None,
    ) -> list[FieldDefinition]:
        canon_keys = self.observation_canonical_keys(observation)
        if not canon_keys:
            return []
        matched: list[FieldDefinition] = []
        for fd in self._registry.all_fields():
            if fd.canonical_key and fd.canonical_key in canon_keys:
                matched.append(fd)
        if doc_type is not None:
            matched = select_fields_for_observation(matched, doc_type=doc_type, parsed=parsed)
        return matched

    def observation_to_field_name(
        self,
        observation: ExtractedObservation,
        *,
        doc_type: DocumentType | None = None,
        parsed: ParsedDocument | None = None,
    ) -> str | None:
        fields = self.fields_for_observation(observation, doc_type=doc_type, parsed=parsed)
        if not fields:
            return None
        return fields[0].field_name

    def build_candidates(
        self,
        *,
        doc_type: DocumentType,
        observations: list[ExtractedObservation],
        parsed: ParsedDocument | None = None,
        normalize_fn=None,
    ) -> list[FieldCandidate]:
        candidates: list[FieldCandidate] = []
        for obs in observations:
            for field_def in self.fields_for_observation(obs, doc_type=doc_type, parsed=parsed):
                if doc_type not in field_def.document_types_allowed:
                    continue
                observation = obs
                if normalize_fn is not None:
                    observation = normalize_fn(obs, field_def)
                candidates.append(
                    FieldCandidate(
                        field_name=field_def.field_name,
                        observation=observation,
                        target_column=field_def.target_column,
                        temporal_anchor=None,
                        confidence=observation.confidence,
                    )
                )
        return candidates

    def attach_field_definitions(self, field_name: str) -> FieldDefinition | None:
        return self._registry.get(field_name)
