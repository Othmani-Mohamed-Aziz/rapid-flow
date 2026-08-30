"""Normalisation des valeurs extraites selon `normalization_rule` du schéma."""

from __future__ import annotations

from app.extraction.imaging_langextract import _normalize_recist_label, _parse_mm_value
from app.schemas.models import ExtractedObservation, FieldDefinition
from app.schemas.study_schema import _RECIST_CANON_CODES


class NormalizationService:
    """Applique les règles déclaratives post-extraction."""

    def apply(
        self, observation: ExtractedObservation, field_def: FieldDefinition
    ) -> ExtractedObservation:
        rule = field_def.normalization_rule
        if not rule:
            return observation
        val = observation.normalized_value
        if rule == "recist_category":
            if isinstance(val, str):
                norm = _normalize_recist_label(val)
                if norm and norm in _RECIST_CANON_CODES:
                    return observation.model_copy(update={"normalized_value": norm})
            return observation
        if rule == "numeric_mm" and val is not None:
            parsed = _parse_mm_value(
                observation.evidence_text or observation.raw_value or "",
                {"value_mm": val, "unit": observation.unit},
            )
            if parsed is None:
                try:
                    parsed = float(str(val).replace(",", "."))
                    parsed = int(parsed) if parsed.is_integer() else parsed
                except (TypeError, ValueError):
                    return observation
            return observation.model_copy(
                update={
                    "normalized_value": parsed,
                    "unit": observation.unit or "mm",
                }
            )
        if rule == "bool_yes_no":
            if isinstance(val, bool):
                return observation
            if isinstance(val, str):
                low = val.lower()
                if low in ("true", "yes", "oui", "1"):
                    return observation.model_copy(update={"normalized_value": True})
                if low in ("false", "no", "non", "0"):
                    return observation.model_copy(update={"normalized_value": False})
        return observation
