"""Désambiguïsation champ eCRF lorsque plusieurs partagent une `canonical_key`."""

from __future__ import annotations

from app.schemas.enums import DocumentType, TemporalScope
from app.schemas.models import FieldDefinition, ParsedDocument

# Ordre de préférence quand plusieurs champs partagent une clé et plusieurs scopes sont plausibles.
_SCOPE_PRIORITY: dict[DocumentType, tuple[TemporalScope, ...]] = {
    DocumentType.LAB_BLOOD_PANEL: (TemporalScope.BASELINE, TemporalScope.FOLLOW_UP),
    DocumentType.IMAGING_REPORT: (
        TemporalScope.FIRST_IMAGING,
        TemporalScope.SECOND_IMAGING,
        TemporalScope.BASELINE,
    ),
    DocumentType.CLINICAL_LETTER: (
        TemporalScope.UNSPECIFIED,
        TemporalScope.BASELINE,
        TemporalScope.FOLLOW_UP,
    ),
}


def preferred_temporal_scopes(
    doc_type: DocumentType,
    parsed: ParsedDocument | None,
) -> set[TemporalScope]:
    """Portées temporelles plausibles pour un document routé."""
    if doc_type == DocumentType.IMAGING_REPORT:
        return {TemporalScope.FIRST_IMAGING, TemporalScope.SECOND_IMAGING, TemporalScope.BASELINE}
    if doc_type == DocumentType.LAB_BLOOD_PANEL:
        return {TemporalScope.BASELINE, TemporalScope.FOLLOW_UP}
    if doc_type == DocumentType.CLINICAL_LETTER:
        return {TemporalScope.UNSPECIFIED, TemporalScope.BASELINE, TemporalScope.FOLLOW_UP}
    if parsed and parsed.document_type_hint == DocumentType.IMAGING_REPORT:
        return {TemporalScope.FIRST_IMAGING, TemporalScope.SECOND_IMAGING}
    return {TemporalScope.UNSPECIFIED, TemporalScope.BASELINE, TemporalScope.FOLLOW_UP}


def select_fields_for_observation(
    matched: list[FieldDefinition],
    *,
    doc_type: DocumentType,
    parsed: ParsedDocument | None = None,
) -> list[FieldDefinition]:
    """
    Réduit les champs candidats quand plusieurs partagent la même clé canonique.

    Priorité aux `temporal_scope` compatibles avec le type de document.
    """
    if len(matched) <= 1:
        return matched
    preferred = preferred_temporal_scopes(doc_type, parsed)
    tier_a = [f for f in matched if f.temporal_scope in preferred]
    if tier_a:
        if len(tier_a) == 1:
            return tier_a
        priority = _SCOPE_PRIORITY.get(doc_type, ())
        for scope in priority:
            scoped = [f for f in tier_a if f.temporal_scope == scope]
            if scoped:
                return scoped
        return tier_a
    tier_b = [f for f in matched if f.temporal_scope == TemporalScope.UNSPECIFIED]
    return tier_b if tier_b else [matched[0]]
