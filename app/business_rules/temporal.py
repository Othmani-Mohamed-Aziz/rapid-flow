from __future__ import annotations

from datetime import date

from app.schemas.enums import TemporalLabel, TemporalScope
from app.schemas.models import ExtractedObservation, FieldDefinition, ParsedDocument, TemporalAnchor


class TemporalMappingService:
    """
    Applique les ancres temporelles aux observations intermédiaires.

    V1 : pour un bilan de baseline, alignement sur `TemporalLabel.BASELINE`
    et date document si disponible.
    """

    def apply_for_field(
        self,
        parsed: ParsedDocument,
        observation: ExtractedObservation,
        field_def: FieldDefinition,
    ) -> ExtractedObservation:
        label = self._label_for_scope(field_def.temporal_scope)
        event_date = parsed.document_date or date.today()
        doc_date = parsed.document_date
        return observation.model_copy(
            update={
                "temporal_label": label,
                "event_date": event_date,
                "document_date": doc_date,
            }
        )

    def build_anchor(self, field_def: FieldDefinition, parsed: ParsedDocument) -> TemporalAnchor:
        return TemporalAnchor(
            label=self._label_for_scope(field_def.temporal_scope),
            scope=field_def.temporal_scope,
            anchor_date=parsed.document_date,
            line_number=self._line_number(field_def.temporal_scope),
        )

    @staticmethod
    def _label_for_scope(scope: TemporalScope) -> TemporalLabel:
        mapping = {
            TemporalScope.BASELINE: TemporalLabel.BASELINE,
            TemporalScope.FOLLOW_UP: TemporalLabel.FOLLOW_UP,
            TemporalScope.FIRST_IMAGING: TemporalLabel.FIRST_IMAGING_EVALUATION,
            TemporalScope.SECOND_IMAGING: TemporalLabel.SECOND_IMAGING_EVALUATION,
            TemporalScope.LINE_L2: TemporalLabel.LINE_L2,
            TemporalScope.LINE_L3: TemporalLabel.LINE_L3,
            TemporalScope.LINE_L4: TemporalLabel.LINE_L4,
            TemporalScope.UNSPECIFIED: TemporalLabel.UNKNOWN,
        }
        return mapping.get(scope, TemporalLabel.UNKNOWN)

    @staticmethod
    def _line_number(scope: TemporalScope) -> int | None:
        if scope == TemporalScope.LINE_L2:
            return 2
        if scope == TemporalScope.LINE_L3:
            return 3
        if scope == TemporalScope.LINE_L4:
            return 4
        return None
