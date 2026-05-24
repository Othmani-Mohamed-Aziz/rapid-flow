from __future__ import annotations

from collections import defaultdict

from app.config.ecrf_fields import EXAMPLE_ECRF_FIELDS
from app.schemas.enums import DocumentType, ExtractionFamily, FieldFamily
from app.schemas.models import FieldDefinition
from app.schemas.study_schema import StudySchema


class FieldRegistry:
    """Accès indexé aux définitions de champs eCRF."""

    def __init__(self, fields: list[FieldDefinition]) -> None:
        self._by_name: dict[str, FieldDefinition] = {f.field_name: f for f in fields}
        self._by_canonical: dict[str, list[FieldDefinition]] = {}
        for f in fields:
            if f.canonical_key:
                self._by_canonical.setdefault(f.canonical_key, []).append(f)

    @classmethod
    def default(cls) -> FieldRegistry:
        return cls(list(EXAMPLE_ECRF_FIELDS))

    @classmethod
    def from_study_schema(cls, study_schema: StudySchema) -> FieldRegistry:
        return cls(study_schema.to_field_definitions())

    def get(self, field_name: str) -> FieldDefinition | None:
        return self._by_name.get(field_name)

    def by_canonical_key(self, canonical_key: str) -> list[FieldDefinition]:
        return list(self._by_canonical.get(canonical_key, []))

    def all_fields(self) -> list[FieldDefinition]:
        return list(self._by_name.values())

    def for_document_type(self, doc_type: DocumentType) -> list[FieldDefinition]:
        return [f for f in self._by_name.values() if doc_type in f.document_types_allowed]

    def by_field_family(self) -> dict[FieldFamily, list[FieldDefinition]]:
        out: dict[FieldFamily, list[FieldDefinition]] = defaultdict(list)
        for f in self._by_name.values():
            out[f.field_family].append(f)
        return dict(out)

    def families_for_extraction(self, family: ExtractionFamily) -> list[FieldFamily]:
        fams: set[FieldFamily] = set()
        for f in self._by_name.values():
            if f.extraction_family == family:
                fams.add(f.field_family)
        return sorted(fams, key=lambda x: x.value)


def get_default_registry() -> FieldRegistry:
    from app.config.study_schema_provider import resolve_study_schema

    return FieldRegistry.from_study_schema(resolve_study_schema())
