"""Schéma d'étude eCRF — contrat interne pour extraction et mapping dynamiques."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.enums import DocumentType, ExtractionFamily, FieldFamily
from app.schemas.extraction_catalog import ExtractionCatalog, LabAnalyteSpec, NarrativeKeywordSpec
from app.schemas.models import FieldDefinition


class ExtractionStrategy(str, Enum):
    """Stratégie d'extraction rattachée à un champ du schéma d'étude."""

    NONE = "none"
    LAB_DETERMINISTIC = "lab_deterministic"
    IMAGING_LANGEXTRACT = "imaging_langextract"
    NARRATIVE_KEYWORDS = "narrative_keywords"


# Classe LangExtract → clé canonique (observation) par défaut.
DEFAULT_LANGEXTRACT_CLASS_TO_CANONICAL: dict[str, str] = {
    "lesion_size_mm": "Size_major_nodule_mm",
    "recist_response": "RECIST_response",
    "lesion_description": "lesion_description",
    "imaging_conclusion": "imaging_conclusion",
}

# Descriptions LangExtract par défaut (surchargées via `ExtractionCatalog`).
DEFAULT_LANGEXTRACT_CLASS_DESCRIPTIONS: dict[str, str] = {
    "lesion_size_mm": (
        "taille d'une lésion ou nodule en millimètres (valeur numérique dans attributes.value_mm)"
    ),
    "lesion_description": "description localisée d'une lésion (segment, organe)",
    "recist_response": (
        "catégorie de réponse tumorale si mentionnée (CR, PR, SD, PD, NE ou libellé français)"
    ),
    "imaging_conclusion": "phrase de conclusion synthétique du radiologue",
}

_RECIST_CANON_CODES = frozenset({"CR", "PR", "SD", "PD", "NE"})

_EXTRACTION_FAMILY_TO_STRATEGY: dict[ExtractionFamily, ExtractionStrategy] = {
    ExtractionFamily.LAB_VALUES: ExtractionStrategy.LAB_DETERMINISTIC,
    ExtractionFamily.IMAGING_RECIST: ExtractionStrategy.IMAGING_LANGEXTRACT,
    ExtractionFamily.NARRATIVE_CLINICAL: ExtractionStrategy.NARRATIVE_KEYWORDS,
}


class StudyFieldDefinition(FieldDefinition):
    """Champ eCRF enrichi pour piloter extraction et retrieval."""

    canonical_key: str | None = None
    extraction_strategy: ExtractionStrategy | None = None
    langextract_class: str | None = Field(
        default=None,
        description="Classe LangExtract (ex. lesion_size_mm) si différente de la clé canonique.",
    )
    retrieval_query: str | None = None

    def resolved_extraction_strategy(self) -> ExtractionStrategy:
        if self.extraction_strategy is not None:
            return self.extraction_strategy
        return _EXTRACTION_FAMILY_TO_STRATEGY.get(self.extraction_family, ExtractionStrategy.NONE)

    def resolved_langextract_class(self) -> str | None:
        if self.langextract_class:
            return self.langextract_class
        if (
            self.canonical_key
            and self.resolved_extraction_strategy() == ExtractionStrategy.IMAGING_LANGEXTRACT
        ):
            for lx_cls, canon in DEFAULT_LANGEXTRACT_CLASS_TO_CANONICAL.items():
                if canon == self.canonical_key:
                    return lx_cls
        return None


class StudySchema(BaseModel):
    """Schéma versionné pour une étude clinique."""

    study_id: str
    schema_version: str = "1"
    fields: list[StudyFieldDefinition] = Field(default_factory=list)
    family_retrieval_queries: dict[str, str] = Field(
        default_factory=dict,
        description="Requêtes RAG par `field_family` (clé = valeur enum).",
    )
    extraction_catalog: ExtractionCatalog = Field(default_factory=ExtractionCatalog)

    def effective_langextract_class_map(self) -> dict[str, str]:
        merged = dict(DEFAULT_LANGEXTRACT_CLASS_TO_CANONICAL)
        merged.update(self.extraction_catalog.langextract_class_map)
        return merged

    def effective_langextract_class_descriptions(self) -> dict[str, str]:
        merged = dict(DEFAULT_LANGEXTRACT_CLASS_DESCRIPTIONS)
        merged.update(self.extraction_catalog.langextract_class_descriptions)
        return merged

    def resolved_lab_analytes(self) -> list[LabAnalyteSpec]:
        if self.extraction_catalog.lab_analytes:
            return list(self.extraction_catalog.lab_analytes)
        from app.extraction.lab_heuristics import builtin_lab_analyte_specs

        return builtin_lab_analyte_specs()

    def lab_analytes_for_keys(
        self, canonical_keys: frozenset[str] | set[str]
    ) -> list[LabAnalyteSpec]:
        if not canonical_keys:
            return self.resolved_lab_analytes()
        return [a for a in self.resolved_lab_analytes() if a.canonical_key in canonical_keys]

    def resolved_narrative_specs(self) -> list[NarrativeKeywordSpec]:
        if self.extraction_catalog.narrative_keywords:
            return list(self.extraction_catalog.narrative_keywords)
        from app.extraction.lab_heuristics import builtin_narrative_specs

        return builtin_narrative_specs()

    def narrative_specs_for_keys(
        self, canonical_keys: frozenset[str] | set[str]
    ) -> list[NarrativeKeywordSpec]:
        specs = self.resolved_narrative_specs()
        if not canonical_keys:
            return specs
        return [s for s in specs if s.canonical_key in canonical_keys]

    def fields_for_document_type(self, doc_type: DocumentType) -> list[StudyFieldDefinition]:
        return [f for f in self.fields if doc_type in f.document_types_allowed]

    def extractable_fields_for_document_type(
        self, doc_type: DocumentType
    ) -> list[StudyFieldDefinition]:
        return [
            f
            for f in self.fields_for_document_type(doc_type)
            if f.resolved_extraction_strategy() != ExtractionStrategy.NONE
        ]

    def retrieval_query_for_family(self, family: FieldFamily) -> str | None:
        key = family.value
        if key in self.family_retrieval_queries:
            return self.family_retrieval_queries[key]
        for f in self.fields:
            if f.field_family == family and f.retrieval_query:
                return f.retrieval_query
        return None

    def to_field_definitions(self) -> list[FieldDefinition]:
        out: list[FieldDefinition] = []
        for f in self.fields:
            data = f.model_dump()
            strat = data.get("extraction_strategy")
            if isinstance(strat, ExtractionStrategy):
                data["extraction_strategy"] = strat.value
            out.append(FieldDefinition(**data))
        return out


class ExtractionJob(BaseModel):
    """Lot d'extraction planifié pour une famille / stratégie."""

    extraction_strategy: ExtractionStrategy
    field_family: FieldFamily
    canonical_keys: list[str] = Field(default_factory=list)
    langextract_classes: list[str] = Field(default_factory=list)
    field_names: list[str] = Field(default_factory=list)
    derive_recist_from_conclusion: bool = False
    study_schema_version: str = "1"

    @property
    def canonical_key_set(self) -> frozenset[str]:
        return frozenset(self.canonical_keys)

    @property
    def langextract_class_set(self) -> frozenset[str]:
        return frozenset(self.langextract_classes)


def validate_study_schema(schema: StudySchema) -> None:
    """Lève `ValueError` si le schéma est incohérent pour l'extraction."""
    errors: list[str] = []
    seen_names: set[str] = set()
    for f in schema.fields:
        if f.field_name in seen_names:
            errors.append(f"field_name dupliqué : {f.field_name!r}")
        seen_names.add(f.field_name)
        strat = f.resolved_extraction_strategy()
        if strat != ExtractionStrategy.NONE and not (f.canonical_key or "").strip():
            errors.append(
                f"{f.field_name!r} : canonical_key obligatoire lorsque extraction_strategy={strat.value}"
            )
        if strat == ExtractionStrategy.IMAGING_LANGEXTRACT and f.canonical_key:
            lx_map = schema.effective_langextract_class_map()
            if f.canonical_key not in lx_map.values() and not f.langextract_class:
                errors.append(
                    f"{f.field_name!r} : canonical_key {f.canonical_key!r} sans langextract_class "
                    f"ni entrée dans extraction_catalog.langextract_class_map"
                )
    if errors:
        raise ValueError("StudySchema invalide :\n- " + "\n- ".join(errors))
