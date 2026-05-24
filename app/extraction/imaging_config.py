"""Configuration d'extraction imagerie dérivée du `StudySchema`."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.study_schema import (
    DEFAULT_LANGEXTRACT_CLASS_DESCRIPTIONS,
    ExtractionJob,
    StudySchema,
)


@dataclass(frozen=True)
class ImagingExtractionConfig:
    """Paramètres LangExtract filtrés par schéma d'étude."""

    allowed_langextract_classes: frozenset[str]
    allowed_canonical_keys: frozenset[str]
    class_to_canonical: dict[str, str] = field(default_factory=dict)
    class_descriptions: dict[str, str] = field(default_factory=dict)
    derive_recist_from_conclusion: bool = True

    @classmethod
    def from_job(cls, job: ExtractionJob, study_schema: StudySchema) -> ImagingExtractionConfig:
        class_to_canon = dict(study_schema.effective_langextract_class_map())
        for ck in job.canonical_keys:
            for lx_cls, canon in class_to_canon.items():
                if canon == ck:
                    class_to_canon[lx_cls] = ck
        allowed_classes = (
            frozenset(job.langextract_classes)
            if job.langextract_classes
            else frozenset(c for c in class_to_canon if class_to_canon[c] in job.canonical_key_set)
        )
        all_descriptions = study_schema.effective_langextract_class_descriptions()
        descriptions = {c: all_descriptions[c] for c in allowed_classes if c in all_descriptions}
        return cls(
            allowed_langextract_classes=allowed_classes,
            allowed_canonical_keys=job.canonical_key_set,
            class_to_canonical=class_to_canon,
            class_descriptions=descriptions,
            derive_recist_from_conclusion=job.derive_recist_from_conclusion,
        )

    @classmethod
    def legacy_default(cls) -> ImagingExtractionConfig:
        from app.schemas.study_schema import DEFAULT_LANGEXTRACT_CLASS_TO_CANONICAL

        return cls(
            allowed_langextract_classes=frozenset(DEFAULT_LANGEXTRACT_CLASS_TO_CANONICAL),
            allowed_canonical_keys=frozenset(DEFAULT_LANGEXTRACT_CLASS_TO_CANONICAL.values()),
            class_to_canonical=dict(DEFAULT_LANGEXTRACT_CLASS_TO_CANONICAL),
            class_descriptions=dict(DEFAULT_LANGEXTRACT_CLASS_DESCRIPTIONS),
            derive_recist_from_conclusion=True,
        )
