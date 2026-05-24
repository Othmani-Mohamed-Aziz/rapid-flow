"""Planification des jobs d'extraction à partir d'un `StudySchema`."""

from __future__ import annotations

from app.schemas.enums import DocumentType, FieldFamily
from app.schemas.study_schema import (
    ExtractionJob,
    ExtractionStrategy,
    StudyFieldDefinition,
    StudySchema,
)


def plan_extraction_jobs(
    study_schema: StudySchema,
    doc_type: DocumentType,
) -> list[ExtractionJob]:
    """
    Regroupe les champs extractibles du schéma par (stratégie, field_family).

    Un job = une stratégie d'extracteur + une famille pour le retrieval / dédup.
    """
    fields = study_schema.extractable_fields_for_document_type(doc_type)
    buckets: dict[tuple[ExtractionStrategy, FieldFamily], list[StudyFieldDefinition]] = {}

    for f in fields:
        strategy = f.resolved_extraction_strategy()
        if strategy == ExtractionStrategy.NONE:
            continue
        key = (strategy, f.field_family)
        buckets.setdefault(key, []).append(f)

    lx_map = study_schema.effective_langextract_class_map()
    jobs: list[ExtractionJob] = []
    for (strategy, family), group in sorted(
        buckets.items(), key=lambda x: (x[0][0].value, x[0][1].value)
    ):
        canonical_keys = sorted({fk for fk in (g.canonical_key for g in group) if fk})
        lx_classes: set[str] = set()
        for g in group:
            lx = g.resolved_langextract_class()
            if lx:
                lx_classes.add(lx)
        if strategy == ExtractionStrategy.IMAGING_LANGEXTRACT:
            for lx_cls, canon in lx_map.items():
                if canon in canonical_keys:
                    lx_classes.add(lx_cls)
            if "lesion_description" in canonical_keys:
                lx_classes.add("lesion_description")
            derive_recist = "RECIST_response" in canonical_keys
            if derive_recist:
                lx_classes.add("imaging_conclusion")
        else:
            derive_recist = False

        jobs.append(
            ExtractionJob(
                extraction_strategy=strategy,
                field_family=family,
                canonical_keys=canonical_keys,
                langextract_classes=sorted(lx_classes),
                field_names=[g.field_name for g in group],
                derive_recist_from_conclusion=derive_recist,
                study_schema_version=study_schema.schema_version,
            )
        )
    return jobs


def field_families_from_jobs(jobs: list[ExtractionJob]) -> list[FieldFamily]:
    return sorted({j.field_family for j in jobs}, key=lambda f: f.value)
