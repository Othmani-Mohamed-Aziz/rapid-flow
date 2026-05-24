"""Chargement du `StudySchema` depuis JSON (fichier étude / export eCRF)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config.ecrf_fields import EXAMPLE_ECRF_FIELDS
from app.schemas.enums import DocumentType, ExtractionFamily, FieldFamily, TemporalScope
from app.schemas.extraction_catalog import ExtractionCatalog, LabAnalyteSpec, NarrativeKeywordSpec
from app.schemas.study_schema import (
    ExtractionStrategy,
    StudyFieldDefinition,
    StudySchema,
    validate_study_schema,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_SCHEMA_PATH = _PROJECT_ROOT / "data" / "study_schema_default.json"

# Clés canoniques lab par défaut (alignées sur `lab_heuristics.LAB_PATTERNS`).
_DEFAULT_LAB_CANONICAL: dict[str, tuple[FieldFamily, ExtractionFamily]] = {
    "AFP": (FieldFamily.INFLAMMATION_BIOMARKERS, ExtractionFamily.LAB_VALUES),
    "AST": (FieldFamily.HEPATIC_BIOCHEMISTRY, ExtractionFamily.LAB_VALUES),
    "ALT": (FieldFamily.HEPATIC_BIOCHEMISTRY, ExtractionFamily.LAB_VALUES),
    "Total_bilirubine": (FieldFamily.HEPATIC_BIOCHEMISTRY, ExtractionFamily.LAB_VALUES),
    "PLT": (FieldFamily.HEMATOLOGY, ExtractionFamily.LAB_VALUES),
}

_DEFAULT_IMAGING_CANONICAL: dict[str, str] = {
    "Size_major_nodule_mm": "lesion_size_mm",
    "RECIST_response": "recist_response",
}

_DEFAULT_FAMILY_QUERIES: dict[str, str] = {
    "hematology": "hématologie hémogramme plaquettes leucocytes",
    "coagulation": "coagulation INR TP TCA fibrinogène",
    "hepatic_biochemistry": "bilan hépatique AST ALT GGT bilirubine",
    "inflammation_biomarkers": "inflammation CRP biomarqueurs AFP",
    "comorbidities": "antécédents comorbidités cirrhose traitements",
    "imaging_recist": (
        "RECIST réponse tumorale cible non cible progression stabilité "
        "scanner TDM IRM mesure lésion millimètres mm"
    ),
}


def study_schema_from_example_fields() -> StudySchema:
    """Construit un schéma à partir de `EXAMPLE_ECRF_FIELDS` (rétrocompat)."""
    fields: list[StudyFieldDefinition] = []
    for fd in EXAMPLE_ECRF_FIELDS:
        canon: str | None = None
        lx_class: str | None = None
        strategy: ExtractionStrategy | None = None

        if fd.field_name == "Cirrhosis":
            canon = "Cirrhosis"
            strategy = ExtractionStrategy.NARRATIVE_KEYWORDS
        elif fd.extraction_family == ExtractionFamily.IMAGING_RECIST:
            for ck, cls in _DEFAULT_IMAGING_CANONICAL.items():
                if ck in fd.field_name or ck in fd.target_column:
                    canon = ck
                    lx_class = cls
                    break
            if canon is None and "RECIST" in fd.field_name:
                canon = "RECIST_response"
                lx_class = "recist_response"
            elif canon is None:
                canon = "Size_major_nodule_mm"
                lx_class = "lesion_size_mm"
            strategy = ExtractionStrategy.IMAGING_LANGEXTRACT
        elif fd.extraction_family == ExtractionFamily.LAB_VALUES:
            for lab_key in _DEFAULT_LAB_CANONICAL:
                if lab_key in fd.field_name:
                    canon = lab_key
                    break
            strategy = ExtractionStrategy.LAB_DETERMINISTIC

        payload = fd.model_dump()
        payload["canonical_key"] = canon
        payload["extraction_strategy"] = strategy
        payload["langextract_class"] = lx_class
        fields.append(StudyFieldDefinition(**payload))
    schema = StudySchema(
        study_id="EXAMPLE",
        schema_version="example-v1",
        fields=fields,
        family_retrieval_queries=dict(_DEFAULT_FAMILY_QUERIES),
    )
    validate_study_schema(schema)
    return schema


def load_study_schema_from_json(path: str | Path) -> StudySchema:
    """Charge et valide un fichier JSON de schéma d'étude."""
    p = Path(path)
    raw: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    fields_raw = raw.get("fields") or []
    fields: list[StudyFieldDefinition] = []
    for item in fields_raw:
        if "extraction_strategy" in item and isinstance(item["extraction_strategy"], str):
            item = {**item, "extraction_strategy": ExtractionStrategy(item["extraction_strategy"])}
        for dt_key in ("document_types_allowed",):
            if dt_key in item:
                item[dt_key] = [DocumentType(v) for v in item[dt_key]]
        for enum_key, enum_cls in (
            ("field_family", FieldFamily),
            ("extraction_family", ExtractionFamily),
            ("temporal_scope", TemporalScope),
        ):
            if enum_key in item and isinstance(item[enum_key], str):
                item[enum_key] = enum_cls(item[enum_key])
        fields.append(StudyFieldDefinition(**item))
    catalog_raw = raw.get("extraction_catalog") or {}
    catalog = _parse_extraction_catalog(catalog_raw)
    schema = StudySchema(
        study_id=str(raw.get("study_id", p.stem)),
        schema_version=str(raw.get("schema_version", "1")),
        fields=fields,
        family_retrieval_queries=dict(raw.get("family_retrieval_queries") or {}),
        extraction_catalog=catalog,
    )
    validate_study_schema(schema)
    return schema


def _parse_extraction_catalog(raw: dict[str, Any]) -> ExtractionCatalog:
    lab_raw = raw.get("lab_analytes") or []
    lab_analytes: list[LabAnalyteSpec] = []
    for item in lab_raw:
        if "field_family" in item and isinstance(item["field_family"], str):
            item = {**item, "field_family": FieldFamily(item["field_family"])}
        lab_analytes.append(LabAnalyteSpec(**item))
    narrative_raw = raw.get("narrative_keywords") or []
    narrative = [NarrativeKeywordSpec(**n) for n in narrative_raw]
    return ExtractionCatalog(
        langextract_class_map=dict(raw.get("langextract_class_map") or {}),
        langextract_class_descriptions=dict(raw.get("langextract_class_descriptions") or {}),
        lab_analytes=lab_analytes,
        narrative_keywords=narrative,
    )


def resolve_study_schema(path: str | Path | None = None) -> StudySchema:
    """Schéma depuis fichier si présent, sinon dérivé des champs exemple."""
    if path is not None:
        return load_study_schema_from_json(path)
    if DEFAULT_STUDY_SCHEMA_PATH.is_file():
        return load_study_schema_from_json(DEFAULT_STUDY_SCHEMA_PATH)
    return study_schema_from_example_fields()
