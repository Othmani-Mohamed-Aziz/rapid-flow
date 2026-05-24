"""
Heuristiques pour bilans biologiques et texte narratif (hors LLM).

Les patterns lab sont pilotés par `StudySchema.extraction_catalog` (ou défauts intégrés).
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from app.schemas.enums import FieldFamily
from app.schemas.extraction_catalog import LabAnalyteSpec, NarrativeKeywordSpec
from app.schemas.models import ExtractedObservation

_BUILTIN_LAB_SPECS: list[LabAnalyteSpec] = [
    LabAnalyteSpec(
        canonical_key="AST",
        pattern=r"\bAST\b\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:UI/L|U/L|iu/l)?",
        unit="U/L",
        field_family=FieldFamily.HEPATIC_BIOCHEMISTRY,
    ),
    LabAnalyteSpec(
        canonical_key="ALT",
        pattern=r"\bALT\b\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:UI/L|U/L|iu/l)?",
        unit="U/L",
        field_family=FieldFamily.HEPATIC_BIOCHEMISTRY,
    ),
    LabAnalyteSpec(
        canonical_key="Total_bilirubine",
        pattern=(
            r"\b(?:bilirubine\s+totale|total\s+bilirubin)\b\s*[:=]?\s*"
            r"([0-9]+(?:[.,][0-9]+)?)\s*(?:µmol/L|umol/l|mg/dl)?"
        ),
        unit="µmol/L",
        field_family=FieldFamily.HEPATIC_BIOCHEMISTRY,
    ),
    LabAnalyteSpec(
        canonical_key="PLT",
        pattern=r"\b(?:plt|plaquettes)\b\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:G/L|g/l|10\^9/L)?",
        unit="G/L",
        field_family=FieldFamily.HEMATOLOGY,
    ),
    LabAnalyteSpec(
        canonical_key="AFP",
        pattern=r"\bAFP\b\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:UI/mL|ng/mL|kUI/L)?",
        unit=None,
        field_family=FieldFamily.INFLAMMATION_BIOMARKERS,
    ),
    LabAnalyteSpec(
        canonical_key="CRP",
        pattern=r"\bCRP\b\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:mg/L|mg/l)?",
        unit="mg/L",
        field_family=FieldFamily.INFLAMMATION_BIOMARKERS,
    ),
    LabAnalyteSpec(
        canonical_key="INR",
        pattern=r"\bINR\b\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)",
        unit=None,
        field_family=FieldFamily.COAGULATION,
    ),
]

_DEFAULT_NARRATIVE_SPECS: list[NarrativeKeywordSpec] = [
    NarrativeKeywordSpec(
        canonical_key="Cirrhosis",
        keywords=["cirrhose", "cirrhosis"],
    ),
]


def builtin_lab_analyte_specs() -> list[LabAnalyteSpec]:
    return list(_BUILTIN_LAB_SPECS)


def builtin_narrative_specs() -> list[NarrativeKeywordSpec]:
    return list(_DEFAULT_NARRATIVE_SPECS)


# Rétrocompat tests / imports historiques.
LAB_PATTERNS: list[tuple[str, re.Pattern[str], str | None, FieldFamily]] = [
    (s.canonical_key, s.compiled_pattern(), s.unit, s.field_family) for s in _BUILTIN_LAB_SPECS
]


def parse_numeric(raw: str) -> float | int:
    cleaned = raw.replace(",", ".")
    val = float(cleaned)
    return int(val) if val.is_integer() else val


def _line_has_negation_before_keyword(
    line: str, keyword: str, negation_patterns: list[str]
) -> bool:
    lower = line.lower()
    pos = lower.find(keyword.lower())
    if pos < 0:
        return False
    prefix = lower[:pos]
    return any(re.search(pat, prefix, re.I) for pat in negation_patterns)


def extract_lab_observations(
    text: str,
    *,
    extraction_method: str,
    model_name: str | None,
    schema_version: str | None,
    source_chunk_id: str | None,
    allowed_canonical_keys: frozenset[str] | set[str] | None = None,
    analyte_specs: list[LabAnalyteSpec] | None = None,
) -> list[ExtractedObservation]:
    specs = analyte_specs if analyte_specs is not None else builtin_lab_analyte_specs()
    obs: list[ExtractedObservation] = []
    for spec in specs:
        if allowed_canonical_keys is not None and spec.canonical_key not in allowed_canonical_keys:
            continue
        pattern = spec.compiled_pattern()
        m = pattern.search(text)
        if not m:
            continue
        raw_val = m.group(1)
        try:
            norm: float | int | str = parse_numeric(raw_val)
        except ValueError:
            norm = raw_val
        evidence = m.group(0).strip()
        obs.append(
            ExtractedObservation(
                observation_id=str(uuid.uuid4()),
                field_family=spec.field_family,
                raw_value=raw_val,
                normalized_value=norm,
                unit=spec.unit,
                confidence=0.82,
                evidence_text=evidence,
                source_chunk_id=source_chunk_id,
                extraction_method=extraction_method,
                model_name=model_name,
                schema_version=schema_version,
                extra={"canonical_lab_key": spec.canonical_key},
            )
        )
    return obs


def extract_narrative_keyword_observations(
    text: str,
    *,
    extraction_method: str,
    model_name: str | None,
    schema_version: str | None,
    source_chunk_id: str | None,
    specs: list[NarrativeKeywordSpec],
    allowed_canonical_keys: frozenset[str] | set[str] | None = None,
    field_family: FieldFamily = FieldFamily.COMORBIDITIES,
) -> list[ExtractedObservation]:
    obs: list[ExtractedObservation] = []
    for spec in specs:
        if allowed_canonical_keys is not None and spec.canonical_key not in allowed_canonical_keys:
            continue
        matched_line: str | None = None
        for line in text.splitlines():
            lower = line.lower()
            if not any(kw.lower() in lower for kw in spec.keywords):
                continue
            hit_kw = next(kw for kw in spec.keywords if kw.lower() in lower)
            if _line_has_negation_before_keyword(line, hit_kw, spec.negation_patterns):
                continue
            matched_line = line.strip()
            break
        if matched_line is None:
            continue
        obs.append(
            ExtractedObservation(
                observation_id=str(uuid.uuid4()),
                field_family=field_family,
                raw_value="mention",
                normalized_value=True,
                unit=None,
                confidence=0.8,
                evidence_text=matched_line,
                source_chunk_id=source_chunk_id,
                extraction_method=extraction_method,
                model_name=model_name,
                schema_version=schema_version,
                extra={"canonical_key": spec.canonical_key, "comorbidity": spec.canonical_key},
            )
        )
    return obs


def extract_comorbidity_observations(
    text: str,
    *,
    extraction_method: str,
    model_name: str | None,
    schema_version: str | None,
    source_chunk_id: str | None,
) -> list[ExtractedObservation]:
    """Rétrocompat : délègue à `extract_narrative_keyword_observations` (Cirrhosis par défaut)."""
    return extract_narrative_keyword_observations(
        text,
        extraction_method=extraction_method,
        model_name=model_name,
        schema_version=schema_version,
        source_chunk_id=source_chunk_id,
        specs=builtin_narrative_specs(),
    )


def build_langextract_schema_stub(family: FieldFamily) -> dict[str, Any]:
    """Schéma JSON minimal pour futur appel LangExtract réel."""
    return {
        "type": "object",
        "properties": {
            "observations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "value": {"type": "number"},
                        "unit": {"type": "string"},
                    },
                    "required": ["code", "value"],
                },
            }
        },
        "x-field-family": family.value,
    }
