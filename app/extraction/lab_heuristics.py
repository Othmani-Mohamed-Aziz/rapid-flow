"""
Heuristiques V1 pour bilans biologiques (hors LLM).

Utilisé par `LangExtractExtractor` en mode mock et comme filet de sécurité.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from app.schemas.enums import FieldFamily
from app.schemas.models import ExtractedObservation

# (canonical_key, regex, unit, field_family)
LAB_PATTERNS: list[tuple[str, re.Pattern[str], str | None, FieldFamily]] = [
    (
        "AST",
        re.compile(r"\bAST\b\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:UI/L|U/L|iu/l)?", re.I),
        "U/L",
        FieldFamily.HEPATIC_BIOCHEMISTRY,
    ),
    (
        "ALT",
        re.compile(r"\bALT\b\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:UI/L|U/L|iu/l)?", re.I),
        "U/L",
        FieldFamily.HEPATIC_BIOCHEMISTRY,
    ),
    (
        "Total_bilirubine",
        re.compile(
            r"\b(?:bilirubine\s+totale|total\s+bilirubin)\b\s*[:=]?\s*"
            r"([0-9]+(?:[.,][0-9]+)?)\s*(?:µmol/L|umol/l|mg/dl)?",
            re.I,
        ),
        "µmol/L",
        FieldFamily.HEPATIC_BIOCHEMISTRY,
    ),
    (
        "PLT",
        re.compile(
            r"\b(?:plt|plaquettes)\b\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:G/L|g/l|10\^9/L)?",
            re.I,
        ),
        "G/L",
        FieldFamily.HEMATOLOGY,
    ),
    (
        "AFP",
        re.compile(r"\bAFP\b\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:UI/mL|ng/mL|kUI/L)?", re.I),
        None,
        FieldFamily.INFLAMMATION_BIOMARKERS,
    ),
    (
        "CRP",
        re.compile(r"\bCRP\b\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:mg/L|mg/l)?", re.I),
        "mg/L",
        FieldFamily.INFLAMMATION_BIOMARKERS,
    ),
    (
        "INR",
        re.compile(r"\bINR\b\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)", re.I),
        None,
        FieldFamily.COAGULATION,
    ),
]


def parse_numeric(raw: str) -> float | int:
    cleaned = raw.replace(",", ".")
    val = float(cleaned)
    return int(val) if val.is_integer() else val


def extract_lab_observations(
    text: str,
    *,
    extraction_method: str,
    model_name: str | None,
    schema_version: str | None,
    source_chunk_id: str | None,
) -> list[ExtractedObservation]:
    obs: list[ExtractedObservation] = []
    for key, pattern, unit, fam in LAB_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        raw_val = m.group(1)
        norm: float | int | str
        try:
            norm = parse_numeric(raw_val)
        except ValueError:
            norm = raw_val
        evidence = m.group(0).strip()
        obs.append(
            ExtractedObservation(
                observation_id=str(uuid.uuid4()),
                field_family=fam,
                raw_value=raw_val,
                normalized_value=norm,
                unit=unit,
                confidence=0.82,
                evidence_text=evidence,
                source_chunk_id=source_chunk_id,
                extraction_method=extraction_method,
                model_name=model_name,
                schema_version=schema_version,
                extra={"canonical_lab_key": key},
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
    """Repères narratifs ultra-simples pour la V1 (remplacé par LLM spécialisé)."""
    obs: list[ExtractedObservation] = []
    lower = text.lower()
    if "cirrhose" in lower or "cirrhosis" in lower:
        obs.append(
            ExtractedObservation(
                observation_id=str(uuid.uuid4()),
                field_family=FieldFamily.COMORBIDITIES,
                raw_value="mention",
                normalized_value=True,
                unit=None,
                confidence=0.8,
                evidence_text=next(
                    (line.strip() for line in text.splitlines() if "cirrh" in line.lower()),
                    "cirrhosis mention",
                ),
                source_chunk_id=source_chunk_id,
                extraction_method=extraction_method,
                model_name=model_name,
                schema_version=schema_version,
                extra={"comorbidity": "Cirrhosis"},
            )
        )
    return obs


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
