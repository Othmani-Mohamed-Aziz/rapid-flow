"""Catalogue d'extraction chargeable depuis le `StudySchema` (patterns lab, classes LangExtract)."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

from app.schemas.enums import FieldFamily


class LabAnalyteSpec(BaseModel):
    """Définition d'un analyte labo (regex + métadonnées)."""

    canonical_key: str
    pattern: str
    unit: str | None = None
    field_family: FieldFamily

    @field_validator("pattern")
    @classmethod
    def _pattern_compiles(cls, v: str) -> str:
        re.compile(v)
        return v

    def compiled_pattern(self) -> re.Pattern[str]:
        return re.compile(self.pattern, re.I)


class NarrativeKeywordSpec(BaseModel):
    """Mots-clés narratifs pour une clé canonique (ex. comorbidité)."""

    canonical_key: str
    keywords: list[str] = Field(min_length=1)
    negation_patterns: list[str] = Field(
        default_factory=lambda: [
            r"\bpas\s+de\s+",
            r"\bsans\s+",
            r"\bno\s+",
            r"\bwithout\s+",
            r"\bn[ée]gatif\s+pour\s+",
        ],
    )


class ExtractionCatalog(BaseModel):
    """Extensions d'extraction par étude (fusionnées avec les défauts code)."""

    langextract_class_map: dict[str, str] = Field(default_factory=dict)
    langextract_class_descriptions: dict[str, str] = Field(default_factory=dict)
    lab_analytes: list[LabAnalyteSpec] = Field(default_factory=list)
    narrative_keywords: list[NarrativeKeywordSpec] = Field(default_factory=list)
