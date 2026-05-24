"""Logique partagée d’indices métier / dates sur texte extrait."""

from __future__ import annotations

import re
from datetime import date

from app.schemas.enums import DocumentType

_GENERIC_DATE_PATTERNS = (
    re.compile(r"\b(?P<y>\d{4})[-/](?P<m>\d{2})[-/](?P<d>\d{2})\b"),
    re.compile(r"\b(?P<d>\d{2})[/](?P<m>\d{2})[/](?P<y>\d{4})\b"),
    re.compile(r"\b(?P<d>\d{2})[/](?P<m>\d{2})[/](?P<y>\d{2})\b"),
)

_PRIORITIZED_DATE_PATTERNS = (
    re.compile(r"prélev[ée]?\s+le\s+(?P<d>\d{2})/(?P<m>\d{2})/(?P<y>\d{2,4})", re.I),
    re.compile(r"prélèvement\s+du\s+(?P<d>\d{2})/(?P<m>\d{2})/(?P<y>\d{2,4})", re.I),
    re.compile(r"edit[ée]?\s+le\s+(?P<d>\d{2})/(?P<m>\d{2})/(?P<y>\d{2,4})", re.I),
    re.compile(
        r"date\s+de\s+l[’']examen\s*[:\-]?\s*(?P<d>\d{2})/(?P<m>\d{2})/(?P<y>\d{2,4})", re.I
    ),
)


def _coerce_year(y: int) -> int:
    if y >= 100:
        return y
    return 2000 + y if y < 70 else 1900 + y


def _build_date(gd: dict[str, str]) -> date | None:
    try:
        y, mo, d = _coerce_year(int(gd["y"])), int(gd["m"]), int(gd["d"])
        return date(y, mo, d)
    except (ValueError, KeyError):
        return None


def extract_document_date(text: str) -> date | None:
    for pat in _PRIORITIZED_DATE_PATTERNS:
        m = pat.search(text)
        if m:
            dt = _build_date(m.groupdict())
            if dt:
                return dt
    for pat in _GENERIC_DATE_PATTERNS:
        m = pat.search(text)
        if m:
            dt = _build_date(m.groupdict())
            if dt:
                return dt
    return None


def guess_document_type(text: str, mime: str) -> DocumentType:  # noqa: ARG001
    """
    API historique : sans sections ni métadonnées de parse, alignée sur le classifieur
    multi-signaux avec **structure vide** (texte seul, rapide).
    """
    from app.parsing.document_type_scoring import classify_document_type

    return classify_document_type(text, [], {}).document_type
