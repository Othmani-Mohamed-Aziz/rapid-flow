"""Utilitaires partagés lecture / écriture gabarit eCRF XLSX."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.etl.empty_sentinels import is_ecrf_cell_empty, parse_empty_sentinels

_LOG = logging.getLogger(__name__)

__all__ = [
    "HeaderMapResult",
    "build_header_map",
    "find_patient_row",
    "is_ecrf_cell_empty",
    "parse_empty_sentinels",
    "patient_id_matches",
    "resolve_column_key",
]


@dataclass
class HeaderMapResult:
    """Résultat du scan de la ligne d'en-tête."""

    mapping: dict[str, int]
    duplicate_headers: list[str] = field(default_factory=list)


def build_header_map(ws: Any, header_row: int) -> HeaderMapResult:
    mapping: dict[str, int] = {}
    duplicates: list[str] = []
    for col_idx, cell in enumerate(ws[header_row], start=1):
        name = cell.value
        if name is None:
            continue
        key = str(name).strip()
        if not key:
            continue
        if key in mapping:
            duplicates.append(key)
            continue
        mapping[key] = col_idx
    if duplicates:
        _LOG.warning(
            "En-têtes dupliqués dans le gabarit eCRF (ligne %s, première occurrence conservée) : %s",
            header_row,
            sorted(set(duplicates)),
        )
    return HeaderMapResult(mapping=mapping, duplicate_headers=sorted(set(duplicates)))


def find_patient_row(
    ws: Any,
    *,
    patient_id: str,
    patient_col: int,
    header_row: int,
) -> int | None:
    for row_idx in range(header_row + 1, ws.max_row + 1):
        if patient_id_matches(ws.cell(row=row_idx, column=patient_col).value, patient_id):
            return row_idx
    return None


def patient_id_matches(cell_value: Any, patient_id: str) -> bool:
    if cell_value is None:
        return False
    left = str(cell_value).strip()
    right = str(patient_id).strip()
    if left == right:
        return True
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def resolve_column_key(
    header_map: dict[str, int], column_key: str, aliases: dict[str, str]
) -> str | None:
    if column_key in header_map:
        return column_key
    alias = aliases.get(column_key)
    if alias and alias in header_map:
        return alias
    return None
