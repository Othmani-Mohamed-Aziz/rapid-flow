"""Alias colonnes StudySchema → en-têtes gabarit MA_Base."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALIASES_PATH = _PROJECT_ROOT / "data" / "column_aliases_ma.json"

# Fallback intégré (écrasé / complété par le fichier JSON si présent).
BUILTIN_COLUMN_ALIASES: dict[str, str] = {
    "Response_at_first_imaging_RECIST": "Response_at_the_first_imaging_RECIST",
    "Response_at_second_imaging_RECIST": "Response_at_the_second_imaging_RECIST",
    "Response_at_first_imaging_mRECIST": "Response_at_the_first_imaging_mRECIST",
    "Response_at_second_imaging_mRECIST": "Response_at_the_second_imaging_mRECIST",
}


def load_column_aliases(path: Path | None = None) -> dict[str, str]:
    """Fusionne alias intégrés + fichier JSON optionnel."""
    merged = dict(BUILTIN_COLUMN_ALIASES)
    candidates = [path, DEFAULT_ALIASES_PATH] if path is None else [path]
    for p in candidates:
        if p is None or not p.is_file():
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("Alias colonnes illisibles (%s) : %s", p, exc)
            continue
        if isinstance(raw, dict):
            merged.update({str(k): str(v) for k, v in raw.items()})
        break
    return merged


def resolve_column_aliases(settings: Any | None = None) -> dict[str, str]:
    if settings is None:
        return load_column_aliases()
    custom = getattr(settings, "export_xls_column_aliases_path", None)
    return load_column_aliases(Path(custom) if custom else None)
