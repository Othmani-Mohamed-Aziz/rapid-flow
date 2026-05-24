"""Réduction des chemins absolus dans exports JSON / dumps (PHI / vie privée)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Clés dont les valeurs string sont typiquement des chemins de fichiers.
_PATH_LIKE_KEYS: frozenset[str] = frozenset(
    {
        "path",
        "source_path",
        "document_path",
        "json",
        "csv",
    }
)

_MAX_DEPTH = 32


def _basename_if_path(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if os.sep not in s and "/" not in s and "\\" not in s:
        return s
    try:
        return Path(s).name
    except (OSError, ValueError, RuntimeError):
        return s


def redact_file_paths_in_jsonable(obj: Any, *, depth: int = 0) -> Any:
    """Retourne une copie avec basenames pour les clés sensibles (récursif)."""
    if depth > _MAX_DEPTH:
        return obj
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in _PATH_LIKE_KEYS and isinstance(v, str):
                out[k] = _basename_if_path(v)
            else:
                out[k] = redact_file_paths_in_jsonable(v, depth=depth + 1)
        return out
    if isinstance(obj, list):
        return [redact_file_paths_in_jsonable(x, depth=depth + 1) for x in obj]
    if isinstance(obj, tuple):
        return tuple(redact_file_paths_in_jsonable(x, depth=depth + 1) for x in obj)
    return obj
