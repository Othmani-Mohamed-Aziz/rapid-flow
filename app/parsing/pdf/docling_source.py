"""Résolution de la source Docling (fichier local vs flux mémoire)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from app.schemas.models import RawDocument


def resolve_docling_conversion_source(raw: RawDocument) -> Any:
    """
    Docling accepte un chemin, une URL ou un `DocumentStream`.

    Si `source_path` pointe vers un fichier existant, on l’utilise ; sinon les
    octets sont passés via `DocumentStream` (comme pypdf avec BytesIO).
    """
    from docling.datamodel.base_models import DocumentStream

    if raw.source_path:
        p = Path(raw.source_path)
        if p.is_file():
            return p
        fname = p.name if p.name else "document.pdf"
    else:
        fname = "document.pdf"
    return DocumentStream(name=fname, stream=BytesIO(raw.content_bytes))
