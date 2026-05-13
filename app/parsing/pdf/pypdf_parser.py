from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from app.parsing.pdf.base import BasePdfParser, PdfParseResult
from app.schemas.models import DocumentSection, RawDocument


class PypdfPdfParser(BasePdfParser):
    """Extraction texte via pypdf (léger, sans service externe)."""

    name = "pypdf"

    def is_available(self) -> bool:
        try:
            import pypdf  # noqa: F401
        except ImportError:
            return False
        return True

    def parse(self, raw: RawDocument) -> PdfParseResult:
        try:
            from pypdf import PdfReader
        except ImportError as e:
            msg = "pypdf n’est pas installé. Ajoutez-le aux dépendances : pip install pypdf"
            raise ImportError(msg) from e

        source: str | Path | BytesIO
        if Path(raw.source_path).is_file():
            source = raw.source_path
        else:
            source = BytesIO(raw.content_bytes)

        reader = PdfReader(source)
        pages: list[str] = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:  # pragma: no cover - robustesse PDF exotiques
                t = ""
            pages.append(t)
        full_text = "\n\n".join(pages).strip()
        meta: dict[str, Any] = {
            "page_count": len(reader.pages),
            "encrypted": bool(getattr(reader, "is_encrypted", False)),
        }
        sections = [DocumentSection(heading=None, body=full_text, metadata={"parser": self.name})]
        return PdfParseResult(full_text=full_text, sections=sections, extra_metadata=meta)
