from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.models import DocumentSection, RawDocument


class PdfParseResult(BaseModel):
    """Sortie normalisée des parseurs PDF (avant fusion dans `ParsedDocument`)."""

    full_text: str
    sections: list[DocumentSection] = Field(default_factory=list)
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


class BasePdfParser(ABC):
    """Abstraction parseur PDF local (Docling, pypdf, etc.)."""

    name: str = "base_pdf"

    @abstractmethod
    def parse(self, raw: RawDocument) -> PdfParseResult:
        """Extrait texte + sections à partir d’un `RawDocument` PDF."""

    def is_available(self) -> bool:
        return True
