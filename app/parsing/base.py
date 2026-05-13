from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.models import ParsedDocument, RawDocument


class ParsingService(ABC):
    """Transforme un `RawDocument` en texte exploitable."""

    @abstractmethod
    def parse(self, raw: RawDocument) -> ParsedDocument:
        """Extraction texte + indices de type document."""
