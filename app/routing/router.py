from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.enums import DocumentType
from app.schemas.models import ParsedDocument


class DocumentRouter(ABC):
    """Détermine le type métier du document pour le pipeline."""

    @abstractmethod
    def route(self, parsed: ParsedDocument) -> DocumentType:
        """Retourne le `DocumentType` effectif (peut affiner `document_type_hint`)."""


class KeywordHeuristicDocumentRouter(DocumentRouter):
    """Routage V1 par mots-clés (bilan sanguin prioritaire)."""

    _LAB = (
        "nfs",
        "hémogramme",
        "hemogramme",
        "plaquette",
        "leucocyte",
        "crp",
        "bilirubin",
        "bilirubine",
        "ast",
        "alt",
        "tp",
        "inr",
    )

    _IMG = ("recist", "tdm", "scanner", "irm", "imagerie", "lésion", "lesion")

    def route(self, parsed: ParsedDocument) -> DocumentType:
        if parsed.document_type_hint != DocumentType.UNKNOWN:
            return parsed.document_type_hint
        lower = parsed.full_text.lower()
        if any(k in lower for k in self._LAB):
            return DocumentType.LAB_BLOOD_PANEL
        if any(k in lower for k in self._IMG):
            return DocumentType.IMAGING_REPORT
        return DocumentType.UNKNOWN
