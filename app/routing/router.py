from __future__ import annotations

from abc import ABC, abstractmethod

from app.parsing.document_type_scoring import classify_document_type
from app.schemas.enums import DocumentType
from app.schemas.models import ParsedDocument


class DocumentRouter(ABC):
    """Détermine le type métier du document pour le pipeline."""

    @abstractmethod
    def route(self, parsed: ParsedDocument) -> DocumentType:
        """Retourne le `DocumentType` effectif (peut affiner `document_type_hint`)."""


class KeywordHeuristicDocumentRouter(DocumentRouter):
    """Routage V1 : hint parseur si connu, sinon même classifieur que le scoring documentaire."""

    def route(self, parsed: ParsedDocument) -> DocumentType:
        if parsed.document_type_hint != DocumentType.UNKNOWN:
            return parsed.document_type_hint
        cls = classify_document_type(
            parsed.full_text,
            parsed.structured_sections,
            parsed.metadata,
        )
        return cls.document_type
