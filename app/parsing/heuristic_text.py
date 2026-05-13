from __future__ import annotations

from typing import Any

from app.parsing.base import ParsingService
from app.parsing.text_enrichment import extract_document_date, guess_document_type
from app.schemas.models import ParsedDocument, RawDocument


class HeuristicTextParser(ParsingService):
    """
    Parseur texte (.txt, .md) et fallback UTF-8.

    Pour les PDF, utilisez `SmartParsingService` : ce parseur conserve
    un placeholder PDF minimal si appelé seul (compatibilité tests).
    """

    def parse(self, raw: RawDocument) -> ParsedDocument:
        text, meta = self._decode_text(raw)
        doc_date = extract_document_date(text)
        return ParsedDocument(
            document_id=raw.document_id,
            patient_id=raw.patient_id,
            study_id=raw.study_id,
            full_text=text,
            document_type_hint=guess_document_type(text, raw.mime_type),
            language=meta.get("language"),
            document_date=doc_date,
            metadata=meta,
            source_path=raw.source_path,
        )

    def _decode_text(self, raw: RawDocument) -> tuple[str, dict[str, Any]]:
        if raw.mime_type in ("text/plain", "text/markdown") or raw.source_path.lower().endswith(
            (".txt", ".md")
        ):
            return raw.content_bytes.decode("utf-8", errors="replace"), {"encoding": "utf-8"}
        if "pdf" in raw.mime_type or raw.source_path.lower().endswith(".pdf"):
            return (
                "[PDF — utiliser SmartParsingService / parseur PDF dédié]",
                {"pdf_placeholder": True},
            )
        try:
            return raw.content_bytes.decode("utf-8", errors="replace"), {
                "encoding": "utf-8-fallback"
            }
        except Exception:  # pragma: no cover
            return "", {"error": "decode_failed"}
