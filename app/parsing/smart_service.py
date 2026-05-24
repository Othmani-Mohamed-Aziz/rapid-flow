from __future__ import annotations

from app.config.settings import Settings
from app.parsing.base import ParsingService
from app.parsing.document_type_scoring import classify_document_type
from app.parsing.errors import DoclingNotInstalledError
from app.parsing.heuristic_text import HeuristicTextParser
from app.parsing.lab_detection import (
    imaging_or_morpho_context,
    is_probable_lab_document,
    lab_document_score,
)
from app.parsing.lab_post_processor import LabReportPostProcessor
from app.parsing.pdf.resilient import ensure_non_empty_text, parse_pdf_bytes_resilient
from app.parsing.structuring import DocumentStructuringService
from app.parsing.text_enrichment import extract_document_date
from app.schemas.models import ParsedDocument, RawDocument


def _is_pdf(raw: RawDocument) -> bool:
    mime = (raw.mime_type or "").lower()
    if "pdf" in mime:
        return True
    path = (raw.source_path or "").lower()
    return path.endswith(".pdf")


def _canonical_full_text_from_sections(parsed: ParsedDocument) -> str | None:
    bodies = [(s.body or "").strip() for s in parsed.structured_sections if (s.body or "").strip()]
    if not bodies:
        return None
    return "\n\n".join(bodies)


def _align_full_text_to_structured_sections(parsed: ParsedDocument) -> ParsedDocument:
    """
    Aligne `parsed.full_text` sur la concaténation des sections (ordre chunking / spans).

    Le texte d’origine (ex. Markdown Docling) reste traçable dans les métadonnées.
    """
    canonical = _canonical_full_text_from_sections(parsed)
    if not canonical or not canonical.strip():
        return parsed
    meta = dict(parsed.metadata)
    meta.setdefault("full_text_original_markdown", parsed.full_text)
    meta["full_text_source"] = "assembled_structured_sections"
    meta["chunk_spans_relative_to"] = "parsed.full_text"
    return parsed.model_copy(update={"full_text": canonical.strip(), "metadata": meta})


class SmartParsingService(ParsingService):
    """
    Routage parsing : PDF (Docling / pypdf) vs texte (heuristique).

    Conserve `ParsedDocument` comme contrat unique pour le pipeline.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        text_parser: HeuristicTextParser | None = None,
        lab_postprocessor: LabReportPostProcessor | None = None,
        structuring: DocumentStructuringService | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self._text = text_parser or HeuristicTextParser()
        self._lab = lab_postprocessor or LabReportPostProcessor()
        self._struct = structuring or DocumentStructuringService()

    def parse(self, raw: RawDocument) -> ParsedDocument:
        if not _is_pdf(raw):
            doc = self._text.parse(raw)
            return self._struct.enrich(doc)

        try:
            pr = parse_pdf_bytes_resilient(raw, self.settings)
        except DoclingNotInstalledError:
            raise
        pr = ensure_non_empty_text(pr, raw)

        meta = dict(pr.extra_metadata)
        meta.setdefault("mime", raw.mime_type)

        sample = pr.full_text[:18000] if pr.full_text else ""

        cls = classify_document_type(pr.full_text, list(pr.sections), meta)
        meta.update(cls.as_metadata())

        parsed = ParsedDocument(
            document_id=raw.document_id,
            patient_id=raw.patient_id,
            study_id=raw.study_id,
            full_text=pr.full_text,
            document_type_hint=cls.document_type,
            document_date=extract_document_date(pr.full_text),
            metadata=meta,
            source_path=raw.source_path,
            structured_sections=list(pr.sections),
        )
        parsed = self._struct.enrich(parsed)

        is_lab = is_probable_lab_document(sample)

        if self.settings.lab_postprocess_pdf_lab_reports and is_lab:
            parsed = self._lab.enrich(parsed)

        meta = dict(parsed.metadata)
        meta["lab_document_score"] = lab_document_score(sample)
        meta["imaging_or_morpho_context"] = imaging_or_morpho_context(sample)

        if _is_pdf(raw) and not is_lab:
            meta["chunking_strategy"] = "sections"
            parsed = parsed.model_copy(update={"metadata": meta})
            parsed = _align_full_text_to_structured_sections(parsed)
        else:
            meta.pop("chunking_strategy", None)
            parsed = parsed.model_copy(update={"metadata": meta})

        return parsed
