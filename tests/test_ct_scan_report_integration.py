"""
Intégration sur le PDF de référence `data/ct_scan_report_liver.pdf`.

Chemin par défaut : répertoire projet + `data/ct_scan_report_liver.pdf`.
Surcharge : variable d'environnement `ECRF_CT_TEST_PDF_PATH` (fichier absolu).

Les tests Docling sont ignorés si le fichier ou l'extra Docling manque (CI sans PDF).
"""

from __future__ import annotations

import pytest

from app.config.settings import Settings
from app.parsing.chunking import DefaultChunkingService, SectionBasedChunkingService
from app.parsing.lab_detection import is_probable_lab_document, lab_document_score
from app.parsing.smart_service import SmartParsingService
from app.schemas.enums import DocumentType
from tests.data_paths import CT_SCAN_REPORT_LIVER_PDF

pytestmark = pytest.mark.ct_integration_pdf


def test_ct_scan_pdf_path_resolves_under_project_data() -> None:
    assert CT_SCAN_REPORT_LIVER_PDF.name == "ct_scan_report_liver.pdf"
    assert CT_SCAN_REPORT_LIVER_PDF.parent.name == "data"


@pytest.mark.skipif(not CT_SCAN_REPORT_LIVER_PDF.is_file(), reason="PDF CT absent")
def test_ct_scan_ingestion_produces_raw_pdf(ct_scan_raw_document) -> None:
    raw = ct_scan_raw_document
    assert "pdf" in (raw.mime_type or "").lower()
    assert len(raw.content_bytes) > 500
    assert raw.source_path


@pytest.mark.skipif(not CT_SCAN_REPORT_LIVER_PDF.is_file(), reason="PDF CT absent")
def test_ct_scan_pypdf_extracts_substantial_text(ct_scan_raw_document) -> None:
    parsed = SmartParsingService(Settings(pdf_parser_backend="pypdf")).parse(ct_scan_raw_document)
    assert parsed.metadata.get("pdf_parser") == "pypdf"
    assert len((parsed.full_text or "").strip()) > 80


@pytest.mark.skipif(not CT_SCAN_REPORT_LIVER_PDF.is_file(), reason="PDF CT absent")
def test_ct_scan_pypdf_routed_as_narrative_chunking(ct_scan_raw_document) -> None:
    parsed = SmartParsingService(Settings(pdf_parser_backend="pypdf")).parse(ct_scan_raw_document)
    assert parsed.metadata.get("chunking_strategy") == "sections"
    chunks = DefaultChunkingService(Settings()).chunk(parsed)
    assert chunks
    assert chunks[0].metadata.get("chunker") == "SectionBasedChunkingService"


def test_docling_parses_ct_scan_metadata(ct_scan_parsed_docling) -> None:
    parsed = ct_scan_parsed_docling
    assert parsed.metadata.get("pdf_parser") == "docling"
    assert parsed.metadata.get("chunking_strategy") == "sections"
    assert parsed.metadata.get("full_text_source") == "assembled_structured_sections"
    assert "lab_document_score" in parsed.metadata


def test_docling_ct_scan_document_type_hint_is_imaging(ct_scan_parsed_docling) -> None:
    parsed = ct_scan_parsed_docling
    assert parsed.document_type_hint == DocumentType.IMAGING_REPORT


def test_docling_ct_scan_first_section_marked_admin(ct_scan_parsed_docling) -> None:
    parsed = ct_scan_parsed_docling
    assert parsed.structured_sections[0].metadata.get("content_kind") == "admin_section"
    chunks = SectionBasedChunkingService().chunk(parsed)
    assert chunks[0].metadata.get("content_kind") == "admin_section"


def test_docling_ct_scan_not_classified_as_lab(ct_scan_parsed_docling) -> None:
    """Le CR CT ne doit pas déclencher la branche labo (score / post-traitement)."""
    parsed = ct_scan_parsed_docling
    original = (parsed.metadata or {}).get("full_text_original_markdown") or ""
    assert original, "le markdown d'origine doit être conservé en métadonnée"
    assert is_probable_lab_document(original[:18000]) is False
    assert parsed.metadata.get("chunking_strategy") == "sections"
    assert not (parsed.structured_lab_lines or [])


def test_docling_ct_scan_has_sections_and_aligned_full_text(ct_scan_parsed_docling) -> None:
    parsed = ct_scan_parsed_docling
    bodies = [(s.body or "").strip() for s in parsed.structured_sections if (s.body or "").strip()]
    assert bodies, "au moins une section avec corps"
    assembled = "\n\n".join(bodies)
    assert parsed.full_text.strip() == assembled


def test_docling_ct_scan_chunking_spans(ct_scan_parsed_docling) -> None:
    parsed = ct_scan_parsed_docling
    chunks = SectionBasedChunkingService().chunk(parsed)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.text.strip()
        assert c.metadata.get("chunker") == "SectionBasedChunkingService"
        if c.metadata.get("char_spans_verified"):
            assert c.char_start is not None and c.char_end is not None
            assert parsed.full_text[c.char_start : c.char_end] == c.text


def test_docling_ct_scan_docling_sectioning_metadata(ct_scan_parsed_docling) -> None:
    parsed = ct_scan_parsed_docling
    assert parsed.metadata.get("docling_sectioning") in (
        "graph",
        "markdown_fallback",
        "full_text_only",
    )


def test_ct_scan_lab_score_is_documented(ct_scan_parsed_docling) -> None:
    parsed = ct_scan_parsed_docling
    sample = (parsed.metadata.get("full_text_original_markdown") or "")[:18000]
    assert isinstance(parsed.metadata.get("lab_document_score"), int)
    assert lab_document_score(sample) == parsed.metadata["lab_document_score"]
