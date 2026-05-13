"""Tests sur la découpe de sections longues (`_split_oversized_body`)."""

from __future__ import annotations

from datetime import date

from app.parsing.chunking import SectionBasedChunkingService, _split_oversized_body
from app.schemas.enums import DocumentType
from app.schemas.models import DocumentSection, ParsedDocument


def test_split_oversized_short_returns_single() -> None:
    assert _split_oversized_body("abc", max_chars=100) == ["abc"]


def test_split_oversized_by_paragraphs() -> None:
    body = "A\n\nB\n\nC"
    parts = _split_oversized_body(body, max_chars=4)
    assert len(parts) >= 2
    assert "A" in parts[0] or parts[0].startswith("A")


def test_section_chunking_verifies_spans_on_aligned_full_text() -> None:
    full = "One\n\nTwo\n\nThree"
    parsed = ParsedDocument(
        document_id="d",
        patient_id="p",
        study_id="s",
        full_text=full,
        document_type_hint=DocumentType.UNKNOWN,
        document_date=date(2024, 1, 1),
        metadata={"chunking_strategy": "sections"},
        structured_sections=[
            DocumentSection(heading=None, body="One", metadata={"source": "docling_graph"}),
            DocumentSection(heading="S2", body="Two", metadata={"source": "docling_graph"}),
            DocumentSection(heading="S3", body="Three", metadata={"source": "docling_graph"}),
        ],
    )
    chunks = SectionBasedChunkingService(max_section_chars=100).chunk(parsed)
    assert len(chunks) == 3
    assert all(c.metadata.get("char_spans_verified") for c in chunks)
    assert parsed.full_text[chunks[1].char_start : chunks[1].char_end] == chunks[1].text


def test_section_chunking_oversized_section_splits_parts() -> None:
    big = "x" * 50 + "\n\n" + "y" * 50
    parsed = ParsedDocument(
        document_id="d2",
        patient_id="p",
        study_id="s",
        full_text=big,
        document_type_hint=DocumentType.UNKNOWN,
        document_date=date(2024, 1, 1),
        metadata={"chunking_strategy": "sections"},
        structured_sections=[
            DocumentSection(heading="H", body=big, metadata={"source": "docling_graph"}),
        ],
    )
    chunks = SectionBasedChunkingService(max_section_chars=40).chunk(parsed)
    assert len(chunks) >= 2
