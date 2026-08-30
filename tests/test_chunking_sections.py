from __future__ import annotations

from datetime import date

from app.parsing.chunking import DefaultChunkingService, SectionBasedChunkingService
from app.schemas.enums import DocumentType
from app.schemas.models import DocumentSection, ParsedDocument, StructuredLabLine


def _parsed_narrative(*, strategy: str | None = "sections") -> ParsedDocument:
    meta = {"pdf_parser": "docling"}
    if strategy:
        meta["chunking_strategy"] = strategy
    return ParsedDocument(
        document_id="d1",
        patient_id="p1",
        study_id="s1",
        full_text="Intro\n\nCorps A\n\nCorps B",
        document_type_hint=DocumentType.UNKNOWN,
        document_date=date(2024, 1, 1),
        metadata=meta,
        structured_sections=[
            DocumentSection(heading=None, body="Intro", metadata={"source": "docling_graph"}),
            DocumentSection(
                heading="Technique", body="Corps A", metadata={"source": "docling_graph"}
            ),
            DocumentSection(
                heading="Conclusion", body="Corps B", metadata={"source": "docling_graph"}
            ),
        ],
    )


def test_section_based_chunking_preserves_headings() -> None:
    parsed = _parsed_narrative()
    chunks = SectionBasedChunkingService().chunk(parsed)
    assert len(chunks) == 3
    headings = [c.metadata.get("section_heading") for c in chunks]
    assert headings == [None, "Technique", "Conclusion"]
    assert all(c.metadata.get("content_kind") == "clinical_section" for c in chunks)
    assert chunks[1].metadata.get("section_index") == 1
    assert all(c.metadata.get("char_spans_verified") for c in chunks)
    assert chunks[0].char_start == 0


def test_default_chunking_routes_sections_when_requested() -> None:
    parsed = _parsed_narrative(strategy="sections")
    out = DefaultChunkingService().chunk(parsed)
    assert len(out) == 3
    assert out[0].metadata.get("chunker") == "SectionBasedChunkingService"


def test_section_based_chunking_propagates_admin_content_kind() -> None:
    parsed = ParsedDocument(
        document_id="d1",
        patient_id="p1",
        study_id="s1",
        full_text="Admin\n\nClinique",
        document_type_hint=DocumentType.IMAGING_REPORT,
        document_date=None,
        metadata={"chunking_strategy": "sections"},
        structured_sections=[
            DocumentSection(
                heading="CENTRE D'IMAGERIE MÉDICALE PARIS",
                body="Service de Radiologie",
                metadata={"source": "docling_graph", "content_kind": "admin_section"},
            ),
            DocumentSection(
                heading="CONCLUSION",
                body="Stabilité.",
                metadata={"source": "docling_graph", "content_kind": "clinical_section"},
            ),
        ],
    )
    chunks = SectionBasedChunkingService().chunk(parsed)
    assert chunks[0].metadata.get("content_kind") == "admin_section"
    assert chunks[1].metadata.get("content_kind") == "clinical_section"


def test_default_chunking_lab_heuristic_when_no_strategy() -> None:
    parsed = _parsed_narrative(strategy=None)
    parsed = parsed.model_copy(
        update={
            "full_text": "AST : 10 U/L\n\nHb : 12 g/dL",
            "structured_sections": [
                DocumentSection(heading=None, body=parsed.full_text, metadata={}),
            ],
        }
    )
    chunks = DefaultChunkingService().chunk(parsed)
    assert chunks[0].metadata.get("chunker") == "HeuristicLabChunkingService"


def test_default_chunking_groups_complete_lab_rows_by_subsection() -> None:
    parsed = _parsed_narrative(strategy="lab_rows").model_copy(
        update={
            "structured_lab_lines": [
                StructuredLabLine(
                    name="AST",
                    value=28.5,
                    unit="U/L",
                    section="Biochimie",
                    subsection="Biochimie",
                    raw_text="AST | 28.5 U/L",
                ),
                StructuredLabLine(
                    name="ALT",
                    value=10,
                    unit="U/L",
                    section="Biochimie",
                    subsection="Biochimie",
                    raw_text="ALT | 10 U/L",
                ),
            ]
        }
    )

    chunks = DefaultChunkingService().chunk(parsed)

    assert len(chunks) == 1
    assert chunks[0].text == "AST | 28.5 U/L\nALT | 10 U/L"
    assert chunks[0].metadata["chunker"] == "LabRowChunkingService"
