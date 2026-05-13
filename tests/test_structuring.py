from __future__ import annotations

from app.parsing.structuring import DocumentStructuringService
from app.parsing.text_enrichment import extract_document_date
from app.schemas.enums import DocumentType
from app.schemas.models import DocumentSection, ParsedDocument


def test_extract_document_date_prioritizes_sample_collection() -> None:
    text = "Date de naissance : 22/11/1993\nPrélevé le 29/07/25 15h00\nEdité le 30/07/25 à 14h00\n"
    dt = extract_document_date(text)
    assert str(dt) == "2025-07-29"


def test_structuring_service_builds_sections_for_lab_report() -> None:
    text = (
        "Header administratif\n"
        "Hematologie\n"
        "Globules rouges....................\n"
        "4.4 10^6/mm3\n"
        "Biochimie\n"
        "AST : 48 U/L\n"
        "ALT : 31 U/L\n"
    )
    parsed = ParsedDocument(
        document_id="d1",
        patient_id="p1",
        study_id="s1",
        full_text=text,
        document_type_hint=DocumentType.LAB_BLOOD_PANEL,
    )
    out = DocumentStructuringService().enrich(parsed)
    headings = [s.heading or "" for s in out.structured_sections]
    assert any("Hematologie" in h for h in headings)
    assert any("Biochimie" in h for h in headings)


def test_structuring_preserves_single_docling_section() -> None:
    """Ne pas écraser une unique section Docling par le segmenteur plein texte."""
    text = "Markdown ou texte long…"
    parsed = ParsedDocument(
        document_id="d1",
        patient_id="p1",
        study_id="s1",
        full_text=text,
        document_type_hint=DocumentType.UNKNOWN,
        structured_sections=[
            DocumentSection(
                heading="Synthèse",
                body="Contenu clinique unique.",
                metadata={"source": "docling_graph"},
            ),
        ],
    )
    out = DocumentStructuringService().enrich(parsed)
    assert len(out.structured_sections) == 1
    assert out.structured_sections[0].heading == "Synthèse"
    assert "docling" in (out.structured_sections[0].metadata.get("source") or "")
