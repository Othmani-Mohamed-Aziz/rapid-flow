from __future__ import annotations

from app.parsing.lab_post_processor import LabReportPostProcessor
from app.schemas.enums import DocumentType
from app.schemas.models import DocumentSection, ParsedDocument


def test_lab_postprocessor_parses_colon_lines() -> None:
    text = """Hépatite
AST : 48 U/L
ALT : 32 U/L
Plaquettes : 145 G/L
"""
    parsed = ParsedDocument(
        document_id="d",
        patient_id="p",
        study_id="s",
        full_text=text,
        document_type_hint=DocumentType.LAB_BLOOD_PANEL,
    )
    out = LabReportPostProcessor().enrich(parsed)
    names = {ln.name.lower() for ln in out.structured_lab_lines}
    assert "ast" in names or any("AST" in ln.name for ln in out.structured_lab_lines)
    assert out.metadata.get("structured_lab_line_count", 0) >= 2


def test_lab_postprocessor_uses_sections_when_present() -> None:
    parsed = ParsedDocument(
        document_id="d",
        patient_id="p",
        study_id="s",
        full_text="ignored body duplicate",
        document_type_hint=DocumentType.LAB_BLOOD_PANEL,
        structured_sections=[
            DocumentSection(heading="Biochimie", body="CRP : 5 mg/L\n", metadata={}),
        ],
    )
    out = LabReportPostProcessor().enrich(parsed)
    assert any(ln.name.upper().startswith("CRP") for ln in out.structured_lab_lines)


def test_lab_postprocessor_ignores_reference_and_page_noise() -> None:
    parsed = ParsedDocument(
        document_id="d",
        patient_id="p",
        study_id="s",
        full_text="",
        document_type_hint=DocumentType.LAB_BLOOD_PANEL,
        structured_sections=[
            DocumentSection(
                heading="Biochimie",
                body="AST....................\n28.5 U/L\nInf. à 55\nPage 1\n",
                metadata={},
            ),
        ],
    )
    out = LabReportPostProcessor().enrich(parsed)
    raw = [ln.raw_text for ln in out.structured_lab_lines]
    assert any("AST" in r for r in raw)
    assert all("Inf. à" not in r for r in raw)
    assert all("Page" not in r for r in raw)
