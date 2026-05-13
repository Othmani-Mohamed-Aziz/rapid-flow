"""
Tests d’intégration sur de vrais fichiers PDF (binaires).

1) PDF généré à la volée (reportlab) — reproductible en CI.
2) PDF fourni par l’utilisateur : variable d’environnement ECRF_TEST_PDF_PATH
   ou fichier tests/fixtures/local_lab.pdf
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

from app.config.settings import Settings
from app.ingestion.service import LocalFileIngestionService
from app.parsing.lab_post_processor import LabReportPostProcessor
from app.parsing.pdf.pypdf_parser import PypdfPdfParser
from app.parsing.smart_service import SmartParsingService
from app.schemas.models import RawDocument
from tests.pdf_fixtures import reportlab_available, write_minimal_lab_pdf


@pytest.fixture
def generated_lab_pdf(tmp_path: Path) -> Path:
    if not reportlab_available():
        pytest.skip('reportlab absent : pip install -e ".[dev]"')
    out = tmp_path / "generated_lab.pdf"
    write_minimal_lab_pdf(out)
    return out


def test_pypdf_reads_real_generated_pdf(generated_lab_pdf: Path) -> None:
    raw = RawDocument(
        document_id="d1",
        patient_id="p1",
        study_id="s1",
        source_path=str(generated_lab_pdf.resolve()),
        mime_type="application/pdf",
        content_bytes=generated_lab_pdf.read_bytes(),
        ingested_at=datetime.now(),
    )
    result = PypdfPdfParser().parse(raw)
    assert "AST" in result.full_text
    assert "48" in result.full_text
    assert "Plaquettes" in result.full_text or "plaquettes" in result.full_text.lower()


def test_smart_parsing_lab_lines_on_real_generated_pdf(generated_lab_pdf: Path) -> None:
    raw = RawDocument(
        document_id="d1",
        patient_id="p1",
        study_id="s1",
        source_path=str(generated_lab_pdf.resolve()),
        mime_type="application/pdf",
        content_bytes=generated_lab_pdf.read_bytes(),
        ingested_at=datetime.now(),
    )
    parsed = SmartParsingService(Settings(pdf_parser_backend="pypdf")).parse(raw)
    assert parsed.metadata.get("pdf_parser") == "pypdf"
    assert len(parsed.structured_lab_lines) >= 2
    canon = {ln.canonical_name or ln.name for ln in parsed.structured_lab_lines}
    assert "AST" in canon or any("AST" in (ln.name or "") for ln in parsed.structured_lab_lines)


def test_lab_postprocessor_on_ingested_real_pdf(generated_lab_pdf: Path) -> None:
    raw = LocalFileIngestionService().ingest(
        str(generated_lab_pdf.resolve()),
        patient_id="p1",
        study_id="s1",
    )
    parsed = SmartParsingService(Settings(pdf_parser_backend="pypdf")).parse(raw)
    assert raw.document_id == parsed.document_id
    re_enriched = LabReportPostProcessor().enrich(parsed, force=True)
    assert len(re_enriched.structured_lab_lines) >= 1


def _optional_pdf_from_env() -> Path | None:
    p = os.environ.get("ECRF_TEST_PDF_PATH", "").strip()
    if not p:
        return None
    path = Path(p)
    return path if path.is_file() else None


def _optional_pdf_from_fixtures() -> Path | None:
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "local_lab.pdf"
    return path if path.is_file() else None


@pytest.mark.optional_user_pdf
def test_parse_user_pdf_from_env_or_fixtures() -> None:
    """
    Lance un parsing réel sur un PDF hors repo si :

    - ECRF_TEST_PDF_PATH pointe vers un fichier, ou
    - tests/fixtures/local_lab.pdf existe (copie locale).
    """
    path = _optional_pdf_from_env() or _optional_pdf_from_fixtures()
    if path is None:
        pytest.skip(
            "Aucun PDF optionnel : définissez ECRF_TEST_PDF_PATH ou ajoutez tests/fixtures/local_lab.pdf"
        )
    raw = LocalFileIngestionService().ingest(str(path.resolve()), patient_id="p1", study_id="s1")
    parsed = SmartParsingService(Settings(pdf_parser_backend="pypdf")).parse(raw)
    assert parsed.full_text.strip(), "full_text vide : PDF peut-être image-only ou protégé"
    assert parsed.metadata.get("pdf_parser") == "pypdf"
    # Bilan : on attend au moins du texte exploitable ; lignes structurées si format compatible
    assert len(parsed.full_text) > 20
