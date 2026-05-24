from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.config.settings import Settings
from app.parsing.errors import DoclingNotInstalledError
from app.parsing.pdf.docling_parser import DoclingPdfParser
from app.parsing.pdf.factory import select_pdf_parser
from app.parsing.pdf.pypdf_parser import PypdfPdfParser
from app.parsing.smart_service import SmartParsingService
from app.schemas.models import RawDocument


def _raw_pdf_stub() -> RawDocument:
    return RawDocument(
        document_id="doc-pdf-1",
        patient_id="p1",
        study_id="s1",
        source_path=str(Path("__not_a_real_path__.pdf")),
        mime_type="application/pdf",
        content_bytes=b"%PDF-fake-bytes",
        ingested_at=datetime.now(timezone.utc),
    )


@patch("pypdf.PdfReader")
def test_pypdf_parser_extracts_text(mock_reader_cls: MagicMock) -> None:
    page = MagicMock()
    page.extract_text.return_value = "AST : 42 U/L\nPlaquettes : 150 G/L"
    reader = MagicMock()
    reader.pages = [page]
    reader.is_encrypted = False
    mock_reader_cls.return_value = reader

    raw = _raw_pdf_stub()
    result = PypdfPdfParser().parse(raw)
    assert "AST" in result.full_text
    assert "42" in result.full_text
    assert result.extra_metadata.get("page_count") == 1


def test_select_pdf_parser_pypdf_forced() -> None:
    s = Settings(pdf_parser_backend="pypdf")
    p = select_pdf_parser(s)
    assert p.name == "pypdf"


def test_select_pdf_parser_docling_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force `Docling indisponible` même si la dépendance est installée."""
    monkeypatch.setattr(DoclingPdfParser, "is_available", lambda self: False)
    s = Settings(pdf_parser_backend="docling")
    with pytest.raises(DoclingNotInstalledError):
        select_pdf_parser(s)


def test_resilient_docling_forced_falls_back_on_hf401(monkeypatch: pytest.MonkeyPatch) -> None:
    """Token HF expiré / 401 → repli pypdf (message type huggingface_hub)."""
    monkeypatch.setattr(DoclingPdfParser, "is_available", lambda self: True)

    def _hf401(self, raw):  # type: ignore[no-untyped-def]
        raise RuntimeError(
            "401 Client Error. Repository Not Found for url: ... "
            'User Access Token "hf_xxx" is expired'
        )

    monkeypatch.setattr(DoclingPdfParser, "parse", _hf401)

    with patch("pypdf.PdfReader") as mock_reader_cls:
        page = MagicMock()
        page.extract_text.return_value = "ALT : 2 U/L"
        reader = MagicMock()
        reader.pages = [page]
        reader.is_encrypted = False
        mock_reader_cls.return_value = reader

        from app.parsing.pdf.resilient import parse_pdf_bytes_resilient

        raw = _raw_pdf_stub()
        out = parse_pdf_bytes_resilient(raw, Settings(pdf_parser_backend="docling"))
        assert "ALT" in out.full_text
        assert out.extra_metadata.get("pdf_parser") == "pypdf"


def test_resilient_docling_forced_falls_back_on_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend docling + échec réseau simulé → repli pypdf (même comportement qu’en prod HF bloqué)."""
    monkeypatch.setattr(DoclingPdfParser, "is_available", lambda self: True)

    def _boom(self, raw):  # type: ignore[no-untyped-def]
        raise ConnectionError("ConnectError: tunnel connection failed")

    monkeypatch.setattr(DoclingPdfParser, "parse", _boom)

    with patch("pypdf.PdfReader") as mock_reader_cls:
        page = MagicMock()
        page.extract_text.return_value = "AST : 1 U/L"
        reader = MagicMock()
        reader.pages = [page]
        reader.is_encrypted = False
        mock_reader_cls.return_value = reader

        from app.parsing.pdf.resilient import parse_pdf_bytes_resilient

        raw = _raw_pdf_stub()
        out = parse_pdf_bytes_resilient(raw, Settings(pdf_parser_backend="docling"))
        assert "AST" in out.full_text
        assert out.extra_metadata.get("pdf_parser") == "pypdf"
        assert any(
            "docling_recoverable_fallback" in w for w in out.extra_metadata.get("warnings", [])
        )


def test_resilient_auto_without_docling_uses_pypdf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DoclingPdfParser, "is_available", lambda self: False)

    with patch("pypdf.PdfReader") as mock_reader_cls:
        page = MagicMock()
        page.extract_text.return_value = "CRP : 2 mg/L"
        reader = MagicMock()
        reader.pages = [page]
        reader.is_encrypted = False
        mock_reader_cls.return_value = reader

        from app.parsing.pdf.resilient import parse_pdf_bytes_resilient

        raw = _raw_pdf_stub()
        out = parse_pdf_bytes_resilient(raw, Settings(pdf_parser_backend="auto"))
        assert "CRP" in out.full_text
        assert out.extra_metadata.get("pdf_parser") == "pypdf"


@patch("pypdf.PdfReader")
def test_smart_parsing_pdf_lab_postprocessor(mock_reader_cls: MagicMock) -> None:
    page = MagicMock()
    page.extract_text.return_value = "Biologie\nAST : 40 U/L\nPlaquettes : 155 G/L"
    reader = MagicMock()
    reader.pages = [page]
    reader.is_encrypted = False
    mock_reader_cls.return_value = reader

    raw = _raw_pdf_stub()
    parsed = SmartParsingService(Settings(pdf_parser_backend="pypdf")).parse(raw)
    assert parsed.metadata.get("pdf_parser") == "pypdf"
    assert len(parsed.structured_lab_lines) >= 1


def test_smart_parsing_non_pdf_delegates_to_heuristic(tmp_path: Path) -> None:
    f = tmp_path / "lab.txt"
    f.write_text("Bilan NFS\nAST 10 U/L\n", encoding="utf-8")
    raw = RawDocument(
        document_id="t1",
        patient_id="p",
        study_id="s",
        source_path=str(f),
        mime_type="text/plain",
        content_bytes=f.read_bytes(),
        ingested_at=datetime.now(timezone.utc),
    )
    parsed = SmartParsingService().parse(raw)
    assert "AST" in parsed.full_text
    assert parsed.source_path == str(f)
