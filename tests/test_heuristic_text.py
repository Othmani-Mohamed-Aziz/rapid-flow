"""Tests du parseur texte heuristique."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.parsing.heuristic_text import HeuristicTextParser
from app.schemas.models import RawDocument


def _raw_txt(path: Path, text: str) -> RawDocument:
    return RawDocument(
        document_id="t1",
        patient_id="p",
        study_id="s",
        source_path=str(path),
        mime_type="text/plain",
        content_bytes=text.encode("utf-8"),
        ingested_at=datetime.now(timezone.utc),
    )


def test_heuristic_text_parses_plain_file(tmp_path: Path) -> None:
    f = tmp_path / "note.txt"
    f.write_text("Conclusion : suivi à 6 mois.\n", encoding="utf-8")
    raw = _raw_txt(f, f.read_text(encoding="utf-8"))
    out = HeuristicTextParser().parse(raw)
    assert "Conclusion" in out.full_text
    assert out.metadata.get("encoding") == "utf-8"


def test_heuristic_text_pdf_placeholder() -> None:
    raw = RawDocument(
        document_id="t2",
        patient_id="p",
        study_id="s",
        source_path="C:/x/document.pdf",
        mime_type="application/pdf",
        content_bytes=b"%PDF",
        ingested_at=datetime.now(timezone.utc),
    )
    out = HeuristicTextParser().parse(raw)
    assert "SmartParsingService" in out.full_text
    assert out.metadata.get("pdf_placeholder") is True
