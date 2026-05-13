"""Tests sur la résolution de source Docling (fichier vs flux)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.parsing.pdf.docling_source import resolve_docling_conversion_source
from app.schemas.models import RawDocument


def test_resolve_docling_source_uses_existing_path(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 minimal\n")
    raw = RawDocument(
        document_id="1",
        patient_id="p",
        study_id="s",
        source_path=str(pdf),
        mime_type="application/pdf",
        content_bytes=pdf.read_bytes(),
        ingested_at=datetime.now(timezone.utc),
    )
    src = resolve_docling_conversion_source(raw)
    assert isinstance(src, Path)
    assert src.resolve() == pdf.resolve()


def test_resolve_docling_source_stream_when_path_missing() -> None:
    raw = RawDocument(
        document_id="2",
        patient_id="p",
        study_id="s",
        source_path=str(Path("__nope__/missing.pdf")),
        mime_type="application/pdf",
        content_bytes=b"%PDF-fake",
        ingested_at=datetime.now(timezone.utc),
    )
    pytest.importorskip("docling")
    src = resolve_docling_conversion_source(raw)
    assert type(src).__name__ == "DocumentStream"
