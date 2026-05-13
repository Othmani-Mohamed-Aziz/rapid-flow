"""Tests unitaires sur les helpers Markdown du parseur Docling (sans exécuter Docling)."""

from __future__ import annotations

from app.parsing.pdf.docling_parser import _sections_from_markdown


def test_sections_from_markdown_single_block() -> None:
    md = "Intro sans titre ##\n"
    secs = _sections_from_markdown(md)
    assert len(secs) == 1
    assert secs[0].heading is None
    assert "Intro" in secs[0].body


def test_sections_from_markdown_with_headings() -> None:
    md = "Preamble\n\n## Technique\n\nTexte technique.\n\n## Conclusion\n\nFin."
    secs = _sections_from_markdown(md)
    assert len(secs) >= 2
    headings = [s.heading for s in secs if s.heading]
    assert "Technique" in headings
    assert "Conclusion" in headings
    assert all(s.metadata.get("source") == "docling_markdown_split" for s in secs if s.heading)


def test_sections_from_markdown_strips_heading_hashes() -> None:
    # Le split attend un saut de ligne avant `##` (comme l’export Docling typique).
    md = "Préambule\n\n##  Titre avec espaces  \n\nCorps."
    secs = _sections_from_markdown(md)
    titled = [s for s in secs if s.heading]
    assert titled and titled[0].heading == "Titre avec espaces"
