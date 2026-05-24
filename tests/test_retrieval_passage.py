"""Tests unitaires sur `passage_for_embedding_and_rerank`."""

from __future__ import annotations

from app.indexing.retrieval_passage import passage_for_embedding_and_rerank
from app.schemas.models import DocumentChunk


def test_passage_heading_and_body() -> None:
    c = DocumentChunk(
        chunk_id="c1",
        document_id="d1",
        text="  Corps clinique.  ",
        metadata={"section_heading": "CONCLUSION"},
    )
    assert passage_for_embedding_and_rerank(c) == "CONCLUSION\nCorps clinique."


def test_passage_heading_only_when_body_empty() -> None:
    c = DocumentChunk(
        chunk_id="c1",
        document_id="d1",
        text="   ",
        metadata={"section_heading": "ENTÊTE"},
    )
    assert passage_for_embedding_and_rerank(c) == "ENTÊTE"


def test_passage_body_only_when_no_heading() -> None:
    c = DocumentChunk(chunk_id="c1", document_id="d1", text="Seul le corps.", metadata={})
    assert passage_for_embedding_and_rerank(c) == "Seul le corps."


def test_passage_ignores_blank_heading() -> None:
    c = DocumentChunk(
        chunk_id="c1",
        document_id="d1",
        text="Corps",
        metadata={"section_heading": "  \t  "},
    )
    assert passage_for_embedding_and_rerank(c) == "Corps"
