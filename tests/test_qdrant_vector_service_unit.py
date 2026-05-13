"""Tests unitaires sur `QdrantHybridVectorService` (avec mocks)."""

from __future__ import annotations

from typing import Sequence
from unittest.mock import MagicMock

import pytest

qdrant_client = pytest.importorskip("qdrant_client")  # extra `vector`

from app.config.settings import Settings  # noqa: E402
from app.indexing.embeddings import EmbeddingService  # noqa: E402
from app.indexing.qdrant_vector_service import (  # noqa: E402
    QdrantHybridVectorService,
    _build_payload,
    _payload_to_chunk,
    _stable_point_id,
)
from app.indexing.reranker import Reranker  # noqa: E402
from app.indexing.sparse import SparseEncoder, SparseVector  # noqa: E402
from app.schemas.enums import FieldFamily  # noqa: E402
from app.schemas.models import DocumentChunk  # noqa: E402


class _FakeEmbeddings(EmbeddingService):
    @property
    def dim(self) -> int:
        return 4

    @property
    def model_id(self) -> str:
        return "fake-emb"

    @property
    def version_tag(self) -> str:
        return "fake-v1"

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class _FakeSparse(SparseEncoder):
    @property
    def model_id(self) -> str:
        return "fake-bm25"

    def encode_passages(self, texts):
        return [SparseVector(indices=[1], values=[1.0]) for _ in texts]

    def encode_query(self, text):
        return SparseVector(indices=[1], values=[1.0])


class _NoopReranker(Reranker):
    def rerank(self, *, query, candidates, top_n):
        return list(candidates)[:top_n]


def _make_service(
    *, with_sparse: bool = False, with_reranker: bool = False
) -> tuple[QdrantHybridVectorService, MagicMock]:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    settings = Settings(
        vector_backend="qdrant",
        enable_sparse_bm25=with_sparse,
        enable_reranker=with_reranker,
    )
    svc = QdrantHybridVectorService(
        settings,
        client=client,
        embeddings=_FakeEmbeddings(),
        sparse=_FakeSparse() if with_sparse else None,
        reranker=_NoopReranker() if with_reranker else None,
    )
    return svc, client


def test_payload_build_and_round_trip() -> None:
    chunk = DocumentChunk(
        chunk_id="c1",
        document_id="d1",
        text="lésion hépatique",
        field_family=FieldFamily.HEPATIC_BIOCHEMISTRY,
        char_start=0,
        char_end=10,
        metadata={
            "section_heading": "Conclusion",
            "section_index": 2,
            "part_index": 0,
            "content_kind": "clinical_section",
            "char_spans_verified": True,
        },
    )
    payload = _build_payload(
        chunk=chunk,
        tenant_id="STUDY-01",
        patient_id="P1",
        study_id="STUDY-01",
        document_type="imaging_report",
        embedding_version="fake-v1",
    )
    assert payload["tenant_id"] == "STUDY-01"
    assert payload["field_family"] == FieldFamily.HEPATIC_BIOCHEMISTRY.value
    assert payload["embedding_version"] == "fake-v1"
    rebuilt = _payload_to_chunk(payload)
    assert rebuilt.chunk_id == "c1"
    assert rebuilt.field_family == FieldFamily.HEPATIC_BIOCHEMISTRY
    assert rebuilt.text == "lésion hépatique"


def test_stable_point_id_is_deterministic() -> None:
    a = _stable_point_id("chk_abc")
    b = _stable_point_id("chk_abc")
    c = _stable_point_id("chk_other")
    assert a == b
    assert a != c


def test_upsert_requires_tenant() -> None:
    svc, _ = _make_service()
    with pytest.raises(ValueError):
        svc.upsert_chunks(
            [DocumentChunk(chunk_id="c", document_id="d", text="x", metadata={})],
            tenant_id="",
        )


def test_upsert_creates_collection_and_calls_qdrant() -> None:
    svc, client = _make_service()
    chunks = [
        DocumentChunk(chunk_id="c1", document_id="d1", text="t1", metadata={}),
        DocumentChunk(chunk_id="c2", document_id="d1", text="t2", metadata={}),
    ]
    svc.upsert_chunks(chunks, tenant_id="STUDY-1", patient_id="P", study_id="STUDY-1")
    client.create_collection.assert_called_once()
    args, kwargs = client.upsert.call_args
    assert kwargs["collection_name"] == "ecrf_chunks__STUDY-1"
    assert len(kwargs["points"]) == 2


def test_search_requires_tenant() -> None:
    svc, _ = _make_service()
    with pytest.raises(ValueError):
        svc.search(query_text="x", tenant_id=None)


def test_search_dense_only_calls_query_points(monkeypatch: pytest.MonkeyPatch) -> None:
    svc, client = _make_service()
    fake_point = MagicMock()
    fake_point.score = 0.9
    fake_point.payload = {
        "chunk_id": "c1",
        "document_id": "d1",
        "text": "fragment",
        "tenant_id": "STUDY-1",
        "field_family": None,
    }
    client.query_points.return_value = MagicMock(points=[fake_point])
    results = svc.search(query_text="lésion", tenant_id="STUDY-1", top_k=3)
    assert len(results) == 1
    assert results[0][0].chunk_id == "c1"
    assert results[0][1] == pytest.approx(0.9)


def test_search_hybrid_uses_prefetch(monkeypatch: pytest.MonkeyPatch) -> None:
    svc, client = _make_service(with_sparse=True)
    client.query_points.return_value = MagicMock(points=[])
    svc.search(query_text="ast", tenant_id="STUDY-1", top_k=2)
    _, kwargs = client.query_points.call_args
    assert "prefetch" in kwargs
    assert len(kwargs["prefetch"]) == 2
