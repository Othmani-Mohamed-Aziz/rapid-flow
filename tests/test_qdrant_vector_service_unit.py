"""Tests unitaires sur `QdrantHybridVectorService` (avec mocks)."""

from __future__ import annotations

import logging
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

_QDRANT_LOGGER = "app.indexing.qdrant_vector_service"


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


class _CapturingEmbeddings(_FakeEmbeddings):
    """Enregistre les textes passés à `embed_passages` (upsert)."""

    def __init__(self) -> None:
        self.passage_texts: list[str] = []

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:  # type: ignore[override]
        self.passage_texts = list(texts)
        return super().embed_passages(texts)


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
    *,
    with_sparse: bool = False,
    with_reranker: bool = False,
    qdrant_read_only: bool = False,
    qdrant_max_retries: int = 3,
) -> tuple[QdrantHybridVectorService, MagicMock]:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    settings = Settings(
        vector_backend="qdrant",
        enable_sparse_bm25=with_sparse,
        enable_reranker=with_reranker,
        qdrant_read_only=qdrant_read_only,
        qdrant_max_retries=qdrant_max_retries,
        # Pas de sleep dans les tests unit (backoff_base=0).
        qdrant_retry_backoff_base=0.0,
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
        embedding_model="fake-model",
        embedding_dim=4,
    )
    assert payload["tenant_id"] == "STUDY-01"
    assert payload["field_family"] == FieldFamily.HEPATIC_BIOCHEMISTRY.value
    assert payload["embedding_version"] == "fake-v1"
    # F1: model + dim doivent être en payload (audit / forensic).
    assert payload["embedding_model"] == "fake-model"
    assert payload["embedding_dim"] == 4
    rebuilt = _payload_to_chunk(payload)
    assert rebuilt.chunk_id == "c1"
    assert rebuilt.field_family == FieldFamily.HEPATIC_BIOCHEMISTRY
    assert rebuilt.text == "lésion hépatique"
    # F1: model + dim doivent aussi survivre au round-trip (metadata du chunk).
    assert rebuilt.metadata.get("embedding_model") == "fake-model"
    assert rebuilt.metadata.get("embedding_dim") == 4


def test_build_filter_field_family_accepts_untagged_or_matching() -> None:
    from qdrant_client import models as qm

    from app.indexing.qdrant_vector_service import QdrantHybridVectorService
    from app.schemas.enums import FieldFamily

    flt = QdrantHybridVectorService._build_filter(
        tenant_id="T1",
        patient_id=None,
        document_id="doc-1",
        document_type="imaging_report",
        field_family=FieldFamily.IMAGING_RECIST,
        embedding_version="e5-v1",
    )
    assert len(flt.must) == 5
    family_clause = flt.must[3]
    assert isinstance(family_clause, qm.Filter)
    assert len(family_clause.should) == 2
    assert any(
        isinstance(c, qm.FieldCondition)
        and c.key == "field_family"
        and c.match.value == FieldFamily.IMAGING_RECIST.value
        for c in family_clause.should
    )
    assert any(isinstance(c, qm.IsNullCondition) for c in family_clause.should)


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


def test_upsert_embeds_heading_plus_body_not_payload_text() -> None:
    """Les vecteurs dense/sparse utilisent titre+corps ; le payload garde le corps seul."""
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    emb = _CapturingEmbeddings()
    settings = Settings(vector_backend="qdrant", enable_sparse_bm25=True)
    svc = QdrantHybridVectorService(settings, client=client, embeddings=emb, sparse=_FakeSparse())
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_id="d1",
            text="corps seul",
            metadata={"section_heading": "RÉSULTATS"},
        )
    ]
    svc.upsert_chunks(chunks, tenant_id="STUDY-1")
    assert emb.passage_texts == ["RÉSULTATS\ncorps seul"]
    pt = client.upsert.call_args.kwargs["points"][0]
    assert pt.payload["text"] == "corps seul"
    assert pt.payload["section_heading"] == "RÉSULTATS"


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


# ─────────────────────────────────────────────────────────────────────────────
# A4 — batching automatique sur upsert
# ─────────────────────────────────────────────────────────────────────────────


def test_upsert_batches_when_over_limit() -> None:
    """7 chunks avec upsert_batch_size=3 → 3 appels (3 + 3 + 1)."""
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    settings = Settings(vector_backend="qdrant", upsert_batch_size=3)
    svc = QdrantHybridVectorService(settings, client=client, embeddings=_FakeEmbeddings())
    chunks = [
        DocumentChunk(chunk_id=f"c{i}", document_id="d", text=f"t{i}", metadata={})
        for i in range(7)
    ]
    svc.upsert_chunks(chunks, tenant_id="STUDY-1")
    assert client.upsert.call_count == 3
    sizes = [len(call.kwargs["points"]) for call in client.upsert.call_args_list]
    assert sizes == [3, 3, 1]
    # Tous les points doivent avoir transité sans collision d'IDs.
    all_ids = [p.id for call in client.upsert.call_args_list for p in call.kwargs["points"]]
    assert len(set(all_ids)) == 7


def test_upsert_single_batch_when_below_limit() -> None:
    """Cas nominal : N <= upsert_batch_size → un seul appel."""
    svc, client = _make_service()  # batch_size par défaut = 256
    chunks = [
        DocumentChunk(chunk_id=f"c{i}", document_id="d", text=f"t{i}", metadata={})
        for i in range(5)
    ]
    svc.upsert_chunks(chunks, tenant_id="STUDY-1")
    assert client.upsert.call_count == 1
    assert len(client.upsert.call_args.kwargs["points"]) == 5


# ─────────────────────────────────────────────────────────────────────────────
# A6 — delete_document retourne le nombre de points supprimés
# ─────────────────────────────────────────────────────────────────────────────


def test_delete_document_returns_count() -> None:
    svc, client = _make_service()
    client.count.return_value = MagicMock(count=3)
    client.delete.return_value = MagicMock(status=MagicMock(value="completed"))
    n = svc.delete_document(tenant_id="STUDY-1", document_id="d42")
    assert n == 3
    client.count.assert_called_once()
    client.delete.assert_called_once()


def test_delete_document_returns_zero_when_nothing_to_delete() -> None:
    """Si aucun point ne matche, on ne fait même pas l'appel delete."""
    svc, client = _make_service()
    client.count.return_value = MagicMock(count=0)
    n = svc.delete_document(tenant_id="STUDY-1", document_id="dghost")
    assert n == 0
    client.delete.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# A2 — filtre embedding_version auto-injecté au search
# ─────────────────────────────────────────────────────────────────────────────


def _filter_keys_from_call(call_kwargs: dict) -> list[str]:
    """Extrait la liste des `key` des FieldCondition d'un Filter passé à query_points."""
    flt = call_kwargs.get("query_filter") or call_kwargs.get("prefetch", [None])[0]
    if hasattr(flt, "filter"):
        flt = flt.filter
    must = getattr(flt, "must", [])
    keys: list[str] = []
    for cond in must:
        key = getattr(cond, "key", None)
        if key:
            keys.append(key)
    return keys


def test_search_dense_injects_embedding_version_filter_by_default() -> None:
    """A2: settings par défaut → filtre `embedding_version` présent."""
    svc, client = _make_service()
    client.query_points.return_value = MagicMock(points=[])
    svc.search(query_text="x", tenant_id="STUDY-1", top_k=2)
    _, kwargs = client.query_points.call_args
    keys = _filter_keys_from_call(kwargs)
    assert "embedding_version" in keys
    assert "tenant_id" in keys


def test_search_does_not_inject_embedding_version_when_disabled() -> None:
    """A2: opt-out via enforce_embedding_version_filter=False (migration)."""
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    settings = Settings(
        vector_backend="qdrant",
        enforce_embedding_version_filter=False,
    )
    svc = QdrantHybridVectorService(settings, client=client, embeddings=_FakeEmbeddings())
    client.query_points.return_value = MagicMock(points=[])
    svc.search(query_text="x", tenant_id="STUDY-1", top_k=2)
    _, kwargs = client.query_points.call_args
    keys = _filter_keys_from_call(kwargs)
    assert "embedding_version" not in keys
    assert "tenant_id" in keys  # tenant reste obligatoire


def test_search_hybrid_also_injects_version_filter_in_prefetch() -> None:
    """A2: en mode hybride, le filtre version doit être appliqué dans chaque
    Prefetch (dense + sparse), sinon BM25 leakerait des points d'autres versions."""
    svc, client = _make_service(with_sparse=True)
    client.query_points.return_value = MagicMock(points=[])
    svc.search(query_text="ast", tenant_id="STUDY-1", top_k=2)
    _, kwargs = client.query_points.call_args
    prefetches = kwargs["prefetch"]
    for pf in prefetches:
        must = getattr(pf.filter, "must", [])
        keys = [getattr(c, "key", None) for c in must]
        assert "embedding_version" in keys, f"prefetch {pf.using}: pas de filtre version"


# ─────────────────────────────────────────────────────────────────────────────
# B1 — Retry exponentiel sur erreurs Qdrant transientes
# ─────────────────────────────────────────────────────────────────────────────


class _FakeQdrantHttpError(Exception):
    """Mimique une `UnexpectedResponse` du client Qdrant (status_code attribute).

    On NE référence PAS `qdrant_client.http.exceptions.UnexpectedResponse`
    directement pour rester découplé de la signature exacte (qui change selon
    les versions du client) ; le code de prod n'introspecte que `status_code`.
    """

    def __init__(self, status_code: int, msg: str = "") -> None:
        super().__init__(msg)
        self.status_code = status_code


def _ok_upsert_result() -> MagicMock:
    return MagicMock(status=MagicMock(value="completed"))


def test_retry_succeeds_after_transient_failures() -> None:
    """ConnectionError x2 puis succès → 3 tentatives, pas d'exception remontée."""
    svc, client = _make_service(qdrant_max_retries=3)
    client.upsert.side_effect = [
        ConnectionError("transient #1"),
        ConnectionError("transient #2"),
        _ok_upsert_result(),
    ]
    chunks = [DocumentChunk(chunk_id="c1", document_id="d", text="t", metadata={})]
    svc.upsert_chunks(chunks, tenant_id="STUDY-1")
    assert client.upsert.call_count == 3


def test_retry_gives_up_after_max_attempts() -> None:
    """Erreur transiente permanente → on raise après max_retries+1 tentatives."""
    svc, client = _make_service(qdrant_max_retries=2)
    client.upsert.side_effect = ConnectionError("permanently broken")
    chunks = [DocumentChunk(chunk_id="c1", document_id="d", text="t", metadata={})]
    with pytest.raises(ConnectionError):
        svc.upsert_chunks(chunks, tenant_id="STUDY-1")
    # max_retries=2 → 1 initiale + 2 retries = 3 appels.
    assert client.upsert.call_count == 3


def test_retry_does_not_retry_on_value_error() -> None:
    """Erreur programmatique → 1 seul appel (pas de retry)."""
    svc, client = _make_service(qdrant_max_retries=5)
    client.upsert.side_effect = ValueError("bad input")
    chunks = [DocumentChunk(chunk_id="c1", document_id="d", text="t", metadata={})]
    with pytest.raises(ValueError):
        svc.upsert_chunks(chunks, tenant_id="STUDY-1")
    assert client.upsert.call_count == 1


def test_retry_does_not_retry_on_4xx() -> None:
    """Status 400 (Bad Request) → pas de retry, on raise direct."""
    svc, client = _make_service(qdrant_max_retries=3)
    client.upsert.side_effect = _FakeQdrantHttpError(400, "bad request")
    chunks = [DocumentChunk(chunk_id="c1", document_id="d", text="t", metadata={})]
    with pytest.raises(_FakeQdrantHttpError):
        svc.upsert_chunks(chunks, tenant_id="STUDY-1")
    assert client.upsert.call_count == 1


def test_retry_retries_on_5xx() -> None:
    """Status 503 (Service Unavailable) → on retry."""
    svc, client = _make_service(qdrant_max_retries=2)
    client.upsert.side_effect = [
        _FakeQdrantHttpError(503, "unavailable"),
        _ok_upsert_result(),
    ]
    chunks = [DocumentChunk(chunk_id="c1", document_id="d", text="t", metadata={})]
    svc.upsert_chunks(chunks, tenant_id="STUDY-1")
    assert client.upsert.call_count == 2


def test_retry_retries_on_429() -> None:
    """Status 429 (Too Many Requests) → on retry (rate-limit transient)."""
    svc, client = _make_service(qdrant_max_retries=2)
    client.upsert.side_effect = [
        _FakeQdrantHttpError(429, "rate limited"),
        _ok_upsert_result(),
    ]
    chunks = [DocumentChunk(chunk_id="c1", document_id="d", text="t", metadata={})]
    svc.upsert_chunks(chunks, tenant_id="STUDY-1")
    assert client.upsert.call_count == 2


def test_retry_is_applied_to_search_too() -> None:
    """Le retry couvre aussi `search` (pas seulement les writes)."""
    svc, client = _make_service(qdrant_max_retries=2)
    client.query_points.side_effect = [
        ConnectionError("flap"),
        MagicMock(points=[]),
    ]
    results = svc.search(query_text="x", tenant_id="STUDY-1", top_k=2)
    assert results == []
    assert client.query_points.call_count == 2


def test_retry_zero_max_means_no_retry() -> None:
    """`qdrant_max_retries=0` → 1 seul appel, échec direct."""
    svc, client = _make_service(qdrant_max_retries=0)
    client.upsert.side_effect = ConnectionError("boom")
    chunks = [DocumentChunk(chunk_id="c1", document_id="d", text="t", metadata={})]
    with pytest.raises(ConnectionError):
        svc.upsert_chunks(chunks, tenant_id="STUDY-1")
    assert client.upsert.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# B2 — Mode read-only (Settings.qdrant_read_only)
# ─────────────────────────────────────────────────────────────────────────────


def test_read_only_blocks_upsert() -> None:
    svc, client = _make_service(qdrant_read_only=True)
    chunks = [DocumentChunk(chunk_id="c1", document_id="d", text="t", metadata={})]
    with pytest.raises(RuntimeError, match="read.?only|read_only|ECRF_QDRANT_READ_ONLY"):
        svc.upsert_chunks(chunks, tenant_id="STUDY-1")
    client.upsert.assert_not_called()


def test_read_only_blocks_delete() -> None:
    svc, client = _make_service(qdrant_read_only=True)
    with pytest.raises(RuntimeError, match="read.?only|read_only|ECRF_QDRANT_READ_ONLY"):
        svc.delete_document(tenant_id="STUDY-1", document_id="d42")
    client.delete.assert_not_called()
    client.count.assert_not_called()


def test_read_only_allows_search() -> None:
    """Search reste fonctionnel en read-only (c'est tout l'intérêt du mode)."""
    svc, client = _make_service(qdrant_read_only=True)
    client.query_points.return_value = MagicMock(points=[])
    results = svc.search(query_text="lésion", tenant_id="STUDY-1", top_k=2)
    assert results == []
    client.query_points.assert_called_once()


def test_read_only_ensure_collection_does_not_create() -> None:
    """En read-only, `ensure_tenant_collection` ne tente PAS la création
    (l'instance n'a peut-être même pas les droits côté Qdrant)."""
    svc, client = _make_service(qdrant_read_only=True)
    name = svc.ensure_tenant_collection("STUDY-X")
    assert name == "ecrf_chunks__STUDY-X"
    client.get_collections.assert_not_called()
    client.create_collection.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# C1 — Préservation du score Qdrant après reranking
# ─────────────────────────────────────────────────────────────────────────────


def _fake_point(*, score: float, chunk_id: str, text: str = "fragment") -> MagicMock:
    p = MagicMock()
    p.score = score
    p.payload = {
        "chunk_id": chunk_id,
        "document_id": "d1",
        "text": text,
        "tenant_id": "STUDY-1",
        "field_family": None,
    }
    return p


class _InvertingReranker(Reranker):
    """Inverse l'ordre des candidats Qdrant et assigne des scores rerank
    décroissants (1.0, 0.5, …). Permet de vérifier que :
      - le rerank touche bien le classement (ordre différent du score Qdrant),
      - le score Qdrant initial reste accessible via `chunk.metadata`."""

    def rerank(self, *, query, candidates, top_n):
        out = list(reversed(list(candidates)))[:top_n]
        return [(c, 1.0 / (i + 1)) for i, (c, _) in enumerate(out)]


def test_search_attaches_retrieval_score_metadata_without_rerank() -> None:
    """Sans reranker, `retrieval_score`/`retrieval_rank` doivent être présents."""
    svc, client = _make_service()
    client.query_points.return_value = MagicMock(
        points=[
            _fake_point(score=0.85, chunk_id="c1"),
            _fake_point(score=0.55, chunk_id="c2"),
        ]
    )
    results = svc.search(query_text="lésion", tenant_id="STUDY-1", top_k=2)
    assert len(results) == 2
    c1, score1 = results[0]
    assert c1.metadata["retrieval_score"] == pytest.approx(0.85)
    assert c1.metadata["retrieval_rank"] == 1
    assert "rerank_score" not in c1.metadata
    assert "rerank_rank" not in c1.metadata
    # API rétrocompatible : tuple score == retrieval_score sans rerank.
    assert score1 == pytest.approx(0.85)


def test_search_preserves_retrieval_score_after_rerank() -> None:
    """Avec reranker, les DEUX scores doivent être dans `chunk.metadata`."""
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    settings = Settings(
        vector_backend="qdrant",
        enable_reranker=True,
        qdrant_retry_backoff_base=0.0,
    )
    svc = QdrantHybridVectorService(
        settings,
        client=client,
        embeddings=_FakeEmbeddings(),
        reranker=_InvertingReranker(),
    )
    client.query_points.return_value = MagicMock(
        points=[
            _fake_point(score=0.9, chunk_id="c1"),
            _fake_point(score=0.7, chunk_id="c2"),
        ]
    )
    results = svc.search(query_text="x", tenant_id="STUDY-1", top_k=2)
    assert len(results) == 2

    # Rerank inverse l'ordre : c2 d'abord, c1 ensuite.
    first_chunk, first_rerank_score = results[0]
    second_chunk, second_rerank_score = results[1]
    assert first_chunk.chunk_id == "c2"
    assert second_chunk.chunk_id == "c1"

    # API rétrocompatible : tuple score == rerank_score.
    assert first_rerank_score == pytest.approx(1.0)
    assert second_rerank_score == pytest.approx(0.5)

    # C1 : retrieval_score (Qdrant) préservé indépendamment du rerank.
    assert first_chunk.metadata["retrieval_score"] == pytest.approx(0.7)
    assert first_chunk.metadata["retrieval_rank"] == 2  # 2e côté Qdrant
    assert first_chunk.metadata["rerank_score"] == pytest.approx(1.0)
    assert first_chunk.metadata["rerank_rank"] == 1  # 1er côté rerank

    assert second_chunk.metadata["retrieval_score"] == pytest.approx(0.9)
    assert second_chunk.metadata["retrieval_rank"] == 1
    assert second_chunk.metadata["rerank_score"] == pytest.approx(0.5)
    assert second_chunk.metadata["rerank_rank"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# C2 — Logging structuré + timings
# ─────────────────────────────────────────────────────────────────────────────


def _find_event(caplog: pytest.LogCaptureFixture, event_name: str) -> logging.LogRecord:
    matches = [r for r in caplog.records if getattr(r, "event", None) == event_name]
    assert matches, f"event '{event_name}' absent. Records: {[r.message for r in caplog.records]}"
    return matches[-1]


def test_search_emits_done_event_with_timings_and_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    svc, client = _make_service()
    client.query_points.return_value = MagicMock(points=[_fake_point(score=0.9, chunk_id="c1")])
    with caplog.at_level(logging.INFO, logger=_QDRANT_LOGGER):
        svc.search(query_text="x", tenant_id="STUDY-1", top_k=3)

    rec = _find_event(caplog, "qdrant.search.done")
    assert rec.tenant == "STUDY-1"
    assert rec.top_k == 3
    assert rec.n_candidates == 1
    assert rec.n_returned == 1
    assert rec.rerank_used is False
    assert rec.hybrid is False
    # Timings cohérents : embed_dense + qdrant_query + total présents et >= 0.
    assert "embed_dense" in rec.timings_ms
    assert "qdrant_query" in rec.timings_ms
    assert "total" in rec.timings_ms
    assert rec.timings_ms["total"] >= 0.0


def test_search_hybrid_log_includes_sparse_timing(caplog: pytest.LogCaptureFixture) -> None:
    svc, client = _make_service(with_sparse=True)
    client.query_points.return_value = MagicMock(points=[])
    with caplog.at_level(logging.INFO, logger=_QDRANT_LOGGER):
        svc.search(query_text="ast", tenant_id="STUDY-1", top_k=2)
    rec = _find_event(caplog, "qdrant.search.done")
    assert rec.hybrid is True
    assert "embed_sparse" in rec.timings_ms


def test_search_rerank_log_includes_rerank_timing(caplog: pytest.LogCaptureFixture) -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    settings = Settings(
        vector_backend="qdrant",
        enable_reranker=True,
        qdrant_retry_backoff_base=0.0,
    )
    svc = QdrantHybridVectorService(
        settings,
        client=client,
        embeddings=_FakeEmbeddings(),
        reranker=_InvertingReranker(),
    )
    client.query_points.return_value = MagicMock(points=[_fake_point(score=0.9, chunk_id="c1")])
    with caplog.at_level(logging.INFO, logger=_QDRANT_LOGGER):
        svc.search(query_text="x", tenant_id="STUDY-1", top_k=2)
    rec = _find_event(caplog, "qdrant.search.done")
    assert rec.rerank_used is True
    assert "rerank" in rec.timings_ms


def test_upsert_emits_done_event_with_n_batches(caplog: pytest.LogCaptureFixture) -> None:
    """upsert_batch_size=2 + 5 chunks → 3 batches, événement avec timings."""
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    settings = Settings(
        vector_backend="qdrant",
        upsert_batch_size=2,
        qdrant_retry_backoff_base=0.0,
    )
    svc = QdrantHybridVectorService(settings, client=client, embeddings=_FakeEmbeddings())
    chunks = [
        DocumentChunk(chunk_id=f"c{i}", document_id="d", text=f"t{i}", metadata={})
        for i in range(5)
    ]
    with caplog.at_level(logging.INFO, logger=_QDRANT_LOGGER):
        svc.upsert_chunks(chunks, tenant_id="STUDY-1")

    rec = _find_event(caplog, "qdrant.upsert.done")
    assert rec.tenant == "STUDY-1"
    assert rec.n_chunks == 5
    assert rec.n_batches == 3
    assert rec.batch_size == 2
    assert "embed_dense" in rec.timings_ms
    assert "qdrant_upsert" in rec.timings_ms
    assert "total" in rec.timings_ms


def test_delete_emits_done_event_with_n_deleted(caplog: pytest.LogCaptureFixture) -> None:
    svc, client = _make_service()
    client.count.return_value = MagicMock(count=3)
    client.delete.return_value = MagicMock(status=MagicMock(value="completed"))
    with caplog.at_level(logging.INFO, logger=_QDRANT_LOGGER):
        n = svc.delete_document(tenant_id="STUDY-1", document_id="d42")
    assert n == 3
    rec = _find_event(caplog, "qdrant.delete.done")
    assert rec.document_id == "d42"
    assert rec.n_deleted == 3
    assert "count" in rec.timings_ms
    assert "delete" in rec.timings_ms
    assert "total" in rec.timings_ms


def test_delete_emits_done_event_even_when_nothing_to_delete(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Idempotence + observabilité : un delete sur doc inexistant produit
    quand même un event (n_deleted=0, pas de timing 'delete' car pas d'appel)."""
    svc, client = _make_service()
    client.count.return_value = MagicMock(count=0)
    with caplog.at_level(logging.INFO, logger=_QDRANT_LOGGER):
        n = svc.delete_document(tenant_id="STUDY-1", document_id="ghost")
    assert n == 0
    rec = _find_event(caplog, "qdrant.delete.done")
    assert rec.n_deleted == 0
    assert "count" in rec.timings_ms
    assert "delete" not in rec.timings_ms  # short-circuit avant l'appel delete


# ─────────────────────────────────────────────────────────────────────────────
# D1 — list_chunks paginé via scroll
# ─────────────────────────────────────────────────────────────────────────────


def _fake_record(*, chunk_id: str, document_id: str = "d1") -> MagicMock:
    rec = MagicMock()
    rec.payload = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "text": f"text-{chunk_id}",
        "tenant_id": "STUDY-1",
        "field_family": None,
    }
    return rec


def test_list_chunks_returns_chunks_and_next_offset() -> None:
    svc, client = _make_service()
    client.scroll.return_value = (
        [_fake_record(chunk_id="c1"), _fake_record(chunk_id="c2")],
        "next-token-42",
    )
    chunks, next_offset = svc.list_chunks(tenant_id="STUDY-1", limit=2)
    assert [c.chunk_id for c in chunks] == ["c1", "c2"]
    assert next_offset == "next-token-42"
    # Le filtre passé à Qdrant ne contient que tenant_id (pas de document_id).
    _, kwargs = client.scroll.call_args
    assert kwargs["collection_name"] == "ecrf_chunks__STUDY-1"
    keys = [getattr(c, "key", None) for c in kwargs["scroll_filter"].must]
    assert keys == ["tenant_id"]
    assert kwargs["with_vectors"] is False


def test_list_chunks_filters_by_document_id() -> None:
    svc, client = _make_service()
    client.scroll.return_value = ([_fake_record(chunk_id="c1")], None)
    chunks, next_offset = svc.list_chunks(tenant_id="STUDY-1", document_id="d42", limit=10)
    assert chunks[0].chunk_id == "c1"
    assert next_offset is None
    _, kwargs = client.scroll.call_args
    keys = [getattr(c, "key", None) for c in kwargs["scroll_filter"].must]
    assert keys == ["tenant_id", "document_id"]


def test_list_chunks_pagination_passes_offset() -> None:
    svc, client = _make_service()
    client.scroll.return_value = ([_fake_record(chunk_id="c3")], None)
    svc.list_chunks(tenant_id="STUDY-1", limit=10, offset="prev-token")
    _, kwargs = client.scroll.call_args
    assert kwargs["offset"] == "prev-token"


def test_list_chunks_requires_tenant() -> None:
    svc, _ = _make_service()
    with pytest.raises(ValueError):
        svc.list_chunks(tenant_id="")


def test_list_chunks_works_in_read_only_mode() -> None:
    """list_chunks est une opération de lecture → autorisée en read-only."""
    svc, client = _make_service(qdrant_read_only=True)
    client.scroll.return_value = ([_fake_record(chunk_id="c1")], None)
    chunks, _ = svc.list_chunks(tenant_id="STUDY-1", limit=5)
    assert len(chunks) == 1


def test_list_chunks_emits_done_event(caplog: pytest.LogCaptureFixture) -> None:
    svc, client = _make_service()
    client.scroll.return_value = ([_fake_record(chunk_id="c1")], "tok-1")
    with caplog.at_level(logging.INFO, logger=_QDRANT_LOGGER):
        svc.list_chunks(tenant_id="STUDY-1", document_id="d42", limit=50)
    rec = _find_event(caplog, "qdrant.list_chunks.done")
    assert rec.tenant == "STUDY-1"
    assert rec.document_id == "d42"
    assert rec.n_returned == 1
    assert rec.has_more is True


# ─────────────────────────────────────────────────────────────────────────────
# D2 — Validation stricte du tenant_id côté service
# ─────────────────────────────────────────────────────────────────────────────


def test_service_rejects_invalid_tenant_id_in_strict_mode() -> None:
    """Un upsert avec tenant invalide en mode strict → ValueError dès le
    `collection_name`, sans toucher Qdrant."""
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    settings = Settings(
        vector_backend="qdrant",
        qdrant_strict_tenant_id=True,
        qdrant_retry_backoff_base=0.0,
    )
    svc = QdrantHybridVectorService(settings, client=client, embeddings=_FakeEmbeddings())
    chunks = [DocumentChunk(chunk_id="c1", document_id="d", text="t", metadata={})]
    with pytest.raises(ValueError, match="tenant_id"):
        svc.upsert_chunks(chunks, tenant_id="STUDY 01")
    client.upsert.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# F2 — health_check
# ─────────────────────────────────────────────────────────────────────────────


def test_health_check_returns_reachable_when_qdrant_responds() -> None:
    svc, client = _make_service()
    client.get_collections.return_value = MagicMock(
        collections=[MagicMock(), MagicMock(), MagicMock()]
    )
    h = svc.health_check()
    assert h["qdrant_reachable"] is True
    assert h["n_collections"] == 3
    assert "error" not in h
    # Champs config systématiquement présents.
    for key in (
        "qdrant_url",
        "embedding_model",
        "embedding_dim",
        "embedding_version",
        "sparse_enabled",
        "reranker_enabled",
        "read_only",
        "collection_prefix",
    ):
        assert key in h


def test_health_check_returns_unreachable_on_error() -> None:
    """Pas de retry sur health_check : un échec immédiat produit
    `qdrant_reachable=False` + `error`."""
    svc, client = _make_service()
    client.get_collections.side_effect = ConnectionError("server down")
    h = svc.health_check()
    assert h["qdrant_reachable"] is False
    assert h["n_collections"] is None
    assert "error" in h
    assert "ConnectionError" in h["error"]
    # Le get_collections ne doit avoir été appelé qu'UNE seule fois (pas de retry).
    assert client.get_collections.call_count == 1


def test_health_check_reflects_config_flags() -> None:
    """sparse_enabled / reranker_enabled / read_only doivent refléter Settings."""
    svc, client = _make_service(with_sparse=True, with_reranker=True, qdrant_read_only=True)
    client.get_collections.return_value = MagicMock(collections=[])
    h = svc.health_check()
    assert h["sparse_enabled"] is True
    assert h["reranker_enabled"] is True
    assert h["read_only"] is True
    assert h["embedding_dim"] == 4  # _FakeEmbeddings.dim
    assert h["embedding_model"] == "fake-emb"
