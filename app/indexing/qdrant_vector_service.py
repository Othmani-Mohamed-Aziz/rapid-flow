"""
QdrantHybridVectorService : implémentation `VectorIndexService` production-ready.

- Tenant obligatoire (Option C : 1 collection par tenant, par défaut `study_id`).
- Dense + sparse BM25 (hybride RRF côté serveur Qdrant) si activé.
- Reranker CrossEncoder optionnel après top‑K.
- Métadonnées payload alignées sur le schéma de `app/indexing/qdrant_bootstrap.py`.
- Upsert idempotent : `point_id = chunk.chunk_id`.
- Suppression `delete_document(document_id, tenant_id)`.

⚠️ Filtres :
  - `tenant_id` est utilisé pour calculer le nom de collection (sécurité structurelle).
  - `patient_id` / `document_id` / `field_family` filtrés au query-time côté payload.

API conservée :
  - `upsert_chunks(chunks, *, tenant_id, ...)` (variante étendue)
  - `search(*, document_id, query_text, ..., tenant_id, patient_id)` étendu
"""

from __future__ import annotations

import uuid

from app.config.settings import Settings
from app.indexing.embeddings import EmbeddingService, build_default_embedding_service
from app.indexing.qdrant_bootstrap import collection_name, ensure_collection
from app.indexing.reranker import BGECrossEncoderReranker, Reranker
from app.indexing.sparse import FastembedBM25SparseEncoder, SparseEncoder
from app.indexing.vector_service import VectorIndexService
from app.schemas.enums import FieldFamily
from app.schemas.models import DocumentChunk

_VEC_DENSE = "text"
_VEC_SPARSE = "text_bm25"


def _stable_point_id(chunk_id: str) -> str:
    """Qdrant accepte UUID ou int. On hash le `chunk_id` métier en UUIDv5 déterministe."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _build_payload(
    *,
    chunk: DocumentChunk,
    tenant_id: str,
    patient_id: str | None,
    study_id: str | None,
    document_type: str | None,
    embedding_version: str,
) -> dict:
    md = dict(chunk.metadata or {})
    return {
        "tenant_id": tenant_id,
        "patient_id": patient_id,
        "study_id": study_id,
        "document_id": chunk.document_id,
        "document_type": document_type,
        "field_family": chunk.field_family.value if chunk.field_family is not None else None,
        "chunk_id": chunk.chunk_id,
        "section_heading": md.get("section_heading"),
        "section_index": md.get("section_index"),
        "part_index": md.get("part_index"),
        "content_kind": md.get("content_kind"),
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "char_spans_verified": bool(md.get("char_spans_verified", False)),
        "embedding_version": embedding_version,
        "text": chunk.text,
    }


def _payload_to_chunk(payload: dict) -> DocumentChunk:
    md = {
        k: payload.get(k)
        for k in (
            "section_heading",
            "section_index",
            "part_index",
            "content_kind",
            "char_spans_verified",
            "embedding_version",
            "tenant_id",
            "patient_id",
            "study_id",
            "document_type",
        )
        if payload.get(k) is not None
    }
    fam_val = payload.get("field_family")
    fam = FieldFamily(fam_val) if fam_val else None
    return DocumentChunk(
        chunk_id=payload.get("chunk_id") or "",
        document_id=payload.get("document_id") or "",
        text=payload.get("text") or "",
        field_family=fam,
        char_start=payload.get("char_start"),
        char_end=payload.get("char_end"),
        metadata=md,
    )


class QdrantNotInstalledError(RuntimeError):
    """L'extra `vector` n'est pas installé."""


class QdrantHybridVectorService(VectorIndexService):
    """
    Vector store hybride : dense (e5/BGE) + sparse BM25 + reranker (optionnels).

    Le `tenant_id` détermine la **collection cible**. Il doit être passé
    explicitement à `upsert_chunks` / `search` / `delete_document`. Une absence
    de `tenant_id` lève `ValueError`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client=None,
        embeddings: EmbeddingService | None = None,
        sparse: SparseEncoder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self._client = client or self._build_client()
        self.embeddings = embeddings or build_default_embedding_service(self.settings)
        self._sparse = sparse or (
            FastembedBM25SparseEncoder(self.settings) if self.settings.enable_sparse_bm25 else None
        )
        self._reranker = reranker or (
            BGECrossEncoderReranker(self.settings) if self.settings.enable_reranker else None
        )

    # ---------- Construction client ------------------------------------------

    def _build_client(self):
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:  # pragma: no cover
            raise QdrantNotInstalledError('Installer l\'extra : pip install -e ".[vector]"') from e
        return QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            prefer_grpc=self.settings.qdrant_prefer_grpc,
            timeout=int(self.settings.qdrant_timeout_s),
        )

    # ---------- Bootstrap (par tenant) ---------------------------------------

    def ensure_tenant_collection(self, tenant_id: str) -> str:
        return ensure_collection(self._client, self.settings, tenant_id)

    # ---------- Upsert -------------------------------------------------------

    def upsert_chunks(  # type: ignore[override]
        self,
        chunks: list[DocumentChunk],
        *,
        tenant_id: str | None = None,
        patient_id: str | None = None,
        study_id: str | None = None,
        document_type: str | None = None,
    ) -> None:
        if not chunks:
            return
        tenant = self._require_tenant(tenant_id)
        coll = self.ensure_tenant_collection(tenant)
        embed_version = self.embeddings.version_tag

        texts = [c.text for c in chunks]
        dense_vecs = self.embeddings.embed_passages(texts)
        sparse_vecs = self._sparse.encode_passages(texts) if self._sparse is not None else None

        from qdrant_client import models as qm

        points: list[qm.PointStruct] = []
        for i, c in enumerate(chunks):
            payload = _build_payload(
                chunk=c,
                tenant_id=tenant,
                patient_id=patient_id,
                study_id=study_id,
                document_type=document_type,
                embedding_version=embed_version,
            )
            vector: dict = {_VEC_DENSE: dense_vecs[i]}
            if sparse_vecs is not None:
                sv = sparse_vecs[i]
                vector[_VEC_SPARSE] = qm.SparseVector(indices=sv.indices, values=sv.values)
            points.append(
                qm.PointStruct(
                    id=_stable_point_id(c.chunk_id),
                    vector=vector,
                    payload=payload,
                )
            )

        self._client.upsert(collection_name=coll, points=points, wait=True)

    # ---------- Delete -------------------------------------------------------

    def delete_document(self, *, tenant_id: str, document_id: str) -> int:
        """Supprime tous les points (tenant_id, document_id). Retourne 1 si le
        serveur a confirmé l’opération (`UpdateStatus.COMPLETED`), 0 sinon."""
        tenant = self._require_tenant(tenant_id)
        coll = collection_name(self.settings, tenant)
        from qdrant_client import models as qm

        flt = qm.Filter(
            must=[
                qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant)),
                qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id)),
            ]
        )
        res = self._client.delete(
            collection_name=coll,
            points_selector=qm.FilterSelector(filter=flt),
            wait=True,
        )
        status = getattr(res, "status", None)
        status_value = getattr(status, "value", status)
        return 1 if str(status_value).lower() == "completed" else 0

    # ---------- Search -------------------------------------------------------

    def search(  # type: ignore[override]
        self,
        *,
        document_id: str | None = None,
        query_text: str,
        field_family: FieldFamily | None = None,
        top_k: int = 5,
        tenant_id: str | None = None,
        patient_id: str | None = None,
        document_type: str | None = None,
        min_score: float | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        if not query_text or not query_text.strip():
            return []
        tenant = self._require_tenant(tenant_id)
        coll = collection_name(self.settings, tenant)

        from qdrant_client import models as qm

        flt = self._build_filter(
            tenant_id=tenant,
            patient_id=patient_id,
            document_id=document_id,
            document_type=document_type,
            field_family=field_family,
        )

        candidate_k = max(top_k, top_k * self.settings.rerank_candidate_multiplier)
        dense_query = self.embeddings.embed_query(query_text)

        if self._sparse is not None:
            sparse_q = self._sparse.encode_query(query_text)
            qd_resp = self._client.query_points(
                collection_name=coll,
                prefetch=[
                    qm.Prefetch(
                        query=dense_query,
                        using=_VEC_DENSE,
                        limit=candidate_k,
                        filter=flt,
                    ),
                    qm.Prefetch(
                        query=qm.SparseVector(indices=sparse_q.indices, values=sparse_q.values),
                        using=_VEC_SPARSE,
                        limit=candidate_k,
                        filter=flt,
                    ),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=candidate_k,
                with_payload=True,
            )
            scored_points = qd_resp.points
        else:
            qd_resp = self._client.query_points(
                collection_name=coll,
                query=dense_query,
                using=_VEC_DENSE,
                limit=candidate_k,
                query_filter=flt,
                with_payload=True,
            )
            scored_points = qd_resp.points

        results: list[tuple[DocumentChunk, float]] = []
        for sp in scored_points:
            payload = dict(sp.payload or {})
            chunk = _payload_to_chunk(payload)
            results.append((chunk, float(sp.score)))

        if min_score is not None:
            results = [(c, s) for c, s in results if s >= min_score]

        if self._reranker is not None and results:
            reranked = self._reranker.rerank(
                query=query_text,
                candidates=results,
                top_n=top_k,
            )
            return reranked

        return results[:top_k]

    # ---------- Helpers ------------------------------------------------------

    @staticmethod
    def _require_tenant(tenant_id: str | None) -> str:
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id obligatoire (Option C : 1 collection par tenant).")
        return tenant_id.strip()

    @staticmethod
    def _build_filter(
        *,
        tenant_id: str,
        patient_id: str | None,
        document_id: str | None,
        document_type: str | None,
        field_family: FieldFamily | None,
    ):
        from typing import Any

        from qdrant_client import models as qm

        # `must` reste typé large pour Qdrant : `Filter(must=...)` accepte
        # un mélange `FieldCondition | Filter | …` mais mypy infère invariant
        # depuis `list[FieldCondition]`.
        must: list[Any] = [qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))]
        if patient_id:
            must.append(qm.FieldCondition(key="patient_id", match=qm.MatchValue(value=patient_id)))
        if document_id:
            must.append(
                qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))
            )
        if document_type:
            must.append(
                qm.FieldCondition(key="document_type", match=qm.MatchValue(value=document_type))
            )
        if field_family is not None:
            must.append(
                qm.FieldCondition(key="field_family", match=qm.MatchValue(value=field_family.value))
            )
        return qm.Filter(must=must)
