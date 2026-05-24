"""
QdrantHybridVectorService : implémentation `VectorIndexService` production-ready.

- Tenant obligatoire (Option C : 1 collection par tenant, par défaut `study_id`).
- Dense + sparse BM25 (hybride RRF côté serveur Qdrant) si activé.
- Reranker CrossEncoder optionnel après top‑K.
- Métadonnées payload alignées sur le schéma de `app/indexing/qdrant_bootstrap.py`.
- Upsert idempotent : `point_id = chunk.chunk_id`.
- Suppression `delete_document(document_id, tenant_id)`.
- **Passage indexé** (dense + sparse + rerank) : `section_heading` + corps via
  `passage_for_embedding_and_rerank` ; le payload Qdrant conserve `text` = corps
  seul pour affichage / audit et cohérence avec les spans du parseur.

⚠️ Filtres :
  - `tenant_id` est utilisé pour calculer le nom de collection (sécurité structurelle).
  - `patient_id` / `document_id` / `field_family` filtrés au query-time côté payload.

API conservée :
  - `upsert_chunks(chunks, *, tenant_id, ...)` (variante étendue)
  - `search(*, document_id, query_text, ..., tenant_id, patient_id)` étendu
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, TypeVar

from app.config.settings import Settings
from app.indexing.embeddings import EmbeddingService, build_default_embedding_service
from app.indexing.qdrant_bootstrap import (
    CollectionConfigMismatchError,
    collection_name,
    ensure_collection,
)
from app.indexing.reranker import BGECrossEncoderReranker, Reranker
from app.indexing.retrieval_passage import passage_for_embedding_and_rerank
from app.indexing.sparse import FastembedBM25SparseEncoder, SparseEncoder
from app.indexing.vector_service import VectorIndexService
from app.schemas.enums import FieldFamily
from app.schemas.models import DocumentChunk

_VEC_DENSE = "text"
_VEC_SPARSE = "text_bm25"

_LOG = logging.getLogger(__name__)

T = TypeVar("T")

#: Exceptions qu'on NE retry JAMAIS : signaler une erreur programmatique ou un
#: mismatch config (`CollectionConfigMismatchError`) → un retry ne ferait que
#: masquer le vrai problème, plus long à diagnostiquer.
_NON_RETRYABLE_TYPES: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    AssertionError,
    KeyError,
    AttributeError,
    CollectionConfigMismatchError,
)


def _is_retryable_qdrant_error(exc: BaseException) -> bool:
    """Décide si une exception du client Qdrant mérite un retry.

    Politique :
    - Erreurs programmatiques (`ValueError`, `TypeError`, …) et
      `CollectionConfigMismatchError` → **jamais** retry.
    - Si l'exception expose `status_code` :
        - 429 (rate-limit) → retry.
        - 5xx (server error / transient) → retry.
        - 4xx autres (bad request, not found, conflict) → **jamais** retry.
    - Sinon (transport error sans status_code : timeout, connection reset,
      `ResponseHandlingException`) → retry (assumé transient)."""
    if isinstance(exc, _NON_RETRYABLE_TYPES):
        return False

    status: Any = getattr(exc, "status_code", None)
    if status is None:
        # Certaines versions du client logent le status dans `content`.
        content = getattr(exc, "content", None)
        if content is not None:
            status = getattr(content, "status", None) or getattr(content, "status_code", None)

    if status is not None:
        try:
            status_int = int(status)
        except (TypeError, ValueError):
            status_int = None
        if status_int is not None:
            if status_int == 429:
                return True
            return 500 <= status_int < 600
        # status présent mais non-entier → on s'abstient pour ne pas masquer une erreur.
        return False

    # Pas de status_code : transport-level error (timeout, connection refused, …).
    return True


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
    embedding_model: str,
    embedding_dim: int,
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
        # ─── traçabilité du vecteur ──────────────────────────────────────────
        # `embedding_version` : tag logique stable (e.g. "e5-base-v1"), utilisé
        # pour le filtre auto au search (cf. settings.enforce_embedding_version_filter).
        # `embedding_model` / `embedding_dim` : info pure pour audit / forensic
        # (savoir d'où vient le vecteur quand plusieurs versions cohabitent).
        "embedding_version": embedding_version,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        # Corps seul (affichage ARC, extraction, spans) — les vecteurs utilisent
        # `passage_for_embedding_and_rerank(chunk)` à l'upsert.
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
            "embedding_model",
            "embedding_dim",
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

    # ---------- Helpers transverses (retry, read-only) -----------------------

    def _check_writable(self, op: str) -> None:
        """Garde-fou écriture : refuse `op` quand l'instance est en read-only.

        Évite qu'un container API "search-only" exposé à Internet puisse muter
        l'index Qdrant via un bug d'orchestration ou un endpoint mal sécurisé."""
        if self.settings.qdrant_read_only:
            raise RuntimeError(
                f"Opération '{op}' interdite : Settings.qdrant_read_only=True. "
                "Cette instance est configurée en lecture seule. Définir "
                "ECRF_QDRANT_READ_ONLY=false pour autoriser l'écriture."
            )

    def _call_with_retry(
        self,
        op_name: str,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Exécute `fn` avec retry exponentiel sur les erreurs Qdrant transientes.

        Voir `_is_retryable_qdrant_error` pour la politique précise.
        Idempotence : tous les call-sites de cette classe utilisent des
        opérations idempotentes (upsert avec point_id déterministe, delete par
        filtre, count, search, ensure_collection)."""
        attempts = max(0, int(self.settings.qdrant_max_retries))
        backoff = max(0.0, float(self.settings.qdrant_retry_backoff_base))
        last_exc: BaseException | None = None
        for attempt in range(attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - on filtre via _is_retryable
                if not _is_retryable_qdrant_error(exc):
                    raise
                last_exc = exc
                if attempt >= attempts:
                    break
                delay = backoff * (2**attempt)
                _LOG.warning(
                    "Qdrant op '%s' échouée (tentative %d/%d, %s). Retry dans %.2fs.",
                    op_name,
                    attempt + 1,
                    attempts + 1,
                    type(exc).__name__,
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
        assert last_exc is not None  # pour mypy : on n'arrive ici qu'après ≥1 échec
        _LOG.error(
            "Qdrant op '%s' abandonnée après %d tentatives : %s",
            op_name,
            attempts + 1,
            type(last_exc).__name__,
        )
        raise last_exc

    # ---------- Bootstrap (par tenant) ---------------------------------------

    def ensure_tenant_collection(self, tenant_id: str) -> str:
        """Crée / valide la collection du tenant.

        En `qdrant_read_only=True`, on ne tente PAS la création (l'instance n'a
        peut-être même pas les droits) : on retourne juste le nom calculé en
        supposant que la collection a été bootstrappée hors-bande.
        """
        if self.settings.qdrant_read_only:
            return collection_name(self.settings, tenant_id)
        return self._call_with_retry(
            "ensure_collection",
            ensure_collection,
            self._client,
            self.settings,
            tenant_id,
        )

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
        """Upsert idempotent. Bat les points par lots de
        `settings.upsert_batch_size` pour éviter les limites Qdrant (timeout,
        msg size gRPC) sur gros documents. Chaque batch est retry-aware."""
        self._check_writable("upsert_chunks")
        if not chunks:
            return
        tenant = self._require_tenant(tenant_id)

        t_total0 = time.perf_counter()
        timings_ms: dict[str, float] = {}

        t0 = time.perf_counter()
        coll = self.ensure_tenant_collection(tenant)
        timings_ms["bootstrap"] = (time.perf_counter() - t0) * 1000.0

        embed_version = self.embeddings.version_tag
        embed_model = self.embeddings.model_id
        embed_dim = int(self.embeddings.dim)

        doc_ids = {c.document_id for c in chunks}
        if (
            self.settings.warn_on_embedding_version_mismatch
            or self.settings.reject_embedding_version_mismatch_upsert
        ):
            for doc_id in doc_ids:
                prior = self._existing_embedding_versions_for_document(tenant, doc_id)
                mismatched = prior - {embed_version}
                if prior and mismatched:
                    msg = (
                        f"Qdrant: le document {doc_id!r} a déjà des points en embedding_version "
                        f"{sorted(prior)!r} ; cet upsert utilise {embed_version!r}. "
                        "Supprimer le document (delete_document) puis ré-indexer, ou désactiver "
                        "ECRF_WARN_ON_EMBEDDING_VERSION_MISMATCH / ECRF_REJECT_EMBEDDING_VERSION_MISMATCH_UPSERT."
                    )
                    if self.settings.reject_embedding_version_mismatch_upsert:
                        raise ValueError(msg)
                    _LOG.warning(msg)

        texts = [passage_for_embedding_and_rerank(c) for c in chunks]

        t0 = time.perf_counter()
        dense_vecs = self.embeddings.embed_passages(texts)
        timings_ms["embed_dense"] = (time.perf_counter() - t0) * 1000.0

        if self._sparse is not None:
            t0 = time.perf_counter()
            sparse_vecs = self._sparse.encode_passages(texts)
            timings_ms["embed_sparse"] = (time.perf_counter() - t0) * 1000.0
        else:
            sparse_vecs = None

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
                embedding_model=embed_model,
                embedding_dim=embed_dim,
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

        batch_size = max(1, int(self.settings.upsert_batch_size))
        n_batches = 0
        t0 = time.perf_counter()
        for start in range(0, len(points), batch_size):
            self._call_with_retry(
                "upsert",
                self._client.upsert,
                collection_name=coll,
                points=points[start : start + batch_size],
                wait=True,
            )
            n_batches += 1
        timings_ms["qdrant_upsert"] = (time.perf_counter() - t0) * 1000.0
        timings_ms["total"] = (time.perf_counter() - t_total0) * 1000.0

        _LOG.info(
            "qdrant.upsert.done",
            extra={
                "event": "qdrant.upsert.done",
                "tenant": tenant,
                "n_chunks": len(chunks),
                "n_batches": n_batches,
                "batch_size": batch_size,
                "hybrid": self._sparse is not None,
                "embedding_version": embed_version,
                "timings_ms": timings_ms,
            },
        )

    # ---------- Delete -------------------------------------------------------

    def delete_document(self, *, tenant_id: str, document_id: str) -> int:
        """Supprime tous les points `(tenant_id, document_id)`.

        Retourne **le nombre de points supprimés** (compté via `client.count`
        AVANT la suppression). En cas d'erreur Qdrant côté delete, on relève
        l'exception ; le compteur reste informatif (l'upsert idempotent peut
        être relancé sans risque)."""
        self._check_writable("delete_document")
        tenant = self._require_tenant(tenant_id)
        coll = collection_name(self.settings, tenant)
        from qdrant_client import models as qm

        t_total0 = time.perf_counter()
        timings_ms: dict[str, float] = {}

        flt = qm.Filter(
            must=[
                qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant)),
                qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id)),
            ]
        )
        t0 = time.perf_counter()
        count_resp = self._call_with_retry(
            "count",
            self._client.count,
            collection_name=coll,
            count_filter=flt,
            exact=True,
        )
        timings_ms["count"] = (time.perf_counter() - t0) * 1000.0

        n_before = int(getattr(count_resp, "count", 0) or 0)
        if n_before == 0:
            timings_ms["total"] = (time.perf_counter() - t_total0) * 1000.0
            _LOG.info(
                "qdrant.delete.done",
                extra={
                    "event": "qdrant.delete.done",
                    "tenant": tenant,
                    "document_id": document_id,
                    "n_deleted": 0,
                    "timings_ms": timings_ms,
                },
            )
            return 0

        t0 = time.perf_counter()
        self._call_with_retry(
            "delete",
            self._client.delete,
            collection_name=coll,
            points_selector=qm.FilterSelector(filter=flt),
            wait=True,
        )
        timings_ms["delete"] = (time.perf_counter() - t0) * 1000.0
        timings_ms["total"] = (time.perf_counter() - t_total0) * 1000.0

        _LOG.info(
            "qdrant.delete.done",
            extra={
                "event": "qdrant.delete.done",
                "tenant": tenant,
                "document_id": document_id,
                "n_deleted": n_before,
                "timings_ms": timings_ms,
            },
        )
        return n_before

    # ---------- Health / admin -----------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Probe synchrone (sans retry) renvoyant un état structuré du service.

        Pas de retry : l'objectif est d'observer la *réalité actuelle*. Un
        retry masquerait un Qdrant down derrière une latence soudaine.

        Returns:
            Dict avec :
              - `qdrant_url`, `qdrant_reachable` (bool), `n_collections` (int|None),
                `error` (str si reachable=False).
              - `embedding_model`, `embedding_dim`, `embedding_version`.
              - `sparse_enabled`, `reranker_enabled`, `read_only`.
              - `collection_prefix`.

        Use cases :
          - probe au boot pour fail-fast si Qdrant injoignable,
          - endpoint Kubernetes `/readyz` / `/healthz` (API service),
          - debug rapide (`python -c "from app... import ...; print(svc.health_check())"`).
        """
        out: dict[str, Any] = {
            "qdrant_url": self.settings.qdrant_url,
            "qdrant_reachable": False,
            "n_collections": None,
            "embedding_model": self.embeddings.model_id,
            "embedding_dim": int(self.embeddings.dim),
            "embedding_version": self.embeddings.version_tag,
            "sparse_enabled": self._sparse is not None,
            "reranker_enabled": self._reranker is not None,
            "read_only": bool(self.settings.qdrant_read_only),
            "collection_prefix": self.settings.qdrant_collection_prefix,
        }
        try:
            cols = self._client.get_collections()
            out["qdrant_reachable"] = True
            out["n_collections"] = len(cols.collections)
        except Exception as exc:  # noqa: BLE001 - probe : on capture TOUT
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    def _existing_embedding_versions_for_document(self, tenant: str, document_id: str) -> set[str]:
        """Versions d'embedding déjà présentes en payload pour ce document (scroll paginé)."""
        versions: set[str] = set()
        offset: Any = None
        for _ in range(100):
            try:
                batch, offset = self.list_chunks(
                    tenant_id=tenant,
                    document_id=document_id,
                    limit=256,
                    offset=offset,
                )
            except (ValueError, TypeError):
                # Client mocké ou réponse scroll inattendue : pas de preflight (tests unitaires).
                return set()
            for ch in batch:
                v = (ch.metadata or {}).get("embedding_version")
                if isinstance(v, str) and v.strip():
                    versions.add(v.strip())
            if offset is None or not batch:
                break
        return versions

    # ---------- List / admin -------------------------------------------------

    def list_chunks(
        self,
        *,
        tenant_id: str,
        document_id: str | None = None,
        limit: int = 100,
        offset: Any = None,
    ) -> tuple[list[DocumentChunk], Any]:
        """Liste paginée des chunks d'un tenant (admin / debug / migration).

        Utilise l'API `scroll` de Qdrant : récupération séquentielle, sans
        scoring, sans rerank. Idéal pour :
          - inspecter ce qui a été indexé pour un document donné,
          - exporter / migrer les chunks d'un tenant,
          - debug d'un retrieval qui retourne du vide ("ai-je bien indexé ?").

        Args:
            tenant_id: tenant cible (requis, valide selon `qdrant_strict_tenant_id`).
            document_id: si fourni, ne retourne que les chunks du document.
            limit: nombre max de chunks par page (1..n, Qdrant n'a pas de cap dur
                mais préférer ≤ 1000 pour éviter les gros payloads).
            offset: `next_offset` retourné par l'appel précédent pour pagination
                (None = première page).

        Returns:
            `(chunks, next_offset)`. `next_offset is None` ⇒ plus de page.

        Note: opération de lecture → autorisée en `qdrant_read_only=True`.
        """
        tenant = self._require_tenant(tenant_id)
        coll = collection_name(self.settings, tenant)
        from qdrant_client import models as qm

        t_total0 = time.perf_counter()

        must: list[Any] = [qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant))]
        if document_id:
            must.append(
                qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))
            )
        flt = qm.Filter(must=must)

        records, next_offset = self._call_with_retry(
            "scroll",
            self._client.scroll,
            collection_name=coll,
            scroll_filter=flt,
            limit=max(1, int(limit)),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        chunks: list[DocumentChunk] = []
        for rec in records:
            payload = dict(rec.payload or {})
            chunks.append(_payload_to_chunk(payload))

        _LOG.info(
            "qdrant.list_chunks.done",
            extra={
                "event": "qdrant.list_chunks.done",
                "tenant": tenant,
                "document_id": document_id,
                "limit": limit,
                "n_returned": len(chunks),
                "has_more": next_offset is not None,
                "duration_ms": (time.perf_counter() - t_total0) * 1000.0,
            },
        )
        return chunks, next_offset

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
        """Recherche hybride dense + sparse + rerank optionnel.

        Le `float` du tuple retourné est le **score final** : score Qdrant brut
        si pas de rerank, score reranker sinon (rétrocompat). Pour ne pas perdre
        l'info, chaque `chunk.metadata` est enrichi (cf. C1) avec :

          - `retrieval_score` / `retrieval_rank` : toujours présents, valeur du
            score Qdrant brut (RRF si hybride) et son rang dans la réponse.
          - `rerank_score` / `rerank_rank` : présents uniquement si le reranker
            a tourné, valeur du score CrossEncoder et son rang final.

        Permet la calibration / debugging du rerank en prod (mesurer le shift
        de classement, identifier les cas où Qdrant et le rerank divergent)."""
        if not query_text or not query_text.strip():
            return []
        tenant = self._require_tenant(tenant_id)
        coll = collection_name(self.settings, tenant)

        from qdrant_client import models as qm

        # `embedding_version` injecté quand on veut garantir que seuls les
        # chunks générés par le modèle courant sont matchés. Désactivable via
        # `Settings.enforce_embedding_version_filter` (migrations multi-versions).
        version_filter = (
            self.embeddings.version_tag if self.settings.enforce_embedding_version_filter else None
        )
        flt = self._build_filter(
            tenant_id=tenant,
            patient_id=patient_id,
            document_id=document_id,
            document_type=document_type,
            field_family=field_family,
            embedding_version=version_filter,
        )

        candidate_k = max(top_k, top_k * self.settings.rerank_candidate_multiplier)

        # ─── C2: timings par phase ──────────────────────────────────────────
        t_total0 = time.perf_counter()
        timings_ms: dict[str, float] = {}

        t0 = time.perf_counter()
        dense_query = self.embeddings.embed_query(query_text)
        timings_ms["embed_dense"] = (time.perf_counter() - t0) * 1000.0

        if self._sparse is not None:
            t0 = time.perf_counter()
            sparse_q = self._sparse.encode_query(query_text)
            timings_ms["embed_sparse"] = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            qd_resp = self._call_with_retry(
                "query_points",
                self._client.query_points,
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
            timings_ms["qdrant_query"] = (time.perf_counter() - t0) * 1000.0
        else:
            t0 = time.perf_counter()
            qd_resp = self._call_with_retry(
                "query_points",
                self._client.query_points,
                collection_name=coll,
                query=dense_query,
                using=_VEC_DENSE,
                limit=candidate_k,
                query_filter=flt,
                with_payload=True,
            )
            timings_ms["qdrant_query"] = (time.perf_counter() - t0) * 1000.0

        scored_points = qd_resp.points

        # ─── C1: attacher le score Qdrant + rang dans le metadata ───────────
        results: list[tuple[DocumentChunk, float]] = []
        for rank, sp in enumerate(scored_points, start=1):
            payload = dict(sp.payload or {})
            chunk = _payload_to_chunk(payload)
            score = float(sp.score)
            chunk.metadata["retrieval_score"] = score
            chunk.metadata["retrieval_rank"] = rank
            results.append((chunk, score))

        n_candidates = len(results)
        if min_score is not None:
            results = [(c, s) for c, s in results if s >= min_score]
        n_above_threshold = len(results)

        if self._reranker is not None and results:
            t0 = time.perf_counter()
            reranked = self._reranker.rerank(
                query=query_text,
                candidates=results,
                top_n=top_k,
            )
            timings_ms["rerank"] = (time.perf_counter() - t0) * 1000.0
            # C1: attacher le score reranker + rang final SANS perdre retrieval_score
            # (qui reste accessible via chunk.metadata).
            for rank, (chunk, rscore) in enumerate(reranked, start=1):
                chunk.metadata["rerank_score"] = float(rscore)
                chunk.metadata["rerank_rank"] = rank
            final = reranked
        else:
            final = results[:top_k]

        timings_ms["total"] = (time.perf_counter() - t_total0) * 1000.0

        # ─── C2: event structuré (ingestible JSON via formatter dédié) ──────
        _LOG.info(
            "qdrant.search.done",
            extra={
                "event": "qdrant.search.done",
                "tenant": tenant,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "n_candidates": n_candidates,
                "n_above_threshold": n_above_threshold,
                "n_returned": len(final),
                "rerank_used": self._reranker is not None and bool(results),
                "hybrid": self._sparse is not None,
                "version_filter": version_filter,
                "timings_ms": timings_ms,
            },
        )

        return final

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
        embedding_version: str | None = None,
    ):
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
            # Chunks issus de `SectionBasedChunkingService` (CR imagerie, lettres)
            # ont souvent `field_family=None` au payload. Aligner sur le backend
            # mémoire : accepter les points non étiquetés OU la famille demandée.
            must.append(
                qm.Filter(
                    should=[
                        qm.FieldCondition(
                            key="field_family",
                            match=qm.MatchValue(value=field_family.value),
                        ),
                        qm.IsNullCondition(is_null=qm.PayloadField(key="field_family")),
                    ]
                )
            )
        if embedding_version:
            must.append(
                qm.FieldCondition(
                    key="embedding_version",
                    match=qm.MatchValue(value=embedding_version),
                )
            )
        return qm.Filter(must=must)
