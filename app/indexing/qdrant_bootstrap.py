"""
Bootstrap idempotent d’une collection Qdrant par tenant (Option C).

- Nom de collection : `{settings.qdrant_collection_prefix}__{tenant_id_normalisé}`.
- Vecteurs : dense (`text`) en `Cosine` + sparse (`text_bm25`) si BM25 activé.
- Payload indexes systématiques pour les filtres applicatifs (perf + sécurité).
"""

from __future__ import annotations

import re

from app.config.settings import Settings

_TENANT_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def normalize_tenant_id(tenant_id: str) -> str:
    if not tenant_id or not tenant_id.strip():
        raise ValueError("tenant_id requis (non vide).")
    out = _TENANT_RE.sub("_", tenant_id.strip())
    return out[:60] or "default"


def collection_name(settings: Settings, tenant_id: str) -> str:
    return f"{settings.qdrant_collection_prefix}__{normalize_tenant_id(tenant_id)}"


def ensure_collection(client, settings: Settings, tenant_id: str) -> str:
    """
    Crée la collection si absente. Idempotent (no-op si déjà conforme).

    Le `client` est un `qdrant_client.QdrantClient` (passé pour faciliter les tests).
    """
    from qdrant_client import models as qm

    name = collection_name(settings, tenant_id)
    existing = {c.name for c in client.get_collections().collections}

    if name not in existing:
        vectors_config = {
            "text": qm.VectorParams(size=settings.embedding_dim, distance=qm.Distance.COSINE),
        }
        sparse_vectors_config = None
        if settings.enable_sparse_bm25:
            sparse_vectors_config = {
                "text_bm25": qm.SparseVectorParams(
                    index=qm.SparseIndexParams(on_disk=False),
                ),
            }
        client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
            hnsw_config=qm.HnswConfigDiff(
                m=settings.qdrant_hnsw_m,
                ef_construct=settings.qdrant_hnsw_ef_construct,
            ),
        )

    _ensure_payload_indexes(client, name)
    return name


def _ensure_payload_indexes(client, collection_name: str) -> None:
    """Indexes Qdrant pour les filtres applicatifs (idempotent : on ignore les erreurs « déjà créé »)."""
    from qdrant_client import models as qm

    keyword_fields = (
        "tenant_id",
        "patient_id",
        "study_id",
        "document_id",
        "document_type",
        "field_family",
        "embedding_version",
        "content_kind",
    )
    integer_fields = ("section_index", "part_index")
    bool_fields = ("char_spans_verified",)

    for f in keyword_fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=f,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception:  # pragma: no cover - déjà existant
            pass
    for f in integer_fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=f,
                field_schema=qm.PayloadSchemaType.INTEGER,
            )
        except Exception:  # pragma: no cover
            pass
    for f in bool_fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=f,
                field_schema=qm.PayloadSchemaType.BOOL,
            )
        except Exception:  # pragma: no cover
            pass
