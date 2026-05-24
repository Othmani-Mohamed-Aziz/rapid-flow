"""
Bootstrap idempotent d’une collection Qdrant par tenant (Option C).

- Nom de collection : `{settings.qdrant_collection_prefix}__{tenant_id_normalisé}`.
- Vecteurs : dense (`text`) en `Cosine` + sparse (`text_bm25`) si BM25 activé.
- Payload indexes systématiques pour les filtres applicatifs (perf + sécurité).
- Garde-fou : si la collection existe mais avec une dim / distance différentes
  de la config courante, on lève une erreur explicite (cf. `CollectionConfigMismatchError`)
  plutôt que de laisser Qdrant échouer en silence à l'upsert.
"""

from __future__ import annotations

import re
from typing import Any

from app.config.settings import Settings

_TENANT_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_STRICT_TENANT_RE = re.compile(r"^[A-Za-z0-9_-]{1,60}$")


class CollectionConfigMismatchError(RuntimeError):
    """La collection Qdrant existe mais sa config diverge de Settings.

    Cas typique : on a bumpé `ECRF_EMBEDDING_MODEL` (donc `embedding_dim`) sans
    recréer la collection. On préfère une erreur claire à un upsert qui crash
    avec un message Qdrant opaque (`Wrong input: Vector dimension error: ...`).
    """


def normalize_tenant_id(tenant_id: str) -> str:
    """Normalise un tenant_id : strip + substitution permissive des caractères
    non `[A-Za-z0-9_-]` par `_`, troncature à 60.

    Comportement historique, **permissif** : `"STUDY 1/2"` → `"STUDY_1_2"`.
    Risque de collision silencieuse → préférer la validation stricte en prod
    via `validate_strict_tenant_id` (cf. `Settings.qdrant_strict_tenant_id`)."""
    if not tenant_id or not tenant_id.strip():
        raise ValueError("tenant_id requis (non vide).")
    out = _TENANT_RE.sub("_", tenant_id.strip())
    return out[:60] or "default"


def validate_strict_tenant_id(tenant_id: str) -> str:
    """Valide qu'un `tenant_id` respecte le format strict `[A-Za-z0-9_-]{1,60}`
    **sans normalisation**.

    Lève `ValueError` avec un message explicite (caractères incriminés, regex
    attendue) sinon. Préférer ce mode strict en prod pour éviter qu'une faute
    de frappe ne crée silencieusement un nouveau tenant.
    """
    if not tenant_id:
        raise ValueError("tenant_id requis (non vide).")
    if not _STRICT_TENANT_RE.match(tenant_id):
        invalid = sorted({c for c in tenant_id if not re.match(r"[A-Za-z0-9_-]", c)})
        details = f" Caractères invalides : {invalid}." if invalid else " (vide ou >60 caractères)."
        raise ValueError(
            f"tenant_id invalide '{tenant_id}' : doit matcher [A-Za-z0-9_-]{{1,60}}."
            f"{details} Désactiver via ECRF_QDRANT_STRICT_TENANT_ID=false pour "
            "réactiver la normalisation permissive (rétrocompat uniquement)."
        )
    return tenant_id


def collection_name(settings: Settings, tenant_id: str) -> str:
    """Nom de collection Qdrant pour un tenant.

    Le mode strict (`Settings.qdrant_strict_tenant_id=True`) court-circuite la
    normalisation et lève `ValueError` si le tenant_id ne matche pas le format
    canonique. Sinon (rétrocompat), on délègue à `normalize_tenant_id` qui
    substitue les caractères invalides."""
    if settings.qdrant_strict_tenant_id:
        normalized = validate_strict_tenant_id(tenant_id)
    else:
        normalized = normalize_tenant_id(tenant_id)
    return f"{settings.qdrant_collection_prefix}__{normalized}"


def _extract_dense_vector_config(info: Any) -> tuple[int, str] | None:
    """Extrait (size, distance) du vecteur dense `text` d'un `CollectionInfo`.

    Le shape varie selon la version du client Qdrant : on tente la voie
    `config.params.vectors` (dict ou `VectorParams` direct), avec fallback
    silencieux si le format change (on ne veut pas casser un boot juste
    parce que la lib client a évolué)."""
    try:
        params = info.config.params.vectors
    except AttributeError:
        return None
    # Cas multivecteurs : params est un dict {name: VectorParams}.
    if isinstance(params, dict):
        vec = params.get("text")
        if vec is None:
            return None
    else:
        # Cas mono-vecteur : params est un VectorParams direct (collection
        # créée sans nom). Non utilisé chez nous mais on garde la lecture.
        vec = params
    size = int(getattr(vec, "size", 0) or 0)
    dist = getattr(vec, "distance", None)
    dist_str = str(getattr(dist, "value", dist) or "").lower()
    return size, dist_str


def _validate_existing_collection(client, name: str, settings: Settings) -> None:
    """Vérifie qu'une collection existante est compatible avec `settings`.

    Lève `CollectionConfigMismatchError` si la dimension ou la distance
    diffère. Tolérant aux variations cosmétiques (`Cosine`/`cosine`)."""
    try:
        info = client.get_collection(collection_name=name)
    except Exception:  # pragma: no cover - dépend du transport client
        # Si on n'arrive pas à lire la config (vieux client, erreur réseau),
        # on laisse passer : le check est best-effort, pas une porte sécu.
        return
    parsed = _extract_dense_vector_config(info)
    if parsed is None:
        return
    actual_size, actual_dist = parsed
    expected_size = int(settings.embedding_dim)
    if actual_size and actual_size != expected_size:
        raise CollectionConfigMismatchError(
            f"Collection Qdrant '{name}' : dimension={actual_size}, attendu={expected_size}. "
            f"Recréer la collection (curl -X DELETE .../collections/{name}) ou remettre "
            "`ECRF_EMBEDDING_MODEL` / `ECRF_EMBEDDING_DIM` à la valeur d'origine."
        )
    if actual_dist and actual_dist != "cosine":
        raise CollectionConfigMismatchError(
            f"Collection Qdrant '{name}' : distance='{actual_dist}', attendu='cosine'. "
            "Le code suppose Cosine pour les embeddings normalisés (e5/bge). "
            f"Recréer la collection (curl -X DELETE .../collections/{name})."
        )


def ensure_collection(client, settings: Settings, tenant_id: str) -> str:
    """
    Crée la collection si absente. Idempotent (no-op si déjà conforme).

    Si la collection existe déjà, on valide que sa dim / distance correspond
    à `Settings` ; sinon `CollectionConfigMismatchError` est levée pour éviter
    une corruption silencieuse à l'upsert (mismatch dense vector).

    Le `client` est un `qdrant_client.QdrantClient` (passé pour faciliter les tests).
    """
    from qdrant_client import models as qm

    name = collection_name(settings, tenant_id)
    existing = {c.name for c in client.get_collections().collections}

    if name in existing:
        _validate_existing_collection(client, name, settings)
    else:
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
