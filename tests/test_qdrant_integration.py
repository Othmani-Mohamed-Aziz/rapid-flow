"""
Test d’intégration Qdrant + embeddings locaux (skip si serveur indisponible).

Pré-requis :
- Qdrant joignable sur `ECRF_QDRANT_URL` (par défaut http://localhost:6333).
- Extra `vector` installé : `pip install -e ".[vector]"`.
- Modèle e5 téléchargeable (Internet) ou déjà en cache HF.

Activation : variable `ECRF_RUN_QDRANT_INTEGRATION=1`.
"""

from __future__ import annotations

import os
import platform
import sys

import pytest

pytest.importorskip("qdrant_client")
pytest.importorskip("sentence_transformers")

from app.config.settings import Settings  # noqa: E402
from app.indexing.qdrant_bootstrap import CollectionConfigMismatchError  # noqa: E402
from app.indexing.qdrant_vector_service import QdrantHybridVectorService  # noqa: E402
from app.schemas.models import DocumentChunk  # noqa: E402

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("ECRF_RUN_QDRANT_INTEGRATION", "0") != "1",
        reason="Activer avec ECRF_RUN_QDRANT_INTEGRATION=1 (et un Qdrant joignable).",
    ),
    pytest.mark.real_vector_backend,
]


def _server_reachable(url: str) -> bool:
    try:
        from qdrant_client import QdrantClient

        c = QdrantClient(url=url, timeout=2.0)
        c.get_collections()
        return True
    except Exception:
        return False


def _fastembed_safe_on_this_runtime() -> bool:
    """fastembed/ONNX runtime crashe (access violation) sur Python 3.14 + Windows.

    On désactive automatiquement la branche BM25 sparse dans ce cas pour ne pas
    tuer le processus pytest. Pour valider le chemin hybride complet, utiliser
    Python 3.12 ou 3.13."""
    if sys.version_info >= (3, 14) and platform.system() == "Windows":
        return False
    return True


def test_qdrant_end_to_end_hybrid(tmp_path) -> None:
    use_sparse = _fastembed_safe_on_this_runtime()
    # Reranker activé pour valider C1 (retrieval_score + rerank_score préservés)
    # en E2E sur la voie hybride complète.
    settings = Settings(
        vector_backend="qdrant",
        qdrant_collection_prefix="ecrf_chunks_itest",
        enable_sparse_bm25=use_sparse,
        enable_reranker=True,
    )
    if not use_sparse:
        pytest.skip(
            "fastembed/ONNX instable sur Python 3.14 + Windows ; "
            "lancer ce test sous Python 3.12/3.13 pour valider la voie hybride. "
            "Une variante dense-only existe : test_qdrant_end_to_end_dense_only."
        )
    if not _server_reachable(settings.qdrant_url):
        pytest.skip(f"Qdrant injoignable sur {settings.qdrant_url}")

    svc = QdrantHybridVectorService(settings)
    tenant = "ITEST-STUDY"
    chunks = [
        DocumentChunk(
            chunk_id="itest-c1",
            document_id="itest-d1",
            text="Lésion hépatique du segment VI, suivi à 6 mois.",
            metadata={"section_heading": "Conclusion"},
        ),
        DocumentChunk(
            chunk_id="itest-c2",
            document_id="itest-d1",
            text="Pas de collection. Reins normaux. Antécédents notables.",
            metadata={"section_heading": "Résultats"},
        ),
    ]
    svc.upsert_chunks(chunks, tenant_id=tenant, patient_id="P1", study_id=tenant)

    hits = svc.search(
        query_text="lésion du foie",
        tenant_id=tenant,
        patient_id="P1",
        top_k=2,
    )
    assert hits
    assert hits[0][0].text.startswith("Lésion hépatique")

    # C1 E2E : retrieval_score (Qdrant) ET rerank_score (CrossEncoder) doivent
    # tous les deux être dans le metadata après rerank. Sans C1, on perdrait le
    # retrieval_score (écrasé par le tuple final).
    for chunk, final_score in hits:
        assert "retrieval_score" in chunk.metadata
        assert "retrieval_rank" in chunk.metadata
        assert "rerank_score" in chunk.metadata
        assert "rerank_rank" in chunk.metadata
        # Le tuple float = rerank_score (la signature publique retourne le
        # score "final"). Le retrieval_score reste accessible via metadata.
        assert float(chunk.metadata["rerank_score"]) == pytest.approx(final_score)

    deleted = svc.delete_document(tenant_id=tenant, document_id="itest-d1")
    assert deleted == 2


def test_qdrant_end_to_end_dense_only(tmp_path) -> None:
    """Variante dense-only : valide upsert + search + delete sans fastembed.

    Utile pour Python 3.14 + Windows où fastembed/ONNX crashe à l'init."""
    settings = Settings(
        vector_backend="qdrant",
        qdrant_collection_prefix="ecrf_chunks_itest_dense",
        enable_sparse_bm25=False,
        enable_reranker=False,
    )
    if not _server_reachable(settings.qdrant_url):
        pytest.skip(f"Qdrant injoignable sur {settings.qdrant_url}")

    svc = QdrantHybridVectorService(settings)
    tenant = "ITEST-DENSE"
    chunks = [
        DocumentChunk(
            chunk_id="itest-c1",
            document_id="itest-d1",
            text="Lésion hépatique du segment VI, suivi à 6 mois.",
            metadata={"section_heading": "Conclusion"},
        ),
        DocumentChunk(
            chunk_id="itest-c2",
            document_id="itest-d1",
            text="Pas de collection. Reins normaux. Antécédents notables.",
            metadata={"section_heading": "Résultats"},
        ),
    ]
    svc.upsert_chunks(chunks, tenant_id=tenant, patient_id="P1", study_id=tenant)

    hits = svc.search(
        query_text="lésion du foie",
        tenant_id=tenant,
        patient_id="P1",
        top_k=2,
    )
    assert hits
    assert hits[0][0].text.startswith("Lésion hépatique")

    # C1 E2E (dense-only, sans rerank) : retrieval_score doit être dans
    # metadata, rerank_score doit être absent.
    for chunk, final_score in hits:
        assert "retrieval_score" in chunk.metadata
        assert "retrieval_rank" in chunk.metadata
        assert "rerank_score" not in chunk.metadata
        assert float(chunk.metadata["retrieval_score"]) == pytest.approx(final_score)

    # A6 : delete_document retourne le nombre exact de points supprimés.
    deleted = svc.delete_document(tenant_id=tenant, document_id="itest-d1")
    assert deleted == 2

    # Idempotence : re-delete sur même doc → 0 (rien à supprimer).
    deleted_again = svc.delete_document(tenant_id=tenant, document_id="itest-d1")
    assert deleted_again == 0


# ─────────────────────────────────────────────────────────────────────────────
# E2E ciblé Catégorie A — isolation tenant + filtre embedding_version
# ─────────────────────────────────────────────────────────────────────────────


def test_qdrant_e2e_multi_tenant_isolation() -> None:
    """Upsert tenant A, search tenant B → 0 résultat. Garantie d'isolation
    structurelle (Option C : 1 collection par tenant)."""
    base = Settings(
        vector_backend="qdrant",
        qdrant_collection_prefix="ecrf_chunks_itest_iso",
        enable_sparse_bm25=False,
        enable_reranker=False,
    )
    if not _server_reachable(base.qdrant_url):
        pytest.skip(f"Qdrant injoignable sur {base.qdrant_url}")

    svc = QdrantHybridVectorService(base)
    tenant_a = "ITEST-ISO-A"
    tenant_b = "ITEST-ISO-B"

    chunks_a = [
        DocumentChunk(
            chunk_id="iso-a-c1",
            document_id="iso-a-d1",
            text="Donnée confidentielle étude A : lésion hépatique.",
            metadata={"section_heading": "Confidential"},
        ),
    ]
    svc.upsert_chunks(chunks_a, tenant_id=tenant_a, patient_id="PA", study_id=tenant_a)

    # B n'a rien upserté → la collection B n'existe pas. La recherche doit donc
    # 1) ne pas trouver de hits ou 2) lever une erreur "collection introuvable".
    # Le second cas est plus protecteur ; on l'accepte aussi.
    try:
        hits_b = svc.search(query_text="lésion hépatique", tenant_id=tenant_b, top_k=5)
    except Exception:
        hits_b = []
    assert not hits_b, "Fuite cross-tenant détectée"

    # Le tenant A, lui, voit bien sa donnée.
    hits_a = svc.search(query_text="lésion hépatique", tenant_id=tenant_a, top_k=5)
    assert hits_a, "Tenant A devrait voir ses propres données"

    # Cleanup
    svc.delete_document(tenant_id=tenant_a, document_id="iso-a-d1")


def test_qdrant_e2e_embedding_version_filter_excludes_old_chunks() -> None:
    """Scénario migration : on upsert avec version 'v1', puis on search avec
    une nouvelle version 'v2' → les anciens chunks doivent être exclus
    (filtre `embedding_version` auto-injecté).
    Avec opt-out (`enforce_embedding_version_filter=False`), ils redeviennent
    visibles."""
    settings_v1 = Settings(
        vector_backend="qdrant",
        qdrant_collection_prefix="ecrf_chunks_itest_ver",
        enable_sparse_bm25=False,
        enable_reranker=False,
        embedding_version="v1",
    )
    if not _server_reachable(settings_v1.qdrant_url):
        pytest.skip(f"Qdrant injoignable sur {settings_v1.qdrant_url}")

    tenant = "ITEST-VER"
    # 1) Upsert avec version v1 (le service écrit `embedding_version=v1` en payload
    #    car la version vient de embeddings.version_tag qui lit settings.embedding_version).
    svc_v1 = QdrantHybridVectorService(settings_v1)
    svc_v1.upsert_chunks(
        [
            DocumentChunk(
                chunk_id="ver-c1",
                document_id="ver-d1",
                text="Lésion hépatique du segment VI, suivi à 6 mois.",
                metadata={"section_heading": "Conclusion"},
            )
        ],
        tenant_id=tenant,
        patient_id="P1",
        study_id=tenant,
    )

    # 2) Search avec settings v2 + filtre actif → 0 résultat (les anciens chunks
    #    ont `embedding_version=v1` et sont écartés du filtre).
    settings_v2 = settings_v1.model_copy(update={"embedding_version": "v2"})
    svc_v2 = QdrantHybridVectorService(settings_v2)
    hits_filtered = svc_v2.search(query_text="lésion du foie", tenant_id=tenant, top_k=5)
    assert not hits_filtered, "v2 ne devrait pas voir les chunks v1 avec filtre actif"

    # 3) Désactivation du filtre → on retrouve l'ancien chunk.
    settings_v2_no_filter = settings_v2.model_copy(
        update={"enforce_embedding_version_filter": False}
    )
    svc_v2_lax = QdrantHybridVectorService(settings_v2_no_filter)
    hits_unfiltered = svc_v2_lax.search(query_text="lésion du foie", tenant_id=tenant, top_k=5)
    assert hits_unfiltered, "Sans filtre, v2 doit retrouver l'ancien chunk v1"

    # Cleanup
    svc_v1.delete_document(tenant_id=tenant, document_id="ver-d1")


def test_qdrant_e2e_list_chunks_paginates_and_filters() -> None:
    """D1: upsert N=5 chunks puis liste page par page (limit=3) avec filter
    document_id. Vérifie pagination + filtre."""
    settings = Settings(
        vector_backend="qdrant",
        qdrant_collection_prefix="ecrf_chunks_itest_list",
        enable_sparse_bm25=False,
        enable_reranker=False,
    )
    if not _server_reachable(settings.qdrant_url):
        pytest.skip(f"Qdrant injoignable sur {settings.qdrant_url}")

    svc = QdrantHybridVectorService(settings)
    tenant = "ITEST-LIST"

    # 5 chunks répartis sur 2 documents.
    chunks = [
        DocumentChunk(
            chunk_id=f"list-c{i}",
            document_id="list-d1" if i < 3 else "list-d2",
            text=f"Fragment {i}.",
            metadata={"section_heading": f"Section {i}"},
        )
        for i in range(5)
    ]
    svc.upsert_chunks(chunks, tenant_id=tenant, patient_id="P1", study_id=tenant)

    # 1) Filtre par document_id : seuls les 3 chunks de list-d1.
    found_d1: list[str] = []
    offset = None
    page = 0
    while True:
        page += 1
        batch, offset = svc.list_chunks(
            tenant_id=tenant, document_id="list-d1", limit=2, offset=offset
        )
        found_d1.extend(c.chunk_id for c in batch)
        if offset is None or page > 10:  # safety
            break
    assert sorted(found_d1) == ["list-c0", "list-c1", "list-c2"]

    # 2) Sans filtre document_id : 5 chunks au total, paginés (limit=3 → 3 + 2).
    page1, offset = svc.list_chunks(tenant_id=tenant, limit=3)
    assert len(page1) == 3
    assert offset is not None
    page2, offset2 = svc.list_chunks(tenant_id=tenant, limit=3, offset=offset)
    assert len(page2) == 2
    assert offset2 is None

    # Cleanup
    svc.delete_document(tenant_id=tenant, document_id="list-d1")
    svc.delete_document(tenant_id=tenant, document_id="list-d2")


def test_qdrant_e2e_read_only_blocks_writes() -> None:
    """B2: une instance configurée `qdrant_read_only=True` doit refuser les
    writes (upsert/delete) ET laisser passer les searches (sur une collection
    déjà bootstrappée par une autre instance writable)."""
    base = Settings(
        vector_backend="qdrant",
        qdrant_collection_prefix="ecrf_chunks_itest_ro",
        enable_sparse_bm25=False,
        enable_reranker=False,
    )
    if not _server_reachable(base.qdrant_url):
        pytest.skip(f"Qdrant injoignable sur {base.qdrant_url}")

    tenant = "ITEST-RO"

    # 1) Bootstrap + seed avec un service writable.
    writer = QdrantHybridVectorService(base)
    writer.upsert_chunks(
        [
            DocumentChunk(
                chunk_id="ro-c1",
                document_id="ro-d1",
                text="Test read-only.",
                metadata={"section_heading": "RO"},
            )
        ],
        tenant_id=tenant,
        patient_id="P1",
        study_id=tenant,
    )

    # 2) Instance read-only : search OK, upsert/delete KO.
    ro_settings = base.model_copy(update={"qdrant_read_only": True})
    reader = QdrantHybridVectorService(ro_settings)

    # Search fonctionne en read-only.
    hits = reader.search(query_text="read-only", tenant_id=tenant, top_k=5)
    assert hits, "Le reader read-only doit pouvoir lire"

    # Upsert refusé.
    with pytest.raises(RuntimeError):
        reader.upsert_chunks(
            [DocumentChunk(chunk_id="ro-c2", document_id="ro-d1", text="x", metadata={})],
            tenant_id=tenant,
            patient_id="P1",
            study_id=tenant,
        )

    # Delete refusé.
    with pytest.raises(RuntimeError):
        reader.delete_document(tenant_id=tenant, document_id="ro-d1")

    # Cleanup via le writer.
    writer.delete_document(tenant_id=tenant, document_id="ro-d1")


def test_qdrant_e2e_dim_mismatch_raises_explicit_error() -> None:
    """A1 E2E : si une collection existe avec une dim/distance différente de
    Settings, `ensure_collection` lève `CollectionConfigMismatchError` AVANT
    tout upsert (au lieu de laisser Qdrant crasher avec un message opaque)."""
    settings = Settings(
        vector_backend="qdrant",
        qdrant_collection_prefix="ecrf_chunks_itest_mismatch",
        enable_sparse_bm25=False,
        enable_reranker=False,
    )
    if not _server_reachable(settings.qdrant_url):
        pytest.skip(f"Qdrant injoignable sur {settings.qdrant_url}")

    from qdrant_client import QdrantClient
    from qdrant_client import models as qm

    raw_client = QdrantClient(url=settings.qdrant_url, timeout=10)
    # On force le nom calculé par collection_name pour matcher.
    from app.indexing.qdrant_bootstrap import collection_name

    coll = collection_name(settings, "MISMATCH")

    # Cleanup d'une éventuelle collection précédente.
    try:
        raw_client.delete_collection(coll)
    except Exception:
        pass

    # 1) Création directe d'une collection avec dim=4 (≠ embedding_dim attendue
    #    par le service, par défaut 768 pour intfloat/multilingual-e5-base).
    raw_client.create_collection(
        collection_name=coll,
        vectors_config={"text": qm.VectorParams(size=4, distance=qm.Distance.COSINE)},
    )

    try:
        # 2) Le service doit détecter le mismatch dès ensure_collection.
        svc = QdrantHybridVectorService(settings)
        chunks = [
            DocumentChunk(chunk_id="mm-c1", document_id="mm-d1", text="x", metadata={}),
        ]
        with pytest.raises(CollectionConfigMismatchError) as exc:
            svc.upsert_chunks(chunks, tenant_id="MISMATCH", patient_id="P1", study_id="MISMATCH")
        msg = str(exc.value)
        assert "dimension=4" in msg
        assert f"attendu={settings.embedding_dim}" in msg
    finally:
        try:
            raw_client.delete_collection(coll)
        except Exception:
            pass


def test_qdrant_e2e_health_check_reachable() -> None:
    """F2 E2E : health_check sur un Qdrant up doit retourner reachable=True
    avec un compteur de collections (>= 0)."""
    settings = Settings(
        vector_backend="qdrant",
        enable_sparse_bm25=False,
        enable_reranker=False,
    )
    if not _server_reachable(settings.qdrant_url):
        pytest.skip(f"Qdrant injoignable sur {settings.qdrant_url}")

    svc = QdrantHybridVectorService(settings)
    h = svc.health_check()
    assert h["qdrant_reachable"] is True
    assert isinstance(h["n_collections"], int) and h["n_collections"] >= 0
    assert "error" not in h
    assert h["embedding_dim"] == settings.embedding_dim
