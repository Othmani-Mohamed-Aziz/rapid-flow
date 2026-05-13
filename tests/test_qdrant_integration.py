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
    settings = Settings(
        vector_backend="qdrant",
        qdrant_collection_prefix="ecrf_chunks_itest",
        enable_sparse_bm25=use_sparse,
        enable_reranker=False,
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

    deleted = svc.delete_document(tenant_id=tenant, document_id="itest-d1")
    assert deleted >= 0


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

    deleted = svc.delete_document(tenant_id=tenant, document_id="itest-d1")
    assert deleted >= 0
