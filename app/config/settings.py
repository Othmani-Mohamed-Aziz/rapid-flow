from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _PROJECT_ROOT / ".env"
if _ENV_FILE.is_file():
    load_dotenv(_ENV_FILE)


class Settings(BaseSettings):
    """Paramètres runtime (extensions RAGFlow, Qdrant, modèles)."""

    model_config = SettingsConfigDict(env_prefix="ECRF_", extra="ignore")

    pipeline_version: str = "0.1.0"
    default_confidence_floor: float = 0.5
    mock_llama_model_name: str = "mock-llama-lab-v1"
    mock_langextract_version: str = "langextract-mock-0.1"

    #: `auto` = Docling si installé, sinon pypdf ; `docling` = Docling obligatoire ;
    #: `pypdf` = moteur texte léger uniquement.
    pdf_parser_backend: Literal["auto", "docling", "pypdf"] = "auto"
    #: Post-traitement bilan sanguin (lignes structurées) après PDF labo détecté.
    lab_postprocess_pdf_lab_reports: bool = True
    #: Taille max. d’un fragment de section avant sous-découpe (`SectionBasedChunkingService`).
    chunk_max_section_chars: int = 12000

    # ─── Indexation vectorielle ────────────────────────────────────────────────
    #: Backend de l’index : `memory` (dev), `qdrant` (prod on-prem).
    vector_backend: Literal["memory", "qdrant"] = "memory"
    #: Préfixe collection Qdrant ; nom final = `{prefix}__{tenant_id}`.
    qdrant_collection_prefix: str = "ecrf_chunks"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_prefer_grpc: bool = False
    qdrant_timeout_s: float = 30.0
    qdrant_hnsw_m: int = 16
    qdrant_hnsw_ef_construct: int = 128

    #: Embeddings denses locaux (Hugging Face).
    embedding_provider: Literal["hf_local"] = "hf_local"
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768
    embedding_batch_size: int = 32
    #: Préfixes E5 / BGE (vide pour modèles non préfixés).
    embedding_passage_prefix: str = "passage: "
    embedding_query_prefix: str = "query: "
    #: Tag de version stocké dans payload Qdrant pour migrations futures.
    embedding_version: str = "e5-base-v1"

    #: Sparse BM25 (fastembed) — active la branche hybride côté upsert + search.
    enable_sparse_bm25: bool = True
    #: Modèle BM25 fastembed (cf. `ECRF_SPARSE_MODEL`).
    sparse_model: str = "Qdrant/bm25"

    #: Reranker CrossEncoder (BGE) appliqué après top‑k Qdrant.
    enable_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_n: int = 5
    rerank_candidate_multiplier: int = 4  # top_k_final * multiplier candidats Qdrant
    reranker_device: Literal["cpu", "cuda", "auto"] = "auto"
