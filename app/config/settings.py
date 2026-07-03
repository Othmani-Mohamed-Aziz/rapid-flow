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

    # ─── Extraction imagerie (LangExtract + Ollama) ───────────────────────────
    #: Active l'appel LangExtract/Ollama sur les chunks imagerie (sinon aucune obs. LLM).
    langextract_enabled: bool = True
    #: Modèle Ollama (`ollama pull <model>`). Ex. gemma2:2b, llama3.1:8b, mistral.
    ollama_model_id: str = "gemma2:2b"
    ollama_url: str = "http://localhost:11434"
    ollama_timeout_s: int = 120
    langextract_schema_version: str = "imaging-recist-langextract-v1"
    #: Nombre max de chunks imagerie traités **en parallèle** par LangExtract/Ollama
    #: (réduit la latence quand plusieurs hits RAG). `1` = séquentiel.
    imaging_extraction_max_workers: int = 8
    #: Chemin JSON du schéma d'étude. Vide = `data/study_schema_default.json` ou dérivé du code exemple.
    study_schema_path: str | None = None

    # ─── Export eCRF XLS (gabarit MA) ─────────────────────────────────────────
    #: Gabarit XLSX MA (colonnes eCRF). Défaut : `data/MA_Base_example.xlsx`.
    export_xlsx_template_path: str | None = None
    #: Active l'export XLSX en plus du JSON / CSV mock.
    export_xls_enabled: bool = True
    export_xls_sheet_name: str = "Global_CC"
    export_xls_header_row: int = 2
    export_xls_patient_id_column: str = "ID_current_base"
    export_xls_overwrite_policy: Literal["empty_only", "always", "never"] = "empty_only"
    #: JSON ``{ "pipeline_patient_id": "ma_row_key" }`` pour ``ID_current_base``.
    export_xls_patient_id_map_path: str | None = None
    #: Si True et gabarit présent : erreur si aucune ligne MA pour la clé résolue.
    export_xls_require_patient_in_ma: bool = True
    #: Met à jour le gabarit MA sur disque (source de vérité). Si False, copie vers outputs/ uniquement.
    export_xls_update_master_workbook: bool = True
    #: Copie optionnelle du gabarit mis à jour dans le dossier ``outputs/<doc_id>/``.
    export_xls_mirror_to_output: bool = False
    #: JSON ``{ "schema_column": "MA_header" }`` — défaut + data/column_aliases_ma.json.
    export_xls_column_aliases_path: str | None = None
    #: Placeholders MA traités comme cellules vides (virgules).
    empty_sentinels: str = "NA,N/A,-,A completer,A compléter,A verifier,A vérifier"

    def parsed_empty_sentinels(self) -> frozenset[str]:
        from app.etl.empty_sentinels import parse_empty_sentinels

        return parse_empty_sentinels(self.empty_sentinels)

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

    # ─── Robustesse upsert / search ────────────────────────────────────────────
    #: Taille max d'un batch Qdrant `upsert`. Au-delà, batching auto. Protège
    #: contre les limites gRPC (msg size) et les timeouts sur gros documents.
    upsert_batch_size: int = 256
    #: Filtre `embedding_version == settings.embedding_version` injecté
    #: automatiquement à chaque `search`. Désactiver UNIQUEMENT pendant une
    #: migration où les deux versions doivent cohabiter.
    enforce_embedding_version_filter: bool = True
    #: Avant `upsert_chunks`, si des points existent déjà pour le document avec une
    #: autre `embedding_version`, émettre un warning (ré-index / `delete_document`).
    warn_on_embedding_version_mismatch: bool = True
    #: Si True, refuse l'upsert tant que des points « ancienne version » coexistent
    #: (garde-fou prod : évite mélange dense/sparse incompatible au search).
    reject_embedding_version_mismatch_upsert: bool = False

    # ─── Retrieval (écarts d’environnement) ───────────────────────────────────
    #: Le reranker CrossEncoder et le sparse BM25 (fastembed) peuvent être absents
    #: ou désactivés selon la plateforme (ex. Windows sans wheel) : les scores et
    #: la latence diffèrent alors de la CI Linux. Voir logs au boot / health_check.

    # ─── Robustesse réseau (retry / read-only) ─────────────────────────────────
    #: Nombre de retries (en plus de la tentative initiale) sur les appels
    #: Qdrant pour les erreurs transientes (timeout, 5xx, 429). 0 = pas de retry.
    qdrant_max_retries: int = 3
    #: Base du backoff exponentiel entre retries Qdrant : delay = base * 2**attempt.
    #: Mettre à 0 dans les tests unitaires pour ne pas ralentir la suite.
    qdrant_retry_backoff_base: float = 0.5
    #: Mode lecture seule : si True, `upsert_chunks` et `delete_document` lèvent
    #: une `RuntimeError`. `ensure_collection` ne tente pas de créer (suppose la
    #: collection déjà bootstrappée). Utile pour un container API search-only.
    qdrant_read_only: bool = False
    #: Validation stricte du `tenant_id` : doit matcher `[A-Za-z0-9_-]{1,60}`,
    #: sinon `ValueError`. Par défaut **False** (rétrocompat : normalisation
    #: permissive qui substitue les caractères invalides par `_`). Activer
    #: **fortement recommandé** en prod : évite des collisions silencieuses
    #: entre tenants (ex: "STUDY 1" et "STUDY-1" qui normalisent en "STUDY_1"
    #: et "STUDY-1" → tenants distincts par accident).
    qdrant_strict_tenant_id: bool = False
