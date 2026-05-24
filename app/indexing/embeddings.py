"""
Service d’embeddings denses (abstraction + implémentation locale Hugging Face).

Garanties :
- Dimension/version cohérentes via `embedding_dim` et `embedding_version` (utilisés en payload Qdrant).
- Préfixes type E5 / BGE séparés `passage:` / `query:` (gros impact qualité).
- Modèle chargé à la **première utilisation** (lazy) pour ne pas pénaliser les imports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

from app.config.settings import Settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingService(ABC):
    """Abstraction d’embeddings — dimension fixe par instance."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def version_tag(self) -> str: ...

    @abstractmethod
    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class EmbeddingsNotInstalledError(RuntimeError):
    """Levée si l’extra `vector` n’est pas installé."""


class HuggingFaceEmbeddingService(EmbeddingService):
    """
    Embeddings via `sentence-transformers` (local, on-prem).

    Modèles cibles :
    - `intfloat/multilingual-e5-base` / `…-large`
    - `BAAI/bge-m3` (sparse + dense multivecteurs : à activer ultérieurement)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._model: SentenceTransformer | None = None
        self._passage_prefix = self.settings.embedding_passage_prefix
        self._query_prefix = self.settings.embedding_query_prefix

    # --- Propriétés exposées --------------------------------------------------

    @property
    def dim(self) -> int:
        return int(self.settings.embedding_dim)

    @property
    def model_id(self) -> str:
        return self.settings.embedding_model

    @property
    def version_tag(self) -> str:
        return self.settings.embedding_version

    # --- Lazy loading --------------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - dépendance optionnelle
            raise EmbeddingsNotInstalledError(
                'Installer l\'extra : pip install -e ".[vector]"'
            ) from e
        device = self._resolve_device()
        self._model = SentenceTransformer(self.model_id, device=device)
        # Validation rapide de la dim (compatible ST 2.x / 3.x / 5.x)
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            getattr(self._model, "get_sentence_embedding_dimension", None),
        )
        actual = int((get_dim() if callable(get_dim) else 0) or 0)
        if actual and actual != self.dim:
            raise RuntimeError(
                f"Embedding dim mismatch : settings={self.dim} mais modèle {self.model_id}={actual}. "
                "Mettre à jour `ECRF_EMBEDDING_DIM` ou changer de modèle."
            )
        return self._model

    def _resolve_device(self) -> str:
        d = self.settings.reranker_device
        if d == "auto":
            try:
                import torch  # noqa: F401
                from torch.cuda import is_available

                return "cuda" if is_available() else "cpu"
            except Exception:
                return "cpu"
        return d

    # --- API publique --------------------------------------------------------

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_model()
        prepared = [f"{self._passage_prefix}{t or ''}" for t in texts]
        vecs = model.encode(
            prepared,
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        vec = model.encode(
            [f"{self._query_prefix}{text or ''}"],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vec[0].tolist()


def build_default_embedding_service(settings: Settings | None = None) -> EmbeddingService:
    s = settings or Settings()
    if s.embedding_provider == "hf_local":
        return HuggingFaceEmbeddingService(s)
    raise ValueError(f"Unsupported embedding_provider: {s.embedding_provider}")
