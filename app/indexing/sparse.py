"""
Encodeur BM25 sparse via `fastembed` (compatible Qdrant `SparseVector`).

Activé par `Settings.enable_sparse_bm25`. Lazy import : `fastembed` reste optionnel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from app.config.settings import Settings


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class SparseEncoder(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def encode_passages(self, texts: Sequence[str]) -> list[SparseVector]: ...

    @abstractmethod
    def encode_query(self, text: str) -> SparseVector: ...


class FastembedBM25SparseEncoder(SparseEncoder):
    """BM25 via `fastembed.SparseTextEmbedding`."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._model: Any = None

    @property
    def model_id(self) -> str:
        return self.settings.sparse_model

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from fastembed import SparseTextEmbedding
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                'fastembed manquant. Installer l\'extra : pip install -e ".[vector]"'
            ) from e
        self._model = SparseTextEmbedding(model_name=self.model_id)
        return self._model

    def _to_sparse(self, emb) -> SparseVector:
        indices = getattr(emb, "indices", None)
        values = getattr(emb, "values", None)
        if indices is None or values is None:
            d = dict(emb) if not isinstance(emb, dict) else emb
            indices = d.get("indices", [])
            values = d.get("values", [])
        return SparseVector(
            indices=[int(i) for i in indices],
            values=[float(v) for v in values],
        )

    def encode_passages(self, texts: Sequence[str]) -> list[SparseVector]:
        model = self._ensure_model()
        return [self._to_sparse(e) for e in model.passage_embed(list(texts))]

    def encode_query(self, text: str) -> SparseVector:
        model = self._ensure_model()
        embs = list(model.query_embed([text]))
        if not embs:
            return SparseVector(indices=[], values=[])
        return self._to_sparse(embs[0])
