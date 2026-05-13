"""
Reranker CrossEncoder (BGE) appliqué sur les candidats Qdrant.

Lazy import — `sentence-transformers` est dans l'extra `vector`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from app.config.settings import Settings
from app.schemas.models import DocumentChunk


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[tuple[DocumentChunk, float]],
        top_n: int,
    ) -> list[tuple[DocumentChunk, float]]: ...


class BGECrossEncoderReranker(Reranker):
    """Reranker `BAAI/bge-reranker-v2-m3` (multilingue, qualité prouvée)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                'sentence-transformers manquant. Installer l\'extra : pip install -e ".[vector]"'
            ) from e
        device = self._resolve_device()
        self._model = CrossEncoder(self.settings.reranker_model, device=device)
        return self._model

    def _resolve_device(self) -> str:
        d = self.settings.reranker_device
        if d == "auto":
            try:
                from torch.cuda import is_available

                return "cuda" if is_available() else "cpu"
            except Exception:
                return "cpu"
        return d

    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[tuple[DocumentChunk, float]],
        top_n: int,
    ) -> list[tuple[DocumentChunk, float]]:
        if not candidates:
            return []
        model = self._ensure_model()
        pairs = [(query, c.text or "") for c, _ in candidates]
        scores = model.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
        ranked = sorted(
            zip([c for c, _ in candidates], (float(s) for s in scores)),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return ranked[:top_n]
