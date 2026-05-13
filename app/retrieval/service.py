from __future__ import annotations

from abc import ABC, abstractmethod

from app.indexing.vector_service import VectorIndexService
from app.schemas.enums import FieldFamily
from app.schemas.models import RetrievalHit


class RetrievalService(ABC):
    """Service de retrieval paramétrable (vectoriel, hybride, distant)."""

    @abstractmethod
    def retrieve_for_family(
        self,
        *,
        document_id: str,
        field_family: FieldFamily,
        query_text: str,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        """Retourne les chunks les plus pertinents pour une famille."""


class VectorBackedRetrievalService(RetrievalService):
    """Retrieval basé sur `VectorIndexService` (mock ou futur Qdrant)."""

    def __init__(self, index: VectorIndexService) -> None:
        self._index = index

    def retrieve_for_family(
        self,
        *,
        document_id: str,
        field_family: FieldFamily,
        query_text: str,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        pairs = self._index.search(
            document_id=document_id,
            query_text=query_text,
            field_family=field_family,
            top_k=top_k,
        )
        hits: list[RetrievalHit] = []
        for rank, (chunk, score) in enumerate(pairs, start=1):
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=score,
                    query_family=field_family,
                    rank=rank,
                )
            )
        return hits
