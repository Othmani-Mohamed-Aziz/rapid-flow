from __future__ import annotations

from abc import ABC, abstractmethod

from app.indexing.retrieval_passage import passage_for_embedding_and_rerank
from app.schemas.enums import FieldFamily
from app.schemas.models import DocumentChunk


class VectorIndexService(ABC):
    """
    Abstraction d'index vectoriel (Qdrant, pgvector, etc.).

    Implémentations actuelles : `InMemoryVectorIndexService` (développement),
    `LlamaIndexVectorService` (même stockage mémoire + métadonnées prêtes LlamaIndex).
    """

    @abstractmethod
    def upsert_chunks(self, chunks: list[DocumentChunk], **kwargs) -> None:
        """Indexe ou met à jour des chunks. kwargs : tenant_id, patient_id, study_id, document_type."""

    @abstractmethod
    def search(
        self,
        *,
        document_id: str | None = None,
        query_text: str,
        field_family: FieldFamily | None = None,
        top_k: int = 5,
        **kwargs,
    ) -> list[tuple[DocumentChunk, float]]:
        """Retourne des paires (chunk, score). kwargs : tenant_id, patient_id, document_type."""


class InMemoryVectorIndexService(VectorIndexService):
    """Index trivial pour développement sans Qdrant."""

    def __init__(self) -> None:
        self._store: dict[str, list[DocumentChunk]] = {}

    def upsert_chunks(self, chunks: list[DocumentChunk], **kwargs) -> None:  # noqa: ARG002
        for c in chunks:
            bucket = self._store.setdefault(c.document_id, [])
            bucket[:] = [x for x in bucket if x.chunk_id != c.chunk_id]
            bucket.append(c)

    def search(
        self,
        *,
        document_id: str | None = None,
        query_text: str,
        field_family: FieldFamily | None = None,
        top_k: int = 5,
        **kwargs,  # noqa: ARG002 - tenant_id ignoré en mémoire
    ) -> list[tuple[DocumentChunk, float]]:
        if document_id is None:
            pool: list[DocumentChunk] = [c for bucket in self._store.values() for c in bucket]
        else:
            pool = [c for c in self._store.get(document_id, [])]
        if field_family is not None:
            pool = [c for c in pool if c.field_family == field_family or c.field_family is None]
        q = query_text.lower()
        scored: list[tuple[DocumentChunk, float]] = []
        for c in pool:
            text = passage_for_embedding_and_rerank(c).lower()
            overlap = sum(1 for tok in q.split() if len(tok) > 2 and tok in text)
            score = min(1.0, 0.2 + 0.15 * overlap)
            scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class LlamaIndexVectorService(InMemoryVectorIndexService):
    """
    Métadonnées alignées sur `llama_index.core.Document` (ids + champs filtrables).

    V1 : recherche identique à `InMemoryVectorIndexService` (score lexical binaire),
    sans embeddings ni Qdrant — préparation pour brancher `VectorStoreIndex` ensuite.
    """

    @staticmethod
    def _normalize_metadata(chunk: DocumentChunk) -> dict[str, object]:
        md = dict(chunk.metadata or {})
        md.setdefault("ref_doc_id", chunk.document_id)
        md.setdefault("ref_chunk_id", chunk.chunk_id)
        if chunk.field_family is not None:
            md.setdefault("field_family", chunk.field_family.value)
        return md

    def upsert_chunks(self, chunks: list[DocumentChunk], **kwargs) -> None:
        wrapped = [
            c.model_copy(
                update={
                    "metadata": self._normalize_metadata(c),
                }
            )
            for c in chunks
        ]
        super().upsert_chunks(wrapped, **kwargs)

    def search(
        self,
        *,
        document_id: str | None = None,
        query_text: str,
        field_family: FieldFamily | None = None,
        top_k: int = 5,
        **kwargs,
    ) -> list[tuple[DocumentChunk, float]]:
        return super().search(
            document_id=document_id,
            query_text=query_text,
            field_family=field_family,
            top_k=top_k,
            **kwargs,
        )
