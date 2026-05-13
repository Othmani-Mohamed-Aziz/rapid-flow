from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.enums import FieldFamily
from app.schemas.models import DocumentChunk, RetrievalHit


class BaseWorkflowOrchestrator(ABC):
    """
    Orchestration haut niveau des workflows documentaires (RAGFlow, local, etc.).

    Cette couche reste volontairement mince : elle coordonne retrieval
    multi-familles sans embarquer la logique métier eCRF.
    """

    @abstractmethod
    def retrieve_for_families(
        self,
        *,
        document_id: str,
        chunks: list[DocumentChunk],
        families: list[FieldFamily],
        top_k_per_family: int = 3,
    ) -> dict[FieldFamily, list[RetrievalHit]]:
        """Retourne des hits par famille de champs."""


class LocalWorkflowOrchestrator(BaseWorkflowOrchestrator):
    """Implémentation locale : filtrage par famille + score lexical léger."""

    def retrieve_for_families(
        self,
        *,
        document_id: str,
        chunks: list[DocumentChunk],
        families: list[FieldFamily],
        top_k_per_family: int = 3,
    ) -> dict[FieldFamily, list[RetrievalHit]]:
        out: dict[FieldFamily, list[RetrievalHit]] = {}
        doc_chunks = [c for c in chunks if c.document_id == document_id]
        for fam in families:
            hits: list[RetrievalHit] = []
            candidates = [c for c in doc_chunks if c.field_family == fam or c.field_family is None]
            for rank, ch in enumerate(candidates[:top_k_per_family], start=1):
                # Les chunks non étiquetés restent exploitables pour familles narratives.
                if ch.field_family == fam:
                    score = 0.88
                elif ch.field_family is None:
                    score = 0.78
                else:
                    score = 0.55
                hits.append(
                    RetrievalHit(
                        chunk=ch,
                        score=score,
                        query_family=fam,
                        rank=rank,
                    )
                )
            out[fam] = hits
        return out


class VectorStoreWorkflowOrchestrator(BaseWorkflowOrchestrator):
    """
    Orchestrateur qui interroge un `VectorIndexService` (Qdrant ou mémoire).

    Le `tenant_id` est requis ; on utilise une **requête générique par famille** comme
    base (mot-clé canonique de la famille) puis le reranker du vector store affine.
    """

    _FAMILY_QUERY: dict[FieldFamily, str] = {
        FieldFamily.HEMATOLOGY: "hématologie hémogramme plaquettes leucocytes",
        FieldFamily.COAGULATION: "coagulation INR TP TCA fibrinogène",
        FieldFamily.HEPATIC_BIOCHEMISTRY: "bilan hépatique AST ALT GGT bilirubine",
        FieldFamily.INFLAMMATION_BIOMARKERS: "inflammation CRP biomarqueurs",
        FieldFamily.COMORBIDITIES: "antécédents comorbidités traitements",
    }

    def __init__(self, vector_index, *, tenant_id: str | None = None) -> None:
        self._index = vector_index
        self._tenant_id = tenant_id

    def retrieve_for_families(
        self,
        *,
        document_id: str,
        chunks: list[DocumentChunk],  # noqa: ARG002 — non utilisé (la source de vérité = le store)
        families: list[FieldFamily],
        top_k_per_family: int = 3,
    ) -> dict[FieldFamily, list[RetrievalHit]]:
        out: dict[FieldFamily, list[RetrievalHit]] = {}
        for fam in families:
            query = self._FAMILY_QUERY.get(fam, fam.value.replace("_", " "))
            try:
                results = self._index.search(
                    document_id=document_id,
                    query_text=query,
                    field_family=fam,
                    top_k=top_k_per_family,
                    tenant_id=self._tenant_id,
                )
            except TypeError:
                results = self._index.search(
                    document_id=document_id,
                    query_text=query,
                    field_family=fam,
                    top_k=top_k_per_family,
                )
            hits = [
                RetrievalHit(chunk=ch, score=float(sc), query_family=fam, rank=i + 1)
                for i, (ch, sc) in enumerate(results)
            ]
            out[fam] = hits
        return out


class RagflowWorkflowOrchestrator(BaseWorkflowOrchestrator):
    """
    Point d'accroche RAGFlow (HTTP / SDK).

    TODO: appeler l'API RAGFlow pour dataset_id / retrieval ciblé par tags
    de famille et réhydrater en `RetrievalHit` avec scores distants.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = base_url
        self.api_key = api_key

    def retrieve_for_families(
        self,
        *,
        document_id: str,
        chunks: list[DocumentChunk],
        families: list[FieldFamily],
        top_k_per_family: int = 3,
    ) -> dict[FieldFamily, list[RetrievalHit]]:
        raise NotImplementedError(
            "TODO: intégration RAGFlow — mapper document_id vers KB RAGFlow "
            "et traduire la réponse JSON en RetrievalHit."
        )
