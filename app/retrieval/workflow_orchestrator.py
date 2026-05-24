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
        tenant_id: str | None = None,
        family_queries: dict[FieldFamily, str] | None = None,
    ) -> dict[FieldFamily, list[RetrievalHit]]:
        """Retourne des hits par famille de champs.

        Args:
            tenant_id: obligatoire pour les orchestrateurs branchés sur un vector store
                partitionné par tenant (ex. Qdrant) ; ignoré en mode purement local.
        """


class LocalWorkflowOrchestrator(BaseWorkflowOrchestrator):
    """Implémentation locale : filtrage par famille + score lexical léger."""

    def retrieve_for_families(
        self,
        *,
        document_id: str,
        chunks: list[DocumentChunk],
        families: list[FieldFamily],
        top_k_per_family: int = 3,
        tenant_id: str | None = None,  # noqa: ARG002 — non utilisé en mémoire locale
        family_queries: dict[FieldFamily, str] | None = None,
    ) -> dict[FieldFamily, list[RetrievalHit]]:
        _ = tenant_id, family_queries
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

    Le `tenant_id` doit être fourni à chaque appel (`retrieve_for_families`) ou
    une seule fois via le constructeur (rétrocompat tests / scripts courts).
    """

    _FAMILY_QUERY: dict[FieldFamily, str] = {
        FieldFamily.HEMATOLOGY: "hématologie hémogramme plaquettes leucocytes",
        FieldFamily.COAGULATION: "coagulation INR TP TCA fibrinogène",
        FieldFamily.HEPATIC_BIOCHEMISTRY: "bilan hépatique AST ALT GGT bilirubine",
        FieldFamily.INFLAMMATION_BIOMARKERS: "inflammation CRP biomarqueurs",
        FieldFamily.COMORBIDITIES: "antécédents comorbidités traitements",
        FieldFamily.IMAGING_RECIST: (
            "RECIST réponse tumorielle cible non cible progression stabilité "
            "scanner TDM IRM mesure lésion millimètres mm"
        ),
        FieldFamily.TREATMENT_LINES: (
            "ligne de traitement chimiothérapie immunothérapie protocole cycle "
            "dose arrêt efficacité toxicité seconde ligne troisième ligne"
        ),
    }

    def __init__(self, vector_index, *, tenant_id: str | None = None) -> None:
        self._index = vector_index
        self._tenant_id = tenant_id

    def retrieve_for_families(
        self,
        *,
        document_id: str,
        chunks: list[DocumentChunk],
        families: list[FieldFamily],
        top_k_per_family: int = 3,
        tenant_id: str | None = None,
        family_queries: dict[FieldFamily, str] | None = None,
    ) -> dict[FieldFamily, list[RetrievalHit]]:
        effective_tenant = (tenant_id or self._tenant_id or "").strip()
        if not effective_tenant:
            raise ValueError(
                "tenant_id obligatoire pour VectorStoreWorkflowOrchestrator "
                "(passer tenant_id=... à retrieve_for_families ou au constructeur)."
            )
        for c in chunks:
            if c.document_id != document_id:
                raise ValueError(
                    "retrieve_for_families: incohérence document_id — "
                    f"chunk {c.chunk_id!r} a document_id={c.document_id!r}, "
                    f"attendu {document_id!r}."
                )
        out: dict[FieldFamily, list[RetrievalHit]] = {}
        for fam in families:
            query = (family_queries or {}).get(fam) or self._FAMILY_QUERY.get(
                fam, fam.value.replace("_", " ")
            )
            try:
                results = self._index.search(
                    document_id=document_id,
                    query_text=query,
                    field_family=fam,
                    top_k=top_k_per_family,
                    tenant_id=effective_tenant,
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

    Non implémenté en V1 : tout appel lève `NotImplementedError`.
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
        tenant_id: str | None = None,  # noqa: ARG002
        family_queries: dict[FieldFamily, str] | None = None,
    ) -> dict[FieldFamily, list[RetrievalHit]]:
        _ = document_id, chunks, families, top_k_per_family, tenant_id, family_queries
        raise NotImplementedError(
            "Intégration RAGFlow non disponible : mapper document_id vers KB RAGFlow "
            "et traduire la réponse JSON en RetrievalHit."
        )
