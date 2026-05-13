from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.enums import FieldFamily
from app.schemas.models import ExtractedObservation


class BaseExtractor(ABC):
    """Contrat générique pour extracteurs spécialisés (Llama, règles, etc.)."""

    name: str = "base"

    @abstractmethod
    def extract_by_field_category(
        self,
        *,
        subtext: str,
        field_family: FieldFamily,
        source_chunk_id: str | None = None,
    ) -> list[ExtractedObservation]:
        """
        Extrait des observations à partir d'un sous-texte ciblé.

        Ne doit pas supposer l'accès au document entier : le pipeline fournit
        uniquement les fragments pertinents (post-routing / retrieval).
        """
