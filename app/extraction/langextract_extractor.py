from __future__ import annotations

from typing import Any

from app.extraction.lab_heuristics import build_langextract_schema_stub, extract_lab_observations
from app.schemas.enums import FieldFamily
from app.schemas.models import ExtractedObservation


class LangExtractExtractor:
    """
    Extracteur guidé par schéma (intégration LangExtract prévue).

    La méthode `extract` accepte un fragment, un schéma de sortie (dict)
    et un contexte optionnel (patient, unités attendues, etc.).

    TODO: remplacer la branche mock par l'appel réel à LangExtract lorsque
    la dépendance et les credentials seront disponibles.
    """

    def __init__(self, *, model_name: str | None = None, schema_version: str | None = None) -> None:
        self.model_name = model_name
        self.schema_version = schema_version

    def extract(
        self,
        text: str,
        output_schema: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        source_chunk_id: str | None = None,
    ) -> list[ExtractedObservation]:
        """
        Exécute une extraction structurée sur un **sous-texte** uniquement.

        En V1, si le schéma porte la famille lab, on applique des heuristiques
        locales plutôt qu'un appel distant.
        """
        _ = context  # réservé aux futurs paramètres LangExtract
        family_token = output_schema.get("x-field-family")
        if family_token:
            _ = FieldFamily(family_token)  # validation précoce du schéma
        # TODO: dispatch selon `output_schema` réel LangExtract
        return extract_lab_observations(
            text,
            extraction_method="langextract_stub",
            model_name=self.model_name,
            schema_version=self.schema_version or "mock-schema",
            source_chunk_id=source_chunk_id,
        )

    @staticmethod
    def default_schema_for_family(family: FieldFamily) -> dict[str, Any]:
        return build_langextract_schema_stub(family)
