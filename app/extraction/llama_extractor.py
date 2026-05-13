from __future__ import annotations

from app.extraction.base import BaseExtractor
from app.extraction.lab_heuristics import extract_comorbidity_observations, extract_lab_observations
from app.schemas.enums import FieldFamily
from app.schemas.models import ExtractedObservation


class LlamaExtractor(BaseExtractor):
    """
    Extracteur spécialisé modèle Llama (fine-tuné labo en prod).

    V1 : délègue aux heuristiques comme **placeholder** pour préserver
    l'architecture ; remplacer par inférence `llama.cpp` / HF / vLLM.

    TODO: charger le modèle fine-tuné et appliquer des prompts courts par catégorie.
    """

    name = "llama_lab_specialist"

    def __init__(self, *, model_name: str, schema_version: str | None = None) -> None:
        self.model_name = model_name
        self.schema_version = schema_version

    def extract_by_field_category(
        self,
        *,
        subtext: str,
        field_family: FieldFamily,
        source_chunk_id: str | None = None,
    ) -> list[ExtractedObservation]:
        if field_family == FieldFamily.COMORBIDITIES:
            return extract_comorbidity_observations(
                subtext,
                extraction_method="llama_stub",
                model_name=self.model_name,
                schema_version=self.schema_version,
                source_chunk_id=source_chunk_id,
            )
        observations = extract_lab_observations(
            subtext,
            extraction_method="llama_stub",
            model_name=self.model_name,
            schema_version=self.schema_version,
            source_chunk_id=source_chunk_id,
        )
        return [o for o in observations if o.field_family == field_family]
