from __future__ import annotations

from app.extraction.langextract_extractor import LangExtractExtractor
from app.extraction.llama_extractor import LlamaExtractor
from app.schemas.enums import DocumentType, ExtractionFamily, FieldFamily
from app.schemas.models import ExtractedObservation, FieldDefinition, RetrievalHit


class ExtractionService:
    """
    Orchestre les extracteurs par famille — **pas** de prompt monolithique.

    Pour un bilan sanguin, chaque famille reçoit uniquement les fragments
    issus du retrieval ciblé.
    """

    def __init__(
        self,
        *,
        langextract: LangExtractExtractor,
        llama: LlamaExtractor,
    ) -> None:
        self._langextract = langextract
        self._llama = llama

    def extract_for_family(
        self,
        *,
        doc_type: DocumentType,
        field_family: FieldFamily,
        hits: list[RetrievalHit],
        field_defs: list[FieldDefinition],
    ) -> list[ExtractedObservation]:
        if doc_type != DocumentType.LAB_BLOOD_PANEL:
            return []
        fam_defs = [d for d in field_defs if d.field_family == field_family]
        if not fam_defs:
            return []
        extraction_family = fam_defs[0].extraction_family
        observations: list[ExtractedObservation] = []
        schema = LangExtractExtractor.default_schema_for_family(field_family)
        merged_parts: list[str] = []
        chunk_ids: list[str | None] = []
        for hit in hits:
            merged_parts.append(hit.chunk.text)
            chunk_ids.append(hit.chunk.chunk_id)
        merged_text = "\n".join(dict.fromkeys(merged_parts))  # préserve l'ordre, supprime doublons
        primary_chunk_id = chunk_ids[0] if chunk_ids else None
        if extraction_family == ExtractionFamily.LAB_VALUES:
            observations.extend(
                self._langextract.extract(
                    merged_text,
                    schema,
                    source_chunk_id=primary_chunk_id,
                )
            )
        elif extraction_family == ExtractionFamily.NARRATIVE_CLINICAL:
            # TODO: prompts narratifs courts par champ
            observations.extend(
                self._llama.extract_by_field_category(
                    subtext=merged_text,
                    field_family=field_family,
                    source_chunk_id=primary_chunk_id,
                )
            )
        # dédoublonnage simple par (famille, evidence)
        seen: set[tuple[str, str | None]] = set()
        deduped: list[ExtractedObservation] = []
        for o in observations:
            if o.field_family != field_family:
                continue
            key = (o.field_family.value, o.evidence_text)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(o)
        return deduped
