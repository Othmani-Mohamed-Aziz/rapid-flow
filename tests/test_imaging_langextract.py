from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.business_rules.confidence import ConfidenceScoringService
from app.config.study_schema_provider import resolve_study_schema
from app.extraction.imaging_langextract import (
    CANONICAL_LESION_SIZE_MM,
    CANONICAL_RECIST_RESPONSE,
    extractions_to_observations,
    flatten_langextract_extractions,
    run_imaging_langextract,
)
from app.extraction.langextract_extractor import LangExtractExtractor
from app.extraction.service import ExtractionService
from app.extraction.strategy_extractors import ExtractorRegistry
from app.schemas.enums import DocumentType, ExtractionFamily, FieldFamily, TemporalScope
from app.schemas.models import DocumentChunk, FieldCandidate, FieldDefinition, RetrievalHit


def _fake_extraction(
    *,
    extraction_class: str,
    extraction_text: str,
    attributes: dict | None = None,
    grounded: bool = True,
) -> SimpleNamespace:
    char_iv = SimpleNamespace(start_pos=0, end_pos=len(extraction_text)) if grounded else None
    return SimpleNamespace(
        extraction_class=extraction_class,
        extraction_text=extraction_text,
        attributes=attributes or {},
        char_interval=char_iv,
    )


def test_extractions_to_observations_lesion_size() -> None:
    ext = _fake_extraction(
        extraction_class="lesion_size_mm",
        extraction_text="28 mm",
        attributes={"value_mm": 28},
    )
    obs = extractions_to_observations(
        [ext],
        source_chunk_id="c1",
        model_name="test-model",
        schema_version="v1",
    )
    assert len(obs) == 1
    assert obs[0].normalized_value == 28
    assert obs[0].unit == "mm"
    assert obs[0].extra["canonical_imaging_key"] == CANONICAL_LESION_SIZE_MM
    assert obs[0].source_chunk_id == "c1"


def test_extractions_to_observations_skips_ungrounded() -> None:
    ext = _fake_extraction(
        extraction_class="lesion_size_mm",
        extraction_text="99 mm",
        attributes={"value_mm": 99},
        grounded=False,
    )
    assert (
        extractions_to_observations(
            [ext], source_chunk_id=None, model_name=None, schema_version="v1"
        )
        == []
    )


def test_extractions_to_observations_recist() -> None:
    ext = _fake_extraction(
        extraction_class="recist_response",
        extraction_text="maladie stable",
    )
    obs = extractions_to_observations(
        [ext], source_chunk_id="c2", model_name="m", schema_version="v1"
    )
    assert obs[0].normalized_value == "SD"
    assert obs[0].extra["canonical_imaging_key"] == CANONICAL_RECIST_RESPONSE


def test_extractions_to_observations_recist_stabilite_fr() -> None:
    ext = _fake_extraction(
        extraction_class="recist_response",
        extraction_text="stabilité de la maladie",
    )
    obs = extractions_to_observations(
        [ext], source_chunk_id="c2", model_name="m", schema_version="v1"
    )
    assert len(obs) == 1
    assert obs[0].normalized_value == "SD"


def test_extractions_to_observations_skips_non_canon_recist() -> None:
    ext = _fake_extraction(
        extraction_class="recist_response",
        extraction_text="considérations radiologiques diverses",
    )
    obs = extractions_to_observations(
        [ext], source_chunk_id="c2", model_name="m", schema_version="v1"
    )
    assert obs == []


def test_extractions_derived_recist_from_imaging_conclusion() -> None:
    ext = _fake_extraction(
        extraction_class="imaging_conclusion",
        extraction_text="stabilité des lésions hépatiques",
    )
    obs = extractions_to_observations(
        [ext], source_chunk_id="c3", model_name="m", schema_version="v1"
    )
    keys = {(o.extra.get("canonical_imaging_key"), o.normalized_value) for o in obs}
    assert (CANONICAL_RECIST_RESPONSE, "SD") in keys
    assert ("imaging_conclusion", "stabilité des lésions hépatiques") in keys
    derived = next(
        o
        for o in obs
        if o.extra.get("derived_from_imaging_conclusion")
        and o.extra.get("canonical_imaging_key") == CANONICAL_RECIST_RESPONSE
    )
    assert derived.confidence <= 0.72


def test_retrieval_score_in_observation_extra() -> None:
    ext = _fake_extraction(extraction_class="recist_response", extraction_text="SD")
    obs = extractions_to_observations(
        [ext],
        source_chunk_id="c4",
        model_name="m",
        schema_version="v1",
        retrieval_hit_score=0.91,
    )
    assert obs[0].extra.get("retrieval_score") == 0.91


def test_confidence_prefers_per_hit_retrieval_score() -> None:
    ext = _fake_extraction(extraction_class="recist_response", extraction_text="SD")
    obs = extractions_to_observations(
        [ext],
        source_chunk_id="c5",
        model_name="m",
        schema_version="v1",
        retrieval_hit_score=0.4,
    )
    cand = FieldCandidate(
        field_name="Response_at_first_imaging_RECIST",
        observation=obs[0],
        target_column="x",
        confidence=obs[0].confidence,
    )
    blended = ConfidenceScoringService.blend_with_retrieval(cand.observation, retrieval_score=0.99)
    assert blended == 0.5 * obs[0].confidence + 0.5 * 0.4


def test_flatten_langextract_nested_documents() -> None:
    inner = SimpleNamespace(
        extractions=[
            _fake_extraction(
                extraction_class="lesion_size_mm",
                extraction_text="10 mm",
                attributes={"value_mm": 10},
            )
        ]
    )
    wrapped = SimpleNamespace(extractions=[inner])
    flat = flatten_langextract_extractions(wrapped)
    assert len(flat) == 1
    assert flat[0].extraction_class == "lesion_size_mm"


@patch("app.extraction.imaging_langextract.build_imaging_examples", return_value=[])
def test_run_imaging_langextract_calls_extract(mock_examples: MagicMock) -> None:
    fake_result = SimpleNamespace(
        extractions=[
            _fake_extraction(
                extraction_class="lesion_size_mm",
                extraction_text="12 mm",
                attributes={"value_mm": 12},
            )
        ]
    )
    mock_extract = MagicMock(return_value=fake_result)

    obs = run_imaging_langextract(
        "Nodule de 12 mm au segment VII.",
        model_id="gemma2:2b",
        model_url="http://localhost:11434",
        source_chunk_id="chunk-1",
        extract_fn=mock_extract,
    )
    assert len(obs) == 1
    mock_extract.assert_called_once()
    call_kw = mock_extract.call_args.kwargs
    assert call_kw["model_id"] == "gemma2:2b"
    assert call_kw["model_url"] == "http://localhost:11434"
    assert call_kw["language_model_params"] == {"timeout": 120}


def test_langextract_extractor_disabled_returns_empty() -> None:
    ext = LangExtractExtractor(langextract_enabled=False, model_name="x")
    schema = LangExtractExtractor.default_schema_for_family(FieldFamily.IMAGING_RECIST)
    assert ext.extract("texte CR", schema) == []


def test_extraction_service_imaging_per_chunk() -> None:
    schema = resolve_study_schema()
    mock_imaging = MagicMock()
    mock_imaging.extract_chunk.side_effect = [[], []]
    extractor_registry = ExtractorRegistry(
        imaging=mock_imaging,
        lab=MagicMock(),
        narrative=MagicMock(),
    )
    svc = ExtractionService(
        extractor_registry=extractor_registry,
        study_schema=schema,
        imaging_extraction_max_workers=1,
    )
    hits = [
        RetrievalHit(
            chunk=DocumentChunk(chunk_id="a", document_id="d1", text="chunk A"),
            score=0.9,
            query_family=FieldFamily.IMAGING_RECIST,
            rank=1,
        ),
        RetrievalHit(
            chunk=DocumentChunk(chunk_id="b", document_id="d1", text="chunk B"),
            score=0.8,
            query_family=FieldFamily.IMAGING_RECIST,
            rank=2,
        ),
    ]
    field_defs = [
        FieldDefinition(
            field_name="Size_major_nodule_mm_start_AtezoBev_D0",
            field_type="numeric",
            document_types_allowed=[DocumentType.IMAGING_REPORT],
            extraction_family=ExtractionFamily.IMAGING_RECIST,
            field_family=FieldFamily.IMAGING_RECIST,
            temporal_scope=TemporalScope.BASELINE,
            target_column="Size_major_nodule_mm_start_AtezoBev_D0",
            canonical_key="Size_major_nodule_mm",
        )
    ]
    svc.extract_for_family(
        doc_type=DocumentType.IMAGING_REPORT,
        field_family=FieldFamily.IMAGING_RECIST,
        hits=hits,
        field_defs=field_defs,
    )
    assert mock_imaging.extract_chunk.call_count == 2
    assert mock_imaging.extract_chunk.call_args_list[0].kwargs["source_chunk_id"] == "a"
    assert mock_imaging.extract_chunk.call_args_list[1].kwargs["source_chunk_id"] == "b"
