"""Fixtures partagées (PDF CT de référence, parsing Docling une fois par session)."""

from __future__ import annotations

import pytest

from tests.data_paths import CT_SCAN_REPORT_LIVER_PDF


@pytest.fixture(autouse=True)
def _isolate_vector_env(monkeypatch, request):
    """Évite que `.env` local OU les variables CI ne contaminent les tests.

    Contexte :
    - `.env` dev peut être en mode `qdrant`+BM25 (cf. README) ;
    - en CI integration, on passe `ECRF_QDRANT_COLLECTION_PREFIX=ecrf_chunks_citest`
      pour ne pas polluer une éventuelle collection prod, mais ça fuirait dans
      les tests unitaires qui font `Settings()` sans prefix explicite.

    - Par défaut : `vector_backend=memory`, sparse BM25 / reranker désactivés,
      `qdrant_collection_prefix` forcé au défaut (`ecrf_chunks`).
    - Opt-in : un test marqué `@pytest.mark.real_vector_backend` (cas de
      `test_qdrant_integration.py`) conserve ses propres `Settings(...)` et
      les variables d'environnement CI."""
    if request.node.get_closest_marker("real_vector_backend"):
        return
    monkeypatch.setenv("ECRF_VECTOR_BACKEND", "memory")
    monkeypatch.setenv("ECRF_ENABLE_SPARSE_BM25", "false")
    monkeypatch.setenv("ECRF_ENABLE_RERANKER", "false")
    monkeypatch.setenv("ECRF_QDRANT_COLLECTION_PREFIX", "ecrf_chunks")


@pytest.fixture(scope="module")
def ct_scan_parsed_docling():
    """
    `ParsedDocument` pour `data/ct_scan_report_liver.pdf` via Docling.

    Skip si fichier absent ou Docling indisponible.
    """
    if not CT_SCAN_REPORT_LIVER_PDF.is_file():
        pytest.skip(f"PDF CT manquant : {CT_SCAN_REPORT_LIVER_PDF}")
    try:
        from app.parsing.pdf.docling_parser import DoclingPdfParser
    except ImportError:
        pytest.skip("docling / docling_core non importables")
    if not DoclingPdfParser().is_available():
        pytest.skip("docling non installé (extra .[docling])")

    from app.config.settings import Settings
    from app.ingestion.service import LocalFileIngestionService
    from app.parsing.smart_service import SmartParsingService

    raw = LocalFileIngestionService().ingest(
        str(CT_SCAN_REPORT_LIVER_PDF.resolve()),
        patient_id="test-ct-patient",
        study_id="test-ct-study",
    )
    return SmartParsingService(Settings(pdf_parser_backend="docling")).parse(raw)


@pytest.fixture(scope="module")
def ct_scan_raw_document():
    """`RawDocument` pour le PDF CT (pypdf / docling)."""
    if not CT_SCAN_REPORT_LIVER_PDF.is_file():
        pytest.skip(f"PDF CT manquant : {CT_SCAN_REPORT_LIVER_PDF}")
    from app.ingestion.service import LocalFileIngestionService

    return LocalFileIngestionService().ingest(
        str(CT_SCAN_REPORT_LIVER_PDF.resolve()),
        patient_id="test-ct-patient",
        study_id="test-ct-study",
    )


@pytest.fixture(scope="module")
def default_study_schema():
    from tests.extraction_fixtures import resolve_study_schema

    return resolve_study_schema()


@pytest.fixture
def default_extraction_service(default_study_schema):
    from tests.extraction_fixtures import build_extraction_service

    return build_extraction_service(default_study_schema, langextract_enabled=False)
