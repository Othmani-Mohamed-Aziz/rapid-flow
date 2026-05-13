from __future__ import annotations

from app.indexing.vector_service import LlamaIndexVectorService
from app.schemas.enums import FieldFamily
from app.schemas.models import DocumentChunk


def test_llama_index_vector_service_metadata_shape() -> None:
    svc = LlamaIndexVectorService()
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_id="doc-1",
            text="lésion hépatique segment VI",
            field_family=None,
            metadata={"section_heading": "Imagerie", "content_kind": "clinical_section"},
        )
    ]
    svc.upsert_chunks(chunks)
    hits = svc.search(document_id="doc-1", query_text="lésion hépatique", top_k=3)
    assert hits
    stored = hits[0][0]
    assert stored.metadata.get("ref_doc_id") == "doc-1"
    assert stored.metadata.get("ref_chunk_id") == "c1"
    assert stored.metadata.get("section_heading") == "Imagerie"


def test_inmemory_upsert_replaces_same_chunk_id() -> None:
    from app.indexing.vector_service import InMemoryVectorIndexService

    svc = InMemoryVectorIndexService()
    c1 = DocumentChunk(
        chunk_id="same",
        document_id="doc-1",
        text="v1",
        field_family=None,
        metadata={},
    )
    c2 = DocumentChunk(
        chunk_id="same",
        document_id="doc-1",
        text="v2",
        field_family=None,
        metadata={},
    )
    svc.upsert_chunks([c1, c2])
    pool = [c for c in svc._store.get("doc-1", [])]
    assert len(pool) == 1
    assert pool[0].text == "v2"


def test_llama_index_vector_service_field_family_string() -> None:
    svc = LlamaIndexVectorService()
    svc.upsert_chunks(
        [
            DocumentChunk(
                chunk_id="c2",
                document_id="doc-2",
                text="CRP",
                field_family=FieldFamily.INFLAMMATION_BIOMARKERS,
                metadata={},
            )
        ]
    )
    hits = svc.search(
        document_id="doc-2", query_text="crp", field_family=FieldFamily.INFLAMMATION_BIOMARKERS
    )
    assert hits[0][0].metadata.get("field_family") == FieldFamily.INFLAMMATION_BIOMARKERS.value
