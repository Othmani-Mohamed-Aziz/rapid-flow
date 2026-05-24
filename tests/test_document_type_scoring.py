"""Tests du classifieur multi-signaux `document_type_scoring`."""

from __future__ import annotations

from app.parsing.document_type_scoring import classify_document_type
from app.schemas.enums import DocumentType
from app.schemas.models import DocumentSection


def test_classifier_ct_like_imaging_wins_over_contraste() -> None:
    text = (
        "COMPTE RENDU DE SCANNER ABDOMINO-PELVIEN\n"
        "Injection de produit de contraste iodé.\n"
        "Critères RECIST.\n"
    )
    sections = [
        DocumentSection(heading="RÉSULTATS", body=text, metadata={"source": "docling_graph"}),
    ]
    meta = {
        "docling_layout_stats": {
            "n_table_items": 1,
            "n_text_items": 6,
            "n_picture_items": 0,
            "n_heading_events": 3,
        }
    }
    r = classify_document_type(text, sections, meta)
    assert r.document_type == DocumentType.IMAGING_REPORT
    assert (
        r.scores[DocumentType.IMAGING_REPORT.value] > r.scores[DocumentType.LAB_BLOOD_PANEL.value]
    )


def test_classifier_lab_pdf_table_like() -> None:
    body = "| NFS | 4.2 | G/L |\n| Hb | 12 | g/dL |\n| Plaquettes | 180 | G/L |\n"
    text = "Laboratoire central\n" + body * 4
    sections = [
        DocumentSection(heading="Hématologie", body=text, metadata={"source": "docling_graph"})
    ]
    meta = {
        "docling_layout_stats": {
            "n_table_items": 4,
            "n_text_items": 2,
            "n_picture_items": 0,
            "n_heading_events": 2,
        }
    }
    r = classify_document_type(text, sections, meta)
    assert r.document_type == DocumentType.LAB_BLOOD_PANEL


def test_classifier_pathology_context() -> None:
    text = (
        "Compte rendu anatomopathologique\n"
        "Biopsie du foie : adénocarcinome. Immunohistochimie positive.\n"
    )
    r = classify_document_type(text, [], {})
    assert r.document_type == DocumentType.PATHOLOGY_REPORT


def test_classifier_unknown_when_no_signal() -> None:
    text = "Courrier administratif interne sans données médicales structurées."
    r = classify_document_type(text, [], {})
    assert r.document_type == DocumentType.UNKNOWN


def test_classifier_metadata_has_scores_and_rationale() -> None:
    r = classify_document_type("Scanner abdominal sans particularité.", [], {})
    md = r.as_metadata()
    assert "document_type_scores" in md
    assert "document_type_features" in md
    assert "document_type_rationale" in md
    assert isinstance(md["document_type_rationale"], list)
