from __future__ import annotations

from app.routing.router import KeywordHeuristicDocumentRouter
from app.schemas.enums import DocumentType
from app.schemas.models import ParsedDocument


def test_router_detects_blood_panel() -> None:
    router = KeywordHeuristicDocumentRouter()
    parsed = ParsedDocument(
        document_id="d1",
        patient_id="p1",
        study_id="s1",
        full_text="NFS et bilan hépatique : AST 40 UI/L",
        document_type_hint=DocumentType.UNKNOWN,
    )
    assert router.route(parsed) == DocumentType.LAB_BLOOD_PANEL


def test_router_respects_hint() -> None:
    router = KeywordHeuristicDocumentRouter()
    parsed = ParsedDocument(
        document_id="d1",
        patient_id="p1",
        study_id="s1",
        full_text="xxxx",
        document_type_hint=DocumentType.IMAGING_REPORT,
    )
    assert router.route(parsed) == DocumentType.IMAGING_REPORT


def test_router_unknown_does_not_match_ast_inside_contraste() -> None:
    """Régression : « ast » ne doit pas matcher dans « contraste » (sous-chaîne)."""
    router = KeywordHeuristicDocumentRouter()
    parsed = ParsedDocument(
        document_id="d1",
        patient_id="p1",
        study_id="s1",
        full_text=(
            "Compte rendu d'examen tomodensitométrique de l'abdomen. "
            "Injection de produit de contraste iodé. Pas de lésion focrale."
        ),
        document_type_hint=DocumentType.UNKNOWN,
        metadata={"pdf_parser": "docling"},
    )
    assert router.route(parsed) == DocumentType.IMAGING_REPORT
