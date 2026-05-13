"""Tests unitaires sur `text_enrichment` (dates, type de document)."""

from __future__ import annotations

from app.parsing.text_enrichment import extract_document_date, guess_document_type
from app.schemas.enums import DocumentType


def test_extract_document_date_prioritized_preleve() -> None:
    text = "Bruit\nPrélevé le 15/03/2024 à 08:00\n"
    assert str(extract_document_date(text)) == "2024-03-15"


def test_extract_document_date_iso_fallback() -> None:
    text = "Rapport du 2024-05-01 sans autre date prioritaire."
    assert str(extract_document_date(text)) == "2024-05-01"


def test_guess_document_type_lab_keywords() -> None:
    text = "Résultats NFS et plaquettes stables."
    assert guess_document_type(text, "application/pdf") == DocumentType.LAB_BLOOD_PANEL


def test_guess_document_type_imaging() -> None:
    text = "Compte rendu d'examen : IRM cérébrale sans injection."
    assert guess_document_type(text, "application/pdf") == DocumentType.IMAGING_REPORT


def test_guess_document_type_unknown() -> None:
    text = "Courrier administratif sans terme clinique spécifique."
    assert guess_document_type(text, "application/pdf") == DocumentType.UNKNOWN
