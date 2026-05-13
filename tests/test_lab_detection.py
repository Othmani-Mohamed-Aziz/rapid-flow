from __future__ import annotations

from app.parsing.lab_detection import is_probable_lab_document, lab_document_score


def test_lab_detection_true_on_typical_blood_panel() -> None:
    text = "Biologie\nAST : 40 U/L\nPlaquettes : 155 G/L\n"
    assert is_probable_lab_document(text) is True
    assert lab_document_score(text) >= 4


def test_lab_detection_false_imaging_with_isolated_crp() -> None:
    text = (
        "Compte rendu d'imagerie — TDM abdominale\n"
        "Discussion : suivi post-chirurgical. CRP élevée notée en biologie antérieure.\n"
        "Pas de collection.\n"
    )
    assert is_probable_lab_document(text) is False


def test_lab_detection_score_zero_on_empty() -> None:
    assert lab_document_score("") == 0
    assert is_probable_lab_document("") is False


def test_lab_detection_true_when_strong_lab_inside_imaging_report() -> None:
    text = "Compte rendu d'imagerie\nAnnexe laboratoire : NFS du jour.\nLeucocytes 12 G/L.\n"
    assert is_probable_lab_document(text) is True
