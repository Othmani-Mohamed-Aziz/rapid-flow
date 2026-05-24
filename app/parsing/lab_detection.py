"""
Heuristique « document de laboratoire » — réduit les faux positifs sur CR imagerie / lettres.

Un score est calculé sur un extrait du texte ; un contexte « imagerie / examen morpho »
applique un seuil plus strict pour éviter d’activer le post-traitement labo sur une simple mention de CRP/AST.
"""

from __future__ import annotations

import re

# Contexte morpho / imagerie : présence → seuil labo plus élevé
_IMAGING_OR_MORPHO_CONTEXT: tuple[str, ...] = (
    "compte rendu d'imagerie",
    "compte-rendu d'imagerie",
    "compte rendu d'examen",
    "tomodensitom",
    "tomodensitomet",
    " tomodensitométrie",
    " irm ",
    " irl ",
    " tdm ",
    "scanner",
    "échographie",
    "echographie",
    "fibroscan",
    " mammographie",
    "tep-",
    "tep ",
    "tep/",
    "recist",
    "compte rendu anatomopathologique",
    "anatomopathologie",
)

# Indices forts de bilan biologique structuré
_STRONG_LAB: tuple[str, ...] = (
    "nfs",
    "hémogramme",
    "hemogram",
    "numération",
    "numeration",
    "formule leucocytaire",
    "ionogramme",
    "bilan lipidique",
    "bilan hépatique",
    "bilan hepatique",
    "bilan rénal",
    "bilan renal",
    "bilan ionique",
    "bilan complet",
    "hémostase",
    "hemostase",
    "cytométrie",
    "cytometrie",
    "hormonologie",
    "immunologie",
)

# Indices moyens
_MEDIUM_LAB: tuple[str, ...] = (
    "laboratoire",
    "biologie médicale",
    "biologie medicale",
    "prélèvement le",
    "prelevement le",
    "tube sec",
    "tube edta",
)

# Faibles : souvent cités dans un CR non labo
_WEAK_LAB: tuple[str, ...] = (
    "crp",
    "ast",
    "alt",
    "plaquette",
    "bilirubine",
    "bilirubin",
    "créatinine",
    "creatinine",
)

_VALUE_UNIT_HINT = re.compile(
    r"\d+(?:[.,]\d+)?\s*(g/l|mg/l|u/l|µmol/l|umol/l|ui/l|mmol/l|%)\b",
    re.I,
)

# Marqueurs courts : sous-chaîne seule → faux positifs (ex. « ast » dans « contraste »).
_WEAK_LAB_BOUNDARY = frozenset({"ast", "alt", "tp", "crp", "hb", "plt"})


def _weak_lab_keyword_present(keyword: str, s: str) -> bool:
    if keyword in _WEAK_LAB_BOUNDARY:
        return bool(re.search(rf"\b{re.escape(keyword)}\b", s))
    return keyword in s


def lab_document_score(sample: str) -> int:
    """Score entier ≥ 0 ; interprétation via seuils selon contexte."""
    s = sample[:18000].lower()
    score = 0
    for k in _STRONG_LAB:
        if k in s:
            score += 3
    for k in _MEDIUM_LAB:
        if k in s:
            score += 2
    if re.search(r"\bbiologie\b", s):
        score += 2
    weak_hits = sum(1 for k in _WEAK_LAB if _weak_lab_keyword_present(k, s))
    score += min(weak_hits, 3)
    if _VALUE_UNIT_HINT.search(s):
        score += 2
    # Ligne résultat typique (analyte + valeur + unité) sans tout le contexte « laboratoire »
    if _VALUE_UNIT_HINT.search(s) and any(
        _weak_lab_keyword_present(k, s) if k in _WEAK_LAB_BOUNDARY else k in s
        for k in (
            "ast",
            "alt",
            "plaquette",
            "crp",
            "creatinine",
            "créatinine",
            "hb ",
            "hémoglobine",
        )
    ):
        score += 2
    return score


def imaging_or_morpho_context(sample: str) -> bool:
    s = sample[:18000].lower()
    return any(k in s for k in _IMAGING_OR_MORPHO_CONTEXT)


def is_probable_lab_document(sample: str) -> bool:
    """
    True si le document ressemble à un bilan / résultats de laboratoire.

    Sous contexte imagerie ou AP, exige un score plus élevé pour limiter les faux positifs.
    """
    s = sample[:18000].lower()
    if not s.strip():
        return False

    score = lab_document_score(sample)
    if imaging_or_morpho_context(sample):
        return score >= 6

    # Hors contexte imagerie : seuil modéré (évite « CRP » seul dans une phrase)
    if "crp" in s and score < 4:
        if not any(
            x in s for x in ("nfs", "hémogram", "hemogram", "laboratoire", "bilan", "ionogramme")
        ):
            return False

    return score >= 4
