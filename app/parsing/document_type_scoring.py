"""
Classifieur documentaire multi-signaux (texte + structure), sans embedding.

Objectif : `document_type_hint` précis, robuste au bruit lexical, coût O(n) sur un
extrait borné (défaut 18k car.) + métadonnées de parse légères (Docling).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.parsing.lab_detection import imaging_or_morpho_context, lab_document_score
from app.schemas.enums import DocumentType
from app.schemas.models import DocumentSection

_DEFAULT_SAMPLE = 18_000
_MAX_LINES_STRUCT = 500

# Lignes type résultat biologique (réutilise la même idée que lab_detection).
_VALUE_UNIT_LINE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(g/l|mg/l|u/l|µmol/l|umol/l|ui/l|mmol/l|%)\b",
    re.I,
)

# Imagerie hors contexte « morpho » déjà couvert par imaging_or_morpho_context.
_IMAGING_EXTRA = (
    ("compte rendu de scanner", 0.28),
    ("compte rendu de tomodensitom", 0.28),
    ("compte rendu d'irm", 0.22),
    ("compte-rendu de scanner", 0.28),
    ("abdomen", 0.08),
    ("pelvien", 0.08),
    ("injection", 0.05),
    ("contraste", 0.05),
    ("reconstruction", 0.06),
    ("mm coronal", 0.07),
    ("mm sagittal", 0.07),
)

_PATH_STRONG = re.compile(
    r"\b(biopsie|biopsies|lame\(s\)?|coloration|immunohistochimi\w*|ihc\b|gleason|"
    r"anatomopatholog\w*|histolog\w*|cytoponction|cytologie\s+asc|reseau\s+nestin|"
    r"who\s+20\d\d)\b",
    re.I,
)

_PATH_CONTEXT = re.compile(
    r"\b(compte\s+rendu\s+anatomopatholog\w*|rapport\s+d['']?\s*anatomopatholog\w*|"
    r"examen\s+anatomopatholog\w*)\b",
    re.I,
)

_LETTER = re.compile(
    r"\b(antécédents?|antecedents?|comorbidit|comorbidités|traitement\s+en\s+cours|"
    r"cher\s+confrère|cher\s+consoeur|cordialement|votre\s+patiente?|"
    r"médecin\s+traitant|medecin\s+traitant)\b",
    re.I,
)


@dataclass(frozen=True)
class DocumentTypeClassification:
    """Résultat du classifieur : type retenu + scores / explications."""

    document_type: DocumentType
    scores: dict[str, float]
    features: dict[str, Any]
    rationale: list[str] = field(default_factory=list)

    def as_metadata(self) -> dict[str, Any]:
        """Champs prêts à fusionner dans `ParsedDocument.metadata`."""
        return {
            "document_type_scores": dict(self.scores),
            "document_type_features": dict(self.features),
            "document_type_rationale": list(self.rationale),
        }


def _sample(text: str, max_chars: int) -> str:
    return (text or "")[:max_chars]


def _structural_line_stats(bodies_text: str) -> dict[str, float]:
    """Statistiques sur les corps de sections (tables Markdown Docling, lignes courtes, etc.)."""
    lines = [ln for ln in bodies_text.splitlines() if ln.strip()][:_MAX_LINES_STRUCT]
    n = len(lines)
    if not n:
        return {
            "struct_n_lines": 0,
            "struct_pipe_line_ratio": 0.0,
            "struct_value_unit_line_ratio": 0.0,
            "struct_avg_line_len": 0.0,
        }
    pipe_hits = sum(1 for ln in lines if ln.count("|") >= 2)
    vu_hits = sum(1 for ln in lines if _VALUE_UNIT_LINE.search(ln))
    lengths = [len(ln) for ln in lines]
    return {
        "struct_n_lines": float(n),
        "struct_pipe_line_ratio": pipe_hits / n,
        "struct_value_unit_line_ratio": vu_hits / n,
        "struct_avg_line_len": sum(lengths) / n,
    }


def _docling_layout_from_meta(meta: Mapping[str, Any] | None) -> dict[str, float]:
    meta = meta or {}
    if meta.get("docling_sectioning") == "markdown_fallback":
        return {
            "layout_n_table_items": 0.0,
            "layout_n_text_items": 0.0,
            "layout_n_picture_items": 0.0,
            "layout_n_heading_events": 0.0,
        }
    raw = (meta or {}).get("docling_layout_stats") or {}
    if not isinstance(raw, dict):
        return {
            "layout_n_table_items": 0.0,
            "layout_n_text_items": 0.0,
            "layout_n_picture_items": 0.0,
            "layout_n_heading_events": 0.0,
        }
    return {
        "layout_n_table_items": float(raw.get("n_table_items", 0) or 0),
        "layout_n_text_items": float(raw.get("n_text_items", 0) or 0),
        "layout_n_picture_items": float(raw.get("n_picture_items", 0) or 0),
        "layout_n_heading_events": float(raw.get("n_heading_events", 0) or 0),
    }


def _bodies_from_sections(sections: Sequence[DocumentSection]) -> str:
    parts = [(s.body or "").strip() for s in sections if (s.body or "").strip()]
    return "\n".join(parts)


def _pathology_scores(sample_lower: str) -> tuple[float, float]:
    """(score contexte AP, score marqueurs forts)."""
    ctx = 0.75 if _PATH_CONTEXT.search(sample_lower) else 0.0
    n_strong = len(_PATH_STRONG.findall(sample_lower))
    strong = min(1.0, 0.28 * n_strong)
    return ctx, strong


def _letter_score(sample_lower: str) -> float:
    return min(1.0, 0.38 * len(_LETTER.findall(sample_lower)))


def _imaging_keyword_bonus(lower: str) -> float:
    acc = 0.0
    for needle, w in _IMAGING_EXTRA:
        if needle in lower:
            acc += w
    return min(1.0, acc)


def classify_document_type(
    text: str,
    sections: Sequence[DocumentSection],
    parse_metadata: Mapping[str, Any] | None = None,
    *,
    max_chars: int = _DEFAULT_SAMPLE,
) -> DocumentTypeClassification:
    """
    Calcule des scores continus par type puis choisit `document_type` avec seuils de marge.

    - **Texte** : `lab_document_score`, contexte imagerie, motifs path / lettre.
    - **Structure** : densité de pipes (tables MD), lignes valeur+unité, stats graphe Docling.
    """
    sample = _sample(text, max_chars)
    lower = sample.lower()
    meta = parse_metadata or {}

    lab_score = lab_document_score(sample)
    img_ctx = imaging_or_morpho_context(sample)

    bodies = _bodies_from_sections(sections)
    struct_lines = _structural_line_stats(bodies if bodies.strip() else sample)
    layout = _docling_layout_from_meta(meta)

    pipe_r = struct_lines["struct_pipe_line_ratio"]
    vu_r = struct_lines["struct_value_unit_line_ratio"]
    avg_len = struct_lines["struct_avg_line_len"]
    n_lines = int(struct_lines["struct_n_lines"])

    n_tab = layout["layout_n_table_items"]
    n_head = layout["layout_n_heading_events"]

    # --- Composantes 0..1 -------------------------------------------------
    lab_text = min(1.0, lab_score / 6.25)
    lab_struct = min(
        1.0,
        0.42 * pipe_r + 0.38 * min(1.0, vu_r * 12.0) + 0.12 * min(1.0, n_tab / 5.0) + 0.08 * vu_r,
    )
    lab_raw = 0.52 * lab_text + 0.35 * lab_struct + 0.13 * min(1.0, (pipe_r + vu_r) / 1.2)

    img_base = 0.86 if img_ctx else min(1.0, 0.22 + _imaging_keyword_bonus(lower))
    narrative = min(
        1.0,
        0.28 * min(1.0, avg_len / 95.0)
        + 0.32 * (1.0 - pipe_r)
        + 0.22 * min(1.0, n_head / 5.0)
        + 0.18 * min(1.0, n_lines / 80.0),
    )
    img_raw = min(1.0, img_base * (0.62 + 0.38 * narrative))

    path_ctx, path_strong = _pathology_scores(lower)
    path_raw = min(1.0, max(path_ctx, path_strong * 1.15, 0.55 * (path_ctx + path_strong)))

    letter_raw = _letter_score(lower)

    rationale: list[str] = [
        f"lab_score_int={lab_score}",
        f"img_ctx={img_ctx}",
        f"pipe_r={pipe_r:.3f}",
        f"vu_r={vu_r:.3f}",
        f"n_tables={int(n_tab)}",
        f"n_headings={int(n_head)}",
    ]

    # --- Contexte imagerie : abaisse le lab sauf preuve tabulaire / labo forte ---
    if img_ctx:
        if lab_raw < 0.58:
            damp = 0.38
            rationale.append(f"lab_dampen_imaging_ctx={damp}")
            lab_raw *= damp
        else:
            damp = 0.82
            rationale.append(f"lab_dampen_imaging_ctx_soft={damp}")
            lab_raw *= damp

    # --- Pathologie vs imagerie (AP : même bucket « morpho » que l'imagerie) ---
    path_specific = max(path_ctx, path_strong)
    if path_ctx >= 0.7:
        img_raw *= 0.52
        rationale.append("pathology_ctx_strong_vs_imaging")
    elif path_specific >= 0.42 and path_raw >= img_raw - 0.14 and lab_raw < 0.62:
        img_raw *= 0.72
        rationale.append("pathology_tiebreak_vs_imaging")

    scores: dict[str, float] = {
        DocumentType.LAB_BLOOD_PANEL.value: round(lab_raw, 4),
        DocumentType.IMAGING_REPORT.value: round(img_raw, 4),
        DocumentType.PATHOLOGY_REPORT.value: round(path_raw, 4),
        DocumentType.CLINICAL_LETTER.value: round(letter_raw, 4),
    }

    typed = (
        DocumentType.LAB_BLOOD_PANEL,
        DocumentType.IMAGING_REPORT,
        DocumentType.PATHOLOGY_REPORT,
        DocumentType.CLINICAL_LETTER,
    )
    best_type = max(typed, key=lambda t: scores[t.value])
    best_val = scores[best_type.value]
    others = sorted((scores[t.value] for t in typed if t != best_type), reverse=True)
    second_val = others[0] if others else 0.0
    margin = best_val - second_val

    tau_win = 0.33
    tau_margin = 0.055
    winner: DocumentType
    if best_val < tau_win or margin < tau_margin:
        winner = DocumentType.UNKNOWN
        rationale.append(f"low_confidence(best={best_val:.3f},margin={margin:.3f})")
    else:
        winner = best_type
        rationale.append(f"winner={winner.value}(score={best_val:.3f},margin={margin:.3f})")

    features: dict[str, Any] = {
        "lab_document_score": lab_score,
        "imaging_or_morpho_context": img_ctx,
        **{k: float(v) if isinstance(v, (int, float)) else v for k, v in struct_lines.items()},
        **layout,
        "score_lab": scores[DocumentType.LAB_BLOOD_PANEL.value],
        "score_imaging": scores[DocumentType.IMAGING_REPORT.value],
        "score_pathology": scores[DocumentType.PATHOLOGY_REPORT.value],
        "score_clinical_letter": scores[DocumentType.CLINICAL_LETTER.value],
        "classifier": "document_type_scoring_v1",
    }

    return DocumentTypeClassification(
        document_type=winner,
        scores=scores,
        features=features,
        rationale=rationale,
    )
