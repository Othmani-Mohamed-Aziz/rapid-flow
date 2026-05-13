"""Génération de PDF réels (fichiers binaires valides) pour les tests d’intégration."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def reportlab_available() -> bool:
    return importlib.util.find_spec("reportlab") is not None


def write_minimal_lab_pdf(path: Path) -> None:
    """
    Écrit un PDF minimal avec texte extractible (vectoriel).

    Nécessite `reportlab` (extra dev).
    """
    if not reportlab_available():
        msg = 'reportlab est requis : pip install -e ".[dev]"'
        raise RuntimeError(msg)

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    _width, height = A4
    y = height - 72
    c.setFont("Helvetica", 11)
    lines = [
        "Bilan biologique - Date prelevement: 2026-01-15",
        "Hematologie",
        "Plaquettes : 155 G/L",
        "Biochimie",
        "AST : 48 U/L",
        "ALT : 31 U/L",
        "CRP : 6 mg/L",
    ]
    for line in lines:
        c.drawString(72, y, line)
        y -= 18
    c.save()
