"""Chemins de données pour les tests d’intégration."""

from __future__ import annotations

import os
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _TESTS_DIR.parent

# Charge `.env` à la racine projet pour exposer `ECRF_CT_TEST_PDF_PATH` (ne casse pas si dotenv absent).
_ENV_FILE = PROJECT_ROOT / ".env"
if _ENV_FILE.is_file():
    try:
        from dotenv import load_dotenv

        load_dotenv(_ENV_FILE, override=False)
    except ImportError:
        pass

# Fichier de référence demandé pour les tests CR / imagerie (voir README projet).
_DEFAULT_CT_SCAN_PDF = PROJECT_ROOT / "data" / "ct_scan_report_liver.pdf"


def ct_scan_report_liver_pdf() -> Path:
    """Chemin absolu du PDF CT ; surcharge par variable d’environnement `ECRF_CT_TEST_PDF_PATH`."""
    override = os.environ.get("ECRF_CT_TEST_PDF_PATH", "").strip().strip('"').strip("'")
    return Path(override) if override else _DEFAULT_CT_SCAN_PDF


CT_SCAN_REPORT_LIVER_PDF = ct_scan_report_liver_pdf()
