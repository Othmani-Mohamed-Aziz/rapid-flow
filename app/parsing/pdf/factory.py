from __future__ import annotations

from app.config.settings import Settings
from app.parsing.errors import DoclingNotInstalledError
from app.parsing.pdf.base import BasePdfParser
from app.parsing.pdf.docling_parser import DoclingPdfParser
from app.parsing.pdf.pypdf_parser import PypdfPdfParser


def select_pdf_parser(settings: Settings) -> BasePdfParser:
    """
    Retourne le parseur PDF effectif selon la configuration.

    - `docling` : impose Docling (lève `DoclingNotInstalledError` si absent).
    - `pypdf` : impose pypdf.
    - `auto` : Docling si disponible, sinon pypdf.
    """
    docling = DoclingPdfParser()
    pypdf = PypdfPdfParser()

    if settings.pdf_parser_backend == "docling":
        if not docling.is_available():
            raise DoclingNotInstalledError()
        return docling
    if settings.pdf_parser_backend == "pypdf":
        if not pypdf.is_available():
            msg = "pypdf n’est pas installé (dépendance principale PDF texte)."
            raise ImportError(msg)
        return pypdf
    # auto
    if docling.is_available():
        return docling
    if not pypdf.is_available():
        msg = "Aucun parseur PDF disponible : installez pypdf ou docling."
        raise ImportError(msg)
    return pypdf
