from __future__ import annotations

from typing import Any

from app.config.settings import Settings
from app.parsing.errors import DoclingNotInstalledError
from app.parsing.pdf.base import PdfParseResult
from app.parsing.pdf.docling_parser import DoclingPdfParser
from app.parsing.pdf.factory import select_pdf_parser
from app.parsing.pdf.pypdf_parser import PypdfPdfParser
from app.schemas.models import DocumentSection, RawDocument


def _is_docling_recoverable_failure(exc: BaseException) -> bool:
    """
    Erreurs pour lesquelles on peut retomber sur pypdf sans casser le flux.

    Inclut réseau (ConnectError, timeout) et **Hugging Face** (401 token expiré,
    403 dépôt restreint, etc.) — HF renvoie parfois « Repository Not Found » avec un mauvais token.
    """
    e: BaseException | None = exc
    for _ in range(10):
        if e is None:
            break
        name = type(e).__name__.lower()
        if "connect" in name or "timeout" in name or name in ("gaierror",):
            return True
        mod = getattr(type(e), "__module__", "") or ""
        if "httpx" in mod and ("connect" in name or "timeout" in name):
            return True
        if "huggingface_hub" in mod and "http" in name:
            return True
        if "urllib" in mod and "error" in mod:
            return True
        e = e.__cause__ or e.__context__
    msg = str(exc).lower()
    if any(
        s in msg
        for s in (
            "connecterror",
            "connection error",
            "timed out",
            "401 client error",
            " 401 ",
            "403 client error",
            "unauthorized",
            "authentication",
            "access token",
            "invalid token",
            "token expired",
            "is expired",
            "gated repo",
        )
    ):
        return True
    return False


def parse_pdf_bytes_resilient(raw: RawDocument, settings: Settings) -> PdfParseResult:
    """
    Extraction PDF avec stratégie `auto` tolérante (Docling → pypdf).

    Pour `pdf_parser_backend=docling`, propage `DoclingNotInstalledError`.
    """
    if settings.pdf_parser_backend == "auto":
        docling = DoclingPdfParser()
        if docling.is_available():
            try:
                pr = docling.parse(raw)
                meta = {**pr.extra_metadata, "pdf_parser": "docling"}
                return pr.model_copy(update={"extra_metadata": meta})
            except Exception as exc:  # pragma: no cover - robustesse runtime Docling
                pypdf = PypdfPdfParser()
                pr_fb = pypdf.parse(raw)
                warn = f"docling_failed:{type(exc).__name__}:{exc}"
                meta = {
                    **pr_fb.extra_metadata,
                    "pdf_parser": "pypdf",
                    "warnings": [*pr_fb.extra_metadata.get("warnings", []), warn],
                }
                return pr_fb.model_copy(update={"extra_metadata": meta})
        pypdf = PypdfPdfParser()
        pr = pypdf.parse(raw)
        return pr.model_copy(
            update={"extra_metadata": {**pr.extra_metadata, "pdf_parser": "pypdf"}}
        )

    parser = select_pdf_parser(settings)
    try:
        pr = parser.parse(raw)
    except DoclingNotInstalledError:
        raise
    except BaseException as exc:
        if getattr(parser, "name", None) == "docling" and _is_docling_recoverable_failure(exc):
            pr_fb = PypdfPdfParser().parse(raw)
            warn = f"docling_recoverable_fallback:{type(exc).__name__}:{exc!s}"
            meta = {
                **pr_fb.extra_metadata,
                "pdf_parser": "pypdf",
                "warnings": [*pr_fb.extra_metadata.get("warnings", []), warn],
            }
            return pr_fb.model_copy(update={"extra_metadata": meta})
        raise
    return pr.model_copy(
        update={"extra_metadata": {**pr.extra_metadata, "pdf_parser": parser.name}}
    )


def ensure_non_empty_text(result: PdfParseResult, raw: RawDocument) -> PdfParseResult:
    """Si le texte est vide, retente une lecture bytes via pypdf (chemins exotiques)."""
    if result.full_text.strip():
        return result
    pypdf = PypdfPdfParser()
    try:
        second = pypdf.parse(raw)
    except Exception:
        return result
    if not second.full_text.strip():
        return result
    meta: dict[str, Any] = {
        **second.extra_metadata,
        "pdf_parser": "pypdf",
        "note": "fallback_after_empty_primary",
    }
    return second.model_copy(
        update={
            "extra_metadata": meta,
            "sections": second.sections
            or [DocumentSection(heading=None, body=second.full_text, metadata=meta)],
        }
    )
