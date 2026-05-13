from __future__ import annotations

import re
from typing import Any

from app.parsing.pdf.base import BasePdfParser, PdfParseResult
from app.parsing.pdf.docling_source import resolve_docling_conversion_source
from app.schemas.models import DocumentSection, RawDocument


def _sections_from_markdown(markdown: str) -> list[DocumentSection]:
    """Découpe un Markdown Docling en sections (titres ##)."""
    parts = re.split(r"\n(?:##+\s+)", markdown)
    if len(parts) == 1:
        body = parts[0].strip()
        return [DocumentSection(heading=None, body=body, metadata={})] if body else []
    sections: list[DocumentSection] = []
    first = parts[0].strip()
    if first:
        sections.append(DocumentSection(heading=None, body=first, metadata={}))
    for block in parts[1:]:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n", 1)
        heading = lines[0].strip().strip("#").strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        sections.append(
            DocumentSection(
                heading=heading or None,
                body=body,
                metadata={"source": "docling_markdown_split"},
            )
        )
    return sections


def _clinical_export_label_set() -> set[Any]:
    """Labels Markdown exportés hors zones répétitives (entête/pied de page)."""
    from docling_core.types.doc.document import DEFAULT_EXPORT_LABELS
    from docling_core.types.doc.labels import DocItemLabel

    return set(DEFAULT_EXPORT_LABELS) - {
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PAGE_FOOTER,
    }


def _sections_from_docling_graph(document: Any) -> tuple[list[DocumentSection], list[str]]:
    """
    Segmente selon le graphe Docling (titres / section_header), tables en Markdown.

    Retourne (sections_cliniques, lignes_admin page_header/footer).
    """
    from docling_core.types.doc.document import (
        CodeItem,
        FormulaItem,
        PictureItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
        TitleItem,
    )
    from docling_core.types.doc.labels import DocItemLabel

    sections: list[DocumentSection] = []
    admin_lines: list[str] = []
    current_heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, current_heading
        body = "\n\n".join(b for b in buffer if b and b.strip()).strip()
        buffer = []
        if not body:
            current_heading = None
            return
        sections.append(
            DocumentSection(
                heading=current_heading,
                body=body,
                metadata={
                    "source": "docling_graph",
                    "docling_segmenter": "iterate_items",
                },
            )
        )
        current_heading = None

    for item, _level in document.iterate_items(with_groups=False):
        if isinstance(item, SectionHeaderItem):
            flush()
            current_heading = (item.text or "").strip() or None
        elif isinstance(item, TitleItem):
            flush()
            current_heading = (item.text or "").strip() or None
        elif isinstance(item, TextItem) and item.label in (
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.PAGE_FOOTER,
        ):
            t = (item.text or "").strip()
            if t:
                admin_lines.append(t)
        elif isinstance(item, TableItem):
            try:
                frag = item.export_to_markdown(doc=document).strip()
            except Exception:
                frag = ""
            if frag:
                buffer.append(frag)
        elif isinstance(item, PictureItem):
            try:
                frag = item.export_to_markdown(doc=document).strip()
            except Exception:
                frag = ""
            buffer.append(frag if frag else "<!-- figure -->")
        elif isinstance(item, (FormulaItem, CodeItem)):
            t = (getattr(item, "text", None) or "").strip()
            if t:
                buffer.append(t)
        elif isinstance(item, TextItem):
            t = (item.text or "").strip()
            if t:
                buffer.append(t)

    flush()

    if not sections:
        return [], admin_lines

    return sections, admin_lines


class DoclingPdfParser(BasePdfParser):
    """
    PDF via Docling : Markdown filtré (sans page header/footer) + sections graphe.

    Sortie alignée avec `DocumentStructuringService`, chunking par sections
    (`chunking_strategy=sections`) et métadonnées prêtes pour un vector store
    type LlamaIndex.
    """

    name = "docling"

    def is_available(self) -> bool:
        try:
            import docling  # noqa: F401
        except ImportError:
            return False
        return True

    def parse(self, raw: RawDocument) -> PdfParseResult:
        if not self.is_available():
            from app.parsing.errors import DoclingNotInstalledError

            raise DoclingNotInstalledError()

        from docling.document_converter import DocumentConverter
        from docling_core.types.doc.base import ImageRefMode

        converter = DocumentConverter()
        source = resolve_docling_conversion_source(raw)
        result = converter.convert(source)
        document = result.document

        clinical_labels = _clinical_export_label_set()
        full_text = document.export_to_markdown(
            labels=clinical_labels,
            image_mode=ImageRefMode.PLACEHOLDER,
        ).strip()

        graph_sections, admin_lines = _sections_from_docling_graph(document)
        sectioning = "graph"
        if len(graph_sections) < 2:
            md_sections = _sections_from_markdown(full_text)
            if len(md_sections) > len(graph_sections):
                graph_sections = md_sections
                sectioning = "markdown_fallback"

        if not graph_sections and full_text:
            graph_sections = [
                DocumentSection(
                    heading=None,
                    body=full_text,
                    metadata={"source": "docling_full_text_only"},
                )
            ]
            sectioning = "full_text_only"

        extra: dict[str, Any] = {
            "docling_status": getattr(result, "status", None),
            "docling_sectioning": sectioning,
            "docling_markdown_labels": "clinical_no_page_header_footer",
        }
        if admin_lines:
            extra["docling_admin_snippets"] = admin_lines

        return PdfParseResult(
            full_text=full_text,
            sections=graph_sections
            or [
                DocumentSection(
                    heading=None, body=full_text, metadata={"source": "docling_fallback"}
                )
            ],
            extra_metadata=extra,
        )
