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


def _sections_from_docling_graph(
    document: Any,
) -> tuple[list[DocumentSection], list[str], dict[str, int]]:
    """
    Segmente selon le graphe Docling (titres / section_header), tables en Markdown.

    Retourne (sections_cliniques, lignes_admin page_header/footer, stats layout légères).
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
    layout: dict[str, int] = {
        "n_table_items": 0,
        "n_text_items": 0,
        "n_picture_items": 0,
        "n_heading_events": 0,
    }
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
            layout["n_heading_events"] += 1
        elif isinstance(item, TitleItem):
            flush()
            current_heading = (item.text or "").strip() or None
            layout["n_heading_events"] += 1
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
                layout["n_table_items"] += 1
        elif isinstance(item, PictureItem):
            layout["n_picture_items"] += 1
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
                layout["n_text_items"] += 1

    flush()

    if not sections:
        return [], admin_lines, layout

    return sections, admin_lines, layout


_INSTITUTION_HEADER_RE = re.compile(
    r"(?is)\b(centre|clinique|cabinet|groupe|institut|h[oô]pital|polyclinique)\b"
    r".{0,96}\b(imagerie|radiologie)\b"
    r"|\b(imagerie|radiologie)\b.{0,48}\b(medicale|médicale)\b"
)


def _looks_like_radiology_dept_stamp(text: str) -> bool:
    t = re.sub(r"\s+", " ", text.strip().lower())
    if len(t) > 120:
        return False
    return "service" in t and "radiologie" in t


def _annotate_imaging_admin_sections(sections: list[DocumentSection]) -> list[DocumentSection]:
    """
    Docling ne classe pas toujours l'en-tête établissement en PAGE_HEADER.

    Première section courte « centre … imagerie » + tampon « service de radiologie »
    → `metadata.content_kind=admin_section` pour chunking / retrieval.
    """
    if not sections:
        return sections
    out: list[DocumentSection] = []
    for i, sec in enumerate(sections):
        md = dict(sec.metadata or {})
        heading = (sec.heading or "").strip()
        body = (sec.body or "").strip()
        combined = f"{heading}\n{body}".strip()

        is_admin = False
        if i == 0 and len(combined) <= 280:
            if heading and _INSTITUTION_HEADER_RE.search(heading):
                is_admin = True
            elif _INSTITUTION_HEADER_RE.search(combined):
                is_admin = True
            elif _looks_like_radiology_dept_stamp(body) and (
                bool(heading and _INSTITUTION_HEADER_RE.search(heading)) or len(heading) < 3
            ):
                is_admin = True

        if is_admin:
            md["content_kind"] = "admin_section"
        else:
            md.setdefault("content_kind", "clinical_section")

        out.append(sec.model_copy(update={"metadata": md}))
    return out


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

        graph_sections, admin_lines, layout_stats = _sections_from_docling_graph(document)
        sectioning = "graph"
        if len(graph_sections) < 2:
            md_sections = _sections_from_markdown(full_text)
            if len(md_sections) > len(graph_sections):
                graph_sections = md_sections
                sectioning = "markdown_fallback"
                # Stats graphe ≠ sections Markdown : ne pas nourrir le classifieur avec un signal incohérent.
                layout_stats = {
                    "n_table_items": 0,
                    "n_text_items": 0,
                    "n_picture_items": 0,
                    "n_heading_events": 0,
                }

        if not graph_sections and full_text:
            graph_sections = [
                DocumentSection(
                    heading=None,
                    body=full_text,
                    metadata={"source": "docling_full_text_only"},
                )
            ]
            sectioning = "full_text_only"

        sections_final = graph_sections or [
            DocumentSection(heading=None, body=full_text, metadata={"source": "docling_fallback"})
        ]
        sections_final = _annotate_imaging_admin_sections(sections_final)

        extra: dict[str, Any] = {
            "docling_status": getattr(result, "status", None),
            "docling_sectioning": sectioning,
            "docling_markdown_labels": "clinical_no_page_header_footer",
            "docling_layout_stats": layout_stats,
        }
        if admin_lines:
            extra["docling_admin_snippets"] = admin_lines

        return PdfParseResult(
            full_text=full_text,
            sections=sections_final,
            extra_metadata=extra,
        )
