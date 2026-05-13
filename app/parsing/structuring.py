from __future__ import annotations

import re

from app.schemas.enums import DocumentType
from app.schemas.models import DocumentSection, ParsedDocument

_LAB_HEADINGS = (
    "hématologie",
    "hematologie",
    "hemostase",
    "hémostase",
    "biochimie",
    "hormonologie",
    "cytologie",
    "immunologie",
)
_GENERIC_MEDICAL_HEADINGS = (
    "antécédents",
    "antecedents",
    "comorbidités",
    "comorbidites",
    "traitement",
    "traitements",
    "conclusion",
    "imagerie",
    "examen clinique",
    "plan",
    "impression",
)
_NOISE_PATTERNS = (
    re.compile(r"^page\s+\d+\b", re.I),
    re.compile(r"^--\s*\d+\s+of\s+\d+\s*--$", re.I),
    re.compile(r"^labo .*tel[:\s]", re.I),
)


def _normalize(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def _is_noise(line: str) -> bool:
    ln = _normalize(line)
    if not ln:
        return True
    return any(pat.search(ln) for pat in _NOISE_PATTERNS)


def _is_heading(line: str, doc_type: DocumentType) -> bool:
    ln = _normalize(line)
    if not ln or len(ln) > 90:
        return False
    lower = ln.lower().strip(":")
    if doc_type == DocumentType.LAB_BLOOD_PANEL:
        if any(h in lower for h in _LAB_HEADINGS):
            return True
    else:
        if any(h in lower for h in _GENERIC_MEDICAL_HEADINGS):
            return True
    if re.fullmatch(r"[A-ZÀ-Ÿ0-9 /()\-]{4,}", ln):
        return True
    return False


class DocumentStructuringService:
    """Construit des sections cohérentes pour bilans et rapports médicaux."""

    def build_sections(
        self,
        *,
        full_text: str,
        existing_sections: list[DocumentSection],
        doc_type: DocumentType,
    ) -> list[DocumentSection]:
        doclingish = bool(
            existing_sections
            and any(
                str((s.metadata or {}).get("source", "")).startswith("docling")
                for s in existing_sections
            )
        )
        cleaned = [self._clean_section(s) for s in existing_sections] if existing_sections else []
        cleaned = [s for s in cleaned if s.body.strip()]
        if doclingish and cleaned:
            return cleaned
        if existing_sections and len(existing_sections) > 1 and len(cleaned) >= 2:
            return cleaned
        return self._segment_from_text(full_text, doc_type)

    def enrich(self, parsed: ParsedDocument) -> ParsedDocument:
        sections = self.build_sections(
            full_text=parsed.full_text,
            existing_sections=parsed.structured_sections,
            doc_type=parsed.document_type_hint,
        )
        return parsed.model_copy(update={"structured_sections": sections})

    def _segment_from_text(self, text: str, doc_type: DocumentType) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        current_heading: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer, current_heading
            body = "\n".join(buffer).strip()
            if body:
                sections.append(
                    DocumentSection(
                        heading=current_heading,
                        body=body,
                        metadata={"source": "structured_segmenter"},
                    )
                )
            buffer = []

        for raw in text.splitlines():
            line = _normalize(raw)
            if _is_noise(line):
                continue
            if _is_heading(line, doc_type):
                flush()
                current_heading = line.strip(":")
                continue
            buffer.append(line)
        flush()
        if not sections:
            return [
                DocumentSection(
                    heading=None, body=text.strip(), metadata={"source": "fallback_full_text"}
                )
            ]
        return sections

    def _clean_section(self, sec: DocumentSection) -> DocumentSection:
        lines = [_normalize(ln) for ln in sec.body.splitlines() if not _is_noise(ln)]
        body = "\n".join(ln for ln in lines if ln)
        return sec.model_copy(update={"body": body})
