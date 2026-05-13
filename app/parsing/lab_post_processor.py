from __future__ import annotations

import re
import unicodedata

from app.schemas.models import DocumentSection, ParsedDocument, StructuredLabLine

_ANALYTE_LINE = re.compile(r"^(?P<name>[A-Za-zÀ-ÿ][A-Za-z0-9À-ÿ.%/°\-\(\) ]{2,70}?)(?:\.+)?\s*$")
_VALUE_UNIT_LINE = re.compile(
    r"^(?P<val>[<>]?\s*[0-9]+(?:[.,][0-9]+)?)\s*(?P<unit>UI/?L|U/?L|µ?mol/?L|umol/?L|mg/dL|mg/L|g/dL|g/L|G/L|fL|pg|mmol/L|mUI/L|ng/mL|%)?$",
    re.I,
)
_INLINE_LINE = re.compile(
    r"^(?P<name>[A-Za-zÀ-ÿ][A-Za-z0-9À-ÿ.%/°\-\(\) ]{2,60}?)\s*[:=]\s*(?P<val>[<>]?\s*[0-9]+(?:[.,][0-9]+)?)\s*(?P<unit>[A-Za-zµ/%0-9\^./-]+)?$",
    re.I,
)
_NOISE_LINE = re.compile(r"^(page\s+\d+|inf\.\s*à\s+\d|labo .*tel|--\s*\d+\s+of\s+\d+\s*--)$", re.I)

_CANONICAL = {
    "asat": "AST",
    "ast": "AST",
    "alat": "ALT",
    "alt": "ALT",
    "plaquettes": "PLT",
    "plt": "PLT",
    "crp": "CRP",
    "inr": "INR",
    "bilirubine totale": "Total_bilirubine",
    "hemoglobine": "Hb",
    "hémoglobine": "Hb",
}


def _normalize_name(raw_name: str) -> str:
    s = raw_name.strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return _CANONICAL.get(s, raw_name.strip())


def _to_number(raw: str) -> float | int:
    v = float(raw.replace(",", ".").replace(" ", "").lstrip("<>").strip())
    return int(v) if v.is_integer() else v


def _iter_section_blocks(parsed: ParsedDocument) -> list[tuple[str | None, str]]:
    if parsed.structured_sections:
        out: list[tuple[str | None, str]] = []
        for sec in parsed.structured_sections:
            body = sec.body.strip()
            if body:
                out.append((sec.heading, body))
        if out:
            return out
    return [(None, parsed.full_text)]


class LabReportPostProcessor:
    """Extraction robuste de lignes labo depuis sections ou texte brut."""

    def enrich(self, parsed: ParsedDocument, *, force: bool = False) -> ParsedDocument:
        if parsed.structured_lab_lines and not force:
            return parsed
        lines_out: list[StructuredLabLine] = []
        for heading, body in _iter_section_blocks(parsed):
            section_name = heading
            pending_analyte: str | None = None
            for raw in body.splitlines():
                line = raw.strip()
                if not line:
                    continue
                if _NOISE_LINE.match(line):
                    continue
                if line.endswith(":") and len(line) < 80 and not any(c.isdigit() for c in line):
                    section_name = line.rstrip(":").strip()
                    pending_analyte = None
                    continue
                inline = _INLINE_LINE.match(line)
                if inline:
                    name = inline.group("name").strip()
                    if name.lower().startswith("inf."):
                        continue
                    val = _to_number(inline.group("val"))
                    unit = (inline.group("unit") or "").strip() or None
                    canon = _normalize_name(name)
                    lines_out.append(
                        StructuredLabLine(
                            name=name,
                            canonical_name=canon if canon != name else None,
                            value=val,
                            unit=unit,
                            section=section_name,
                            raw_text=line,
                            confidence=0.88 if unit else 0.78,
                        )
                    )
                    pending_analyte = None
                    continue
                analyte_match = _ANALYTE_LINE.match(line)
                if analyte_match and not any(ch.isdigit() for ch in line):
                    pending_analyte = analyte_match.group("name").strip(". ").strip()
                    continue
                if pending_analyte:
                    vu = _VALUE_UNIT_LINE.match(line)
                    if vu:
                        val = _to_number(vu.group("val"))
                        unit = (vu.group("unit") or "").strip() or None
                        canon = _normalize_name(pending_analyte)
                        lines_out.append(
                            StructuredLabLine(
                                name=pending_analyte,
                                canonical_name=canon if canon != pending_analyte else None,
                                value=val,
                                unit=unit,
                                section=section_name,
                                raw_text=f"{pending_analyte} | {line}",
                                confidence=0.84 if unit else 0.74,
                            )
                        )
                    pending_analyte = None

        meta = dict(parsed.metadata)
        meta["lab_postprocessor"] = "LabReportPostProcessorV2"
        meta["structured_lab_line_count"] = len(lines_out)
        return parsed.model_copy(update={"structured_lab_lines": lines_out, "metadata": meta})

    def sections_from_lab_lines(self, lines: list[StructuredLabLine]) -> list[DocumentSection]:
        buckets: dict[str | None, list[str]] = {}
        for ln in lines:
            buckets.setdefault(ln.section, []).append(ln.raw_text)
        return [
            DocumentSection(heading=k, body="\n".join(v), metadata={"kind": "lab_aggregate"})
            for k, v in buckets.items()
        ]
