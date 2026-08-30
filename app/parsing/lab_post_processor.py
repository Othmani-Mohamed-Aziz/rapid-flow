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
_DOT_ROW = re.compile(r"(?P<name>[^\n|]{2,120}?)\.{3,}", re.I)
_LEADING_VALUE = re.compile(
    r"^\s*\|?\s*(?P<val>[<>]?\s*-?\d+(?:[.,]\d+)?)[ \t]*"
    r"(?P<unit>%|10\^\d+/mm3|/[A-Za-z0-9³]+|[A-Za-zµ][A-Za-z0-9µ/^.²³-]*)?",
    re.I,
)
_UNIT = r"%|10\^\d+/mm3|/[A-Za-z0-9³]+|[A-Za-zµ][A-Za-z0-9µ/^.²³-]*"
_INEQUALITY_VALUE = re.compile(
    r"^\s*\|?\s*(?P<val>(?:Inf(?!\.)\s*à|Sup(?!\.)\s*à|[<>≤≥])\s*-?\d+(?:[.,]\d+)?)"
    rf"[ \t]*(?P<unit>{_UNIT})?",
    re.I,
)
_TEXT_VALUE = re.compile(
    r"^\s*\|?\s*(?P<val>Tr[èe]s\s+nombreuses|Nombreuses|Quelques|Mod[ée]r[ée]es?|Rares?"
    r"|Absent(?:es|e|s)?|Pr[ée]sent(?:es|e|s)?)\b",
    re.I,
)
# Une plage de référence (« 3.4 à 20.5 », « Inf. à 8.6 ») n'est jamais un résultat.
# Le point après « Inf » distingue la plage du résultat qualitatif « Inf à 0.8 ».
_RANGE_START = re.compile(
    r"^\s*\|?\s*(?:Inf\.\s*à|Sup\.\s*à|-?\d+(?:[.,]\d+)?\s*à)\s*-?\d",
    re.I,
)
_SKIPPABLE_LINE = re.compile(
    r"^(?:#{1,6}\s|Analyses effectuées|Dossier validé|LABO |Page\s+\d|<!--)"
    r"|^(?:H[ée]matologie|H[ée]mostase|Biochimie|CytologieUrinaire|Hormonologie"
    r"|NumerationGlobulaire|FormuleLeucocytaire|NumerationPlaquettaire)$",
    re.I,
)
_VALUE_ONLY_LINE = re.compile(rf"^(?P<val>-?\d+(?:[.,]\d+)?)[ \t]*(?P<unit>{_UNIT})?$")
# Couples molaire → massique émis en doublon par les automates (61.9 umol/L puis 5.47 mg/L).
_DUAL_UNIT_PAIRS = {
    ("umol/l", "mg/l"),
    ("umol/l", "ug/l"),
    ("mmol/l", "g/l"),
    ("mmol/l", "mg/l"),
    ("nmol/l", "ug/l"),
}
_REFERENCE_PREFIX = re.compile(
    r"^.*(?:\bInf\.?\s*à\s*-?\d+(?:[.,]\d+)?|\b-?\d+(?:[.,]\d+)?\s*à\s*-?\d+(?:[.,]\d+)?)\s+",
    re.I,
)
_LEADING_RESULT = re.compile(
    r"^(?:[<>]?\s*-?\d+(?:[.,]\d+)?\s*"
    r"(?:%|10\^\d+/mm3|/[A-Za-z0-9³]+|[A-Za-zµ][A-Za-z0-9µ/^.²³-]*)?\s+)+",
    re.I,
)
_SECTION_LABELS = {
    "hematologie": "Hematologie",
    "hématologie": "Hematologie",
    "hemostase": "Hemostase",
    "hémostase": "Hemostase",
    "biochimie": "Biochimie",
    "cytologieurinaire": "CytologieUrinaire",
    "hormonologie": "Hormonologie",
}
_SECTION_RE = re.compile(
    r"\b(H[ée]matologie|H[ée]mostase|Biochimie|CytologieUrinaire|Hormonologie)\b",
    re.I,
)
_CAMEL_SUBSECTION_RE = re.compile(
    r"\b(NumerationGlobulaire|FormuleLeucocytaire|NumerationPlaquettaire)\b",
)

# Analytes dont la section ne peut pas être déduite du marqueur le plus proche : dans les
# tableaux Markdown, ils suivent souvent un bloc Biochimie sans en-tête intermédiaire.
_HORMONOLOGY_ANALYTES = {
    "25 oh vitamine d",
    "index homa",
    "insuline",
    "parathormone",
    "t s h ultra-sensible",
    "testosterone",
    "thyroxine libre (t4l)",
}

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


def _fold_name(raw_name: str) -> str:
    folded = unicodedata.normalize("NFD", raw_name.strip().casefold())
    folded = "".join(c for c in folded if unicodedata.category(c) != "Mn")
    return re.sub(r"[\s.]+", " ", folded).strip()


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


def _display_section(raw: str) -> str:
    return _SECTION_LABELS.get(raw.casefold(), raw)


def _context_markers(text: str) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    section_matches = list(_SECTION_RE.finditer(text))
    sections = [(match.start(), _display_section(match.group(1))) for match in section_matches]
    subsections: list[tuple[int, str]] = []
    for match in _CAMEL_SUBSECTION_RE.finditer(text):
        token = match.group(0)
        if _SECTION_RE.fullmatch(token):
            continue
        if any(section.start() <= match.start() for section in section_matches):
            subsections.append((match.start(), token))
    return sections, subsections


def _nearest_context(
    position: int,
    sections: list[tuple[int, str]],
    subsections: list[tuple[int, str]],
) -> tuple[str | None, str | None]:
    section_pos, section = max(
        ((pos, label) for pos, label in sections if pos <= position),
        default=(-1, None),
    )
    subsection = next(
        (label for pos, label in reversed(subsections) if section_pos < pos <= position),
        None,
    )
    return section, subsection or section


def _clean_dot_row_name(raw: str) -> str:
    candidate = raw.replace("\r", "\n").split("\n")[-1].split("|")[-1].strip(" #:-")
    candidate = re.sub(
        r"^.*\b(?:H[ée]matologie|H[ée]mostase|Biochimie|CytologieUrinaire|"
        r"Hormonologie|NumerationGlobulaire|FormuleLeucocytaire|"
        r"NumerationPlaquettaire)\b\s*",
        "",
        candidate,
        flags=re.I,
    )
    candidate = _REFERENCE_PREFIX.sub("", candidate)
    if not re.match(r"^25\s+OH\s+vitamine\s+D\b", candidate, re.I):
        candidate = _LEADING_RESULT.sub("", candidate)
    candidate = re.sub(r"^(?:Page\s+\d+\s+)+", "", candidate, flags=re.I)
    candidate = candidate.strip(" |#:-")
    if re.search(r"(?:[A-Za-zÀ-ÿ]\.)+[A-Za-zÀ-ÿ]$", candidate):
        candidate += "."
    return candidate


_ParsedValue = tuple[str | float | int, str | None, float]


def _normalize_inequality(raw: str) -> str:
    text = re.sub(r"\s+", " ", raw.strip())
    return re.sub(r"^(Inf|Sup)\s*à", lambda m: f"{m.group(1).capitalize()} à", text, flags=re.I)


def _parse_value(rest: str) -> _ParsedValue | None:
    """Premier résultat exploitable en tête de `rest` (jamais une plage de référence)."""
    if _RANGE_START.match(rest):
        return None
    numeric = _LEADING_VALUE.match(rest)
    if numeric is not None:
        unit = (numeric.group("unit") or "").strip() or None
        if unit and unit.casefold().startswith("inf"):
            unit = None
        return _to_number(numeric.group("val")), unit, 0.9 if unit else 0.82
    inequality = _INEQUALITY_VALUE.match(rest)
    if inequality is not None:
        unit = (inequality.group("unit") or "").strip() or None
        return _normalize_inequality(inequality.group("val")), unit, 0.85
    textual = _TEXT_VALUE.match(rest)
    if textual is not None:
        return textual.group("val"), None, 0.82
    return None


def _delayed_value(rest: str, *, budget: int = 1500) -> _ParsedValue | None:
    """Valeur repoussée après des tableaux d'aide (ex. Testostérone et ses normes)."""
    for raw in rest[:budget].splitlines():
        line = raw.strip()
        if not line or line.startswith("|") or _SKIPPABLE_LINE.match(line):
            continue
        if _RANGE_START.match(line):
            continue
        return _parse_value(line)
    return None


def _value_run(rest: str, expected: int) -> list[_ParsedValue]:
    """Colonne de résultats d'une mise en page « tous les noms puis toutes les valeurs »."""
    values: list[_ParsedValue] = []
    for raw in rest.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _RANGE_START.match(line):
            # Les plages de référence closent la colonne de résultats.
            if values:
                break
            continue
        numeric = _VALUE_ONLY_LINE.match(line)
        if numeric is not None:
            unit = (numeric.group("unit") or "").strip() or None
            values.append((_to_number(numeric.group("val")), unit, 0.85))
            continue
        textual = _TEXT_VALUE.fullmatch(line)
        if textual is not None:
            values.append((textual.group("val"), None, 0.8))
            continue
        inequality = _INEQUALITY_VALUE.fullmatch(line)
        if inequality is not None:
            unit = (inequality.group("unit") or "").strip() or None
            values.append((_normalize_inequality(inequality.group("val")), unit, 0.85))
    return _drop_alternate_units(values, len(values) - expected)


def _preceding_values(region: str, expected: int) -> list[_ParsedValue]:
    """Colonne de résultats placée *avant* ses libellés (réordonnancement Docling)."""
    values: list[_ParsedValue] = []
    for raw in region.splitlines():
        line = raw.strip()
        if not line or _RANGE_START.match(line):
            continue
        numeric = _VALUE_ONLY_LINE.match(line)
        if numeric is not None:
            unit = (numeric.group("unit") or "").strip() or None
            values.append((_to_number(numeric.group("val")), unit, 0.8))
            continue
        textual = _TEXT_VALUE.fullmatch(line)
        if textual is not None:
            values.append((textual.group("val"), None, 0.78))
    return values[-expected:]


def _drop_alternate_units(values: list[_ParsedValue], surplus: int) -> list[_ParsedValue]:
    """Retire juste assez de doublons d'unité pour réaligner la colonne sur les libellés."""
    if surplus <= 0:
        return values
    dropped: set[int] = set()
    for index in range(1, len(values)):
        if len(dropped) == surplus:
            break
        previous_unit = (values[index - 1][1] or "").casefold()
        unit = (values[index][1] or "").casefold()
        if index - 1 not in dropped and (previous_unit, unit) in _DUAL_UNIT_PAIRS:
            dropped.add(index)
    return [value for index, value in enumerate(values) if index not in dropped]


def _build_row(
    name: str,
    parsed: _ParsedValue,
    section: str | None,
    subsection: str | None,
) -> StructuredLabLine:
    value, unit, confidence = parsed
    canon = _normalize_name(name)
    rendered = f"{name} | {value}"
    if unit:
        rendered += f" {unit}"
    return StructuredLabLine(
        name=name,
        canonical_name=canon if canon != name else None,
        value=value,
        unit=unit,
        section=section,
        subsection=subsection,
        raw_text=rendered,
        confidence=confidence,
    )


def _named_markers(text: str) -> list[tuple[re.Match[str], str]]:
    """Marqueurs à libellé exploitable ; les faux points de suite (pieds de page) sont écartés."""
    named: list[tuple[re.Match[str], str]] = []
    for marker in _DOT_ROW.finditer(text):
        name = _clean_dot_row_name(marker.group("name"))
        if not name or len(name) > 80 or _NOISE_LINE.match(name):
            continue
        named.append((marker, name))
    return named


def _dot_rows(text: str) -> list[StructuredLabLine]:
    markers = _named_markers(text)
    sections, subsections = _context_markers(text)
    rows: list[StructuredLabLine] = []
    pending: list[tuple[str, str | None, str | None]] = []
    pending_start = 0
    flushed_end = 0
    for index, (marker, name) in enumerate(markers):
        if index + 1 < len(markers):
            next_marker = markers[index + 1][0]
            rest = text[marker.end() : next_marker.start()]
            # Sur une mise en page compacte, le résultat est capté par le libellé suivant.
            if "\n" not in rest:
                rest += next_marker.group("name")
        else:
            rest = text[marker.end() :]

        section, subsection = _nearest_context(marker.end(), sections, subsections)
        if _fold_name(name) in _HORMONOLOGY_ANALYTES:
            section = subsection = "Hormonologie"
        elif (
            section == "Hormonologie"
            and marker.group("name").strip() == name
            and "|" in text[max(0, marker.start() - 4) : marker.start() + 2]
        ):
            section = subsection = "Biochimie"

        if len(pending) >= 2:
            names = [*pending, (name, section, subsection)]
            columnar = list(zip(names, _value_run(rest, len(names))))
            if len(columnar) >= 2:
                rows.extend(_row_batch(columnar))
                pending = []
                flushed_end = marker.end()
                continue

        parsed = _parse_value(rest)
        if parsed is None and not pending:
            parsed = _delayed_value(rest)
        if parsed is None:
            if not pending:
                pending_start = marker.start()
            pending.append((name, section, subsection))
            continue

        # La série de libellés orphelins s'arrête ici : ses valeurs précédaient les libellés.
        rows.extend(_flush_preceding(pending, text[flushed_end:pending_start]))
        pending = []
        flushed_end = marker.end()
        rows.append(_build_row(name, parsed, section, subsection))

    rows.extend(_flush_preceding(pending, text[flushed_end:pending_start]))
    return rows


def _row_batch(
    columnar: list[tuple[tuple[str, str | None, str | None], _ParsedValue]],
) -> list[StructuredLabLine]:
    return [_build_row(entry[0], value, entry[1], entry[2]) for entry, value in columnar]


def _flush_preceding(
    pending: list[tuple[str, str | None, str | None]],
    region: str,
) -> list[StructuredLabLine]:
    if len(pending) < 2:
        return []
    values = _preceding_values(region, len(pending))
    return _row_batch(list(zip(pending, values))) if len(values) >= 2 else []


def _inline_rows(parsed: ParsedDocument) -> list[StructuredLabLine]:
    rows: list[StructuredLabLine] = []
    for heading, body in _iter_section_blocks(parsed):
        section_name = _display_section(heading) if heading else None
        if heading and _SECTION_RE.fullmatch(heading.strip()) is None:
            continue
        for raw in body.splitlines():
            line = raw.strip()
            if not line or _NOISE_LINE.match(line):
                continue
            inline = _INLINE_LINE.match(line)
            if inline is None:
                continue
            name = inline.group("name").strip()
            if name.lower().startswith("inf."):
                continue
            unit = (inline.group("unit") or "").strip() or None
            canon = _normalize_name(name)
            rows.append(
                StructuredLabLine(
                    name=name,
                    canonical_name=canon if canon != name else None,
                    value=_to_number(inline.group("val")),
                    unit=unit,
                    section=section_name,
                    subsection=section_name,
                    raw_text=line,
                    confidence=0.88 if unit else 0.78,
                )
            )
    return rows


class LabReportPostProcessor:
    """Extraction robuste de lignes labo depuis sections ou texte brut."""

    def enrich(self, parsed: ParsedDocument, *, force: bool = False) -> ParsedDocument:
        if parsed.structured_lab_lines and not force:
            return parsed
        source = parsed.full_text or "\n\n".join(body for _, body in _iter_section_blocks(parsed))
        lines_out = _dot_rows(source)
        seen = {(line.name.casefold(), str(line.value), line.unit) for line in lines_out}
        for line in _inline_rows(parsed):
            key = (line.name.casefold(), str(line.value), line.unit)
            if key not in seen:
                lines_out.append(line)
                seen.add(key)

        meta = dict(parsed.metadata)
        meta["lab_postprocessor"] = "LabReportPostProcessorV3"
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
