from __future__ import annotations

import hashlib
import re
import uuid
from abc import ABC, abstractmethod

from app.config.settings import Settings
from app.schemas.enums import FieldFamily
from app.schemas.models import DocumentChunk, DocumentSection, ParsedDocument

_SECTION_CHUNK_RESERVED_METADATA: frozenset[str] = frozenset(
    {
        "chunker",
        "content_kind",
        "section_heading",
        "section_index",
        "part_index",
        "char_spans_verified",
    }
)


class ChunkingService(ABC):
    """Découpe un document parsé en chunks indexables."""

    @abstractmethod
    def chunk(self, parsed: ParsedDocument) -> list[DocumentChunk]:
        """Retourne la liste ordonnée des chunks."""


class HeuristicLabChunkingService(ChunkingService):
    """
    Chunking par paragraphes + heuristique de famille (labo).

    Utilisé aussi comme repli générique (texte non PDF ou PDF sans `chunking_strategy=sections`).
    """

    _FAMILY_KEYWORDS: dict[FieldFamily, tuple[str, ...]] = {
        FieldFamily.HEMATOLOGY: (
            "hb",
            "hémoglobine",
            "hemoglobine",
            "ht",
            "hématocrite",
            "hematocrite",
            "leucocyte",
            "plaquette",
            "plt",
            "nfs",
            "gb",
        ),
        FieldFamily.COAGULATION: (
            "inr",
            "tp",
            "tca",
            "fibrinogène",
            "fibrinogene",
            "d-dimère",
            "ddimer",
        ),
        FieldFamily.HEPATIC_BIOCHEMISTRY: (
            "ast",
            "alt",
            "ggt",
            "pal",
            "bilirubin",
            "bilirubine",
            "albumine",
            "dfh",
        ),
        FieldFamily.INFLAMMATION_BIOMARKERS: ("crp", "vs", "vsp", "pcr", "afp", "ca19", "ca 19"),
    }

    def chunk(self, parsed: ParsedDocument) -> list[DocumentChunk]:
        parts = [p.strip() for p in re.split(r"\n\s*\n+", parsed.full_text) if p.strip()]
        if len(parts) <= 1:
            parts = [p.strip() for p in parsed.full_text.splitlines() if p.strip()]
        chunks: list[DocumentChunk] = []
        offset = 0
        for part in parts:
            fam = self._infer_family(part)
            start: int | None = parsed.full_text.find(part, offset)
            end: int | None = start + len(part) if start is not None and start >= 0 else None
            if start is not None and start < 0:
                start = None
            cid = self._stable_chunk_id(parsed.document_id, part)
            chunks.append(
                DocumentChunk(
                    chunk_id=cid,
                    document_id=parsed.document_id,
                    text=part,
                    field_family=fam,
                    char_start=start,
                    char_end=end,
                    metadata={"chunker": "HeuristicLabChunkingService"},
                )
            )
            offset = end or offset
        if not chunks:
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=parsed.document_id,
                    text=parsed.full_text,
                    field_family=None,
                    metadata={"chunker": "HeuristicLabChunkingService", "fallback": True},
                )
            )
        return chunks

    def _infer_family(self, text: str) -> FieldFamily | None:
        lower = text.lower()
        scores: dict[FieldFamily, int] = {fam: 0 for fam in self._FAMILY_KEYWORDS}
        for fam, kws in self._FAMILY_KEYWORDS.items():
            scores[fam] = sum(1 for kw in kws if kw in lower)
        best = max(scores.items(), key=lambda kv: kv[1])
        return best[0] if best[1] > 0 else None

    @staticmethod
    def _stable_chunk_id(document_id: str, text: str) -> str:
        h = hashlib.sha256(f"{document_id}:{text}".encode("utf-8")).hexdigest()[:16]
        return f"chk_{h}"


class LabRowChunkingService(ChunkingService):
    """Keep reconstructed analyte labels and values together, grouped by subsection."""

    def __init__(self) -> None:
        self._family_inference = HeuristicLabChunkingService()

    def chunk(self, parsed: ParsedDocument) -> list[DocumentChunk]:
        grouped: dict[tuple[str | None, str | None], list[str]] = {}
        for line in parsed.structured_lab_lines:
            grouped.setdefault((line.section, line.subsection), []).append(line.raw_text)
        chunks: list[DocumentChunk] = []
        for index, ((section, subsection), rows) in enumerate(grouped.items()):
            text = "\n".join(rows)
            chunk_id = HeuristicLabChunkingService._stable_chunk_id(
                f"{parsed.document_id}:lab_rows:{index}",
                text,
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=parsed.document_id,
                    text=text,
                    field_family=self._family_inference._infer_family(text),
                    metadata={
                        "chunker": "LabRowChunkingService",
                        "content_kind": "structured_lab_rows",
                        "section_heading": section,
                        "subsection_heading": subsection,
                        "row_count": len(rows),
                        "char_spans_verified": False,
                    },
                )
            )
        return chunks or self._family_inference.chunk(parsed)


def _split_oversized_body(text: str, max_chars: int) -> list[str]:
    """Découpe un corps de section long sans couper au milieu d’un paragraphe si possible."""
    if len(text) <= max_chars:
        return [text]
    parts = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(parts) <= 1:
        chunks: list[str] = []
        cur = ""
        for line in text.splitlines():
            if len(cur) + len(line) + 1 > max_chars and cur:
                chunks.append(cur.strip())
                cur = line
            else:
                cur = (cur + "\n" + line).strip() if cur else line
        if cur.strip():
            chunks.append(cur.strip())
        return chunks or [text[:max_chars]]

    out: list[str] = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}".strip() if buf else p
        else:
            if buf:
                out.append(buf)
            if len(p) > max_chars:
                out.extend(_split_oversized_body(p, max_chars))
                buf = ""
            else:
                buf = p
    if buf:
        out.append(buf)
    return out


class ImagingReportChunkingService(ChunkingService):
    """Keep short imaging reports intact so related RECIST evidence stays together."""

    def __init__(self, *, max_chars: int = 12000) -> None:
        self._max_chars = max_chars

    def chunk(self, parsed: ParsedDocument) -> list[DocumentChunk]:
        full = (parsed.full_text or "").strip()
        if not full:
            return []
        parts = _split_oversized_body(full, self._max_chars)
        chunks: list[DocumentChunk] = []
        offset = 0
        for index, part in enumerate(parts):
            start = full.find(part, offset)
            end = start + len(part) if start >= 0 else None
            chunk_id = HeuristicLabChunkingService._stable_chunk_id(
                f"{parsed.document_id}:imaging:{index}",
                part,
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=parsed.document_id,
                    text=part,
                    field_family=FieldFamily.IMAGING_RECIST,
                    char_start=start if start >= 0 else None,
                    char_end=end,
                    metadata={
                        "chunker": "ImagingReportChunkingService",
                        "content_kind": "imaging_report",
                        "part_index": index,
                        "char_spans_verified": start >= 0,
                    },
                )
            )
            if end is not None:
                offset = end
        return chunks


class SectionBasedChunkingService(ChunkingService):
    """
    Un chunk par fragment de section (compte rendu, lettre) avec métadonnées stables.

    Les offsets `char_start` / `char_end` sont relatifs à `parsed.full_text` lorsque
    celui-ci correspond à la concaténation des sections (voir `SmartParsingService`).
    """

    def __init__(self, *, max_section_chars: int = 12000) -> None:
        self._max_section_chars = max_section_chars

    def chunk(self, parsed: ParsedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        full = parsed.full_text or ""

        active: list[tuple[int, DocumentSection]] = [
            (i, s) for i, s in enumerate(parsed.structured_sections) if (s.body or "").strip()
        ]
        bodies = [(s.body or "").strip() for _, s in active]
        sec_starts: list[int] = []
        pos = 0
        for i, b in enumerate(bodies):
            if i > 0:
                pos += 2
            sec_starts.append(pos)
            pos += len(b)

        for k, (sec_idx, sec) in enumerate(active):
            body = bodies[k]
            sec_base = sec_starts[k]
            pieces = _split_oversized_body(body, self._max_section_chars)
            off_in_body = 0
            for part_i, part in enumerate(pieces):
                p = part.strip()
                if not p:
                    continue
                part_verified = False
                rel = body.find(p, off_in_body)
                if rel < 0:
                    ch_s = ch_e = None
                else:
                    ch_s = sec_base + rel
                    ch_e = sec_base + rel + len(p)
                    if (
                        full
                        and ch_s is not None
                        and ch_e is not None
                        and ch_e <= len(full)
                        and full[ch_s:ch_e] == p
                    ):
                        part_verified = True
                    else:
                        ch_s = ch_e = None
                    off_in_body = rel + len(p)

                md: dict[str, object] = {
                    "chunker": "SectionBasedChunkingService",
                    "content_kind": (sec.metadata or {}).get("content_kind")
                    if (sec.metadata or {}).get("content_kind")
                    in ("admin_section", "clinical_section")
                    else "clinical_section",
                    "section_heading": sec.heading,
                    "section_index": sec_idx,
                    "part_index": part_i,
                    "char_spans_verified": part_verified,
                }
                for mk, mv in (sec.metadata or {}).items():
                    if mk not in _SECTION_CHUNK_RESERVED_METADATA:
                        md[mk] = mv

                cid = HeuristicLabChunkingService._stable_chunk_id(
                    f"{parsed.document_id}:{sec_idx}:{part_i}",
                    p,
                )
                chunks.append(
                    DocumentChunk(
                        chunk_id=cid,
                        document_id=parsed.document_id,
                        text=p,
                        field_family=None,
                        char_start=ch_s,
                        char_end=ch_e,
                        metadata=md,
                    )
                )
        if not chunks and full.strip():
            cid = HeuristicLabChunkingService._stable_chunk_id(parsed.document_id, full)
            chunks.append(
                DocumentChunk(
                    chunk_id=cid,
                    document_id=parsed.document_id,
                    text=full.strip(),
                    field_family=None,
                    metadata={
                        "chunker": "SectionBasedChunkingService",
                        "content_kind": "clinical_fallback",
                        "fallback": True,
                        "char_spans_verified": False,
                    },
                )
            )
        return chunks


class DefaultChunkingService(ChunkingService):
    """
    Routage : rapports narratifs / imagerie → sections ; bilans labo → heuristique mots-clés.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._lab = HeuristicLabChunkingService()
        self._lab_rows = LabRowChunkingService()
        self._imaging = ImagingReportChunkingService(
            max_chars=self.settings.chunk_max_section_chars,
        )
        self._sections = SectionBasedChunkingService(
            max_section_chars=self.settings.chunk_max_section_chars,
        )

    def chunk(self, parsed: ParsedDocument) -> list[DocumentChunk]:
        strategy = parsed.metadata.get("chunking_strategy")
        if strategy == "sections":
            return self._sections.chunk(parsed)
        if strategy == "lab_rows":
            return self._lab_rows.chunk(parsed)
        if strategy == "imaging_full":
            return self._imaging.chunk(parsed)
        return self._lab.chunk(parsed)
