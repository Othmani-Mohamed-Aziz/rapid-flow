"""
Extraction RECIST / imagerie via LangExtract + Ollama (texte d'un chunk RAG).

Few-shots en français ; seules les extractions ancrées (`char_interval`) sont conservées.
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from app.extraction.imaging_config import ImagingExtractionConfig
from app.schemas.enums import FieldFamily
from app.schemas.models import ExtractedObservation
from app.schemas.study_schema import (
    _RECIST_CANON_CODES,
    DEFAULT_LANGEXTRACT_CLASS_DESCRIPTIONS,
    DEFAULT_LANGEXTRACT_CLASS_TO_CANONICAL,
)

_LOG = logging.getLogger(__name__)

SCHEMA_VERSION = "imaging-recist-langextract-v1"

# Clés `extra["canonical_imaging_key"]` reconnues par `FieldMappingService`.
CANONICAL_LESION_SIZE_MM = "Size_major_nodule_mm"
CANONICAL_RECIST_RESPONSE = "RECIST_response"

_SIZE_WITH_UNIT_RE = re.compile(
    r"(?P<num>\d+(?:[.,]\d+)?)\s*" r"(?P<unit>mm|millim(?:[eè]tre)s?|cm|centim(?:[eè]tre)s?)\b",
    re.I,
)
_CM_UNIT_RE = re.compile(r"\b(?:cm|centim(?:[eè]tre)s?)\b", re.I)
_MM_UNIT_RE = re.compile(r"\b(?:mm|millim(?:[eè]tre)s?)\b", re.I)
_RECIST_BOUNDARY_RE = re.compile(r"\b(CR|PR|SD|PD|NE)\b", re.I)
_CONCLUSION_HEADING_RE = re.compile(
    r"(?:"
    r"(?:^|\n)\s*conclusion\b|"
    r"(?:^|\n)\s*au total\b|"
    r"bilan de fin d['’][eé]valuation|"
    r"cat[eé]gorie recist retenue|"
    r"selon recist\s*1(?:[.,]1)?"
    r")",
    re.I,
)
_BASELINE_SIZE_CUES = re.compile(
    r"\b(?:j0|d0)\b|inclusion|scanner initial|d[eé]marrage du traitement|"
    r"bilan d['’]inclusion|r[eé]f[eé]rence",
    re.I,
)
_CURRENT_SIZE_CUES = re.compile(
    r"examen actuel|contr[oô]le du jour|ne pas confondre",
    re.I,
)
_NONTARGET_SIZE_CUES = re.compile(
    r"non[\s-]cible|non retenu|l[eé]sion nouvelle|apparition d['’]une",
    re.I,
)
_NE_PHRASES = (
    "non évaluable",
    "non evaluable",
    "réponse non évaluable",
    "reponse non evaluable",
    "ne peuvent pas être appliqués",
    "ne peuvent pas etre appliques",
    "critères recist ne peuvent pas",
    "criteres recist ne peuvent pas",
)
_BASELINE_SIZE_NEAR_CUE = (
    re.compile(
        r"(?:scanner initial|bilan d['’]inclusion|au d[eé]marrage du traitement|"
        r"r[eé]f[eé]rence j0|j0\s*/\s*d0|inclusion).{0,140}?"
        r"(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>mm|millim(?:[eè]tre)s?|cm|centim(?:[eè]tre)s?)",
        re.I | re.S,
    ),
    re.compile(
        r"(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>mm|millim(?:[eè]tre)s?|cm|centim(?:[eè]tre)s?)"
        r".{0,90}?(?:scanner initial|bilan d['’]inclusion)",
        re.I | re.S,
    ),
)

_CONF_SIZE_DEFAULT = 0.82
_CONF_SIZE_BASELINE = 0.93
_CONF_SIZE_CURRENT = 0.55
_CONF_SIZE_NONTARGET = 0.40
_CONF_RECIST_CONCLUSION = 0.93
_CONF_RECIST_DEFAULT = 0.82
_CONF_SOURCE_FALLBACK = 0.94

_ollama_model_ok: set[tuple[str, str]] = set()
_ollama_model_lock = threading.Lock()


def build_imaging_prompt_description(
    allowed_classes: frozenset[str],
    class_descriptions: dict[str, str] | None = None,
) -> str:
    descriptions = class_descriptions or DEFAULT_LANGEXTRACT_CLASS_DESCRIPTIONS
    lines = [
        "Extraire les informations d'imagerie utiles pour un eCRF oncologie (RECIST) depuis un "
        "extrait de compte rendu (scanner, TDM, IRM).",
        "Règles :",
        "- Utiliser le texte exact du document pour `extraction_text` (pas de paraphrase).",
        "- Lister les extractions dans l'ordre d'apparition.",
        "- Classes autorisées :",
    ]
    for cls in sorted(allowed_classes):
        desc = descriptions.get(cls, cls.replace("_", " "))
        lines.append(f"  * {cls} : {desc}")
    lines.extend(
        [
            "- Ne pas inventer de mesures absentes du texte.",
            "- `lesion_size_mm` : uniquement la cible de référence (J0/D0/inclusion) ; "
            "value_mm toujours en millimètres (×10 si le texte est en cm).",
            "- `recist_response` : uniquement la catégorie officielle de la conclusion.",
        ]
    )
    return "\n".join(lines)


IMAGING_PROMPT = build_imaging_prompt_description(frozenset(DEFAULT_LANGEXTRACT_CLASS_DESCRIPTIONS))


def build_imaging_examples(allowed_classes: frozenset[str] | None = None) -> list[Any]:
    """Few-shots LangExtract (import paresseux de `langextract.data`)."""
    import langextract as lx

    allowed = allowed_classes or frozenset(DEFAULT_LANGEXTRACT_CLASS_DESCRIPTIONS)
    easy = [
        lx.data.Extraction(
            extraction_class="lesion_description",
            extraction_text="nodule hypervasculaire du segment VI",
            attributes={"organ": "foie", "segment": "VI"},
        ),
        lx.data.Extraction(
            extraction_class="lesion_size_mm",
            extraction_text="28 mm",
            attributes={"value_mm": 28, "organ": "foie"},
        ),
        lx.data.Extraction(
            extraction_class="recist_response",
            extraction_text="lésion hépatique stable",
            attributes={"category": "SD"},
        ),
        lx.data.Extraction(
            extraction_class="imaging_conclusion",
            extraction_text="lésion hépatique stable",
            attributes={"stability": "stable"},
        ),
    ]
    hard = [
        lx.data.Extraction(
            extraction_class="lesion_size_mm",
            extraction_text="5,3 cm",
            attributes={"value_mm": 53, "organ": "foie"},
        ),
        lx.data.Extraction(
            extraction_class="recist_response",
            extraction_text="non évaluable (NE)",
            attributes={"category": "NE"},
        ),
        lx.data.Extraction(
            extraction_class="imaging_conclusion",
            extraction_text="réponse officielle retenue est non évaluable (NE)",
            attributes={"category": "NE"},
        ),
    ]
    examples = [
        lx.data.ExampleData(
            text=(
                "Foie : nodule hypervasculaire du segment VI mesurant 28 mm de grand axe. "
                "Pas d'autre lésion focale. CONCLUSION : lésion hépatique stable."
            ),
            extractions=[e for e in easy if e.extraction_class in allowed],
        ),
        lx.data.ExampleData(
            text=(
                "Indication : suspicion clinique de PD. "
                "Lésion cible n°1 (référence J0 / D0) : nodule du foie mesuré à 5,3 cm "
                "au scanner initial. Sur l'examen actuel, la même cible mesure 61 mm. "
                "Lésion non cible : adénopathie médiastinale mesurant 18 mm. "
                "CONCLUSION : Selon RECIST 1.1, la réponse officielle retenue est "
                "non évaluable (NE)."
            ),
            extractions=[e for e in hard if e.extraction_class in allowed],
        ),
    ]
    return [ex for ex in examples if ex.extractions]


def _as_mm_number(value: float) -> float | int:
    return int(value) if value.is_integer() else value


def _to_float(raw: Any) -> float | None:
    try:
        return float(str(raw).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _unit_is_cm(unit: str) -> bool:
    lowered = (unit or "").casefold()
    return lowered.startswith("cm") or "centim" in lowered


def _scale_to_mm(value: float, unit: str | None) -> float:
    return value * 10.0 if unit and _unit_is_cm(unit) else value


def _parse_mm_value(extraction_text: str, attributes: dict[str, Any] | None) -> float | int | None:
    """Parse a lesion size and always return millimetres."""
    attrs = attributes or {}
    text = extraction_text or ""
    match = _SIZE_WITH_UNIT_RE.search(text)
    if match:
        parsed = _to_float(match.group("num"))
        if parsed is not None:
            return _as_mm_number(_scale_to_mm(parsed, match.group("unit")))

    unit_attr = str(attrs.get("unit") or attrs.get("unité") or "")
    raw = attrs.get("value_mm")
    if raw is None and attrs.get("value_cm") is not None:
        raw = attrs.get("value_cm")
        unit_attr = unit_attr or "cm"
    if raw is not None:
        parsed = _to_float(raw)
        if parsed is not None:
            if _CM_UNIT_RE.search(text) and not _MM_UNIT_RE.search(text):
                unit_attr = unit_attr or "cm"
            return _as_mm_number(_scale_to_mm(parsed, unit_attr))

    bare = re.search(r"(\d+(?:[.,]\d+)?)", text)
    if not bare:
        return None
    parsed = _to_float(bare.group(1))
    if parsed is None:
        return None
    if _CM_UNIT_RE.search(text) and not _MM_UNIT_RE.search(text):
        parsed *= 10.0
    return _as_mm_number(parsed)


def _normalize_recist_label(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return None
    lower = t.lower()
    if any(phrase in lower for phrase in _NE_PHRASES) or "non évalu" in lower:
        return "NE"
    codes = [match.group(1).upper() for match in _RECIST_BOUNDARY_RE.finditer(t)]
    if codes:
        return codes[-1]
    if (
        "réponse complète" in lower
        or "reponse complete" in lower
        or "complète" in lower
        or "complete" in lower
    ):
        return "CR"
    if "partielle" in lower or "partial" in lower or "réponse partielle" in lower:
        return "PR"
    if (
        "stabilité" in lower
        or "stabilite" in lower
        or "stable" in lower
        or "stabile" in lower
        or "sans progression" in lower
        or "absence de progression" in lower
        or "maladie stabilisée" in lower
        or "maladie stabilisee" in lower
    ):
        return "SD"
    if "progression" in lower or "progressive" in lower:
        return "PD"
    return t[:64]


def _char_start(char_iv: Any) -> int | None:
    start = getattr(char_iv, "start_pos", None)
    return int(start) if isinstance(start, (int, float)) else None


def _conclusion_start(source_text: str | None) -> int | None:
    last: int | None = None
    for match in _CONCLUSION_HEADING_RE.finditer(source_text or ""):
        last = match.start()
    return last


def _in_conclusion(source_text: str | None, char_iv: Any) -> bool | None:
    heading = _conclusion_start(source_text)
    if heading is None:
        return None
    start = _char_start(char_iv)
    if start is None:
        return False
    return start >= heading


def _containing_sentence(source_text: str | None, char_iv: Any, extraction_text: str) -> str:
    if not source_text:
        return extraction_text
    start = _char_start(char_iv)
    if start is None:
        return extraction_text
    end = getattr(char_iv, "end_pos", None)
    stop = int(end) if isinstance(end, (int, float)) else start
    left = max(source_text.rfind(token, 0, start) for token in (".", "!", "?", "\n"))
    lo = 0 if left < 0 else left + 1
    rights = [source_text.find(token, stop) for token in (".", "!", "?", "\n")]
    rights = [pos for pos in rights if pos >= 0]
    hi = (min(rights) + 1) if rights else len(source_text)
    sentence = source_text[lo:hi].strip()
    return sentence or extraction_text


def _size_confidence(source_text: str | None, char_iv: Any, extraction_text: str) -> float:
    window = _containing_sentence(source_text, char_iv, extraction_text)
    if _NONTARGET_SIZE_CUES.search(window):
        return _CONF_SIZE_NONTARGET
    if _CURRENT_SIZE_CUES.search(window):
        return _CONF_SIZE_CURRENT
    if _BASELINE_SIZE_CUES.search(window):
        return _CONF_SIZE_BASELINE
    return _CONF_SIZE_DEFAULT


class _CharSpan:
    def __init__(self, start: int, end: int) -> None:
        self.start_pos = start
        self.end_pos = end


def flatten_langextract_extractions(result: Any) -> list[Any]:
    """Homogénéise diverses formes du résultat LangExtract (liste plate ou documents imbriqués)."""
    if isinstance(result, (list, tuple)) and result:
        el0 = result[0]
        has_cls = getattr(el0, "extraction_class", None) is not None
        has_txt = getattr(el0, "extraction_text", None) is not None
        if has_cls or has_txt:
            return list(result)
    raw = getattr(result, "extractions", None)
    if raw is None:
        return []
    collected: list[Any] = []

    def walk(cur: Any) -> None:
        if cur is None:
            return
        if isinstance(cur, (list, tuple)):
            for x in cur:
                walk(x)
            return
        inner = getattr(cur, "extractions", None)
        if (
            isinstance(inner, (list, tuple))
            and len(inner) > 0
            and getattr(cur, "extraction_class", None) is None
        ):
            walk(inner)
            return
        extracted_cls = getattr(cur, "extraction_class", None)
        extracted_text = getattr(cur, "extraction_text", None)
        if extracted_cls is not None or extracted_text is not None:
            collected.append(cur)

    walk(raw)
    return collected


def _extra_with_retrieval(ex: dict[str, Any], retrieval_hit_score: float | None) -> dict[str, Any]:
    if retrieval_hit_score is None:
        return ex
    return {**ex, "retrieval_score": float(retrieval_hit_score)}


def _add_source_fallbacks(
    observations: list[ExtractedObservation],
    *,
    source_text: str,
    cfg: ImagingExtractionConfig,
    canon_for_class: Callable[[str], str],
    should_emit_canonical: Callable[[str], bool],
    emit: Callable[..., None],
    conclusion_recist: bool,
) -> None:
    size_canon = canon_for_class("lesion_size_mm")
    recist_canon = canon_for_class("recist_response")
    if should_emit_canonical(size_canon):
        has_baseline = any(
            (o.extra or {}).get("canonical_imaging_key") == size_canon
            and o.confidence >= _CONF_SIZE_BASELINE
            for o in observations
        )
        if not has_baseline:
            for pattern in _BASELINE_SIZE_NEAR_CUE:
                match = pattern.search(source_text)
                if not match:
                    continue
                parsed = _to_float(match.group("num"))
                if parsed is None:
                    continue
                mm = _as_mm_number(_scale_to_mm(parsed, match.group("unit")))
                already = any(
                    (o.extra or {}).get("canonical_imaging_key") == size_canon
                    and o.normalized_value == mm
                    for o in observations
                )
                if already:
                    break
                emit(
                    cls="lesion_size_mm",
                    canon_key=size_canon,
                    text=match.group(0).strip(),
                    value=mm,
                    char_iv=_CharSpan(match.start(), match.end()),
                    attrs={"value_mm": mm, "source": "baseline_cue"},
                    confidence=_CONF_SOURCE_FALLBACK,
                    unit="mm",
                    extra_flags={"source_fallback": "baseline_size"},
                    method="imaging_source_fallback",
                )
                break

    if not (
        cfg.derive_recist_from_conclusion
        and should_emit_canonical(recist_canon)
        and not conclusion_recist
    ):
        return
    heading = _conclusion_start(source_text)
    if heading is None:
        return
    section = source_text[heading:]
    norm = _normalize_recist_label(section)
    if norm not in _RECIST_CANON_CODES:
        return
    emit(
        cls="recist_response",
        canon_key=recist_canon,
        text=section.strip()[:240],
        value=norm,
        char_iv=_CharSpan(heading, len(source_text)),
        attrs={"category": norm},
        confidence=_CONF_SOURCE_FALLBACK,
        extra_flags={"source_fallback": "conclusion_recist"},
        method="imaging_source_fallback",
    )


def extractions_to_observations(
    extractions: Sequence[Any],
    *,
    source_chunk_id: str | None,
    model_name: str | None,
    schema_version: str,
    require_grounding: bool = True,
    retrieval_hit_score: float | None = None,
    imaging_config: ImagingExtractionConfig | None = None,
    source_text: str | None = None,
) -> list[ExtractedObservation]:
    """Convertit les `lx.data.Extraction` en `ExtractedObservation` (famille IMAGING_RECIST)."""
    cfg = imaging_config or ImagingExtractionConfig.legacy_default()
    class_to_canon = cfg.class_to_canonical or dict(DEFAULT_LANGEXTRACT_CLASS_TO_CANONICAL)
    observations: list[ExtractedObservation] = []
    conclusion_recist = False

    def canon_for_class(cls: str) -> str:
        return class_to_canon.get(cls, cls)

    def should_emit_canonical(canon: str) -> bool:
        if not cfg.allowed_canonical_keys:
            return True
        return canon in cfg.allowed_canonical_keys

    def make_extra(cls: str, attrs: dict[str, Any], char_iv: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "langextract_class": cls,
            "attributes": attrs,
            "grounded": char_iv is not None,
        }
        if char_iv is not None:
            base["char_interval"] = {
                "start": getattr(char_iv, "start_pos", None),
                "end": getattr(char_iv, "end_pos", None),
            }
        return base

    def recist_code(text: str, attrs: dict[str, Any] | None = None) -> str | None:
        norm = _normalize_recist_label(text)
        if norm in _RECIST_CANON_CODES:
            return norm
        attr_cat = str((attrs or {}).get("category") or "").strip().upper()
        return attr_cat if attr_cat in _RECIST_CANON_CODES else None

    def emit(
        *,
        cls: str,
        canon_key: str,
        text: str,
        value: Any,
        char_iv: Any,
        attrs: dict[str, Any],
        confidence: float,
        unit: str | None = None,
        extra_flags: dict[str, Any] | None = None,
        method: str = "langextract_ollama",
    ) -> None:
        extra = make_extra(cls, attrs, char_iv)
        extra["canonical_imaging_key"] = canon_key
        if extra_flags:
            extra.update(extra_flags)
        observations.append(
            ExtractedObservation(
                observation_id=str(uuid.uuid4()),
                field_family=FieldFamily.IMAGING_RECIST,
                raw_value=text,
                normalized_value=value,
                unit=unit,
                confidence=confidence,
                evidence_text=text,
                source_chunk_id=source_chunk_id,
                extraction_method=method,
                model_name=model_name,
                schema_version=schema_version,
                extra=_extra_with_retrieval(extra, retrieval_hit_score),
            )
        )

    for ext in extractions:
        if require_grounding and getattr(ext, "char_interval", None) is None:
            continue
        cls = (getattr(ext, "extraction_class", None) or "").strip()
        text = (getattr(ext, "extraction_text", None) or "").strip()
        if not text:
            continue
        attrs = dict(getattr(ext, "attributes", None) or {})
        char_iv = getattr(ext, "char_interval", None)
        grounded_conf = 0.82 if char_iv is not None else 0.55

        if cls == "lesion_size_mm":
            canon_key = canon_for_class(cls)
            if not should_emit_canonical(canon_key):
                continue
            num = _parse_mm_value(text, attrs)
            if num is None:
                continue
            emit(
                cls=cls,
                canon_key=canon_key,
                text=text,
                value=num,
                char_iv=char_iv,
                attrs=attrs,
                confidence=_size_confidence(source_text, char_iv, text),
                unit="mm",
            )
        elif cls == "recist_response":
            canon_key = canon_for_class(cls)
            if not should_emit_canonical(canon_key):
                continue
            norm = recist_code(text, attrs)
            if norm is None:
                continue
            in_conc = _in_conclusion(source_text, char_iv)
            if in_conc is False:
                continue
            recist_conf = _CONF_RECIST_CONCLUSION if in_conc else grounded_conf
            if in_conc:
                conclusion_recist = True
            emit(
                cls=cls,
                canon_key=canon_key,
                text=text,
                value=norm,
                char_iv=char_iv,
                attrs=attrs,
                confidence=recist_conf,
            )
        elif cls in ("lesion_description", "imaging_conclusion"):
            canon_key = canon_for_class(cls)
            if not should_emit_canonical(canon_key):
                continue
            emit(
                cls=cls,
                canon_key=canon_key,
                text=text,
                value=text,
                char_iv=char_iv,
                attrs=attrs,
                confidence=min(grounded_conf, 0.75),
            )

    recist_canon = canon_for_class("recist_response")
    if (
        cfg.derive_recist_from_conclusion
        and should_emit_canonical(recist_canon)
        and not conclusion_recist
    ):
        derived_text: str | None = None
        derived_norm: str | None = None
        derived_attrs: dict[str, Any] = {}
        derived_char_iv: Any = None
        derived_conf = 0.65
        for ext in extractions:
            if require_grounding and getattr(ext, "char_interval", None) is None:
                continue
            cls = (getattr(ext, "extraction_class", None) or "").strip()
            if cls != "imaging_conclusion":
                continue
            text = (getattr(ext, "extraction_text", None) or "").strip()
            if not text:
                continue
            norm = recist_code(text, dict(getattr(ext, "attributes", None) or {}))
            if norm is None:
                continue
            attrs = dict(getattr(ext, "attributes", None) or {})
            char_iv = getattr(ext, "char_interval", None)
            in_conc = _in_conclusion(source_text, char_iv)
            if in_conc is False:
                continue
            conf = 0.82 if char_iv is not None else 0.55
            derived_text = text
            derived_norm = norm
            derived_attrs = attrs
            derived_char_iv = char_iv
            derived_conf = _CONF_RECIST_CONCLUSION if in_conc else min(conf, 0.72)

        if derived_text and derived_norm:
            conclusion_recist = True
            emit(
                cls="recist_response",
                canon_key=recist_canon,
                text=derived_text,
                value=derived_norm,
                char_iv=derived_char_iv,
                attrs=derived_attrs,
                confidence=derived_conf,
                extra_flags={"derived_from_imaging_conclusion": True},
            )

    if source_text:
        _add_source_fallbacks(
            observations,
            source_text=source_text,
            cfg=cfg,
            canon_for_class=canon_for_class,
            should_emit_canonical=should_emit_canonical,
            emit=emit,
            conclusion_recist=conclusion_recist,
        )

    return observations


class LangExtractNotInstalledError(RuntimeError):
    """Extra `extraction` non installé (`pip install -e '.[extraction]'`)."""


class OllamaModelNotAvailableError(ValueError):
    """Le serveur Ollama sur `model_url` ne propose pas le `model_id` demandé."""


def list_ollama_models(model_url: str) -> list[str]:
    """Noms retournés par `GET {model_url}/api/tags` (source de vérité pour l'API generate)."""
    import requests

    base = model_url.rstrip("/")
    resp = requests.get(f"{base}/api/tags", timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    names: list[str] = []
    for entry in payload.get("models") or []:
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("model")
            if name:
                names.append(str(name))
    return names


def clear_ollama_model_registered_cache() -> None:
    """Tests / changement de modèle : invalide le cache du pré-contrôle `/api/tags`."""
    with _ollama_model_lock:
        _ollama_model_ok.clear()


def assert_ollama_model_registered(model_id: str, model_url: str) -> None:
    """
    Vérifie que le modèle est visible par l'API HTTP (évite un 404 opaque côté LangExtract).

    `ollama list` (CLI) peut afficher des modèles que le daemon sur `model_url` n'expose pas
    encore — redémarrer Ollama ou lancer `ollama run <model>` après un `pull`.
    Le résultat positif est mis en cache par (URL, model_id) pour les extractions parallèles.
    """
    base = model_url.rstrip("/")
    key = (base, model_id)
    with _ollama_model_lock:
        if key in _ollama_model_ok:
            return
    names = list_ollama_models(model_url)
    if model_id in names:
        with _ollama_model_lock:
            _ollama_model_ok.add(key)
        return
    base_id = model_id.split(":")[0]
    for n in names:
        if n == model_id or n.startswith(f"{model_id}:") or n.split(":")[0] == base_id:
            with _ollama_model_lock:
                _ollama_model_ok.add(key)
            return
    raise OllamaModelNotAvailableError(
        f"Ollama sur {model_url} ne connaît pas le modèle {model_id!r} "
        f"(modèles API : {names or 'aucun'}). "
        f"Corrections possibles : `ollama run {model_id}` ; redémarrer l'application Ollama "
        f"après `ollama pull` ; ou `ECRF_OLLAMA_MODEL_ID` = un nom listé par "
        f"`curl {model_url.rstrip('/')}/api/tags`."
    )


def run_imaging_langextract(
    chunk_text: str,
    *,
    model_id: str,
    model_url: str,
    timeout: int = 120,
    schema_version: str = SCHEMA_VERSION,
    source_chunk_id: str | None = None,
    retrieval_hit_score: float | None = None,
    imaging_config: ImagingExtractionConfig | None = None,
    show_progress: bool = False,
    extract_fn: Callable[..., Any] | None = None,
) -> list[ExtractedObservation]:
    """
    Appelle LangExtract (Ollama) sur le texte d'un chunk et retourne des observations.

    `extract_fn` : injectable pour les tests (défaut : `langextract.extract`).
    """
    text = (chunk_text or "").strip()
    if not text:
        return []

    cfg = imaging_config or ImagingExtractionConfig.legacy_default()
    allowed_classes = cfg.allowed_langextract_classes or frozenset(
        DEFAULT_LANGEXTRACT_CLASS_DESCRIPTIONS
    )
    prompt = build_imaging_prompt_description(allowed_classes, cfg.class_descriptions)

    if extract_fn is None:
        try:
            import langextract as lx
            from langextract.providers import ollama
        except ImportError as e:
            raise LangExtractNotInstalledError('Installer : pip install -e ".[extraction]"') from e
        fn = lx.extract
        assert_ollama_model_registered(model_id, model_url)
        resolver_params: dict[str, Any] = {"format_handler": ollama.OLLAMA_FORMAT_HANDLER}
    else:
        fn = extract_fn
        resolver_params = {}

    # LangExtract ≥1.4 : `timeout` passe via `language_model_params`, pas en kwarg direct.
    result = fn(
        text_or_documents=text,
        prompt_description=prompt,
        examples=build_imaging_examples(allowed_classes),
        model_id=model_id,
        model_url=model_url,
        language_model_params={"timeout": timeout},
        resolver_params=resolver_params,
        show_progress=show_progress,
    )
    flat = flatten_langextract_extractions(result)
    return extractions_to_observations(
        flat,
        source_chunk_id=source_chunk_id,
        model_name=model_id,
        schema_version=schema_version,
        require_grounding=True,
        retrieval_hit_score=retrieval_hit_score,
        imaging_config=cfg,
        source_text=text,
    )
