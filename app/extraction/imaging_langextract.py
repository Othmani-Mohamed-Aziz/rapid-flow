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

_MM_IN_TEXT = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:mm|millim(?:ètre|etre)s?)\b",
    re.I,
)
_RECIST_BOUNDARY_RE = re.compile(r"\b(CR|PR|SD|PD|NE)\b", re.I)

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
    lines.append("- Ne pas inventer de mesures absentes du texte.")
    return "\n".join(lines)


IMAGING_PROMPT = build_imaging_prompt_description(frozenset(DEFAULT_LANGEXTRACT_CLASS_DESCRIPTIONS))


def build_imaging_examples(allowed_classes: frozenset[str] | None = None) -> list[Any]:
    """Few-shots LangExtract (import paresseux de `langextract.data`)."""
    import langextract as lx

    allowed = allowed_classes or frozenset(DEFAULT_LANGEXTRACT_CLASS_DESCRIPTIONS)
    sample_extractions = [
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
    filtered = [e for e in sample_extractions if e.extraction_class in allowed]
    return [
        lx.data.ExampleData(
            text=(
                "Foie : nodule hypervasculaire du segment VI mesurant 28 mm de grand axe. "
                "Pas d'autre lésion focale. CONCLUSION : lésion hépatique stable."
            ),
            extractions=filtered,
        ),
    ]


def _parse_mm_value(extraction_text: str, attributes: dict[str, Any] | None) -> float | int | None:
    attrs = attributes or {}
    raw = attrs.get("value_mm")
    if raw is not None:
        try:
            val = float(str(raw).replace(",", "."))
            return int(val) if val.is_integer() else val
        except (TypeError, ValueError):
            pass
    m = _MM_IN_TEXT.search(extraction_text or "")
    if not m:
        return None
    val = float(m.group(1).replace(",", "."))
    return int(val) if val.is_integer() else val


def _normalize_recist_label(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return None
    m_boundary = _RECIST_BOUNDARY_RE.search(t)
    if m_boundary:
        return m_boundary.group(1).upper()
    lower = t.lower()
    # Ordre — progression avant « stabilité » pour éviter des cas ambivalents rares.
    if "progression" in lower or "progressive" in lower:
        return "PD"
    if "complète" in lower or "complete" in lower or "réponse complète" in lower:
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
    if "non évaluable" in lower or "non evaluable" in lower or "non évalu" in lower:
        return "NE"
    return t[:64]


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


def extractions_to_observations(
    extractions: Sequence[Any],
    *,
    source_chunk_id: str | None,
    model_name: str | None,
    schema_version: str,
    require_grounding: bool = True,
    retrieval_hit_score: float | None = None,
    imaging_config: ImagingExtractionConfig | None = None,
) -> list[ExtractedObservation]:
    """Convertit les `lx.data.Extraction` en `ExtractedObservation` (famille IMAGING_RECIST)."""
    cfg = imaging_config or ImagingExtractionConfig.legacy_default()
    class_to_canon = cfg.class_to_canonical or dict(DEFAULT_LANGEXTRACT_CLASS_TO_CANONICAL)
    observations: list[ExtractedObservation] = []
    explicit_recist_canon = False

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

    for ext in extractions:
        if require_grounding and getattr(ext, "char_interval", None) is None:
            continue
        cls = (getattr(ext, "extraction_class", None) or "").strip()
        text = (getattr(ext, "extraction_text", None) or "").strip()
        if not text:
            continue
        attrs = dict(getattr(ext, "attributes", None) or {})
        confidence = 0.82 if getattr(ext, "char_interval", None) else 0.55
        char_iv = getattr(ext, "char_interval", None)
        extra_base = make_extra(cls, attrs, char_iv)

        if cls == "lesion_size_mm":
            canon_key = canon_for_class(cls)
            if not should_emit_canonical(canon_key):
                continue
            num = _parse_mm_value(text, attrs)
            if num is None:
                continue
            canon_ex = _extra_with_retrieval(
                {**extra_base, "canonical_imaging_key": canon_key}, retrieval_hit_score
            )
            observations.append(
                ExtractedObservation(
                    observation_id=str(uuid.uuid4()),
                    field_family=FieldFamily.IMAGING_RECIST,
                    raw_value=text,
                    normalized_value=num,
                    unit="mm",
                    confidence=confidence,
                    evidence_text=text,
                    source_chunk_id=source_chunk_id,
                    extraction_method="langextract_ollama",
                    model_name=model_name,
                    schema_version=schema_version,
                    extra=canon_ex,
                )
            )
        elif cls == "recist_response":
            canon_key = canon_for_class(cls)
            if not should_emit_canonical(canon_key):
                continue
            norm = _normalize_recist_label(text)
            if norm is None or norm not in _RECIST_CANON_CODES:
                continue
            explicit_recist_canon = True
            canon_ex = _extra_with_retrieval(
                {**extra_base, "canonical_imaging_key": canon_key}, retrieval_hit_score
            )
            observations.append(
                ExtractedObservation(
                    observation_id=str(uuid.uuid4()),
                    field_family=FieldFamily.IMAGING_RECIST,
                    raw_value=text,
                    normalized_value=norm,
                    confidence=confidence,
                    evidence_text=text,
                    source_chunk_id=source_chunk_id,
                    extraction_method="langextract_ollama",
                    model_name=model_name,
                    schema_version=schema_version,
                    extra=canon_ex,
                )
            )
        elif cls in ("lesion_description", "imaging_conclusion"):
            canon_key = canon_for_class(cls)
            if not should_emit_canonical(canon_key):
                continue
            canon_ex = _extra_with_retrieval(
                {**extra_base, "canonical_imaging_key": canon_key}, retrieval_hit_score
            )
            observations.append(
                ExtractedObservation(
                    observation_id=str(uuid.uuid4()),
                    field_family=FieldFamily.IMAGING_RECIST,
                    raw_value=text,
                    normalized_value=text,
                    confidence=min(confidence, 0.75),
                    evidence_text=text,
                    source_chunk_id=source_chunk_id,
                    extraction_method="langextract_ollama",
                    model_name=model_name,
                    schema_version=schema_version,
                    extra=canon_ex,
                )
            )

    if cfg.derive_recist_from_conclusion and not explicit_recist_canon:
        recist_canon = canon_for_class("recist_response")
        if not should_emit_canonical(recist_canon):
            return observations
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
            norm = _normalize_recist_label(text)
            if norm is None or norm not in _RECIST_CANON_CODES:
                continue
            attrs = dict(getattr(ext, "attributes", None) or {})
            char_iv = getattr(ext, "char_interval", None)
            conf = 0.82 if char_iv is not None else 0.55
            derived_text = text
            derived_norm = norm
            derived_attrs = attrs
            derived_char_iv = char_iv
            derived_conf = min(conf, 0.72)

        if derived_text and derived_norm:
            extra_base = make_extra("recist_response", derived_attrs, derived_char_iv)
            canon_ex = _extra_with_retrieval(
                {
                    **extra_base,
                    "canonical_imaging_key": recist_canon,
                    "derived_from_imaging_conclusion": True,
                },
                retrieval_hit_score,
            )
            observations.append(
                ExtractedObservation(
                    observation_id=str(uuid.uuid4()),
                    field_family=FieldFamily.IMAGING_RECIST,
                    raw_value=derived_text,
                    normalized_value=derived_norm,
                    confidence=derived_conf,
                    evidence_text=derived_text,
                    source_chunk_id=source_chunk_id,
                    extraction_method="langextract_ollama",
                    model_name=model_name,
                    schema_version=schema_version,
                    extra=canon_ex,
                )
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
    )
