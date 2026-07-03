"""Résolution ``patient_id`` pipeline → clé ligne MA (``ID_current_base``)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from app.etl.overwrite_policy import EcrfTemplateSnapshot

_LOG = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATIENT_ID_MAP_PATH = _PROJECT_ROOT / "data" / "patient_id_map.demo.json"


class MaPatientResolutionSource(str, Enum):
    EXPLICIT = "explicit"
    MAP_FILE = "map_file"
    DIRECT_MATCH = "direct_match"
    PIPELINE_ID = "pipeline_id"


@dataclass(frozen=True)
class MaPatientResolution:
    """Résultat de la résolution vers la clé MA."""

    pipeline_patient_id: str
    ma_patient_key: str
    source: MaPatientResolutionSource
    found_in_template: bool

    def to_metadata(self) -> dict[str, str | bool]:
        return {
            "pipeline_patient_id": self.pipeline_patient_id,
            "ma_patient_key": self.ma_patient_key,
            "ma_patient_resolution_source": self.source.value,
            "ma_patient_found_in_template": self.found_in_template,
        }


class MaPatientNotFoundError(ValueError):
    """Aucune ligne patient correspondante dans le gabarit eCRF MA."""


def load_patient_id_map(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"Cartographie patient invalide (objet JSON attendu) : {path}"
        raise ValueError(msg)
    return {str(k): str(v) for k, v in raw.items()}


def _template_path_from_settings(settings: Any) -> Path:
    from app.etl.export import DEFAULT_ECRF_TEMPLATE_XLSX

    configured = settings.export_xlsx_template_path
    return Path(configured) if configured else DEFAULT_ECRF_TEMPLATE_XLSX


def _patient_row_exists(ma_key: str, settings: Any) -> bool:
    template = _template_path_from_settings(settings)
    if not template.is_file():
        return False
    snap = EcrfTemplateSnapshot.from_settings(settings, ma_patient_key=ma_key)
    snap.load()
    return snap.patient_row_index is not None


def resolve_ma_patient_key(
    pipeline_patient_id: str,
    settings: Any,
    *,
    ma_patient_key: str | None = None,
) -> MaPatientResolution:
    """
    Détermine la clé à utiliser pour ``ID_current_base`` (ou colonne configurée).

    Priorité :
    1. ``ma_patient_key`` explicite (argument ``run_pipeline``)
    2. Entrée dans le fichier JSON ``ECRF_EXPORT_XLS_PATIENT_ID_MAP_PATH``
    3. ``pipeline_patient_id`` si une ligne existe déjà dans le gabarit
    4. ``pipeline_patient_id`` tel quel (avec avertissement si absent du gabarit)
    """
    pid = pipeline_patient_id.strip()
    if not pid:
        msg = "patient_id pipeline vide"
        raise ValueError(msg)

    if ma_patient_key is not None:
        key = str(ma_patient_key).strip()
        if not key:
            msg = "ma_patient_key explicite vide"
            raise ValueError(msg)
        found = _patient_row_exists(key, settings)
        return MaPatientResolution(
            pipeline_patient_id=pid,
            ma_patient_key=key,
            source=MaPatientResolutionSource.EXPLICIT,
            found_in_template=found,
        )

    map_path: Path | None = None
    configured_map = settings.export_xls_patient_id_map_path
    if configured_map:
        map_path = Path(configured_map)
    elif DEFAULT_PATIENT_ID_MAP_PATH.is_file():
        map_path = DEFAULT_PATIENT_ID_MAP_PATH

    if map_path is not None and map_path.is_file():
        mapping = load_patient_id_map(map_path)
        if pid in mapping:
            key = mapping[pid]
            found = _patient_row_exists(key, settings)
            return MaPatientResolution(
                pipeline_patient_id=pid,
                ma_patient_key=key,
                source=MaPatientResolutionSource.MAP_FILE,
                found_in_template=found,
            )

    if _patient_row_exists(pid, settings):
        return MaPatientResolution(
            pipeline_patient_id=pid,
            ma_patient_key=pid,
            source=MaPatientResolutionSource.DIRECT_MATCH,
            found_in_template=True,
        )

    resolution = MaPatientResolution(
        pipeline_patient_id=pid,
        ma_patient_key=pid,
        source=MaPatientResolutionSource.PIPELINE_ID,
        found_in_template=False,
    )
    _LOG.warning(
        "patient_id pipeline %r introuvable dans le gabarit MA (colonne %s). "
        "Préfiltre empty_only inefficace ; risque d'ajout de ligne. "
        "Utilisez ma_patient_key=... ou ECRF_EXPORT_XLS_PATIENT_ID_MAP_PATH.",
        pid,
        settings.export_xls_patient_id_column,
    )
    return resolution


def ensure_ma_patient_in_template(resolution: MaPatientResolution, settings: Any) -> None:
    """Lève ``MaPatientNotFoundError`` si le gabarit existe et exige une ligne patient."""
    if not settings.export_xls_require_patient_in_ma:
        return
    template = _template_path_from_settings(settings)
    if not template.is_file():
        return
    if resolution.found_in_template:
        return
    msg = (
        f"Patient MA introuvable : pipeline_patient_id={resolution.pipeline_patient_id!r}, "
        f"ma_patient_key={resolution.ma_patient_key!r} "
        f"(source={resolution.source.value}). "
        f"Vérifiez {settings.export_xls_patient_id_column} dans {template} "
        f"ou la cartographie ECRF_EXPORT_XLS_PATIENT_ID_MAP_PATH."
    )
    raise MaPatientNotFoundError(msg)
