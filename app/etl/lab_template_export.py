from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas.models import ExtractedObservation


def build_lab_template(observations: list[ExtractedObservation]) -> dict[str, Any]:
    """Build Section/Subsection/Analyte JSON; later duplicate rows replace earlier ones."""
    payload: dict[str, Any] = {}
    for observation in observations:
        if observation.extraction_method != "lab_document_rows":
            continue
        extra = observation.extra or {}
        analyte = str(extra.get("analyte_name") or "").strip()
        if not analyte:
            continue
        section = str(extra.get("lab_section") or "Uncategorized")
        subsection = str(extra.get("lab_subsection") or section)
        leaf: dict[str, Any] = {"valeur": observation.normalized_value}
        numeric = isinstance(observation.normalized_value, (int, float)) and not isinstance(
            observation.normalized_value, bool
        )
        # Les résultats qualitatifs sans unité (« Absent ») restent sans clé « unité » ;
        # un résultat textuel unité-porteur (« Inf à 0.6 » ng/ml) la conserve.
        if numeric or observation.unit:
            leaf["unité"] = observation.unit or ""
        payload.setdefault(section, {}).setdefault(subsection, {})[analyte] = leaf
    return payload


def export_lab_template(
    observations: list[ExtractedObservation],
    output_dir: str | Path,
    *,
    source_path: str | Path,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{Path(source_path).stem}_template.json"
    destination.write_text(
        json.dumps(build_lab_template(observations), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return destination
