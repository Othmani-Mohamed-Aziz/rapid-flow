from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas.enums import FieldFamily
from app.schemas.models import EcrfCellUpdate

IMAGING_TEMPLATE_SECTION = "Imaging_RECIST"


def build_imaging_template(updates: list[EcrfCellUpdate]) -> dict[str, Any]:
    """Build evaluator-compatible imaging predictions from validated eCRF updates."""
    fields: dict[str, dict[str, Any]] = {}
    for update in updates:
        observation = update.provenance.observation
        if observation.field_family != FieldFamily.IMAGING_RECIST:
            continue
        if not update.validated or update.value is None:
            continue
        leaf: dict[str, Any] = {"valeur": update.value}
        if observation.unit:
            leaf["unité"] = observation.unit
        fields[update.column_key] = leaf
    return {IMAGING_TEMPLATE_SECTION: fields}


def export_imaging_template(
    updates: list[EcrfCellUpdate],
    output_dir: str | Path,
    *,
    source_path: str | Path,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{Path(source_path).stem}_template.json"
    destination.write_text(
        json.dumps(build_imaging_template(updates), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return destination
