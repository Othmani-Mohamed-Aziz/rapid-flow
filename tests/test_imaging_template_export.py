from __future__ import annotations

import json
from pathlib import Path

from app.etl.imaging_template_export import build_imaging_template, export_imaging_template
from app.schemas.enums import FieldFamily
from app.schemas.models import EcrfCellUpdate, ExtractedObservation, FieldCandidate


def _update(
    *,
    column: str,
    value: str | int,
    unit: str | None,
    validated: bool = True,
) -> EcrfCellUpdate:
    observation = ExtractedObservation(
        observation_id=column,
        field_family=FieldFamily.IMAGING_RECIST,
        raw_value=str(value),
        normalized_value=value,
        unit=unit,
        confidence=0.9,
        extraction_method="langextract_ollama",
    )
    candidate = FieldCandidate(
        field_name=column,
        observation=observation,
        target_column=column,
        confidence=0.9,
    )
    return EcrfCellUpdate(
        patient_id="P1",
        study_id="EXAMPLE",
        column_key=column,
        value=value,
        provenance=candidate,
        validated=validated,
    )


def test_build_imaging_template_matches_recist_evaluation_contract() -> None:
    payload = build_imaging_template(
        [
            _update(
                column="Size_major_nodule_mm_start_AtezoBev_D0",
                value=45,
                unit="mm",
            ),
            _update(
                column="Response_at_first_imaging_RECIST",
                value="SD",
                unit=None,
            ),
            _update(column="ignored_unvalidated", value="PD", unit=None, validated=False),
        ]
    )

    assert payload == {
        "Imaging_RECIST": {
            "Size_major_nodule_mm_start_AtezoBev_D0": {
                "valeur": 45,
                "unité": "mm",
            },
            "Response_at_first_imaging_RECIST": {"valeur": "SD"},
        }
    }


def test_export_imaging_template_uses_evaluation_filename(tmp_path: Path) -> None:
    destination = export_imaging_template(
        [
            _update(
                column="Response_at_first_imaging_RECIST",
                value="PR",
                unit=None,
            )
        ],
        tmp_path,
        source_path="sample_0001.pdf",
    )

    assert destination.name == "sample_0001_template.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "Imaging_RECIST": {
            "Response_at_first_imaging_RECIST": {"valeur": "PR"},
        }
    }
