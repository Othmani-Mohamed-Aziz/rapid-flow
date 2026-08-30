from __future__ import annotations

from app.etl.lab_template_export import build_lab_template
from app.schemas.enums import FieldFamily
from app.schemas.models import ExtractedObservation


def _observation(
    *,
    analyte: str,
    value: str | float | int,
    unit: str | None,
) -> ExtractedObservation:
    return ExtractedObservation(
        observation_id=analyte,
        field_family=FieldFamily.HEMATOLOGY,
        raw_value=str(value),
        normalized_value=value,
        unit=unit,
        confidence=0.9,
        extraction_method="lab_document_rows",
        extra={
            "lab_section": "Hematologie",
            "lab_subsection": "FormuleLeucocytaire",
            "analyte_name": analyte,
        },
    )


def test_build_lab_template_matches_nested_contract_and_last_row_wins() -> None:
    payload = build_lab_template(
        [
            _observation(analyte="Leucocytes", value=6.47, unit="G/L"),
            _observation(analyte="Leucocytes", value=6609, unit="/mm3"),
            _observation(analyte="CRISTAUX", value="Absent", unit=None),
            _observation(analyte="Testostérone", value="Inf à 0.6", unit="ng/ml"),
        ]
    )

    subsection = payload["Hematologie"]["FormuleLeucocytaire"]
    assert subsection["Leucocytes"] == {"valeur": 6609, "unité": "/mm3"}
    assert subsection["CRISTAUX"] == {"valeur": "Absent"}
    assert subsection["Testostérone"] == {"valeur": "Inf à 0.6", "unité": "ng/ml"}
