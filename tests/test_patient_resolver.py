from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config.settings import Settings
from app.etl.patient_resolver import (
    MaPatientNotFoundError,
    MaPatientResolutionSource,
    ensure_ma_patient_in_template,
    resolve_ma_patient_key,
)
from tests.test_etl_export import make_minimal_ma_template


def test_resolve_via_demo_map_file(tmp_path: Path) -> None:
    template = tmp_path / "template.xlsx"
    make_minimal_ma_template(template)
    map_path = tmp_path / "map.json"
    map_path.write_text(
        json.dumps({"PAT-DEMO-LAB": "220"}),
        encoding="utf-8",
    )
    settings = Settings(
        export_xlsx_template_path=str(template),
        export_xls_patient_id_map_path=str(map_path),
        export_xls_require_patient_in_ma=True,
    )
    res = resolve_ma_patient_key("PAT-DEMO-LAB", settings)
    assert res.ma_patient_key == "220"
    assert res.source == MaPatientResolutionSource.MAP_FILE
    assert res.found_in_template is True
    ensure_ma_patient_in_template(res, settings)


def test_resolve_direct_numeric_match(tmp_path: Path) -> None:
    template = tmp_path / "template.xlsx"
    make_minimal_ma_template(template)
    settings = Settings(
        export_xlsx_template_path=str(template),
        export_xls_patient_id_map_path=str(tmp_path / "missing.json"),
        export_xls_require_patient_in_ma=True,
    )
    res = resolve_ma_patient_key("220", settings)
    assert res.ma_patient_key == "220"
    assert res.source == MaPatientResolutionSource.DIRECT_MATCH
    assert res.found_in_template is True


def test_explicit_ma_patient_key(tmp_path: Path) -> None:
    template = tmp_path / "template.xlsx"
    make_minimal_ma_template(template)
    settings = Settings(export_xlsx_template_path=str(template))
    res = resolve_ma_patient_key("PAT-X", settings, ma_patient_key="220")
    assert res.ma_patient_key == "220"
    assert res.source == MaPatientResolutionSource.EXPLICIT


def test_require_patient_raises_when_unknown(tmp_path: Path) -> None:
    template = tmp_path / "template.xlsx"
    make_minimal_ma_template(template)
    settings = Settings(
        export_xlsx_template_path=str(template),
        export_xls_patient_id_map_path=str(tmp_path / "nomap.json"),
        export_xls_require_patient_in_ma=True,
    )
    res = resolve_ma_patient_key("UNKNOWN-ID", settings)
    assert res.found_in_template is False
    with pytest.raises(MaPatientNotFoundError):
        ensure_ma_patient_in_template(res, settings)


@pytest.mark.skipif(
    not Path("data/patient_id_map.demo.json").is_file(),
    reason="Fichier demo absent",
)
def test_demo_map_resolves_pat_demo_lab() -> None:
    if not Path("data/MA_Base_example.xlsx").is_file():
        pytest.skip("MA_Base absent")
    settings = Settings()
    res = resolve_ma_patient_key("PAT-DEMO-LAB", settings)
    assert res.ma_patient_key == "220"
    assert res.found_in_template is True
