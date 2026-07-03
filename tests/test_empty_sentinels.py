from __future__ import annotations

import openpyxl

from app.config.settings import Settings
from app.etl.empty_sentinels import is_ecrf_cell_empty, parse_empty_sentinels
from app.etl.overwrite_policy import EcrfTemplateSnapshot
from tests.test_etl_export import make_minimal_ma_template


def test_default_sentinels_treat_na_as_empty() -> None:
    assert is_ecrf_cell_empty("NA") is True
    assert is_ecrf_cell_empty("n/a") is True
    assert is_ecrf_cell_empty("-") is True
    assert is_ecrf_cell_empty("A completer") is True
    assert is_ecrf_cell_empty("A compléter") is True


def test_numeric_zero_not_empty() -> None:
    assert is_ecrf_cell_empty(0) is False
    assert is_ecrf_cell_empty(0.0) is False


def test_custom_sentinels_from_env_string() -> None:
    custom = parse_empty_sentinels("FOO,BAR")
    assert is_ecrf_cell_empty("FOO", sentinels=custom) is True
    assert is_ecrf_cell_empty("NA", sentinels=custom) is False


def test_settings_parsed_empty_sentinels() -> None:
    s = Settings(empty_sentinels="NA,N/A,-,A completer")
    assert is_ecrf_cell_empty("NA", sentinels=s.parsed_empty_sentinels()) is True


def test_snapshot_plt_na_treated_as_unfilled(tmp_path) -> None:
    """PLT=NA dans le gabarit → colonne non remplie → autofill autorisé."""
    template = tmp_path / "template.xlsx"
    make_minimal_ma_template(template)
    wb = openpyxl.load_workbook(template)
    ws = wb["Global_CC"]
    ws.cell(row=3, column=3, value="NA")
    wb.save(template)
    wb.close()

    snap = EcrfTemplateSnapshot(
        template_path=template,
        ma_patient_key="220",
        empty_sentinels=parse_empty_sentinels(None),
    )
    assert snap.is_column_filled("PLT_start_AtezoBev_D0") is False
