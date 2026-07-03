from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from app.etl.column_aliases import resolve_column_aliases
from app.etl.empty_sentinels import parse_empty_sentinels
from app.etl.overwrite_policy import XlsOverwritePolicy
from app.etl.xls_export import XlsEcrfExporter, XlsExportError
from app.schemas.models import PipelineResult
from app.utils.safe_paths import redact_file_paths_in_jsonable

_LOG = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ECRF_TEMPLATE_XLSX = _PROJECT_ROOT / "data" / "MA_Base_example.xlsx"


class EcrfExportService:
    """
    Export JSON + CSV mock + mise à jour du gabarit MA (source de vérité).

    Le fichier ``export_xlsx_template_path`` est mis à jour en place ; copie optionnelle
    vers ``outputs/`` si ``export_xls_mirror_to_output`` est activé.
    """

    def __init__(
        self,
        *,
        template_xlsx_path: Path | str | None = None,
        export_xls_enabled: bool = True,
        export_xls_update_master_workbook: bool = True,
        export_xls_mirror_to_output: bool = False,
        xls_sheet_name: str = "Global_CC",
        xls_header_row: int = 2,
        xls_patient_id_column: str = "ID_current_base",
        overwrite_policy: XlsOverwritePolicy | str = XlsOverwritePolicy.EMPTY_ONLY,
        empty_sentinels: frozenset[str] | None = None,
        column_aliases: dict[str, str] | None = None,
    ) -> None:
        self.export_xls_enabled = export_xls_enabled
        self.export_xls_update_master_workbook = export_xls_update_master_workbook
        self.export_xls_mirror_to_output = export_xls_mirror_to_output
        self.template_xlsx_path = (
            Path(template_xlsx_path)
            if template_xlsx_path is not None
            else DEFAULT_ECRF_TEMPLATE_XLSX
        )
        self.xls_sheet_name = xls_sheet_name
        self.xls_header_row = xls_header_row
        self.xls_patient_id_column = xls_patient_id_column
        self.overwrite_policy = (
            overwrite_policy
            if isinstance(overwrite_policy, XlsOverwritePolicy)
            else XlsOverwritePolicy(str(overwrite_policy))
        )
        self._empty_sentinels = (
            empty_sentinels if empty_sentinels is not None else parse_empty_sentinels(None)
        )
        self._column_aliases = column_aliases or resolve_column_aliases()

    @classmethod
    def from_settings(cls, settings: Any) -> EcrfExportService:
        template = settings.export_xlsx_template_path
        return cls(
            template_xlsx_path=Path(template) if template else DEFAULT_ECRF_TEMPLATE_XLSX,
            export_xls_enabled=settings.export_xls_enabled,
            export_xls_update_master_workbook=settings.export_xls_update_master_workbook,
            export_xls_mirror_to_output=settings.export_xls_mirror_to_output,
            xls_sheet_name=settings.export_xls_sheet_name,
            xls_header_row=settings.export_xls_header_row,
            xls_patient_id_column=settings.export_xls_patient_id_column,
            overwrite_policy=settings.export_xls_overwrite_policy,
            empty_sentinels=settings.parsed_empty_sentinels(),
            column_aliases=resolve_column_aliases(settings),
        )

    def export(
        self,
        result: PipelineResult,
        output_dir: str | Path,
        *,
        basename: str = "pipeline_output",
    ) -> dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"{basename}.json"
        csv_path = out / f"{basename}_ecrf_mock.csv"

        payload = redact_file_paths_in_jsonable(result.model_dump(mode="json"))
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        self._write_csv_mock(csv_path, result)
        paths: dict[str, str] = {
            "json": str(json_path.resolve()),
            "csv": str(csv_path.resolve()),
        }
        xlsx_paths = self._export_xlsx(result, out, basename=basename)
        paths.update(xlsx_paths)
        return paths

    def _export_xlsx(
        self,
        result: PipelineResult,
        output_dir: Path,
        *,
        basename: str,
    ) -> dict[str, str]:
        if not self.export_xls_enabled:
            return {}
        if not self.template_xlsx_path.is_file():
            _LOG.warning(
                "Export XLS ignoré : gabarit absent (%s). "
                "Placez MA_Base_example.xlsx dans data/ ou définissez ECRF_EXPORT_XLSX_TEMPLATE_PATH.",
                self.template_xlsx_path,
            )
            return {}

        exporter = XlsEcrfExporter(
            template_path=self.template_xlsx_path,
            sheet_name=self.xls_sheet_name,
            header_row=self.xls_header_row,
            patient_id_column=self.xls_patient_id_column,
            overwrite_policy=self.overwrite_policy,
            empty_sentinels=self._empty_sentinels,
            column_aliases=self._column_aliases,
        )
        paths: dict[str, str] = {}
        try:
            if self.export_xls_update_master_workbook:
                master = exporter.write_to_master_workbook(result=result)
                paths["xlsx"] = str(master.resolve())
                paths["xlsx_master"] = paths["xlsx"]
            else:
                dest = output_dir / f"{basename}_ecrf.xlsx"
                written = exporter.export_patient_row(dest_path=dest, result=result)
                paths["xlsx"] = str(written.resolve())
        except XlsExportError as exc:
            _LOG.warning("Export XLS échoué : %s", exc)
            return paths

        if self.export_xls_mirror_to_output and self.export_xls_update_master_workbook:
            mirror = output_dir / f"{basename}_ecrf.xlsx"
            shutil.copy2(self.template_xlsx_path, mirror)
            paths["xlsx_mirror"] = str(mirror.resolve())
        return paths

    def _write_csv_mock(self, path: Path, result: PipelineResult) -> None:
        row: dict[str, Any] = {
            "patient_id": result.patient_id,
            "ma_patient_key": result.ma_patient_key,
            "study_id": result.study_id,
            "document_id": result.document_id,
            "document_type": result.document_type.value,
        }
        for upd in result.cell_updates:
            row[upd.column_key] = upd.value
        fieldnames = list(row.keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
