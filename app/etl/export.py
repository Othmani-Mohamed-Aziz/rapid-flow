from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.schemas.models import PipelineResult
from app.utils.safe_paths import redact_file_paths_in_jsonable


class EcrfExportService:
    """
    Export JSON complet + CSV « mock eCRF ».

    TODO: brancher `openpyxl` / `xlsxwriter` pour écriture XLS réelle.
    """

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
        return {"json": str(json_path.resolve()), "csv": str(csv_path.resolve())}

    def _write_csv_mock(self, path: Path, result: PipelineResult) -> None:
        row: dict[str, Any] = {
            "patient_id": result.patient_id,
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


class XlsExportPlaceholder:
    """TODO: implémenter l'export XLS binaire à partir de `PipelineResult`."""

    def write(self, result: PipelineResult, path: Path) -> None:
        raise NotImplementedError("TODO: export XLS via openpyxl.")
