from __future__ import annotations

from pathlib import Path

from app.config.settings import Settings
from app.orchestration.pipeline import PipelineOrchestrator
from app.parsing.base import ParsingService
from app.schemas.enums import DocumentType
from app.schemas.models import ParsedDocument, RawDocument, StructuredLabLine


class _StructuredLabParser(ParsingService):
    def parse(self, raw: RawDocument) -> ParsedDocument:
        return ParsedDocument(
            document_id=raw.document_id,
            patient_id=raw.patient_id,
            study_id=raw.study_id,
            full_text="Biochimie\nTransaminase ASAT (S.G.O.T) | 28.5 U/L",
            document_type_hint=DocumentType.LAB_BLOOD_PANEL,
            metadata={"chunking_strategy": "lab_rows"},
            source_path=raw.source_path,
            structured_lab_lines=[
                StructuredLabLine(
                    name="Transaminase ASAT (S.G.O.T).",
                    value=28.5,
                    unit="U/L",
                    section="Biochimie",
                    subsection="Biochimie",
                    raw_text="Transaminase ASAT (S.G.O.T). | 28.5 U/L",
                    confidence=0.9,
                )
            ],
        )


def test_pipeline_txt_still_runs_end_to_end() -> None:
    root = Path(__file__).resolve().parents[1]
    sample = root / "scripts" / "sample_data" / "mock_blood_panel.txt"
    result = PipelineOrchestrator(
        settings=Settings(export_xls_require_patient_in_ma=False),
    ).run(str(sample), patient_id="P1", study_id="S1")
    assert result.document_type.value == "lab_blood_panel"
    assert result.cell_updates


def test_pipeline_projects_document_lab_rows_and_exports_template(tmp_path: Path) -> None:
    report = tmp_path / "sample_lab.txt"
    report.write_text("placeholder", encoding="utf-8")
    result = PipelineOrchestrator(
        settings=Settings(
            export_xls_enabled=False,
            export_xls_require_patient_in_ma=False,
            vector_backend="memory",
        ),
        parsing=_StructuredLabParser(),
    ).run(str(report), patient_id="P1", study_id="EXAMPLE")

    assert any(update.column_key == "AST_start_AtezoBev_D0" for update in result.cell_updates)
    assert Path(result.export_paths["lab_template_json"]).is_file()
