from __future__ import annotations

from app.parsing.lab_post_processor import LabReportPostProcessor
from app.schemas.enums import DocumentType
from app.schemas.models import DocumentSection, ParsedDocument


def test_lab_postprocessor_parses_colon_lines() -> None:
    text = """Hépatite
AST : 48 U/L
ALT : 32 U/L
Plaquettes : 145 G/L
"""
    parsed = ParsedDocument(
        document_id="d",
        patient_id="p",
        study_id="s",
        full_text=text,
        document_type_hint=DocumentType.LAB_BLOOD_PANEL,
    )
    out = LabReportPostProcessor().enrich(parsed)
    names = {ln.name.lower() for ln in out.structured_lab_lines}
    assert "ast" in names or any("AST" in ln.name for ln in out.structured_lab_lines)
    assert out.metadata.get("structured_lab_line_count", 0) >= 2


def test_lab_postprocessor_uses_sections_when_present() -> None:
    parsed = ParsedDocument(
        document_id="d",
        patient_id="p",
        study_id="s",
        full_text="ignored body duplicate",
        document_type_hint=DocumentType.LAB_BLOOD_PANEL,
        structured_sections=[
            DocumentSection(heading="Biochimie", body="CRP : 5 mg/L\n", metadata={}),
        ],
    )
    out = LabReportPostProcessor().enrich(parsed)
    assert any(ln.name.upper().startswith("CRP") for ln in out.structured_lab_lines)


def test_lab_postprocessor_ignores_reference_and_page_noise() -> None:
    parsed = ParsedDocument(
        document_id="d",
        patient_id="p",
        study_id="s",
        full_text="",
        document_type_hint=DocumentType.LAB_BLOOD_PANEL,
        structured_sections=[
            DocumentSection(
                heading="Biochimie",
                body="AST....................\n28.5 U/L\nInf. à 55\nPage 1\n",
                metadata={},
            ),
        ],
    )
    out = LabReportPostProcessor().enrich(parsed)
    raw = [ln.raw_text for ln in out.structured_lab_lines]
    assert any("AST" in r for r in raw)
    assert all("Inf. à" not in r for r in raw)
    assert all("Page" not in r for r in raw)


def test_lab_postprocessor_parses_collapsed_rows_and_subsections() -> None:
    text = (
        "Hematologie NumerationGlobulaire Globules rouges.................... 4.4 10^6/mm3 "
        "4.2 à 5.4 FormuleLeucocytaire Neutrophiles.................... 25% : 1652 /mm3 "
        "Hemostase INR.................... 0.9 1.0 à 1.0 "
        "Biochimie Transaminase ASAT (S.G.O.T).................... 28.5 U/L 5 à 34"
    )
    parsed = ParsedDocument(
        document_id="d",
        patient_id="p",
        study_id="s",
        full_text=text,
        document_type_hint=DocumentType.LAB_BLOOD_PANEL,
    )

    rows = LabReportPostProcessor().enrich(parsed).structured_lab_lines

    by_name = {row.name: row for row in rows}
    assert by_name["Globules rouges"].subsection == "NumerationGlobulaire"
    assert by_name["Neutrophiles"].value == 25
    assert by_name["INR"].section == "Hemostase"
    assert by_name["Transaminase ASAT (S.G.O.T)"].value == 28.5


def test_lab_postprocessor_parses_markdown_table_rows() -> None:
    parsed = ParsedDocument(
        document_id="d",
        patient_id="p",
        study_id="s",
        full_text=(
            "Hormonologie\nPage 2\n"
            "| Magnésium plasmatique.................... | 1.8 mg/L | 1.8 à 2 |\n"
            "| Albumine.................... | 44.6 g/L | 35 à 50 |"
        ),
        document_type_hint=DocumentType.LAB_BLOOD_PANEL,
    )

    rows = LabReportPostProcessor().enrich(parsed).structured_lab_lines

    assert [(row.name, row.value, row.unit) for row in rows] == [
        ("Magnésium plasmatique", 1.8, "mg/L"),
        ("Albumine", 44.6, "g/L"),
    ]
    assert all(row.section == "Biochimie" for row in rows)


def test_lab_postprocessor_keeps_hormonology_analytes_out_of_biochemistry() -> None:
    parsed = ParsedDocument(
        document_id="d",
        patient_id="p",
        study_id="s",
        full_text=(
            "## Hormonologie\n"
            "| Albumine.................... | 44.6 g/L | 35 à 50 |\n"
            "| T.S.H. ultra-sensible.................... | 2.1 mUI/L | 0.35 à 4.94 |\n"
            "| 25 OH vitamine D.................... | 24.2 ng/mL | 30 à 60 |\n"
        ),
        document_type_hint=DocumentType.LAB_BLOOD_PANEL,
    )

    rows = LabReportPostProcessor().enrich(parsed).structured_lab_lines

    sections = {row.name: (row.section, row.subsection) for row in rows}
    assert sections["Albumine"] == ("Biochimie", "Biochimie")
    assert sections["T.S.H. ultra-sensible"] == ("Hormonologie", "Hormonologie")
    assert sections["25 OH vitamine D"] == ("Hormonologie", "Hormonologie")


def _lab_rows(text: str) -> dict[str, tuple[object, str | None]]:
    parsed = ParsedDocument(
        document_id="d",
        patient_id="p",
        study_id="s",
        full_text=text,
        document_type_hint=DocumentType.LAB_BLOOD_PANEL,
    )
    rows = LabReportPostProcessor().enrich(parsed).structured_lab_lines
    return {row.name: (row.value, row.unit) for row in rows}


def test_lab_postprocessor_keeps_inequality_results_but_not_reference_ranges() -> None:
    rows = _lab_rows(
        "Hormonologie\n\nTestostérone....................\n\nInf à 0.8 ng/ml\n\n0.3 à 1\n\n"
        "Parathormone....................\n\n2.3 pmol/L\n\n0.69 à 3.90\n"
    )

    assert rows["Testostérone"] == ("Inf à 0.8", "ng/ml")
    assert rows["Parathormone"] == (2.3, "pmol/L")


def test_lab_postprocessor_recovers_value_pushed_after_reference_tables() -> None:
    rows = _lab_rows(
        "Hormonologie\n\nTestostérone....................\n\n"
        "## Testostérone - Valeurs usuelles\n\n"
        "| Fille (ng/mL)        | Garçon (ng/mL)         |\n"
        "|----------------------|------------------------|\n"
        "| Nouveau-née : < 0,03 | Nouveau-né : 0,4 - 8,6 |\n\n"
        "Inf à 0.4 ng/ml\n\n0.3 à 1\n"
    )

    assert rows["Testostérone"] == ("Inf à 0.4", "ng/ml")


def test_lab_postprocessor_parses_qualitative_urine_results() -> None:
    rows = _lab_rows(
        "CytologieUrinaire\n\nCELLULES EPITHELIALES....................\n\nTrès nombreuses\n\n"
        "CRISTAUX....................\n\nRares\n\nCYLINDRES....................\n\nAbsentes\n"
    )

    assert rows["CELLULES EPITHELIALES"] == ("Très nombreuses", None)
    assert rows["CRISTAUX"] == ("Rares", None)
    assert rows["CYLINDRES"] == ("Absentes", None)


def test_lab_postprocessor_aligns_column_of_labels_with_column_of_results() -> None:
    rows = _lab_rows(
        "Hemostase\n\nTaux de prothrombine....................\n\nINR....................\n\n"
        "Ratio patient/témoin....................\n\nFibrinogène....................\n\n"
        "Biochimie\n\nCréatinine....................\n\n"
        "Biochimie\n\n121.0 %\n\n1.0\n\n1.5\n\n3.4 g/L\n\n61.9 umol/L\n\n5.47 mg/L\n\n"
        "70 à 100\n\n1.0 à 1.0\n\nInf. à 1.20\n\n2.0 à 4.0\n\n49 à 90\n"
    )

    assert rows["Taux de prothrombine"] == (121, "%")
    assert rows["INR"] == (1, None)
    assert rows["Ratio patient/témoin"] == (1.5, None)
    assert rows["Fibrinogène"] == (3.4, "g/L")
    # Le doublon massique 5.47 mg/L ne doit pas décaler la colonne.
    assert rows["Créatinine"] == (61.9, "umol/L")


def test_lab_postprocessor_aligns_results_printed_before_their_labels() -> None:
    rows = _lab_rows(
        "CytologieUrinaire\n\n739.4 /mL\n\n1000 à 2000000\n\nTrès nombreuses\n\nAbsent\n\n"
        "LEUCOCYTES....................\n\nCELLULES EPITHELIALES....................\n\n"
        "CRISTAUX....................\n"
    )

    assert rows["LEUCOCYTES"] == (739.4, "/mL")
    assert rows["CELLULES EPITHELIALES"] == ("Très nombreuses", None)
    assert rows["CRISTAUX"] == ("Absent", None)


def test_lab_postprocessor_ignores_dotted_footer_sentences() -> None:
    rows = _lab_rows(
        "Hemostase\n\nINR....................\n\nRatio patient/témoin....................\n\n"
        "Fibrinogène....................\n\n"
        "Les informations de ce document sont enregistrées informatiquement. "
        "Vous disposez d'un droit d'accès et de rectification ...\n\n"
        "0.8\n\n1.5\n\n2.0 g/L\n\n1.0 à 1.0\n\nInf. à 1.20\n\n2.0 à 4.0\n"
    )

    assert rows["INR"] == (0.8, None)
    assert rows["Ratio patient/témoin"] == (1.5, None)
    assert rows["Fibrinogène"] == (2, "g/L")
