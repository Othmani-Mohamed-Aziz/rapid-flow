from __future__ import annotations

from enum import Enum


class DocumentType(str, Enum):
    """Types documentaires routés par le pipeline."""

    UNKNOWN = "unknown"
    LAB_BLOOD_PANEL = "lab_blood_panel"
    IMAGING_REPORT = "imaging_report"
    PATHOLOGY_REPORT = "pathology_report"
    CLINICAL_LETTER = "clinical_letter"


class FieldFamily(str, Enum):
    """Familles de champs / domaines cliniques."""

    HEMATOLOGY = "hematology"
    COAGULATION = "coagulation"
    HEPATIC_BIOCHEMISTRY = "hepatic_biochemistry"
    INFLAMMATION_BIOMARKERS = "inflammation_biomarkers"
    IMAGING_RECIST = "imaging_recist"
    COMORBIDITIES = "comorbidities"
    TREATMENT_LINES = "treatment_lines"


class ExtractionFamily(str, Enum):
    """Famille d'extraction (routage vers extracteurs / prompts)."""

    LAB_VALUES = "lab_values"
    IMAGING_RECIST = "imaging_recist"
    NARRATIVE_CLINICAL = "narrative_clinical"


class TemporalScope(str, Enum):
    """Portée temporelle déclarative pour un champ eCRF."""

    BASELINE = "baseline"
    FOLLOW_UP = "follow_up"
    FIRST_IMAGING = "first_imaging"
    SECOND_IMAGING = "second_imaging"
    LINE_L2 = "line_l2"
    LINE_L3 = "line_l3"
    LINE_L4 = "line_l4"
    UNSPECIFIED = "unspecified"


class TemporalLabel(str, Enum):
    """Libellé temporel attaché à une observation."""

    BASELINE = "baseline"
    FOLLOW_UP = "follow_up"
    FIRST_IMAGING_EVALUATION = "first_imaging_evaluation"
    SECOND_IMAGING_EVALUATION = "second_imaging_evaluation"
    LINE_L2 = "line_l2"
    LINE_L3 = "line_l3"
    LINE_L4 = "line_l4"
    UNKNOWN = "unknown"
