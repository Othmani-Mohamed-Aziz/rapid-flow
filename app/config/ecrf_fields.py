"""
Champs eCRF exemple (bootstrap uniquement).

Source de vérité runtime : `data/study_schema_default.json` (ou `ECRF_STUDY_SCHEMA_PATH`).
Ce module sert à générer le JSON initial via `study_schema_from_example_fields()`.
"""

from __future__ import annotations

from app.schemas.enums import DocumentType, ExtractionFamily, FieldFamily, TemporalScope
from app.schemas.models import FieldDefinition

EXAMPLE_ECRF_FIELDS: list[FieldDefinition] = [
    FieldDefinition(
        field_name="AFP_start_AtezoBev_D0",
        field_type="numeric",
        document_types_allowed=[DocumentType.LAB_BLOOD_PANEL],
        extraction_family=ExtractionFamily.LAB_VALUES,
        field_family=FieldFamily.INFLAMMATION_BIOMARKERS,
        temporal_scope=TemporalScope.BASELINE,
        normalization_rule="numeric_mg_l_or_ug_l",
        target_column="AFP_start_AtezoBev_D0",
        autofill_threshold=0.7,
    ),
    FieldDefinition(
        field_name="AST_start_AtezoBev_D0",
        field_type="numeric",
        document_types_allowed=[DocumentType.LAB_BLOOD_PANEL],
        extraction_family=ExtractionFamily.LAB_VALUES,
        field_family=FieldFamily.HEPATIC_BIOCHEMISTRY,
        temporal_scope=TemporalScope.BASELINE,
        normalization_rule="numeric_u_l",
        target_column="AST_start_AtezoBev_D0",
        autofill_threshold=0.65,
    ),
    FieldDefinition(
        field_name="ALT_start_AtezoBev_D0",
        field_type="numeric",
        document_types_allowed=[DocumentType.LAB_BLOOD_PANEL],
        extraction_family=ExtractionFamily.LAB_VALUES,
        field_family=FieldFamily.HEPATIC_BIOCHEMISTRY,
        temporal_scope=TemporalScope.BASELINE,
        normalization_rule="numeric_u_l",
        target_column="ALT_start_AtezoBev_D0",
        autofill_threshold=0.65,
    ),
    FieldDefinition(
        field_name="Total_bilirubine_D0",
        field_type="numeric",
        document_types_allowed=[DocumentType.LAB_BLOOD_PANEL],
        extraction_family=ExtractionFamily.LAB_VALUES,
        field_family=FieldFamily.HEPATIC_BIOCHEMISTRY,
        temporal_scope=TemporalScope.BASELINE,
        normalization_rule="numeric_umol_l",
        target_column="Total_bilirubine_D0",
        autofill_threshold=0.65,
    ),
    FieldDefinition(
        field_name="PLT_start_AtezoBev_D0",
        field_type="numeric",
        document_types_allowed=[DocumentType.LAB_BLOOD_PANEL],
        extraction_family=ExtractionFamily.LAB_VALUES,
        field_family=FieldFamily.HEMATOLOGY,
        temporal_scope=TemporalScope.BASELINE,
        normalization_rule="numeric_g_l",
        target_column="PLT_start_AtezoBev_D0",
        autofill_threshold=0.65,
    ),
    FieldDefinition(
        field_name="Cirrhosis",
        field_type="boolean",
        document_types_allowed=[
            DocumentType.CLINICAL_LETTER,
            DocumentType.LAB_BLOOD_PANEL,
        ],
        extraction_family=ExtractionFamily.NARRATIVE_CLINICAL,
        field_family=FieldFamily.COMORBIDITIES,
        temporal_scope=TemporalScope.UNSPECIFIED,
        normalization_rule="bool_yes_no",
        target_column="Cirrhosis",
        autofill_threshold=0.75,
    ),
    FieldDefinition(
        field_name="Size_major_nodule_mm_start_AtezoBev_D0",
        field_type="numeric",
        document_types_allowed=[DocumentType.IMAGING_REPORT],
        extraction_family=ExtractionFamily.IMAGING_RECIST,
        field_family=FieldFamily.IMAGING_RECIST,
        temporal_scope=TemporalScope.BASELINE,
        normalization_rule="numeric_mm",
        target_column="Size_major_nodule_mm_start_AtezoBev_D0",
        autofill_threshold=0.7,
    ),
    FieldDefinition(
        field_name="Response_at_first_imaging_RECIST",
        field_type="categorical",
        document_types_allowed=[DocumentType.IMAGING_REPORT],
        extraction_family=ExtractionFamily.IMAGING_RECIST,
        field_family=FieldFamily.IMAGING_RECIST,
        temporal_scope=TemporalScope.FIRST_IMAGING,
        normalization_rule="recist_category",
        target_column="Response_at_first_imaging_RECIST",
        autofill_threshold=0.75,
    ),
]
