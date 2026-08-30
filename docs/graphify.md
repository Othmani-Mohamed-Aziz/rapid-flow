# Graphify: rapid-flow

Generated from the current `app/` package imports and the documented pipeline.

## Package Dependency Graph

```mermaid
flowchart LR
  orchestration["orchestration<br/>pipeline entrypoint"]
  ingestion["ingestion<br/>local documents"]
  parsing["parsing<br/>text/PDF, structuring, chunking"]
  routing["routing<br/>document type"]
  indexing["indexing<br/>memory/Qdrant, dense, BM25, rerank"]
  retrieval["retrieval<br/>workflow services"]
  extraction["extraction<br/>planner, strategies, LangExtract"]
  business_rules["business_rules<br/>mapping, normalization, validation"]
  etl["etl<br/>JSON, CSV/XLS export"]
  config["config<br/>settings, fields, study schema"]
  schemas["schemas<br/>Pydantic contracts"]
  utils["utils<br/>audit, path redaction"]

  orchestration --> ingestion
  orchestration --> parsing
  orchestration --> routing
  orchestration --> indexing
  orchestration --> retrieval
  orchestration --> extraction
  orchestration --> business_rules
  orchestration --> etl
  orchestration --> config
  orchestration --> schemas
  orchestration --> utils

  parsing --> config
  parsing --> schemas
  parsing --> parsing_pdf["parsing.pdf<br/>Docling/pypdf"]
  parsing_pdf --> config
  parsing_pdf --> schemas

  routing --> parsing
  routing --> schemas

  indexing --> config
  indexing --> schemas

  retrieval --> indexing
  retrieval --> schemas

  extraction --> schemas
  extraction --> extraction

  business_rules --> config
  business_rules --> extraction
  business_rules --> schemas

  etl --> schemas
  etl --> utils
  etl --> etl

  config --> schemas
  utils --> schemas

  classDef core fill:#e8f4fc,stroke:#1a73e8,color:#111827
  classDef shared fill:#e6f4ea,stroke:#137333,color:#111827
  classDef output fill:#fef7e0,stroke:#f9ab00,color:#111827
  class orchestration core
  class schemas,config shared
  class etl output
```

## Runtime Flow

```mermaid
flowchart TB
  input["Input document<br/>PDF/TXT/MD"] --> ingest["LocalFileIngestionService"]
  ingest --> parse["SmartParsingService"]
  parse --> route["DocumentRouter"]
  route --> chunk["ChunkingService"]
  chunk --> vector["VectorIndexService<br/>memory or Qdrant"]
  vector --> jobs["plan_extraction_jobs<br/>from StudySchema"]
  jobs --> retrieve["WorkflowOrchestrator / RetrievalService"]
  retrieve --> extract["ExtractionService<br/>lab, narrative, imaging"]
  extract --> map["FieldMappingService"]
  map --> normalize["NormalizationService"]
  normalize --> score["ConfidenceScoringService"]
  score --> validate["ValidationService"]
  validate --> export["EcrfExportService<br/>JSON/CSV/XLS"]

  schema["StudySchema<br/>data/study_schema_default.json"] --> jobs
  settings["Settings<br/>.env / ECRF_*"] --> parse
  settings --> vector
  settings --> export

  audit["AuditTrailService"] -. records .-> export
```

## Main Hubs

| Hub | Role | Key collaborators |
| --- | --- | --- |
| `app/orchestration/pipeline.py` | Assembles the end-to-end pipeline | ingestion, parsing, routing, indexing, retrieval, extraction, business rules, ETL |
| `app/schemas/*` | Shared contracts and enums | nearly every package |
| `app/config/study_schema_provider.py` | Loads the dynamic study schema | extraction planner, field registry, pipeline |
| `app/extraction/service.py` | Runs extraction jobs over retrieval hits | planner, strategy registry, schemas |
| `app/indexing/qdrant_vector_service.py` | Production vector backend | settings, embeddings, sparse encoder, reranker, schemas |
| `app/etl/export.py` | Materializes pipeline results | XLS helpers, overwrite policy, path redaction |

## Architectural Notes

- `schemas` is the stable inner layer; it should avoid depending on operational packages.
- `orchestration` intentionally has the broadest fan-in because it wires the application.
- `business_rules.normalization` imports RECIST helpers from `extraction.imaging_langextract`; if that grows, consider moving shared RECIST normalization into `business_rules` or `schemas`.
- `etl` and `extraction` contain several internal imports, which is expected because both packages have strategy/helper submodules.
