# eCRF Autofill Pipeline — Internship Technical Handover

## 1. Purpose and scope

This document summarizes the engineering work completed during the internship on
`rapid-flow`, explains the changes that improved extraction metrics, and provides
reproducible instructions for running the pipeline and its evaluation.

The project converts local clinical documents (PDF, TXT, or Markdown) into
traceable eCRF field updates. The implemented scope focuses on:

- blood-test reports;
- imaging reports with baseline lesion size and RECIST response;
- local or Qdrant-backed retrieval;
- JSON, CSV, and Excel-compatible eCRF outputs;
- deterministic offline evaluation against paired gold templates.

> **Important:** the reported results are from synthetic or locally staged evaluation
> datasets. They measure the tested formats, not clinical performance on an independent
> hospital dataset.

---

## 2. Current pipeline

The production entry point is `run_pipeline()` in `app/orchestration/pipeline.py`.

```text
Document
  -> local ingestion
  -> PDF/text parsing
  -> document classification and structuring
  -> document-specific chunking
  -> memory or Qdrant indexing
  -> retrieval by field family
  -> StudySchema-driven extraction
  -> normalization and temporal mapping
  -> confidence thresholding
  -> field validation
  -> overwrite policy
  -> JSON / CSV / XLS / evaluation-template export
```

Main modules:

- `app/ingestion/`: loads local source documents.
- `app/parsing/`: pypdf/Docling parsing, document classification, section recovery,
  lab-row reconstruction, and document-specific chunking.
- `app/indexing/`: in-memory retrieval and Qdrant hybrid retrieval.
- `app/retrieval/`: retrieval orchestration by field family.
- `app/extraction/`: extraction planner and lab, narrative, and imaging strategies.
- `app/business_rules/`: mapping, normalization, temporal scope, confidence, and validation.
- `app/etl/`: JSON/CSV/XLS exports and evaluation-compatible prediction templates.
- `app/evaluation/`: dataset discovery, field/value metrics, retrieval metrics, and reports.
- `app/orchestration/pipeline.py`: complete end-to-end orchestration and audit trail.

---

## 3. Changes completed during the internship

### 3.1 Modular end-to-end architecture

The initial implementation established a modular pipeline instead of applying one
prompt to an entire document:

- Pydantic contracts for documents, chunks, observations, candidates, updates, and audit
  records.
- Local ingestion for PDF and text files.
- A single `ParsedDocument` contract shared by all parser backends.
- Heuristic routing into document types and field families.
- Dependency-injected services so parsing, retrieval, and extraction can be tested
  independently.
- Audit records for every major pipeline stage.
- Configurable runtime through `ECRF_*` environment variables.

This separation made it possible to improve lab and imaging extraction without changing
unrelated stages.

### 3.2 PDF parsing and document classification

The parsing layer was made robust to different environments and document layouts:

- `pypdf` is the lightweight parser and fallback.
- Docling is used when available for structured PDF content and tables.
- `auto` mode falls back to pypdf when Docling is unavailable or fails.
- Docling imports are lazy, avoiding startup errors when the optional dependency is absent.
- Parsed sections are aligned with `full_text` so character spans remain traceable.
- Contextual document scoring prevents CT/morphology documents from being misclassified as
  blood-test reports.
- Document dates and parser metadata are retained in the parsed result.

Relevant files:

- `app/parsing/smart_service.py`
- `app/parsing/document_type_scoring.py`
- `app/parsing/pdf/`
- `app/parsing/structuring.py`
- `app/parsing/text_enrichment.py`

### 3.3 Document-specific chunking

Three specialized strategies now replace generic line splitting:

1. `SectionBasedChunkingService` preserves clinical sections and verified character spans.
2. `LabRowChunkingService` groups complete reconstructed analyte rows by subsection.
3. `ImagingReportChunkingService` keeps a short imaging report intact, preserving the
   relationship between baseline measurements, current measurements, distractor lesions,
   and the final RECIST conclusion.

This imaging change fixed empty or schema-invalid LangExtract results caused by splitting
related evidence into separate chunks.

Relevant files:

- `app/parsing/chunking.py`
- `app/parsing/smart_service.py`

### 3.4 Hybrid retrieval and tenant isolation

The retrieval layer supports:

- an in-memory backend for fast development and deterministic tests;
- Qdrant collections isolated by tenant/study;
- multilingual E5 dense embeddings;
- BM25 sparse vectors;
- reciprocal-rank fusion in Qdrant;
- optional BGE cross-encoder reranking;
- mandatory tenant filters;
- embedding-version filters and mismatch safeguards;
- retry/backoff for transient Qdrant errors;
- optional read-only mode;
- deletion by document after temporary evaluation indexing.

The extraction planner obtains retrieval queries from the active `StudySchema` and only
retrieves the field families needed for the current document.

Relevant files:

- `app/indexing/qdrant_vector_service.py`
- `app/indexing/qdrant_bootstrap.py`
- `app/indexing/embeddings.py`
- `app/indexing/sparse.py`
- `app/indexing/reranker.py`
- `app/retrieval/workflow_orchestrator.py`

### 3.5 StudySchema-driven extraction

Extraction configuration was moved into a versioned study schema:

- fields declare canonical keys, document types, extraction strategy, temporal scope,
  normalization rule, target column, and autofill threshold;
- extraction jobs are planned dynamically for the routed document type;
- extraction classes and canonical mappings can be extended through an extraction catalog;
- filled eCRF columns can be excluded before retrieval and extraction.

This avoids hard-coding one study into the orchestration code.

Relevant files:

- `data/study_schema_default.json`
- `app/schemas/study_schema.py`
- `app/config/study_schema_provider.py`
- `app/extraction/planner.py`
- `app/extraction/strategy_extractors.py`

### 3.6 Blood-test extraction improvements

The lab path became primarily deterministic:

- PDF text and Docling Markdown tables are reconstructed into `StructuredLabLine` records.
- Section and subsection are preserved.
- Dotted rows, inline rows, split label/value columns, and values printed before labels are
  supported.
- Reference ranges, headers, footers, and page noise are filtered.
- Numeric, inequality, and qualitative results are supported.
- Dual-unit duplicate results are prevented from shifting subsequent values.
- Common French/English analyte aliases map to canonical study keys.
- Complete structured rows bypass the top-k retrieval gate and are projected directly to
  field candidates.
- Lab predictions are exported in the nested evaluator format:
  `Section/Subsection/Analyte/{valeur, unité}`.

Relevant files:

- `app/parsing/lab_post_processor.py`
- `app/parsing/chunking.py`
- `app/extraction/lab_heuristics.py`
- `app/extraction/strategy_extractors.py`
- `app/etl/lab_template_export.py`

### 3.7 Imaging and RECIST extraction

Imaging extraction uses LangExtract with local Ollama:

- model: `gemma2:2b`;
- URL: `http://localhost:11434`;
- timeout: 120 seconds;
- schema: `imaging-recist-langextract-v1`;
- default imaging chunk concurrency: 8, reduced to 1 during the benchmark runs.

The two evaluated fields are:

- `Size_major_nodule_mm_start_AtezoBev_D0`
- `Response_at_first_imaging_RECIST`

The major accuracy fixes were:

1. **Unit-aware normalization**
   - `cm`, `centimètre`, `mm`, and `millimètre` forms are recognized.
   - Centimetres are converted to millimetres before field export.

2. **Baseline target-lesion selection**
   - measurements near `J0`, `D0`, `inclusion`, `scanner initial`, and `référence` receive
     higher confidence;
   - current-exam measurements are down-ranked;
   - non-target and new-lesion measurements are down-ranked;
   - a grounded source-text fallback recovers an omitted baseline measurement.

3. **Conclusion-only RECIST**
   - RECIST mentions in the indication and intermediate findings are ignored when a
     conclusion exists;
   - official conclusion cues are detected;
   - `non évaluable` is normalized to `NE` before searching for misleading mentions of
     progression;
   - a deterministic conclusion fallback is used if LangExtract misses the final category.

4. **Better extraction instructions**
   - the prompt explicitly requests the baseline target and final conclusion;
   - a hard few-shot demonstrates `5.3 cm -> 53 mm`;
   - the few-shot contains a misleading `PD` in the indication and a final `NE` conclusion.

5. **Evaluator-compatible export**
   - only validated imaging updates are exported;
   - paths and units match the gold-template contract.

Relevant files:

- `app/extraction/imaging_langextract.py`
- `app/extraction/imaging_config.py`
- `app/business_rules/normalization.py`
- `app/schemas/study_schema.py`
- `app/etl/imaging_template_export.py`

### 3.8 Mapping, confidence, temporal logic, and validation

After extraction:

- observations map through canonical keys rather than source labels;
- values are normalized according to the field definition;
- temporal scopes such as baseline and first imaging are attached;
- extraction confidence can be blended with retrieval confidence;
- the highest-scoring validated candidate is retained per target column;
- field type and output constraints are checked before export;
- every decision retains source evidence and provenance.

### 3.9 eCRF export and overwrite safety

The ETL layer was extended beyond the original mock CSV:

- real XLS updates through `openpyxl`;
- patient-row resolution and optional patient-ID mapping;
- aliases between schema columns and MA workbook headers;
- duplicate-header detection;
- configurable empty sentinels;
- overwrite policies: `empty_only`, `always`, and `never`;
- extraction prefiltering for already-filled columns;
- optional update of the master workbook or an output copy;
- JSON and CSV pipeline outputs;
- nested lab and imaging prediction templates for evaluation.

Relevant files:

- `app/etl/xls_export.py`
- `app/etl/overwrite_policy.py`
- `app/etl/patient_resolver.py`
- `app/etl/column_aliases.py`
- `app/etl/lab_template_export.py`
- `app/etl/imaging_template_export.py`

### 3.10 Evaluation framework

An offline evaluator was added to separate field presence from value correctness:

- discovers report, empty-template, filled-template, and prediction quartets;
- validates nested JSON contracts and rejects duplicate keys;
- flattens fields by their complete nested path;
- normalizes text and unit aliases;
- supports numeric absolute and relative tolerances;
- reports TP, FP, FN, precision, recall, F1, autofill, false-autofill, validation pass
  rate, and value accuracy;
- reports per-sample mismatches and out-of-template paths;
- computes micro and macro aggregates;
- optionally computes silver Recall@K for retrieval;
- supports command-line quality thresholds for CI;
- writes a concise `summary.txt` and full `evaluation_report.json`;
- includes a small deterministic fixture in `tests/fixtures/evaluation/`.

Relevant files:

- `app/evaluation/`
- `scripts/run_evaluation.py`
- `tests/test_evaluation_metrics.py`
- `tests/test_evaluation_integration.py`
- `.github/workflows/ci.yml`

### 3.11 Testing, CI, and developer environment

The project includes:

- unit tests for parsing, chunking, extraction, mapping, normalization, validation, ETL,
  evaluation, and Qdrant behavior;
- integration tests with optional real PDFs;
- a deterministic evaluation marker in CI;
- Python 3.12 and 3.13 unit-test jobs;
- a Python 3.13 full integration job with Qdrant;
- Ruff, MyPy, pip-audit, and pre-commit;
- Docker Compose files for Qdrant, development, tests, and optional Ollama;
- a protected-main branch workflow based on pull requests.

---

## 4. Metric progression and causes

### 4.1 Blood-test extraction

The comparable 120-sample subset changed as follows:

| Metric | Before structured-row recall work | After |
|---|---:|---:|
| Precision | 1.0000 | 1.0000 |
| Recall | 0.8086 | 0.9569 |
| F1 | 0.8942 | 0.9780 |
| Value accuracy | 0.9452 | 0.9320 |
| False-autofill | 0.0000 | 0.0000 |

The recall and F1 gain came from reconstructing complete lab rows, retaining
section/subsection context, supporting more PDF layouts, mapping aliases, and bypassing the
top-k retrieval gate for structured rows.

The slight value-accuracy reduction is not hidden: more previously missed values became
eligible for comparison, exposing additional value mismatches.

Final 1,000-sample lab result:

- precision: **1.0000**
- recall: **0.9425**
- F1: **0.9704**
- value accuracy: **0.9264**
- false-autofill: **0.0000**
- report:
  `outputs/evaluation/full_after_recall/f303c1539a814c6199be3f9d97bfbc4d/`

### 4.2 Imaging extraction

The first easy synthetic imaging set produced value accuracy 1.0000 and F1 0.9962 on
1,000 samples. This result is useful as a regression check but is optimistic because the
reports and gold data used highly regular co-generated phrasing.

A harder synthetic set was therefore created with:

- cm/mm variation;
- baseline and current measurements;
- one target plus one to three non-target lesions;
- new-lesion distractors;
- misleading informal response wording;
- an official RECIST conclusion that can conflict with earlier text.

Comparable hard-set result on the same 152 completed samples:

| Metric | Before targeted fixes | After targeted fixes |
|---|---:|---:|
| Precision | 1.0000 | 1.0000 |
| Recall | 0.9671 | 1.0000 |
| F1 | 0.9833 | 1.0000 |
| Value accuracy | 0.6395 | 0.9901 |
| False-autofill | 0.0000 | 0.0000 |

Field-level result after the fixes:

- baseline size: **152/152 correct**;
- RECIST response: **149/152 correct**;
- both fields correct in the same report: **149/152**;
- remaining errors: two `CR -> SD` and one `CR -> PR`.

The before/after reports are:

- before:
  `outputs/evaluation/imaging_hard_200/521fe3c3d93b41978977fd8349935e98/`
- after:
  `outputs/evaluation/imaging_hard_200/aa51f9e137b543aeaa9744db5d0a4fd3/`

> The hard run was requested for 200 reports, but extraction was stopped after 152 had
> completed. The evaluator was run on those 152 complete quartets only.

### 4.3 How to interpret the metrics

- **Precision/recall/F1** describe whether expected schema fields were filled.
- **Value accuracy** describes whether a filled value matched the gold value.
- **Validation pass rate** checks type, path, and declared-unit compatibility.
- **False-autofill** measures fields filled when the gold template was empty.

A high F1 does not imply correct values. The hard imaging baseline showed this clearly:
F1 was 0.9833 while value accuracy was only 0.6395.

---

## 5. Supervisor setup

### 5.1 Clone and select the handover branch

```powershell
git clone git@github.com:sbenhadj/rapid-flow.git
cd rapid-flow
git switch internship-handover
```

### 5.2 Create the Python environment

Python 3.12 or 3.13 is recommended. Python 3.14 on Windows is unstable with the full
FastEmbed/ONNX stack.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,docling,extraction]"
Copy-Item .env.example .env
```

For Qdrant hybrid retrieval, also install:

```powershell
pip install -e ".[dev,vector,docling,extraction]"
python -X utf8 scripts\prefetch_vector_models.py
docker compose -f docker-compose.qdrant.yml up -d
```

### 5.3 Ollama for imaging extraction

Ollama is required to regenerate imaging predictions, but not to score predictions that
already exist.

```powershell
ollama pull gemma2:2b
ollama serve
```

Expected `.env` values:

```ini
ECRF_LANGEXTRACT_ENABLED=true
ECRF_OLLAMA_MODEL_ID=gemma2:2b
ECRF_OLLAMA_URL=http://localhost:11434
ECRF_OLLAMA_TIMEOUT_S=120
ECRF_LANGEXTRACT_SCHEMA_VERSION=imaging-recist-langextract-v1
```

The application does not define a separate Ollama Modelfile or system message.
LangExtract receives the extraction description and two few-shot examples from
`app/extraction/imaging_langextract.py`.

---

## 6. Evaluation dataset contract

Every evaluated sample must have a complete quartet:

```text
<dataset-root>/
  reports/
    sample_0001.pdf
  filtered_templates/
    empty/
      sample_0001_template_empty.json
    filled/
      sample_0001_template_filled.json
  extracted/
    sample_0001_template.json
```

Meaning:

- `reports/`: source document;
- `empty/`: schema fields eligible for evaluation;
- `filled/`: verified gold values;
- `extracted/`: pipeline predictions only.

Never copy a filled gold template into `extracted/`; that would create evaluation leakage.
If any labelled sample is missing part of the quartet, the CLI writes a report but returns
exit code 2 and lists the incomplete sample.

---

## 7. How to run the evaluation

### 7.1 Fast deterministic evaluator test

This test uses the small committed fixture and needs no external data, Ollama, or Qdrant:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest -q -m evaluation
```

### 7.2 Score saved predictions

If `extracted/` is already populated:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\run_evaluation.py `
  --dataset-root "C:\path\to\dataset" `
  --skip-retrieval `
  --output-dir "outputs\evaluation\supervisor_run"
```

Example using the local hard-imaging subset:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\run_evaluation.py `
  --dataset-root "outputs\imaging_eval_ready" `
  --skip-retrieval `
  --output-dir "outputs\evaluation\imaging_hard_supervisor"
```

### 7.3 Regenerate predictions, then score them

Use `scripts/run_dataset_extraction.py` to run the complete pipeline on all labelled PDFs:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\run_dataset_extraction.py `
  --dataset-root "C:\path\to\dataset" `
  --study-id EXAMPLE `
  --vector-backend memory `
  --workers 2 `
  --imaging-workers 1 `
  --overwrite
```

Then evaluate:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\run_evaluation.py `
  --dataset-root "C:\path\to\dataset" `
  --skip-retrieval `
  --output-dir "outputs\evaluation\supervisor_run"
```

Useful extraction options:

- `--limit 20`: process only the first 20 labelled reports;
- `--workers 2`: process two reports concurrently;
- `--imaging-workers 1`: keep LangExtract calls sequential inside each report;
- omit `--overwrite`: resume and skip predictions already present;
- `--vector-backend qdrant`: exercise the configured Qdrant backend.

For the 2B model on a normal workstation, start with two report workers and one imaging
worker. Higher concurrency can increase memory pressure and reduce stability.

### 7.4 Apply quality thresholds

The evaluator can fail CI when a metric is below a required value:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\run_evaluation.py `
  --dataset-root "C:\path\to\dataset" `
  --skip-retrieval `
  --min-precision 0.95 `
  --min-recall 0.90 `
  --min-f1 0.92 `
  --min-value-accuracy 0.90 `
  --max-false-autofill-rate 0.01
```

Exit codes:

- `0`: evaluation and thresholds passed;
- `1`: at least one metric threshold failed;
- `2`: invalid or incomplete evaluation input.

### 7.5 Optional retrieval evaluation

Remove `--skip-retrieval` to parse and index each report and compute silver Recall@K:

```powershell
$env:ECRF_VECTOR_BACKEND = "memory"
$env:ECRF_ENABLE_SPARSE_BM25 = "false"
$env:ECRF_ENABLE_RERANKER = "false"

.\.venv\Scripts\python.exe -X utf8 scripts\run_evaluation.py `
  --dataset-root "C:\path\to\dataset" `
  --k 1 3 5 `
  --output-dir "outputs\evaluation\retrieval_run"
```

Silver qrels are inferred by locating an exact label/value pair in a unique chunk. They are
not manually annotated relevance judgements. Unmatched or ambiguous fields are excluded
and reported.

---

## 8. Data access

### 8.1 What is available from Git

The branch contains:

- source code;
- configuration examples;
- unit and integration tests;
- the tiny synthetic fixture under `tests/fixtures/evaluation/`;
- `ctreport.py`, the Colab-exported hard CT generator;
- this handover document.

### 8.2 What is intentionally not in Git

The following are local-only:

- `data/` medical documents;
- `outputs/` pipeline and evaluation results;
- `Synthetic_Body_CT_Reports/` generated corpus;
- `.env` and credentials.

This is intentional. The project policy forbids committing medical documents, including
documents treated as synthetic test data, and all runtime outputs are gitignored.

The supervisor must obtain the evaluation dataset through the approved encrypted internal
storage/Drive location and place it anywhere locally using the contract in Section 6.
The evaluator accepts an absolute `--dataset-root`, so the data does not need to be copied
into the repository.

### 8.3 Existing local data used during development

On the internship workstation:

- full hard CT generator output:
  `Synthetic_Body_CT_Reports/`;
- staged 200-report hard CT selection:
  `outputs/imaging_hard_200/`;
- complete 152-report before/after comparison set:
  `outputs/imaging_eval_ready/`;
- lab evaluation dataset root recorded by the final report:
  `data/`.

These paths are not available after cloning the repository unless the corresponding data is
transferred separately.

### 8.4 Regenerating the hard imaging data

`ctreport.py` is a Google Colab export. It:

- installs `fpdf2` and `faker`;
- uses random seed `20260821`;
- generates 1,000 synthetic CT PDFs;
- creates matching empty and filled templates;
- writes gold baseline target size in mm and official RECIST conclusion;
- copies the generated files to:
  `/content/drive/MyDrive/Synthetic_Body_CT_Reports/`.

Run it as a Colab notebook/export because it intentionally contains Colab syntax such as
`!pip` and `drive.mount`. The generated `extracted/` directory starts empty; only the
pipeline should populate it.

---

## 9. Output access

### 9.1 Pipeline output

Each pipeline run writes under:

```text
outputs/<document_id>/
```

Depending on configuration and document type, this can include:

- `pipeline_output.json`: complete observations, updates, metadata, and audit trail;
- `pipeline_output_ecrf_mock.csv`: mock eCRF update export;
- an XLS workbook or updated master workbook;
- `<source_stem>_template.json`: evaluator-compatible lab or imaging prediction.

The returned `PipelineResult.export_paths` contains the exact paths.

### 9.2 Evaluation output

Every evaluation creates a unique run directory:

```text
outputs/evaluation/<optional-group>/<run_id>/
  summary.txt
  evaluation_report.json
```

- `summary.txt` contains aggregate metrics.
- `evaluation_report.json` contains settings, counts, micro/macro metrics, per-sample
  mismatches, contract errors, extra paths, and optional retrieval results.

Example:

```powershell
Get-Content "outputs\evaluation\supervisor_run\<run_id>\summary.txt"
code "outputs\evaluation\supervisor_run\<run_id>\evaluation_report.json"
```

These results are also local and gitignored. Copy the chosen report directory to the
approved handover storage if it must be retained outside the workstation.

---

## 10. Verification commands

Before accepting a change:

```powershell
ruff check app tests scripts
ruff format --check app tests scripts
mypy app
pytest -q
```

Focused evaluation and imaging tests:

```powershell
pytest -q -m evaluation
pytest -q tests\test_imaging_langextract.py `
  tests\test_extraction_normalization.py `
  tests\test_imaging_template_export.py
```

The main branch is protected. Changes should follow:

```text
feature branch -> commit -> push -> pull request -> CI -> review -> merge
```

---

## 11. Known limitations and next steps

1. The reported datasets are synthetic or locally staged; independent clinical validation
   is still required.
2. The hard imaging before/after benchmark contains 152 completed samples, not the full
   requested 200.
3. Three hard-set RECIST complete-response cases remained incorrect.
4. Silver retrieval relevance is inferred, not manually annotated.
5. Validation currently checks mainly schema/type/unit contracts; deeper clinical
   cross-field validation is still needed.
6. A human review workflow should precede any production eCRF write.
7. Pseudonymization must be implemented before indexing real identifying data.
8. Multi-document patient conflict resolution is not yet implemented.
9. The current workflow is script-based; an HTTP API/job layer remains future work.
10. Operational telemetry, latency metrics, and model-drift monitoring remain to be added.

The detailed backlog is in `docs/BACKLOG_PIPELINE.md`.

---

## 12. Handover checklist

- [ ] Supervisor can clone and check out `internship-handover`.
- [ ] Python 3.12/3.13 environment installs successfully.
- [ ] `pytest -q -m evaluation` passes without private data.
- [ ] Approved dataset is transferred separately.
- [ ] Dataset contains complete report/empty/filled/extracted quartets.
- [ ] Saved predictions can be evaluated without Ollama.
- [ ] Ollama and `gemma2:2b` are available before regenerating imaging predictions.
- [ ] `summary.txt` and `evaluation_report.json` are archived after supervisor testing.
- [ ] Synthetic results are not presented as independent clinical validation.
