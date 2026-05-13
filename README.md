# eCRF Autofill — backend modulaire (V1)

Pipeline **orientée routing, extraction spécialisée par famille de champs et traçabilité**, sans prompt monolithique sur l'intégralité du document.

> Toutes les commandes opérationnelles (Docker, CI, pre-commit, debug) sont listées dans [`useful_commands.md`](./useful_commands.md).

---

## Architecture

```
ingestion ─► parsing ─► structuring ─► chunking ─► indexing ─► retrieval ─► extraction ─► business_rules ─► etl
            (PDF: pypdf | docling)   │           │ (memory | Qdrant)
                                     │           └─ embeddings dense (e5)
                                     │              + sparse BM25
                                     │              + reranker BGE
                                     └─ section-based ou heuristique lab
```

- **Ingestion / parsing / chunking** : `app/ingestion`, `app/parsing` (`SmartParsingService`, PDF via **pypdf** ou **Docling**, post-traitement labo via `LabReportPostProcessor`).
- **Structurer + segmenter** : `app/parsing/structuring.py` + `DoclingPdfParser` graph-aware ; fallback Markdown si la structure Docling est trop pauvre.
- **Chunking** : `SectionBasedChunkingService` (sections cliniques) ou `HeuristicLabChunkingService` (paragraphes labo). Span verification (`char_start`/`char_end`) garantie pour la traçabilité.
- **Routage documentaire** : `app/routing` (`DocumentRouter`).
- **Index / retrieval** : `app/indexing` (`InMemoryVectorIndexService` pour dev, `QdrantHybridVectorService` pour prod), `app/retrieval` (`RetrievalService`, orchestrateurs de workflow).
- **Extraction** : `app/extraction` (`LangExtractExtractor`, `LlamaExtractor`, `ExtractionService`) — sous-textes seulement.
- **Règles métier** : `app/business_rules` (temporalité, mapping champs, validation, score).
- **ETL sortie** : `app/etl` (JSON + CSV mock ; XLS prévu).
- **Orchestration** : `app/orchestration/pipeline.py` (`run_pipeline`).
- **Schémas** : `app/schemas` (Pydantic v2).
- **Registry champs** : `app/config/ecrf_fields.py` + `FieldRegistry`.

Workflow documentaire :

- `BaseWorkflowOrchestrator` + `LocalWorkflowOrchestrator` (V1, in-memory).
- `VectorStoreWorkflowOrchestrator` (V2, branché si `ECRF_VECTOR_BACKEND=qdrant`).
- `RagflowWorkflowOrchestrator` (TODO).

---

## Prérequis

| Élément | Recommandation |
|---|---|
| Python | **3.12 ou 3.13** (matrice CI testée). 3.11 fonctionne mais non testé en CI. **3.14 instable sur Windows** pour la stack hybride (fastembed/ONNX) — utilisez `dense_only` ou Docker. |
| Docker Desktop | Requis pour : Qdrant local, exécution conteneurisée de la suite (parité CI). |
| Disque | ~6 GiB libres (cache Hugging Face : e5 + bge-reranker + bm25). |
| Token HF | **Non nécessaire** pour les modèles publics. Uniquement si vous ajoutez un modèle « gated ». |

---

## Installation

### Locale (venv)

```powershell
cd "c:\Saima Work\AI4Cure\Code\rapid-flow"
python -m venv .venv
.\.venv\Scripts\activate

# Base + dev tools (pytest, ruff, mypy, pip-audit, pre-commit)
pip install -e ".[dev]"

# Stack vector store (Qdrant + embeddings + BM25 + reranker)
pip install -e ".[dev,vector]"

# Docling (parsing PDF complexes, OCR optionnel)
pip install -e ".[docling]"

# (optionnel) Couche LlamaIndex au-dessus de Qdrant
pip install -e ".[llamaindex]"
```

Copiez `.env.example` vers `.env` si besoin. Le fichier `.env` est chargé automatiquement (`app.config.settings`).

### Pré-télécharger les modèles HF (une fois, ~1.2 GiB)

```powershell
python -X utf8 scripts\prefetch_vector_models.py
```

Le script gère automatiquement : token expiré, fichier persisté `~/.cache/huggingface/token`, erreurs réseau Windows (`WinError 10054`/`10038`), absence du Mode Développeur (`WinError 1314`), et crashes ONNX au teardown.

### Activer pre-commit (optionnel mais recommandé)

```powershell
pre-commit install
pre-commit run --all-files   # premier check sur tout le repo
```

---

## Parsing PDF et bilans sanguins

- **Contrat** : le pipeline consomme toujours un `ParsedDocument` (`full_text`, `metadata`, `structured_sections`, `structured_lab_lines`, `source_path`, …).
- **Orchestrateur** : `SmartParsingService` route les **PDF** vers une pile PDF locale, les **.txt/.md** vers `HeuristicTextParser`.
- **PDF parsers** :
  - `PypdfPdfParser` — défaut / secours, 100 % local.
  - `DoclingPdfParser` — recommandé si installé (sectionnement graph-aware, headers/footers filtrés, tables matérialisées).
  - Mode **`auto`** : tente Docling puis retombe sur pypdf en cas d'erreur (réseau / 401 / absence de modèles).
- **Sectionnement** : `DoclingPdfParser` priorise `document.iterate_items()` (graph) pour produire des `DocumentSection` propres. Fallback Markdown `##` si le graph est trop pauvre.
- **Chunking** : `SectionBasedChunkingService` (sections cliniques) — taille max `ECRF_CHUNK_MAX_SECTION_CHARS` (défaut 12 000 caractères). Pour bilans sanguins : `HeuristicLabChunkingService`.
- **Post-traitement labo** : `LabReportPostProcessor` remplit `structured_lab_lines` pour les PDF « bilan » détectés (scoring contextuel par `is_probable_lab_document`).

### Démo parsing seul

```powershell
python scripts\run_pdf_demo.py "chemin\vers\rapport.pdf"
python scripts\run_pdf_demo.py "chemin\vers\rapport.pdf" --backend pypdf --json-out parsed.json
```

---

## Vector store : Qdrant + dense + BM25 + reranker

### Démarrer Qdrant local

```powershell
docker compose -f docker-compose.qdrant.yml up -d
curl http://localhost:6533/healthz   # attendu : "healthz check passed"
start http://localhost:6533/dashboard
```

Notre conteneur écoute sur **6533** (REST) / **6534** (gRPC) pour cohabiter avec un autre Qdrant local (typiquement starter-kit n8n sur 6333).

### Configurer la pipeline pour Qdrant

Dans `.env` :

```ini
ECRF_VECTOR_BACKEND=qdrant
ECRF_QDRANT_URL=http://localhost:6533
ECRF_ENABLE_SPARSE_BM25=true     # désactiver sur Python 3.14 + Windows
ECRF_ENABLE_RERANKER=true
```

### Isolation tenant

Une **collection par tenant** (option C) : nom = `{ECRF_QDRANT_COLLECTION_PREFIX}__{tenant_id_normalisé}`. Le `tenant_id` est obligatoire dans `upsert_chunks` / `search` / `delete_document` (`ValueError` sinon). Par défaut le pipeline passe `study_id` comme `tenant_id`.

### Recherche hybride

```
query
  │
  ├─► e5 (dense, cosine)  ┐
  │                        ├─► RRF fusion (Qdrant native) ─► top-K ─► [reranker BGE] ─► résultats
  └─► BM25 (sparse)       ┘
```

---

## Lancer la V1 (démo bilan sanguin)

```powershell
python scripts\run_local_demo.py
```

Sorties dans `outputs/<document_id>/` : `pipeline_output.json` et `pipeline_output_ecrf_mock.csv`.

---

## Tests

### Suite par défaut (rapide, sans Qdrant ni Docker)

```powershell
# Tout (mode `memory`, isolé via fixture autouse)
pytest -q

# Avec couverture (gate à 70 %)
pytest -q --cov=app --cov-fail-under=70

# Verbose + raison des skips
pytest -v -rs

# Par marker
pytest -m ct_integration_pdf
pytest -m real_vector_backend
```

### Tests d'intégration Qdrant (opt-in)

```powershell
docker compose -f docker-compose.qdrant.yml up -d
$env:ECRF_RUN_QDRANT_INTEGRATION="1"
pytest -v tests\test_qdrant_integration.py
Remove-Item Env:ECRF_RUN_QDRANT_INTEGRATION
```

Deux tests :
- `test_qdrant_end_to_end_hybrid` : dense + BM25 + RRF (auto-skip sur Python 3.14 + Windows).
- `test_qdrant_end_to_end_dense_only` : dense seul (compatible toutes plateformes).

### Tests conteneurisés (parité avec la CI)

```powershell
# Suite par défaut dans Python 3.13 Linux
docker compose -f docker-compose.test.yml run --rm test-runner

# Intégration Qdrant complète (la voie hybride passe vraiment ici)
docker compose -f docker-compose.test.yml --profile integration up -d qdrant
docker compose -f docker-compose.test.yml --profile integration run --rm test-integration
```

---

## Qualité de code

| Outil | Rôle | Config |
|---|---|---|
| **ruff** | lint + format | `[tool.ruff]` dans `pyproject.toml` |
| **mypy** | type check | `[tool.mypy]` (mode souple, `python_version = "3.12"`) |
| **pip-audit** | CVE deps | déclenché dans CI (`--strict --skip-editable`) |
| **pytest-cov** | couverture | gate **70 %** au démarrage, à monter |
| **pre-commit** | hooks Git | `.pre-commit-config.yaml` (ruff + mypy + hygiène fichiers) |

Commandes :

```powershell
ruff check app tests scripts            # lint
ruff format --check app tests scripts   # format
mypy app                                # types
pip-audit --strict --skip-editable      # CVE
```

---

## CI GitHub Actions

Trois workflows dans `.github/workflows/` :

| Workflow | Trigger | Effet |
|---|---|---|
| `ci.yml` | `push main` + `pull_request` | lint+type+audit / tests-unit matrix (3.12+3.13) / tests-integration Qdrant |
| `release.yml` | tag `v*.*.*` | rejoue CI puis push image `runtime` sur `ghcr.io/<owner>/<repo>` |

Optimisations CI :
- Cache pip (clé : hash de `pyproject.toml`)
- Cache HuggingFace (clé : hash de `app/config/settings.py`)
- `concurrency` : annule les anciens runs sur la même PR

Reproduire en local : voir [`useful_commands.md`](./useful_commands.md#10-ci-github-actions).

---

## Brancher plus tard les intégrations

| Brique | Emplacement | Action suivante |
|--------|-------------|-----------------|
| **RAGFlow** | `app/retrieval/workflow_orchestrator.py` → `RagflowWorkflowOrchestrator` | Implémenter appels HTTP/SDK ; mapper JSON → `RetrievalHit`. |
| **LangExtract** | `app/extraction/langextract_extractor.py` | Remplacer la branche mock par l'API LangExtract ; conserver `extract(text, schema, context=...)`. |
| **Llama fine-tuné** | `app/extraction/llama_extractor.py` | Charger le modèle (vLLM, llama.cpp, HF) ; prompts **courts** par `FieldFamily`. |
| **LlamaIndex layer** | extra `[llamaindex]` | `VectorStoreIndex` + `QdrantVectorStore` ; filtres metadata (`document_id`, `field_family`). Optionnel : la couche bas-niveau (`QdrantHybridVectorService`) suffit déjà. |
| **XLS réel** | `app/etl/export.py` | Implémenter `XlsExportPlaceholder` avec `openpyxl`. |
| **Pseudonymisation HDS** | TODO | Hooks pré-indexation pour anonymiser les identifiants nominatifs. |
| **CLI réindexation** | TODO | Outil pour rejouer l'indexation après changement d'`embedding_version`. |
| **OpenTelemetry** | TODO | Traces + métriques pipeline (Qdrant client, embeddings, retrieval). |

---

## Point d'entrée pipeline

```python
from app.orchestration.pipeline import run_pipeline

result = run_pipeline("chemin/vers/doc.txt", patient_id="P1", study_id="S1")
```

Les observations restent des `ExtractedObservation` (intermédiaires) jusqu'au mapping `FieldCandidate` → `EcrfCellUpdate`.

---

## Variables d'environnement

Préfixe `ECRF_`. Liste complète dans `app/config/settings.py`. Les plus utilisées sont documentées dans la section 7 de [`useful_commands.md`](./useful_commands.md#7-variables-denvironnement-utiles-préfixe-ecrf_).

---

## Arborescence

```
app/
  ingestion/
  parsing/
    pdf/
  indexing/          # InMemory + QdrantHybridVectorService + embeddings + sparse + reranker
  retrieval/
  extraction/
  routing/
  business_rules/
  etl/
  api/
  schemas/
  config/
  utils/
  orchestration/
tests/
scripts/             # demos + prefetch_vector_models.py
docs/
.github/
  workflows/         # ci.yml + release.yml
Dockerfile           # multi-stages: base / runtime / vector / test
docker-compose.qdrant.yml   # Qdrant local seul
docker-compose.test.yml     # parité CI conteneurisée
```
