# Useful commands — `rapid-flow`

Aide-mémoire pour développer, tester et opérer la pipeline en local sur Windows
(PowerShell). Les commandes Linux/macOS sont équivalentes : remplacer
`$env:VAR="value"` par `export VAR=value` et `.\.venv\Scripts\python.exe` par
`./.venv/bin/python`.

---

## 1. Environnement Python

```powershell
# Création du venv (une fois)
python -m venv .venv

# Activation
.\.venv\Scripts\activate

# Dépendances de base + tests
pip install -e ".[dev]"

# Stack vector store (Qdrant + embeddings + BM25 + reranker)
pip install -e ".[dev,vector]"

# Optionnel : couche LlamaIndex (Index/QueryEngine au-dessus de Qdrant)
pip install -e ".[llamaindex]"

# Docling (parsing PDF complexes)
pip install -e ".[docling]"
```

> **Note Python 3.14 + Windows** : `fastembed` / `onnxruntime` crashent
> (access violation) à l'init. Pour la voie hybride complète, utilisez
> **Python 3.12 ou 3.13**. Sur 3.14 : la voie **dense-only** fonctionne.

---

## 2. Pré-télécharger les modèles HuggingFace (une fois, ~1.2 GiB)

```powershell
python -X utf8 scripts\prefetch_vector_models.py
```

Options utiles :

```powershell
# Sauter une étape
python -X utf8 scripts\prefetch_vector_models.py --skip-sparse
python -X utf8 scripts\prefetch_vector_models.py --skip-reranker
python -X utf8 scripts\prefetch_vector_models.py --skip-embeddings

# Forcer le device
python -X utf8 scripts\prefetch_vector_models.py --device cpu
```

Le script gère automatiquement :
- Token HF expiré (variables d'env + fichier persisté `~/.cache/huggingface/token`)
- WinError 10054 / 10038 (retry incrémental sur erreurs réseau Windows)
- WinError 1314 (message explicite : activer le **Mode Développeur** Windows)
- Crashes ONNX au teardown (`os._exit` final)

---

## 3. Qdrant (Docker Compose)

### Démarrer / Vérifier

```powershell
# Démarrer le conteneur
docker compose -f docker-compose.qdrant.yml up -d

# Vérifier que le conteneur tourne
docker ps --filter "name=rapid_flow_qdrant" --format "table {{.Names}}`t{{.Image}}`t{{.Status}}`t{{.Ports}}"

# Healthz REST (200 attendu)
curl http://localhost:6533/healthz

# Lister les collections
curl http://localhost:6533/collections

# Dashboard web
start http://localhost:6533/dashboard
```

### Arrêter / Redémarrer

```powershell
# Arrêter (garde le volume / données)
docker compose -f docker-compose.qdrant.yml stop

# Redémarrer
docker compose -f docker-compose.qdrant.yml start

# Arrêter + supprimer le conteneur (garde le volume)
docker compose -f docker-compose.qdrant.yml down

# Tout purger (conteneur + données)
docker compose -f docker-compose.qdrant.yml down -v
```

### Ports

Le compose utilise des ports décalés pour cohabiter avec un autre Qdrant
(typiquement le starter-kit n8n sur 6333/6334) :

| Service | Host | Container |
|---|---|---|
| REST    | **6533** | 6333 |
| gRPC    | **6534** | 6334 |

Aligner `.env` :

```ini
ECRF_QDRANT_URL=http://localhost:6533
```

### Inspection / Debug

```powershell
# Logs du conteneur
docker logs rapid_flow_qdrant --tail 50

# Logs en suivi continu
docker logs rapid_flow_qdrant -f

# Stats live
docker stats rapid_flow_qdrant

# Shell dans le conteneur
docker exec -it rapid_flow_qdrant /bin/sh

# Supprimer une collection précise (sans arrêter Qdrant)
curl -X DELETE http://localhost:6533/collections/ecrf_chunks__DEMO

# Compter les points d'une collection
curl -X POST http://localhost:6533/collections/ecrf_chunks__DEMO/points/count `
     -H "Content-Type: application/json" `
     -d '{\"exact\": true}'
```

---

## 4. Tests pytest

### Suite par défaut (rapide, sans dépendance externe)

```powershell
# Suite complète
python -X utf8 -m pytest -q

# Suite verbose
python -X utf8 -m pytest -v

# Montrer la raison des skips
python -X utf8 -m pytest -q -rs

# Un seul fichier
python -X utf8 -m pytest tests\test_pdf_parsing.py -v

# Un seul test
python -X utf8 -m pytest tests\test_pdf_parsing.py::test_pypdf_parser_extracts_text -v

# Arrêter au 1er échec
python -X utf8 -m pytest -x

# Filtrer par nom (pattern)
python -X utf8 -m pytest -k "qdrant and not hybrid"

# Filtrer par marker
python -X utf8 -m pytest -m ct_integration_pdf
python -X utf8 -m pytest -m optional_user_pdf
python -X utf8 -m pytest -m real_vector_backend
```

### Tests d'intégration (opt-in)

```powershell
# Intégration Qdrant — nécessite Qdrant up sur ECRF_QDRANT_URL
$env:ECRF_RUN_QDRANT_INTEGRATION="1"
python -X utf8 -m pytest tests\test_qdrant_integration.py -v
Remove-Item Env:ECRF_RUN_QDRANT_INTEGRATION

# Intégration CT scan PDF (Docling) — défini dans .env via ECRF_CT_TEST_PDF_PATH
python -X utf8 -m pytest tests\test_ct_scan_report_integration.py -v
```

### Isolation des tests vis-à-vis de `.env`

Une fixture autouse (`tests/conftest.py::_isolate_vector_env`) force
`ECRF_VECTOR_BACKEND=memory` + sparse/reranker désactivés pour **tous** les
tests sauf ceux marqués `@pytest.mark.real_vector_backend`. Cela permet de
laisser `.env` en mode Qdrant pour le dev sans casser la CI.

---

## 5. Démos / Scripts utilitaires

```powershell
# Démo parsing seul (sans pipeline complet)
python scripts\run_pdf_demo.py "c:\chemin\vers\rapport.pdf"
python scripts\run_pdf_demo.py "c:\chemin\vers\rapport.pdf" --backend pypdf --json-out parsed.json

# Démo pipeline V1 (bilan sanguin mock)
python scripts\run_local_demo.py
```

---

## 6. Smoke-tests manuels (REPL Python)

### Vérifier Qdrant + Settings

```powershell
python -X utf8 -c "
from app.config.settings import Settings
from qdrant_client import QdrantClient
s = Settings()
print('Settings.qdrant_url =', s.qdrant_url)
c = QdrantClient(url=s.qdrant_url, timeout=5.0)
print('collections =', [x.name for x in c.get_collections().collections])
"
```

### Upsert + search end-to-end

```python
from app.config.settings import Settings
from app.indexing.qdrant_vector_service import QdrantHybridVectorService
from app.schemas.models import DocumentChunk

s = Settings(vector_backend="qdrant",
             qdrant_collection_prefix="ecrf_smoke",
             enable_sparse_bm25=False,
             enable_reranker=False)
svc = QdrantHybridVectorService(s)

svc.upsert_chunks(
    [DocumentChunk(chunk_id="c1", document_id="d1",
                   text="Lesion hepatique segment VI, suivi 6 mois.",
                   metadata={"section_heading": "Conclusion"})],
    tenant_id="DEMO", patient_id="P1", study_id="DEMO",
)

for chunk, score in svc.search(query_text="foie", tenant_id="DEMO", top_k=3):
    print(round(score, 3), chunk.text[:80])

svc.delete_document(tenant_id="DEMO", document_id="d1")
```

---

## 7. Variables d'environnement utiles (préfixe `ECRF_`)

| Variable | Défaut | Effet |
|---|---|---|
| `ECRF_PDF_PARSER_BACKEND` | `auto` | `auto` / `docling` / `pypdf` |
| `ECRF_LAB_POSTPROCESS_PDF_LAB_REPORTS` | `true` | Post-traitement lignes labo |
| `ECRF_CHUNK_MAX_SECTION_CHARS` | `12000` | Taille max d'une sous-section |
| `ECRF_VECTOR_BACKEND` | `memory` | `memory` (dev) / `qdrant` (prod) |
| `ECRF_QDRANT_URL` | `http://localhost:6333` | URL Qdrant REST |
| `ECRF_QDRANT_API_KEY` | (vide) | Token Qdrant si auth activée |
| `ECRF_QDRANT_COLLECTION_PREFIX` | `ecrf_chunks` | Nom = `{prefix}__{tenant_id}` |
| `ECRF_EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Modèle dense |
| `ECRF_EMBEDDING_DIM` | `768` | Doit matcher le modèle |
| `ECRF_EMBEDDING_VERSION` | `e5-base-v1` | Tag stocké en payload (migrations) |
| `ECRF_ENABLE_SPARSE_BM25` | `true` | Branche BM25 hybride |
| `ECRF_SPARSE_MODEL` | `Qdrant/bm25` | Modèle fastembed |
| `ECRF_ENABLE_RERANKER` | `true` | CrossEncoder après top-K |
| `ECRF_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker |
| `ECRF_RERANKER_DEVICE` | `auto` | `cpu` / `cuda` / `auto` |
| `ECRF_CT_TEST_PDF_PATH` | (absent) | PDF de référence pour tests CT |
| `ECRF_RUN_QDRANT_INTEGRATION` | `0` | `1` pour activer le test E2E Qdrant |
| `HF_TOKEN` | (absent) | Token HuggingFace (modèles gated uniquement) |

---

## 8. Qualité de code : ruff + mypy + pip-audit + pre-commit

### Lancement local (mêmes outils que la CI)

```powershell
# Lint
.\.venv\Scripts\python.exe -m ruff check app tests scripts

# Lint + autofix
.\.venv\Scripts\python.exe -m ruff check app tests scripts --fix

# Format check / format apply
.\.venv\Scripts\python.exe -m ruff format --check app tests scripts
.\.venv\Scripts\python.exe -m ruff format app tests scripts

# Type check
.\.venv\Scripts\python.exe -m mypy app

# Audit CVE (sur les deps installées en prod, hors editable)
.\.venv\Scripts\python.exe -m pip_audit --strict --skip-editable
```

### Pre-commit hooks (recommandé)

```powershell
# Installation des hooks (une fois)
.\.venv\Scripts\python.exe -m pre_commit install

# Lancer manuellement sur tout le repo
.\.venv\Scripts\python.exe -m pre_commit run --all-files

# Bypass en cas d'urgence
git commit --no-verify -m "..."
```

---

## 9. Stack conteneurisée (parité CI)

### Build des images multi-stages

```powershell
# Image runtime (deps prod uniquement, ~700 MiB) — future cible de déploiement
docker build --target runtime -t rapid-flow:runtime .

# Image vector (runtime + RAG stack, ~4 GiB)
docker build --target vector -t rapid-flow:vector .

# Image test (vector + dev + docling, ~5 GiB) — utilisée en CI
docker build --target test -t rapid-flow:test .

# Forcer une autre version Python (par défaut 3.13)
docker build --target test --build-arg PYTHON_VERSION=3.12 -t rapid-flow:test-py312 .
```

### Exécuter la suite de tests dans le conteneur

```powershell
# Suite par défaut (memory backend, rapide) — équivalent du job tests-unit en CI
docker compose -f docker-compose.test.yml run --rm test-runner

# Suite + intégration Qdrant hybride (dense + BM25 + RRF) — Python 3.13 Linux,
# pas de crash fastembed, donc hybrid passe vraiment
docker compose -f docker-compose.test.yml --profile integration up -d qdrant
docker compose -f docker-compose.test.yml --profile integration run --rm test-integration

# Nettoyage complet
docker compose -f docker-compose.test.yml --profile integration down -v
```

### Volumes utiles

| Volume | Usage |
|---|---|
| `rapid_flow_hf_cache` | cache modèles HF (e5, bge-reranker, bm25) — partagé entre runs |
| `qdrant_storage` | données Qdrant (compose `docker-compose.qdrant.yml`) |

```powershell
docker volume ls --filter "name=rapid_flow"
docker volume inspect rapid_flow_hf_cache
```

---

## 10. CI GitHub Actions

Le workflow `.github/workflows/ci.yml` se déclenche sur `push main` + toute PR :

| Job | OS | Python | Durée | Bloque CI |
|---|---|---|---|---|
| `lint-type-audit` | ubuntu-latest | 3.13 | ~1 min | ruff + format + mypy + pip-audit |
| `tests-unit` (matrix) | ubuntu-latest | **3.12** & **3.13** | ~2-3 min/run | pytest --cov-fail-under=70 |
| `tests-integration` | ubuntu-latest + qdrant service | 3.13 | ~10-15 min | upsert/search/delete Qdrant hybride |

Optimisations actives :
- Cache pip (`actions/setup-python@v5` avec `cache: pip`)
- Cache HuggingFace (`actions/cache@v4` keyé sur `app/config/settings.py`)
- `concurrency` annule les runs obsolètes sur la même PR

Le workflow `.github/workflows/release.yml` se déclenche sur tag `v*.*.*` :
- Rejoue la CI complète sur le tag
- Build + push image **runtime** sur `ghcr.io/<owner>/<repo>:<version>`
- Provenance + SBOM attachés

### Reproduire la CI en local

```powershell
# Job lint-type-audit
ruff check app tests scripts
ruff format --check app tests scripts
mypy app
pip-audit --strict --skip-editable

# Job tests-unit
$env:ECRF_VECTOR_BACKEND="memory"; $env:ECRF_ENABLE_SPARSE_BM25="false"; $env:ECRF_ENABLE_RERANKER="false"
pytest -q --cov=app --cov-fail-under=70

# Job tests-integration (= conteneurisé)
docker compose -f docker-compose.test.yml --profile integration up -d qdrant
docker compose -f docker-compose.test.yml --profile integration run --rm test-integration
```

---

## 11. Workflow Git — branche `main` protégée (Ruleset)

> ⚠️ La branche `main` est protégée par un **GitHub Ruleset** ("Protect main") :
> push direct interdit, force-push bloqué, suppression bloquée, PR obligatoire,
> les 4 checks CI doivent être verts avant merge :
> `Lint + type + audit`, `Unit tests (Py 3.12)`, `Unit tests (Py 3.13)`,
> `Full suite + Qdrant integration (Py 3.13)`.
>
> Toute modification suit donc le cycle **branche → PR → CI verte → merge**.

### 11.1 Cycle complet d'une modification

```powershell
# 1) Récupérer le dernier main
git checkout main
git pull --ff-only origin main

# 2) Créer une branche de travail (préfixes conseillés :
#    feat/, fix/, docs/, chore/, refactor/, test/, ci/)
git checkout -b fix/xyz

# 3) Faire vos modifications, vérifier en local (mêmes outils que la CI)
ruff check app tests scripts
ruff format --check app tests scripts
mypy app
pytest -q

# 4) Commit (pre-commit s'exécute si installé)
git add .
git commit -m "fix: description claire (impératif présent)"

# 5) Push de la branche sur GitHub
git push -u origin fix/xyz

# 6) Ouvrir la PR depuis le terminal (titre = msg de commit, body = template)
gh pr create --fill
#  Variante : pousser un titre / body explicites
gh pr create --title "fix: description" --body "Contexte + ce qui change + impact."

# 7) Attendre que la CI passe (~7 min) — surveille en live
gh pr checks --watch

# 8) Merger en squash + suppression auto de la branche locale et distante
gh pr merge --squash --delete-branch

# 9) Resynchroniser main local
git checkout main
git pull --ff-only origin main
```

### 11.2 Variantes utiles

```powershell
# Voir l'état de la PR courante (numéro, checks, reviewers)
gh pr status
gh pr view                       # détails de la PR de la branche courante
gh pr view --web                 # ouvrir dans le navigateur

# Lister les PR ouvertes / fermées récentes
gh pr list
gh pr list --state closed --limit 10

# Demander une review (si plusieurs collaborateurs)
gh pr edit --add-reviewer "<user1>,<user2>"

# Modifier le titre / body d'une PR
gh pr edit --title "..." --body "..."

# Forcer un re-run CI (parfois utile sur erreur réseau)
gh run rerun --failed                # re-run uniquement les jobs failed
gh run watch                         # suivre en live le run en cours
```

### 11.3 Diagnostiquer une CI rouge depuis le terminal

```powershell
# Liste des derniers runs (status + durée)
gh run list --limit 5

# Détails + jobs d'un run
gh run view <run-id>

# Logs des steps en échec (concis, va droit au but)
gh run view <run-id> --log-failed

# Logs complets d'un job précis
gh run view <run-id> --job <job-id> --log

# Annuler un run en cours (urgence)
gh run cancel <run-id>
```

### 11.4 Erreurs fréquentes

| Symptôme | Cause / Solution |
|---|---|
| `! [remote rejected] main -> main` lors d'un `git push` direct | Le Ruleset bloque le push direct sur `main`. **Passer par une PR** (workflow ci-dessus). |
| `Merge not allowed because checks are required` | Au moins un check CI n'est pas vert. `gh pr checks` pour voir lequel. |
| `Branch fix/xyz is out of date with main` | Le `main` distant a avancé. **Resync** : `git fetch origin && git rebase origin/main && git push --force-with-lease` (force-with-lease, pas `--force`). |
| Le hook `mixed-line-ending` corrige `ci.yml` et bloque le commit | Pre-commit a réparé le fichier. Re-stage + re-commit : `git add . && git commit -m "..."`. |
| Coup d'urgence : besoin de bypasser pre-commit | `git commit --no-verify -m "..."` (à éviter, déclenche un commit non-validé). |
| Coup d'urgence : besoin de bypasser le Ruleset | Si vous êtes admin et avez activé "Bypass list" : possible via merge admin. Sinon impossible — c'est voulu. |

### 11.5 Hotfix / annulation après merge

```powershell
# Inverser le merge d'une PR cassée (crée un commit de revert)
gh pr view <pr-number>             # vérifier le numéro
git checkout -b revert/<pr-number>
git revert -m 1 <merge-commit-sha>
git push -u origin revert/<pr-number>
gh pr create --fill --title "revert: PR #<pr-number>"
# … cycle PR + CI + merge habituel
```

### 11.6 Première release `v0.1.0`

```powershell
# Pré-requis : `main` à jour, CI verte sur le commit HEAD
git checkout main
git pull --ff-only origin main

# Créer le tag annoté
git tag v0.1.0 -m "v0.1.0 — initial release with Docling parsing + Qdrant hybrid RAG"

# Pousser le tag sur le remote → déclenche .github/workflows/release.yml
git push origin v0.1.0

# Suivre le run de release
gh run watch

# Vérifier que l'image GHCR est publiée
gh api /user/packages/container/rapid-flow/versions | ConvertFrom-Json |
    Select-Object name, @{n="tags";e={$_.metadata.container.tags}}
```

Après publication, l'image est tirable via :

```powershell
# Login GHCR (HTTPS, utilise le token gh)
gh auth token | docker login ghcr.io -u sbenhadj --password-stdin

# Pull
docker pull ghcr.io/sbenhadj/rapid-flow:0.1.0
docker pull ghcr.io/sbenhadj/rapid-flow:latest
```

### 11.7 Erreurs sur tag / release

```powershell
# Annuler un tag local (avant push)
git tag -d v0.1.0

# Supprimer un tag distant + release brouillon GHCR
git push origin :refs/tags/v0.1.0
gh release delete v0.1.0 --yes        # si une release GitHub a été créée
gh api -X DELETE /user/packages/container/rapid-flow/versions/<version-id>
```

---

## 12. Diagnostic rapide

```powershell
# Quels processus utilisent quel port ? (utile pour les 6333/6334 occupés)
Get-NetTCPConnection -LocalPort 6333 -ErrorAction SilentlyContinue
Get-NetTCPConnection -LocalPort 6533 -ErrorAction SilentlyContinue

# Lister tous les conteneurs (running + stopped)
docker ps -a --format "table {{.Names}}`t{{.Image}}`t{{.Status}}`t{{.Ports}}"

# Espace disque cache HuggingFace
$cache = Join-Path $HOME ".cache\huggingface\hub"
"{0:N1} GiB" -f ((Get-ChildItem $cache -Recurse -File | Measure-Object Length -Sum).Sum / 1GB)

# Supprimer le cache HF (force re-téléchargement complet)
Remove-Item (Join-Path $HOME ".cache\huggingface\hub") -Recurse -Force

# Token HF expiré : mettre de côté le fichier persisté
Move-Item (Join-Path $HOME ".cache\huggingface\token") `
          (Join-Path $HOME ".cache\huggingface\token.expired") -Force
```
