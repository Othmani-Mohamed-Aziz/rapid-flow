# syntax=docker/dockerfile:1.7
#
# ============================================================================
# rapid-flow — Dockerfile multi-stages
# ============================================================================
#
# Targets disponibles (`docker build --target <name>`) :
#
#   base     — Python 3.13 slim + deps systèmes minimales (poppler, libs ONNX).
#   runtime  — base + projet + dépendances **production** (sans extras).
#              Image attendue ~700 MiB. Cible déploiement futur.
#   vector   — runtime + extra `[vector]` (torch CPU, sentence-transformers,
#              fastembed, qdrant-client). ~4 GiB. Cible workers RAG.
#   test     — vector + extras `[dev,docling]` + sources tests/scripts.
#              ~5 GiB. Cible CI / dev conteneurisé.
#
# Build args :
#   PYTHON_VERSION (default: 3.13)
#   PIP_EXTRA_INDEX_URL (optionnel, pour roues CPU torch)
#
# Build examples :
#   docker build --target test    -t rapid-flow:test    .
#   docker build --target runtime -t rapid-flow:runtime .
#   docker build --target vector  -t rapid-flow:vector  --build-arg PYTHON_VERSION=3.12 .
#
# Notes :
# - On utilise les cache mounts BuildKit (`--mount=type=cache`) sur ~/.cache/pip
#   pour accélérer les rebuilds. Active BuildKit par défaut sur Docker 23+.
# - Les modèles HF NE SONT PAS bakés dans l'image. Ils sont récupérés au
#   premier run et cachés dans le volume `huggingface_cache` (cf. compose).
# ============================================================================

ARG PYTHON_VERSION=3.13

# ─── Stage 1 : base ────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Deps systèmes pour : pypdf (rien), Docling (poppler-utils, libgl), onnxruntime (libgomp).
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      poppler-utils \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Stage 2 : runtime (deps production uniquement) ────────────────────────
FROM base AS runtime

# On copie d'abord pyproject.toml seul → meilleur cache des deps.
COPY pyproject.toml README.md ./
# `setuptools.packages.find` a besoin de voir le code pour résoudre les
# packages, donc on copie aussi le code applicatif tôt.
COPY app/ ./app/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel \
 && pip install -e "."

ENTRYPOINT ["python"]
CMD ["-c", "import app; print('rapid-flow runtime OK')"]

# ─── Stage 3 : vector (RAG hybride) ────────────────────────────────────────
FROM runtime AS vector

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ".[vector]"

ENV ECRF_VECTOR_BACKEND=qdrant

# ─── Stage 4 : test (CI + dev conteneurisé) ────────────────────────────────
FROM vector AS test

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ".[dev,docling]"

COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY .env.example* ./

# En CI les tests généralistes utiliseront `memory` (fixture autouse). Le job
# d'intégration override via env (`ECRF_VECTOR_BACKEND=qdrant`).
ENV ECRF_VECTOR_BACKEND=memory \
    ECRF_ENABLE_SPARSE_BM25=false \
    ECRF_ENABLE_RERANKER=false

ENTRYPOINT []
CMD ["pytest", "-q"]
