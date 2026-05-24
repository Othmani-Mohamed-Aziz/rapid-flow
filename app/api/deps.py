"""
Dépendances FastAPI (à activer lorsque le extra `api` est installé).

L’exposition HTTP du pipeline n’est pas implémentée : utiliser
`app.orchestration.pipeline.run_pipeline` depuis un script ou un worker.
"""

from __future__ import annotations

from typing import Any


def get_pipeline_orchestrator() -> Any:
    """Réservé à une future route `POST /pipeline/run` — non branché en V1."""
    raise NotImplementedError(
        "API HTTP pipeline non disponible : instancier PipelineOrchestrator ou "
        "appeler run_pipeline() depuis le code applicatif."
    )
