from app.indexing.vector_service import (
    InMemoryVectorIndexService,
    LlamaIndexVectorService,
    VectorIndexService,
)


def build_default_vector_service(settings=None):
    """Factory dépendant de `Settings.vector_backend`."""
    from app.config.settings import Settings

    s = settings or Settings()
    if s.vector_backend == "qdrant":
        # Import différé pour ne pas exiger l'extra `vector` en mode mémoire.
        from app.indexing.qdrant_vector_service import QdrantHybridVectorService

        return QdrantHybridVectorService(s)
    return InMemoryVectorIndexService()


__all__ = [
    "InMemoryVectorIndexService",
    "LlamaIndexVectorService",
    "VectorIndexService",
    "build_default_vector_service",
]
