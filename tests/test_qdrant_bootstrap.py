"""Tests unitaires du bootstrap Qdrant (sans serveur).

Mix volontaire :
- les helpers Python purs (`normalize_tenant_id`, `validate_strict_tenant_id`,
  `collection_name`) s'exécutent partout, y compris dans le job CI "Unit tests
  smoke" qui n'installe pas l'extra `[vector]` ;
- les tests qui appellent `ensure_collection` requièrent `qdrant_client`
  (importé au runtime par la fonction) et sont skip via `requires_qdrant_client`
  si le module est absent.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.config.settings import Settings
from app.indexing.qdrant_bootstrap import (
    CollectionConfigMismatchError,
    collection_name,
    ensure_collection,
    normalize_tenant_id,
    validate_strict_tenant_id,
)

try:
    import qdrant_client  # noqa: F401

    _HAS_QDRANT_CLIENT = True
except ImportError:
    _HAS_QDRANT_CLIENT = False

requires_qdrant_client = pytest.mark.skipif(
    not _HAS_QDRANT_CLIENT,
    reason="qdrant_client requis (extra [vector]). Skip en CI smoke.",
)


def test_normalize_tenant_id_basic() -> None:
    assert normalize_tenant_id("STUDY-001") == "STUDY-001"
    assert normalize_tenant_id("  study/AB c ") == "study_AB_c"


def test_normalize_tenant_id_rejects_empty() -> None:
    with pytest.raises(ValueError):
        normalize_tenant_id("")
    with pytest.raises(ValueError):
        normalize_tenant_id("   ")


def test_collection_name_uses_prefix_and_tenant() -> None:
    s = Settings(qdrant_collection_prefix="ecrf_chunks")
    assert collection_name(s, "STUDY-X") == "ecrf_chunks__STUDY-X"
    assert collection_name(s, "S 1/2") == "ecrf_chunks__S_1_2"


# ─────────────────────────────────────────────────────────────────────────────
# A1 — Garde-fou dim/distance mismatch sur collection existante
# ─────────────────────────────────────────────────────────────────────────────


def _mock_client_with_existing_collection(*, size: int, distance: str = "Cosine") -> MagicMock:
    """Fabrique un client Qdrant mock qui simule une collection existante avec
    `text` -> VectorParams(size=..., distance=...).
    """
    client = MagicMock()
    client.get_collections.return_value = MagicMock(
        collections=[MagicMock(name="ecrf_chunks__STUDY")]
    )
    # Mock du `name` (MagicMock met sa propre valeur par défaut pour `name`).
    client.get_collections.return_value.collections[0].name = "ecrf_chunks__STUDY"

    vec_params = MagicMock(size=size)
    vec_params.distance = MagicMock(value=distance)
    info = MagicMock()
    info.config.params.vectors = {"text": vec_params}
    client.get_collection.return_value = info
    return client


@requires_qdrant_client
def test_ensure_collection_passes_when_existing_collection_matches() -> None:
    s = Settings(qdrant_collection_prefix="ecrf_chunks", embedding_dim=768)
    client = _mock_client_with_existing_collection(size=768, distance="Cosine")
    name = ensure_collection(client, s, "STUDY")
    assert name == "ecrf_chunks__STUDY"
    client.create_collection.assert_not_called()


@requires_qdrant_client
def test_ensure_collection_raises_on_dim_mismatch() -> None:
    """Collection existe avec dim=384, settings veut 768 → erreur explicite."""
    s = Settings(qdrant_collection_prefix="ecrf_chunks", embedding_dim=768)
    client = _mock_client_with_existing_collection(size=384, distance="Cosine")
    with pytest.raises(CollectionConfigMismatchError) as exc:
        ensure_collection(client, s, "STUDY")
    assert "dimension=384" in str(exc.value)
    assert "attendu=768" in str(exc.value)


@requires_qdrant_client
def test_ensure_collection_raises_on_distance_mismatch() -> None:
    """Collection existe en Dot, on attend Cosine → erreur explicite."""
    s = Settings(qdrant_collection_prefix="ecrf_chunks", embedding_dim=768)
    client = _mock_client_with_existing_collection(size=768, distance="Dot")
    with pytest.raises(CollectionConfigMismatchError) as exc:
        ensure_collection(client, s, "STUDY")
    assert "distance='dot'" in str(exc.value).lower()


@requires_qdrant_client
def test_ensure_collection_creates_when_absent() -> None:
    s = Settings(qdrant_collection_prefix="ecrf_chunks", embedding_dim=768)
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    name = ensure_collection(client, s, "STUDY")
    assert name == "ecrf_chunks__STUDY"
    client.create_collection.assert_called_once()
    # get_collection ne doit pas être appelé puisqu'on est dans la branche "create".
    client.get_collection.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# D2 — Validation stricte du tenant_id
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "valid_id",
    [
        "STUDY-01",
        "study_001",
        "ABC-123_v2",
        "a",
        "A" * 60,
    ],
)
def test_validate_strict_tenant_id_accepts_valid(valid_id: str) -> None:
    assert validate_strict_tenant_id(valid_id) == valid_id


@pytest.mark.parametrize(
    "invalid_id",
    [
        "",  # vide
        " STUDY-01",  # whitespace en début
        "STUDY 01",  # espace au milieu
        "STUDY/01",  # slash
        "étude-01",  # caractère non ASCII
        "STUDY.01",  # point
        "A" * 61,  # trop long
    ],
)
def test_validate_strict_tenant_id_rejects_invalid(invalid_id: str) -> None:
    with pytest.raises(ValueError) as exc:
        validate_strict_tenant_id(invalid_id)
    msg = str(exc.value)
    assert "tenant_id" in msg


def test_collection_name_uses_strict_validation_when_setting_enabled() -> None:
    """En mode strict, un tenant invalide doit faire échouer `collection_name`
    (au lieu de normaliser silencieusement)."""
    s_strict = Settings(qdrant_collection_prefix="ecrf_chunks", qdrant_strict_tenant_id=True)
    with pytest.raises(ValueError):
        collection_name(s_strict, "STUDY 01")
    # Cas valide → OK.
    assert collection_name(s_strict, "STUDY-01") == "ecrf_chunks__STUDY-01"


def test_collection_name_falls_back_to_permissive_by_default() -> None:
    """Comportement historique préservé sans le setting strict."""
    s = Settings(qdrant_collection_prefix="ecrf_chunks")
    assert s.qdrant_strict_tenant_id is False  # défaut
    assert collection_name(s, "STUDY 01") == "ecrf_chunks__STUDY_01"
