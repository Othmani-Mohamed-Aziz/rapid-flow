"""Tests unitaires du bootstrap Qdrant (sans serveur)."""

from __future__ import annotations

import pytest

from app.config.settings import Settings
from app.indexing.qdrant_bootstrap import collection_name, normalize_tenant_id


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
