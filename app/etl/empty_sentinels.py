"""Valeurs eCRF considérées comme cellules vides (placeholders MA)."""

from __future__ import annotations

from typing import Any

# Placeholders courants dans MA_Base (saisie à compléter / non applicable).
DEFAULT_ECRF_EMPTY_SENTINELS: tuple[str, ...] = (
    "NA",
    "N/A",
    "N/A.",
    "-",
    "—",
    "–",
    "?",
    "A completer",
    "A compléter",
    "A COMPLETER",
    "a completer",
    "A verifier",
    "A vérifier",
)

_DEFAULT_FROZEN = frozenset(DEFAULT_ECRF_EMPTY_SENTINELS)


def parse_empty_sentinels(raw: str | None) -> frozenset[str]:
    """
    Parse ``ECRF_EMPTY_SENTINELS`` (liste séparée par des virgules).

    Ex. ``NA,N/A,-,A completer`` — comparaison insensible à la casse à l'usage.
    """
    if raw is None or not str(raw).strip():
        return _DEFAULT_FROZEN
    parts = [p.strip().strip('"').strip("'") for p in str(raw).split(",")]
    tokens = [p for p in parts if p]
    return frozenset(tokens) if tokens else _DEFAULT_FROZEN


def _normalized_empty_token(token: str) -> str:
    return token.strip().casefold()


def build_sentinel_lookup(sentinels: frozenset[str]) -> frozenset[str]:
    return frozenset(_normalized_empty_token(s) for s in sentinels)


def is_ecrf_cell_empty(
    value: Any,
    *,
    sentinels: frozenset[str] | None = None,
) -> bool:
    """
    Vrai si la cellule eCRF est vide ou ne contient qu'un placeholder configuré.

    Les nombres (y compris ``0``) et les booléens ne sont jamais considérés vides.
    """
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return False

    lookup = build_sentinel_lookup(sentinels or _DEFAULT_FROZEN)
    text = str(value).strip()
    if not text:
        return True
    return _normalized_empty_token(text) in lookup
