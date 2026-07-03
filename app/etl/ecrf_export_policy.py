"""Filtrage unifié des ``cell_updates`` selon la politique eCRF / état MA."""

from __future__ import annotations

from typing import Any

from app.etl.empty_sentinels import is_ecrf_cell_empty
from app.etl.overwrite_policy import EcrfTemplateSnapshot, XlsOverwritePolicy
from app.etl.xls_utils import resolve_column_key
from app.schemas.models import EcrfCellUpdate


def columns_to_skip_for_policy(
    policy: XlsOverwritePolicy,
    snapshot: EcrfTemplateSnapshot,
    target_columns: set[str],
) -> set[str]:
    if policy == XlsOverwritePolicy.NEVER:
        return set(target_columns)
    if policy == XlsOverwritePolicy.EMPTY_ONLY:
        return snapshot.filled_target_columns(target_columns)
    return set()


def partition_cell_updates_for_ecrf(
    updates: list[EcrfCellUpdate],
    *,
    policy: XlsOverwritePolicy,
    snapshot: EcrfTemplateSnapshot,
    column_aliases: dict[str, str],
) -> tuple[list[EcrfCellUpdate], list[dict[str, Any]]]:
    """
    Sépare les mises à jour exportables (JSON/CSV/XLS alignés) des suppressions.

    Raisons ``reason`` : ``policy_never``, ``column_already_filled``, ``always``.
    """
    if policy == XlsOverwritePolicy.NEVER:
        return [], [
            {"column_key": u.column_key, "reason": "policy_never", "value": u.value}
            for u in updates
        ]

    snapshot.load()
    header_map = snapshot._header_map  # noqa: SLF001
    accepted: list[EcrfCellUpdate] = []
    suppressed: list[dict[str, Any]] = []

    for upd in updates:
        if policy == XlsOverwritePolicy.ALWAYS:
            accepted.append(upd)
            continue

        header = resolve_column_key(header_map, upd.column_key, column_aliases)
        if header is None:
            accepted.append(upd)
            continue

        if snapshot.is_column_filled(upd.column_key):
            suppressed.append(
                {
                    "column_key": upd.column_key,
                    "reason": "column_already_filled",
                    "existing_value": snapshot.value_for_column(upd.column_key),
                    "proposed_value": upd.value,
                }
            )
            continue

        accepted.append(upd)

    return accepted, suppressed


def should_write_cell_value(
    live_value: Any,
    *,
    policy: XlsOverwritePolicy,
    empty_sentinels: frozenset[str] | None,
) -> bool:
    if policy == XlsOverwritePolicy.NEVER:
        return False
    if policy == XlsOverwritePolicy.ALWAYS:
        return True
    return is_ecrf_cell_empty(live_value, sentinels=empty_sentinels)
