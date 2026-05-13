from __future__ import annotations

from app.config.field_registry import FieldRegistry
from app.schemas.models import EcrfCellUpdate


class ValidationService:
    """Contrôles basiques avant écriture eCRF."""

    def __init__(self, registry: FieldRegistry) -> None:
        self._registry = registry

    def validate_update(self, update: EcrfCellUpdate) -> tuple[bool, list[str]]:
        errors: list[str] = []
        field = self._registry.get(update.column_key)
        if field is None:
            field = self._registry.get(update.provenance.field_name)
        if field is None:
            errors.append("champ inconnu dans le registry")
            return False, errors
        if (
            field.field_type == "numeric"
            and update.value not in (None, "")
            and not isinstance(update.value, (int, float))
        ):
            errors.append(f"{field.field_name} attend une valeur numérique")
        if field.field_type == "boolean" and update.value not in (None, "", True, False):
            errors.append(f"{field.field_name} attend un booléen")
        return len(errors) == 0, errors
