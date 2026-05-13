from __future__ import annotations


class DoclingNotInstalledError(ImportError):
    """Le backend Docling est demandé mais le package n’est pas installé."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message or "Docling n’est pas installé. Exemple : pip install 'ecrf-autofill[docling]'"
        )
