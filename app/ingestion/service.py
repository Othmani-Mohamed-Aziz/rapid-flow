from __future__ import annotations

import mimetypes
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from app.schemas.models import RawDocument


class DocumentIngestionService(ABC):
    """Charge un document source et produit un `RawDocument` traçable."""

    @abstractmethod
    def ingest(self, document_path: str, patient_id: str, study_id: str) -> RawDocument:
        """Lit le fichier et retourne le contenu brut + métadonnées."""


class LocalFileIngestionService(DocumentIngestionService):
    """Ingestion locale (fichier sur disque)."""

    def ingest(self, document_path: str, patient_id: str, study_id: str) -> RawDocument:
        path = Path(document_path)
        if not path.is_file():
            msg = f"Document introuvable: {document_path}"
            raise FileNotFoundError(msg)
        mime_type, _ = mimetypes.guess_type(str(path))
        content = path.read_bytes()
        return RawDocument(
            document_id=str(uuid.uuid4()),
            patient_id=patient_id,
            study_id=study_id,
            source_path=str(path.resolve()),
            mime_type=mime_type or "application/octet-stream",
            content_bytes=content,
        )
