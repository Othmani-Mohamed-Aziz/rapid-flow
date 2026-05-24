"""
Démo **parsing + chunking** — sans indexation, retrieval ni extraction.

Étape pipeline couverte : ingestion → parsing → chunking (aperçu console).

Scripts complémentaires :
  - index Qdrant + retrieval (sans extraction) : ``run_demo_retrieval.py``
  - pipeline produit complète (extraction + export eCRF) : ``run_demo_e2e.py``

Usage :
    python scripts/run_demo_parsing.py chemin/vers/fichier.pdf
    python scripts/run_demo_parsing.py chemin/vers/fichier.pdf --backend pypdf
    python scripts/run_demo_parsing.py chemin/vers/fichier.pdf --json-out sortie.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

from app.config.settings import Settings
from app.ingestion.service import LocalFileIngestionService
from app.parsing.chunking import DefaultChunkingService
from app.parsing.smart_service import SmartParsingService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse PDF → ParsedDocument (+ labo structuré).")
    parser.add_argument(
        "document",
        type=Path,
        help="Chemin vers un PDF (ou .txt pour comparaison heuristique).",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "pypdf", "docling"),
        default="auto",
        help="ECRF_PDF_PARSER_BACKEND équivalent pour cette exécution.",
    )
    parser.add_argument(
        "--patient-id",
        default="DEMO-PDF",
        help="Identifiant patient fictif pour l'ingestion.",
    )
    parser.add_argument(
        "--study-id",
        default="DEMO-STUDY",
        help="Identifiant étude fictif pour l'ingestion.",
    )
    parser.add_argument(
        "--no-lab-postprocess",
        action="store_true",
        help="Désactive le post-traitement bilan (ECRF_LAB_POSTPROCESS_PDF_LAB_REPORTS=false).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Écrit le JSON complet du ParsedDocument.",
    )
    args = parser.parse_args(argv)

    path = args.document
    if not path.is_file():
        print(f"Fichier introuvable : {path}", file=sys.stderr)
        return 2

    backend: Literal["auto", "pypdf", "docling"] = args.backend  # type: ignore[assignment]
    settings = Settings(
        pdf_parser_backend=backend,
        lab_postprocess_pdf_lab_reports=not args.no_lab_postprocess,
    )
    raw = LocalFileIngestionService().ingest(
        str(path.resolve()),
        patient_id=args.patient_id,
        study_id=args.study_id,
    )
    try:
        parsed = SmartParsingService(settings).parse(raw)
    except Exception as exc:
        print(f"Erreur parsing : {exc}", file=sys.stderr)
        return 1

    payload: dict[str, Any] = parsed.model_dump(mode="json")
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text[:12000])
    if len(text) > 12000:
        print("\n[… troncature console ; utiliser --json-out pour le document complet …]\n")

    if args.json_out:
        args.json_out.write_text(text, encoding="utf-8")
        print(f"JSON écrit : {args.json_out.resolve()}")

    print("\n--- Résumé ---")
    print("pdf_parser:", parsed.metadata.get("pdf_parser"))
    print("document_type_hint:", parsed.document_type_hint.value)
    scores = parsed.metadata.get("document_type_scores")
    if isinstance(scores, dict):
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        print("document_type_scores (top 3):", top)
    print("chunking_strategy:", parsed.metadata.get("chunking_strategy"))
    print("sections:", len(parsed.structured_sections))
    print("lab_lines:", len(parsed.structured_lab_lines))
    chks = DefaultChunkingService(settings).chunk(parsed)
    print("chunks:", len(chks))
    for i, c in enumerate(chks[:8]):
        head = (c.metadata or {}).get("section_heading")
        print(f"  [{i}] {c.chunk_id[:20]}… heading={head!r} chars={len(c.text)}")
    warns = parsed.metadata.get("warnings") or []
    if warns:
        print("avertissements:", warns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
