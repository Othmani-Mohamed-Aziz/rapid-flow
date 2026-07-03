"""
Démo **pipeline produit de bout en bout** via ``run_pipeline()`` (``PipelineOrchestrator``).

Exécute deux documents par défaut :
  1. Bilan labo mock (``scripts/sample_data/mock_blood_panel.txt``) — extraction ``lab_deterministic``
  2. CR imagerie foie (``data/ct_scan_report_liver.pdf``) — extraction ``imaging_langextract`` (Ollama)

Backend vectoriel : ``memory`` ou Qdrant selon ``ECRF_VECTOR_BACKEND`` dans ``.env``.
Imagerie : ``ECRF_LANGEXTRACT_ENABLED=true`` et Ollama joignable (``ECRF_OLLAMA_URL``).

Usage (depuis la racine du dépôt) :
    python scripts/run_demo_e2e.py
    python scripts/run_demo_e2e.py --skip-imaging
    python scripts/run_demo_e2e.py --lab-only
    python scripts/run_demo_e2e.py --imaging-only
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

from app.config.settings import Settings
from app.config.study_schema_provider import resolve_study_schema
from app.orchestration.pipeline import run_pipeline
from app.schemas.models import PipelineResult

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, io.UnsupportedOperation):
        pass


def _print_section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _summarize_result(label: str, result: PipelineResult, *, preview_chars: int) -> None:
    methods = Counter(o.extraction_method for o in result.observations)
    canonical_keys: set[str] = set()
    for o in result.observations:
        extra = o.extra or {}
        for key in ("canonical_lab_key", "canonical_imaging_key", "canonical_key"):
            if val := extra.get(key):
                canonical_keys.add(str(val))

    _print_section(f"Résultat — {label}")
    print(f"  document_id     : {result.document_id}")
    print(f"  document_type   : {result.document_type.value}")
    print(f"  observations    : {len(result.observations)}")
    print(f"  par méthode     : {dict(methods)}")
    if canonical_keys:
        print(f"  clés canoniques : {sorted(canonical_keys)}")
    print(f"  cell_updates    : {len(result.cell_updates)}")
    if result.cell_updates:
        print("  colonnes eCRF   :", ", ".join(u.column_key for u in result.cell_updates))
    print(f"  export          : {result.export_paths}")

    payload = result.model_dump(mode="json")
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if preview_chars > 0:
        print(f"\n--- Aperçu JSON ({min(len(text), preview_chars)} car.) ---")
        print(text[:preview_chars])
        if len(text) > preview_chars:
            print("\n[… troncature ; voir pipeline_output.json dans export …]")


def _run_document(
    *,
    label: str,
    path: Path,
    patient_id: str,
    study_id: str,
    preview_chars: int,
    ma_patient_key: str | None = None,
) -> PipelineResult | None:
    if not path.is_file():
        print(f"⚠️  {label} : fichier introuvable — {path}", file=sys.stderr)
        return None
    _print_section(f"Pipeline — {label}")
    print(f"  fichier : {path.resolve()}")
    if ma_patient_key:
        print(f"  patient_id (pipeline) : {patient_id}")
        print(f"  ma_patient_key (MA)   : {ma_patient_key}")
    else:
        print(f"  patient_id : {patient_id}")
    result = run_pipeline(
        str(path.resolve()),
        patient_id=patient_id,
        study_id=study_id,
        ma_patient_key=ma_patient_key,
    )
    _summarize_result(label, result, preview_chars=preview_chars)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--patient-id-lab",
        default="PAT-DEMO-LAB",
        help="Identifiant patient pour le bilan labo mock.",
    )
    parser.add_argument(
        "--patient-id-imaging",
        default="PAT-DEMO-IMG",
        help="Identifiant patient pour le CR imagerie (traçabilité pipeline).",
    )
    parser.add_argument(
        "--ma-patient-key",
        default=None,
        help=(
            "Clé ligne MA (colonne ID_current_base), ex. 220. "
            "Sinon résolution via data/patient_id_map.demo.json ou correspondance directe."
        ),
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=2500,
        help="Caractères max du JSON affiché par document (0 = rien).",
    )
    parser.add_argument("--lab-only", action="store_true", help="Bilan labo uniquement.")
    parser.add_argument("--imaging-only", action="store_true", help="CR imagerie uniquement.")
    parser.add_argument(
        "--skip-imaging",
        action="store_true",
        help="Alias de --lab-only (rétrocompat).",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    lab_path = root / "scripts" / "sample_data" / "mock_blood_panel.txt"
    imaging_path = root / "data" / "ct_scan_report_liver.pdf"

    settings = Settings()
    study_schema = resolve_study_schema(settings.study_schema_path)
    study_id = study_schema.study_id

    run_lab = not args.imaging_only
    run_imaging = not (args.lab_only or args.skip_imaging)

    if run_imaging and not settings.langextract_enabled:
        print(
            "⚠️  ECRF_LANGEXTRACT_ENABLED=false — le run imagerie ne produira "
            "pas d'observations LLM.",
            file=sys.stderr,
        )
    if run_imaging:
        print(
            f"Imagerie : Ollama attendu sur {settings.ollama_url} "
            f"(modèle {settings.ollama_model_id})",
        )

    results: list[PipelineResult] = []

    if run_lab:
        r = _run_document(
            label="Bilan labo mock (lab_deterministic)",
            path=lab_path,
            patient_id=args.patient_id_lab,
            study_id=study_id,
            preview_chars=args.preview_chars,
            ma_patient_key=args.ma_patient_key,
        )
        if r is not None:
            results.append(r)

    if run_imaging:
        r = _run_document(
            label="CR TDM foie (imaging_langextract)",
            path=imaging_path,
            patient_id=args.patient_id_imaging,
            study_id=study_id,
            preview_chars=args.preview_chars,
            ma_patient_key=args.ma_patient_key,
        )
        if r is not None:
            results.append(r)

    if not results:
        print("❌ Aucun document traité (fichiers manquants ?).", file=sys.stderr)
        return 2

    _print_section("Synthèse")
    for r in results:
        methods = ", ".join(sorted({o.extraction_method for o in r.observations}) or ["—"])
        print(
            f"  {r.document_type.value:20s}  obs={len(r.observations):2d}  "
            f"updates={len(r.cell_updates):2d}  [{methods}]"
        )
        print(f"    → {r.export_paths.get('json', r.export_paths)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
