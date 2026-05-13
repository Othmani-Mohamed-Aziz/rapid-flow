"""
Exemple minimal exécutable : bilan sanguin mock -> JSON + CSV.

Usage (depuis la racine du dépôt) :
    python scripts/run_local_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

from app.orchestration.pipeline import run_pipeline


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sample = root / "scripts" / "sample_data" / "mock_blood_panel.txt"
    result = run_pipeline(str(sample), patient_id="PAT-DEMO-001", study_id="STUDY-AUTO-01")
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False)[:4000])
    print("\n--- Export paths ---")
    print(result.export_paths)


if __name__ == "__main__":
    main()
