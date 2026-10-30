"""Select the documented ideal-backend slice for a bounded Qolumbina task."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/qolumbina_detection"
OUT = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/qolumbina_detection_ideal"

def main():
    data = pd.read_csv(BASE / "normalized_manifest.csv")
    data = data[data.backend.eq("ideal")].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "normalized_manifest.csv", index=False)
    original = json.loads((BASE / "task_definition.json").read_text(encoding="utf-8"))
    original.update({"status": "task_manifest_prepared", "task": "predict whether Qolumbina DOSS detects each seeded buggy variant on the ideal simulator in a majority of available random-seed runs",
        "scope": "ideal backend only; one row per mutant variant", "observations_after_variant_backend_aggregation": int(len(data)),
        "program_groups": int(data.program.nunique()), "backends": ["ideal"],
        "class_counts": {str(k): int(v) for k, v in Counter(data.label).items()},
        "source_manifest": str((BASE / "normalized_manifest.csv").relative_to(ROOT))})
    (OUT / "task_definition.json").write_text(json.dumps(original, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(data), "groups": data.bug_id.nunique(),
                      "class_counts": {str(k): int(v) for k,v in Counter(data.label).items()}}, indent=2))

if __name__ == "__main__": main()
