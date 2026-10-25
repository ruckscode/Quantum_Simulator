import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path("data/qmutbench_ml/final")
report = json.loads((root / "threshold_tuning_report.json").read_text(encoding="utf-8"))
pred = pd.read_csv(root / "tuned_test_predictions.csv")
model = np.load(root / "numpy_logistic_model_tuned.npz")

actual = pred["actual_label"].to_numpy()
guessed = pred["predicted_label"].to_numpy()
normal = actual == "NORMAL"
fault = actual == "PROGRAM_FAULT"

summary = {
    "test_records": int(len(pred)),
    "test_normal_records": int(normal.sum()),
    "test_fault_records": int(fault.sum()),
    "normal_correctly_detected": int(((actual == guessed) & normal).sum()),
    "fault_correctly_detected": int(((actual == guessed) & fault).sum()),
    "false_normal_predictions": int(((guessed == "NORMAL") & fault).sum()),
    "false_fault_predictions": int(((guessed == "PROGRAM_FAULT") & normal).sum()),
    "selected_threshold": float(model["threshold"]),
    "feature_count": int(len(model["feature_names"])),
    "group_overlap": report["group_overlap"],
    "held_out_test_metrics": report["held_out_test_metrics"],
    "scope": "QMutBench-derived program-fault classification; not hardware-fault validation"
}

(root / "final_project_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
print("\nSaved:", root / "final_project_summary.json")
