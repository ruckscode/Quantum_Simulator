"""Colab-native, report-backed QMutBench classifier comparison.

This script only reads saved experiment reports. It does not train models or
claim live inference unless the actual serialized model bundle is present.
Run from the repository root:
    %run qmutbench_ml/colab_results/show_results.py
"""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = Path.cwd()
REPORT = ROOT / "reports" / "experiments" / "qmutbench_final_research_report_20261026.json"
MODEL_DIRS = [
    ROOT / "data" / "qmutbench_ml" / "mutation_aware_v1",
    ROOT / "qmutbench_ml" / "colab_results" / "models",
]
MODEL_FILE = next((p / "best_model.joblib" for p in MODEL_DIRS if (p / "best_model.joblib").is_file()), MODEL_DIRS[0] / "best_model.joblib")
META_FILE = next((p / "best_model_metadata.json" for p in MODEL_DIRS if (p / "best_model_metadata.json").is_file()), MODEL_FILE.with_name("best_model_metadata.json"))

if not REPORT.is_file():
    raise FileNotFoundError(
        f"Saved research report not found at {REPORT}. "
        "Run this cell from the Quantum_Simulator repository root."
    )

with REPORT.open(encoding="utf-8") as f:
    report = json.load(f)

audit = report["baseline_and_feature_audit"]["reproduced_oof"]
nested = report["nested_grouped_evaluation"]["nested_selected_metrics"]

evaluations = [
    ("Baseline — OOF", audit["baseline_metrics"]),
    ("Mutation-aware — OOF", audit["mutation_aware_metrics"]),
    ("Nested grouped evaluation", nested),
]

rows = []
for name, m in evaluations:
    rows.append({
        "Evaluation": name,
        "Accuracy (%)": 100 * m["accuracy"],
        "Balanced accuracy (%)": 100 * m["balanced_accuracy"],
        "Macro F1 (%)": 100 * m["macro_f1"],
        "ROC-AUC (%)": 100 * m["roc_auc"],
    })

df = pd.DataFrame(rows)

display(Markdown("# QMutBench — Classifier Comparison"))
display(Markdown(
    f"**Report date:** {report.get('date', 'not recorded')}  \n"
    f"**Development rows:** {report['baseline_and_feature_audit']['source']['development_rows']:,}  \n"
    f"**Original-circuit groups:** {report['baseline_and_feature_audit']['source']['development_origins']:,}"
))
display(df.style.format({
    "Accuracy (%)": "{:.2f}",
    "Balanced accuracy (%)": "{:.2f}",
    "Macro F1 (%)": "{:.2f}",
    "ROC-AUC (%)": "{:.2f}",
}).hide(axis="index"))

fig, ax = plt.subplots(figsize=(11, 5.5))
df.set_index("Evaluation")[[
    "Accuracy (%)", "Balanced accuracy (%)", "Macro F1 (%)", "ROC-AUC (%)"
]].plot(kind="bar", ax=ax)
ax.set_title("Recorded QMutBench Evaluation Metrics")
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 100)
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=12)
ax.grid(axis="y", alpha=0.25)
ax.legend(loc="lower right")
fig.tight_layout()
plt.show()

base_f1 = audit["baseline_metrics"]["macro_f1"] * 100
mut_f1 = audit["mutation_aware_metrics"]["macro_f1"] * 100
nested_f1 = nested["macro_f1"] * 100

display(Markdown(
    f"**OOF Macro F1 change:** {mut_f1 - base_f1:+.2f} percentage points "
    f"({base_f1:.2f}% → {mut_f1:.2f}%).  \n"
    f"**Nested grouped Macro F1:** {nested_f1:.2f}%. This is a separate "
    "development evaluation procedure, not independent external validation. "
    "The report explicitly records that no compatible independent external "
    "dataset was verified and the 85% Macro F1 target remains unmet."
))

display(Markdown("## Model artifact integrity and inference-loader check"))

if not MODEL_FILE.is_file() or not META_FILE.is_file():
    display(Markdown(
        "**Live inference unavailable:** the model bundle is incomplete in this runtime. "
        "Restore both files from the verified original training environment; this viewer "
        "will not retrain or fabricate predictions."
    ))
else:
    try:
        metadata = json.loads(META_FILE.read_text(encoding="utf-8"))
        digest = hashlib.sha256()
        with MODEL_FILE.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        expected_sha256 = metadata.get("model_sha256")
        report_sha256 = report.get("baseline_and_feature_audit", {}).get("artifacts", {}).get("mutation_aware_model_sha256")
        if not expected_sha256:
            raise ValueError("metadata is missing model_sha256")
        if actual_sha256 != expected_sha256:
            raise ValueError("model SHA-256 does not match metadata")
        if report_sha256 and actual_sha256 != report_sha256:
            raise ValueError("model SHA-256 does not match the committed research report")
        if metadata.get("model_version") != "qmutbench-mutation-aware-v1":
            raise ValueError(f"unexpected model version: {metadata.get('model_version')!r}")
        if len(metadata.get("feature_schema", [])) != 23:
            raise ValueError("metadata does not declare the expected 23-feature schema")
        if float(metadata.get("inference_threshold", -1)) != 0.5:
            raise ValueError("metadata does not declare the selected 0.5 inference threshold")

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from qmutbench_ml import QMutBenchOutcomeModel
        predictor = QMutBenchOutcomeModel(MODEL_FILE, META_FILE)
        if len(predictor.features) != 23:
            raise ValueError(f"inference loader returned {len(predictor.features)} features, expected 23")

        display(Markdown(
            "**Bundle verified and inference loader passed.** This validates artifact integrity and "
            "the model contract; it does not constitute independent evaluation or a real-record "
            "prediction in this runtime."
        ))
        print("Model:", MODEL_FILE)
        print("Metadata:", META_FILE)
        print("Model version:", predictor.model_version)
        print("Feature count:", len(predictor.features))
        print("Threshold:", predictor.threshold)
        print("Model size (bytes):", MODEL_FILE.stat().st_size)
        print("SHA-256:", actual_sha256)
        print("Metadata checksum match: PASS")
        print("Research report checksum match:", "PASS" if report_sha256 == actual_sha256 else "not available")
    except Exception as exc:
        display(Markdown(f"**Bundle validation failed:** `{type(exc).__name__}: {exc}`"))
        raise
