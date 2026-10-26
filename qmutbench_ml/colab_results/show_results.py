"""Colab-native, report-backed QMutBench classifier comparison.

This script only reads saved experiment reports. It does not train models or
claim live inference unless the actual serialized model bundle is present.
Run from the repository root:
    %run qmutbench_ml/colab_results/show_results.py
"""
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = Path.cwd()
REPORT = ROOT / "reports" / "experiments" / "qmutbench_final_research_report_20261026.json"
MODEL_DIR = ROOT / "qmutbench_ml" / "colab_results" / "models"
MODEL_FILE = MODEL_DIR / "best_model.joblib"
META_FILE = MODEL_DIR / "best_model_metadata.json"

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

display(Markdown("## Model artifact availability"))
for path in (MODEL_FILE, META_FILE):
    print(("FOUND: " if path.is_file() else "MISSING: ") + str(path))

if MODEL_FILE.is_file() and META_FILE.is_file():
    display(Markdown(
        "Both expected files exist. Before inference, validate the metadata, "
        "feature schema, model version, and artifact hash against the report."
    ))
else:
    display(Markdown(
        "**Live inference is not enabled by this viewer.** The report records a "
        "historical inference smoke test, but that does not mean the serialized "
        "model bundle is currently available in this checkout. This script will "
        "not fabricate predictions or retrain silently."
    ))
