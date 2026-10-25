import json
from pathlib import Path
import pandas as pd

root = Path("data/qmutbench_ml")
out = root / "final"
summary = json.loads((out / "final_project_summary.json").read_text(encoding="utf-8"))
tuning = json.loads((out / "threshold_tuning_report.json").read_text(encoding="utf-8"))
df = pd.read_csv(root / "qmutbench_results.csv")
failures_path = root / "qmutbench_failures.csv"
failures = pd.read_csv(failures_path) if failures_path.exists() else pd.DataFrame()

m = summary["held_out_test_metrics"]
report = f"""# Hardware-Aware Data-Driven Testing and Fault Localization for Quantum Programs

## 1. Project objective

Develop a data-driven approach to analyse quantum-program execution traces and identify potential program-level faults. The broader research goal is to distinguish program faults from variations caused by quantum hardware noise.

## 2. Current implementation

- Quantum circuit execution and trace extraction using the project's simulator.
- QMutBench-derived mutant and original-circuit records.
- Five trace-based features:
  - Maximum probability difference
  - Mean probability difference
  - Median probability difference
  - Deviation concentration
  - Fraction of high-deviation observations
- NumPy-based logistic regression classifier.
- Group-based train/validation/test separation using original-circuit groups.
- Validation-based threshold selection.

## 3. Dataset summary

| Item | Result |
|---|---:|
| Processed records | {len(df):,} |
| Mutant records | {(df["record_type"] == "mutant").sum():,} |
| Original records | {(df["record_type"] == "original").sum():,} |
| Recorded execution failures | {len(failures):,} |
| Features | {summary["feature_count"]} |
| Decision threshold | {summary["selected_threshold"]:.2f} |

The original records were assigned the `NORMAL` label and mutant records the `PROGRAM_FAULT` label. These are dataset-construction labels, not independent proof that every original behaves correctly or every mutant manifests an observable fault.

## 4. Evaluation protocol

The dataset was divided into training, validation, and held-out test groups by original-circuit identity. The recorded group overlap was zero for all three pairwise comparisons. The classification threshold was selected using the validation data and evaluated on the held-out test set.

## 5. Held-out test results

| Metric | Result |
|---|---:|
| Accuracy | {m["accuracy"]:.2%} |
| Balanced accuracy | {m["balanced_accuracy"]:.2%} |
| Macro F1 | {m["macro_f1"]:.2%} |
| NORMAL precision | {m["normal_precision"]:.2%} |
| NORMAL recall | {m["normal_recall"]:.2%} |
| PROGRAM_FAULT precision | {m["program_fault_precision"]:.2%} |
| PROGRAM_FAULT recall | {m["program_fault_recall"]:.2%} |

Confusion matrix (actual rows; predicted columns), with class order `PROGRAM_FAULT`, `NORMAL`:

| Actual / Predicted | PROGRAM_FAULT | NORMAL |
|---|---:|---:|
| PROGRAM_FAULT | {m["confusion_matrix_actual_rows_predicted_columns"][0][0]} | {m["confusion_matrix_actual_rows_predicted_columns"][0][1]} |
| NORMAL | {m["confusion_matrix_actual_rows_predicted_columns"][1][0]} | {m["confusion_matrix_actual_rows_predicted_columns"][1][1]} |

## 6. Interpretation and limitations

All five trace features are zero for the generated original records, while mutant records have nonzero values for many of these features. This creates a strong relationship between record construction and the labels. The classifier may exploit this distinction rather than learn a generalizable quantum-program fault signature.

The test set contains {summary["test_normal_records"]} NORMAL records and {summary["test_fault_records"]} PROGRAM_FAULT records. Although all {summary["normal_correctly_detected"]} NORMAL records were detected, {summary["false_normal_predictions"]} mutant records were incorrectly classified as NORMAL. NORMAL precision is therefore low.

The current experiment supports a preliminary evaluation of QMutBench-derived program-fault classification only. It does **not** validate hardware-fault localization, identify faulty physical qubits, or establish that an anomaly is caused by hardware rather than software.

The recorded failures should be reported as execution failures, not silently treated as successful classifications.

## 7. Next research milestone

1. Validate the labels against independently defined program-behaviour outcomes.
2. Establish features that do not simply encode whether a record is an original or mutant.
3. Evaluate against held-out circuit families and report per-class precision, recall, F1, and confusion matrices.
4. Separately collect ideal-simulator and real-hardware execution evidence.
5. Develop and validate hardware-aware fault attribution using backend calibration, readout errors, and qubit/connectivity information.

## 8. Reproducibility artifacts

The `data/qmutbench_ml/` directory contains the processed dataset, manifests, and failure records. The `data/qmutbench_ml/final/` directory contains the evaluation summaries, predictions, threshold analysis, and saved NumPy model.

**Conclusion:** The current pipeline and preliminary group-separated evaluation are complete for the present QMutBench sample. Reliable program-fault detection and real-hardware fault localization remain research objectives requiring independent labels and hardware evidence.
"""

path = out / "FINAL_PROJECT_REPORT.md"
path.write_text(report, encoding="utf-8")
print("Created:", path)
print("Dataset rows:", len(df))
print("Failure records:", len(failures))
print("Held-out accuracy:", f'{m["accuracy"]:.2%}')
print("Held-out macro F1:", f'{m["macro_f1"]:.2%}')
print("Report generation complete.")
