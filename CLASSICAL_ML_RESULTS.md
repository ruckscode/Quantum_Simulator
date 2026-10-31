# Classical ML benchmark update — 2026-10-30

## Status

Classical ML work is complete for the currently prepared source adapters. The new source benchmarks use nested, group-disjoint cross-validation: inner GroupKFold (3 folds) chooses among 104 word/character TF-IDF SVM, logistic, SGD, and ComplementNB settings plus 32 sparse cosine-KNN settings; outer GroupKFold (5 folds) produces pooled out-of-fold (OOF) predictions. Model selection never sees an outer validation group. Frozen test sets were not accessed.

The tasks have different labels and meanings. Their accuracies must be read per task; there is no scientifically valid single accuracy obtained by averaging them.

## Per-source and pooled-source OOF results

| Task | Rows / groups | Accuracy | Macro-F1 | Balanced accuracy | Interpretation |
|---|---:|---:|---:|---:|---|
| QBugs QASM correct vs. buggy | 1,496 / 748 | **92.71%** | 92.69% | 92.71% | Best standalone program-correctness source |
| Quantum-platform comment labels | 319 / 304 | **94.04%** | 92.67% | 91.29% | Comment classification, not hardware diagnosis |
| Three-source pooled bug correctness | 1,620 / 810 | **89.63%** | 89.60% | 89.63% | Correct/fixed vs. buggy; pooled Bugs4Q + QBugs task |
| QSimBench exploratory | 300 / 150 | **82.33%** | 82.33% | 82.33% | Backend-identity proxy; descriptor collisions make this exploratory |
| LintQ context, conflict excluded | 287 / 216 | **79.44%** | 67.92% | 65.91% | Warning true-positive vs. false-positive; conflicts excluded |
| Qolumbina ideal-backend outcome | 182 / 26 | **84.07%** | 54.16% | 54.69% | Highly imbalanced; majority baseline is 82.42% |
| Bugs4Q robust-execution pass/fail | 2,220 / 37 | **56.76%** | 56.73% | 56.76% | Few independent issue groups; repeated executions are not independent groups |
| Bugs4Q core source-state label | 62 / 31 | **51.61%** | 51.56% | 51.61% | Small sample; near chance |
| Bugs4Q-NA source-state label | 62 / 31 | **50.00%** | 48.92% | 50.00% | Small sample; near chance |
| Bugs4Q core + NA source-state label | 124 / 62 | **49.19%** | 47.97% | 49.19% | Pooling these source-state examples did not help |

The merged correct/fixed-vs-buggy task reaches 89.63%, below QBugs QASM alone at 92.71%. Pooling sources did not improve this benchmark. Other source targets (comments, backend identity, detector outcome, execution pass/fail, mutation outcome) are different prediction problems and were not blended into a misleading aggregate score.

## QMutBench mutation-outcome benchmark

The established full-feature mutation-aware ExtraTrees model remains the strongest QMutBench result: **79.33% accuracy, 79.33% macro-F1, and 79.41% balanced accuracy**, from grouped five-fold development OOF across 578,117 rows and 305 origins. This is not an untouched external test result; the 85% target was not reached.

For a feature-matched control using the same four fold-local selected/scaled features and saved origin-disjoint folds used by the QNN experiment, eight classical estimators were freshly trained. This is a classical baseline report only; it does not integrate QML into the simulator.

| Classifier | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| ExtraTrees | **57.00%** | 56.61% | 56.73% |
| HistGradientBoosting | 56.93% | 56.48% | 56.64% |
| RandomForest | 56.79% | 56.46% | 56.54% |
| SGD logistic | 55.83% | 55.39% | 55.54% |
| Logistic regression | 55.73% | 55.00% | 55.34% |
| Linear SVC | 55.69% | 54.95% | 55.30% |
| Balanced logistic regression | 55.47% | 55.41% | 55.68% |
| Prior-only dummy | 51.84% | 34.14% | 50.00% |

The 57.00% matched-feature result is not directly interchangeable with the 79.33% full-feature result: the feature sets differ substantially. The matched benchmark measures classical performance on the narrow QNN-compatible input representation.

## Files and reproducibility

- This report: `CLASSICAL_ML_RESULTS.md` (linked from the repository README)
- Per-task pooled OOF metrics and fold-selection manifests: `reports/experiments/classical_ml_benchmarks_20261030/source_tasks/`
- Fold-level OOF predictions for the nested source/classifier searches: `reports/experiments/classical_nested_combined_search_v2_20261030/` and `reports/experiments/classical_sparse_knn_nested_20261030/`
- Historical grouped classical suite fold predictions and summaries: `reports/experiments/local_grouped_classical_suite_*/`
- Matched-feature QMutBench metrics and audit: `reports/experiments/classical_ml_benchmarks_20261030/qmutbench_matched_control/`
- Nested source model search: `experiments/run_classical_source_nested_search.py`
- Sparse cosine KNN search: `experiments/run_classical_sparse_knn_nested.py`
- Source-task configuration and expected input hashes: `reports/experiments/classical_ml_benchmarks_20261030/inputs/classical_nine_source_v2_config.json`

Raw external datasets and cached feature tensors are not part of this publication update. Fold-level OOF prediction CSVs for the nested source searches and grouped classical suites are included beside their metrics and manifests. Large QMutBench per-record prediction files and model intermediates remain local-only. The manifests retain task protocols and source hashes; rerunning source searches requires the matching locally prepared input CSVs. QMutBench raw data and model artifacts remain governed by their existing local-data rules.

## Reliability limits

A separate controlled simulator integration run classified 80/80 constructed synthetic scenarios correctly across its recorded seeds. This is a simulator-only synthetic control; it is not a learned-model benchmark or evidence of real-hardware accuracy. Its record is `reports/experiments/classical_simulator_fault_accuracy_20261030.json`.

Scores estimate performance on held-out groups from these prepared datasets, not on arbitrary future circuits or real hardware. Tiny group counts make the Bugs4Q estimates noisy. Qolumbina's high accuracy hides weak minority-class performance. QSimBench is exploratory due to collisions. Hardware-anomaly accuracy cannot be inferred from these program/classical-ML datasets; that requires a separately validated calibration-backed hardware benchmark.
