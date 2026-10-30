# QMutBench QML reduced-feature pilot

## Status and scope

This is a completed small development-only pilot, not a full-scale model selection or deployment result. It uses the verified historical mutation-outcome target (WOO/OPO detected; Survivor undetected). The source CSV is scanned in bounded chunks to retain saved development IDs; frozen-test rows are not retained for features, labels, prediction, or evaluation.

- Sample: 72 records from 578,117 development rows and 305 origins; pilot used 9 seeded origins.
- Evaluation: 3-fold GroupKFold over sampled origins; per-origin cap 8; no origin overlap between train and validation.
- Feature selection: four encoded scalar inputs chosen by mutual information within each training fold only; fold-selected names are in `run_manifest.json`.
- Source data and cached QASM-derived features were read-only; source SHA-256 verified against the synchronized ML record.
- Wall time: 105.07s; simulator: PennyLane 0.45.1 `default.qubit`, analytic statevector execution.

## Results

Accuracy, balanced accuracy, and macro F1 are computed on pooled out-of-fold pilot predictions. Each QML model is compared with classical controls trained on the same selected four inputs and the same fold records.

| Model | Accuracy | Balanced accuracy | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| vqc | 50.0% | 48.6% | 48.6% | 48.9% |
| qsvc | 43.1% | 42.1% | 42.2% | 36.5% |
| extra_trees_4f | 34.7% | 36.4% | 34.6% | 36.9% |
| logistic_4f | 50.0% | 49.5% | 49.4% | 45.4% |
| rbf_svc_4f | 47.2% | 44.8% | 44.5% | 40.0% |

**Pilot finding:** neither QML model outperformed the strongest matched classical control. VQC macro F1 was 48.6% versus 49.4% for Logistic Regression; QSVC macro F1 was 42.2% and was below the reduced-feature RBF SVC at 44.5%. The paired VQC-minus-RBF intervals include zero. These tiny-sample results are exploratory and do not establish a QML advantage.

### Paired origin bootstrap: QML minus matched classical model

Differences use QML metric minus matched classical metric and are reported in percentage points (pp); intervals resample the nine validation origins, 500 times.

| QML model | Matched baseline | Metric | Difference (pp) | 95% percentile interval (pp) |
|---|---|---|---:|---:|
| vqc | extra_trees_4f | accuracy | +14.4 | [-9.1, +41.0] |
| vqc | extra_trees_4f | balanced_accuracy | +10.6 | [-11.1, +34.2] |
| vqc | extra_trees_4f | macro_f1 | +12.8 | [-9.7, +37.4] |
| vqc | rbf_svc_4f | accuracy | +2.7 | [-20.8, +23.6] |
| vqc | rbf_svc_4f | balanced_accuracy | +3.1 | [-14.3, +21.2] |
| vqc | rbf_svc_4f | macro_f1 | +3.6 | [-16.1, +22.1] |
| qsvc | extra_trees_4f | accuracy | +7.4 | [-4.2, +27.8] |
| qsvc | extra_trees_4f | balanced_accuracy | +4.5 | [-7.6, +18.6] |
| qsvc | extra_trees_4f | macro_f1 | +6.4 | [-5.7, +21.5] |
| qsvc | rbf_svc_4f | accuracy | -4.2 | [-12.5, +0.0] |
| qsvc | rbf_svc_4f | balanced_accuracy | -2.7 | [-10.3, +3.1] |
| qsvc | rbf_svc_4f | macro_f1 | -2.3 | [-10.3, +5.2] |

## Context from the established full-feature ML work

- Existing full 23-feature mutation-aware Extra Trees: accuracy 78.9%, balanced accuracy 79.0%, macro F1 78.9% on saved 3-fold grouped OOF over all 578,117 development records.
- Existing nested grouped development result: accuracy 79.3%, balanced accuracy 79.3%, macro F1 79.3%.
- These larger full-feature figures use different data coverage/protocols. They are context only and are not a direct QML comparison. This pilot does not establish improvement or quantum advantage.

## Costs, limits, and reproducibility

- Circuit/training costs by model are listed below; execution counts are PennyLane device execution totals, not physical shots.

| Model | Train time (s, total) | Inference time (s, total) | Train circuits | Inference circuits | Qubits |
|---|---:|---:|---:|---:|---:|
| vqc | 72.29 | 1.48 | 2880 | 72 | 4 |
| qsvc | 20.55 | 10.17 | 6912 | 3456 | 4 |
| extra_trees_4f | 0.21 | 0.02 | 0 | 0 | - |
| logistic_4f | 0.01 | 0.00 | 0 | 0 | - |
| rbf_svc_4f | 0.00 | 0.00 | 0 | 0 | - |

- VQC: 4 qubits, 2 variational layers, 20 Adam updates per fold, analytic shots (`None`), backpropagated simulator gradients. Loss decreased in 3/3 folds; this is not a formal convergence criterion.
- QSVC: 4-qubit analytic fidelity kernel and classical precomputed-kernel SVC. No quantum hardware or finite-shot noise was used.
- Per-model timings cover estimator fit/inference, not feature loading, fold-local encoding, or selection; total elapsed time covers the run. Memory records process RSS before and after the run; they are snapshots, not peak RSS. Small sample size, only nine origins, and noisy fold-local feature rankings limit conclusions.
- The QASM alignment features inherit the documented common-prefix/common-suffix alignment limitations. Only features selected among the same 23 inputs are used; `feature_parity.csv` documents categorical/numeric treatment and the train-only reduction.
- Frozen test was deliberately not evaluated because this is a pilot and no candidate was frozen. No test prediction/metric is claimed.
- Process RSS snapshots: 185.5 MiB before and 747.5 MiB after; these are not peak memory.
- Reproduce in the isolated env: `python -m venv qmutbench_qml/.venv`, install `qmutbench_qml/requirements-lock.txt`, then run `qmutbench_qml/.venv/Scripts/python.exe -m qmutbench_qml.training.run_pilot` from the repository root.
- Dependencies are isolated under `qmutbench_qml/.venv`; production ML, simulator, and historical reports were not modified.
