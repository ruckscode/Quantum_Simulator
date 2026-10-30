# QMutBench additional QML family pilot

This extension reuses the completed VQC, QSVC, and matched classical predictions from the nine-origin pilot, then evaluates data re-uploading and quantum reservoir classifiers on the identical 72 records, origin assignments, fold assignments, and fold-local four-feature inputs. Frozen-test records and labels were not read.

| Model | Accuracy | Balanced accuracy | Macro F1 | Fold macro F1 mean +/- SD | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| vqc | 50.0% | 48.6% | 48.6% | 46.4% +/- 9.4% | 48.9% |
| qsvc | 43.1% | 42.1% | 42.2% | 41.0% +/- 12.6% | 36.5% |
| extra_trees_4f | 34.7% | 36.4% | 34.6% | 31.5% +/- 5.9% | 36.9% |
| logistic_4f | 50.0% | 49.5% | 49.4% | 48.9% +/- 14.0% | 45.4% |
| rbf_svc_4f | 47.2% | 44.8% | 44.5% | 44.6% +/- 12.2% | 40.0% |
| data_reupload | 48.6% | 53.1% | 46.9% | 45.5% +/- 20.7% | 55.8% |
| quantum_reservoir | 37.5% | 42.1% | 34.7% | 32.3% +/- 8.4% | 44.0% |

## Family accounting

VQC is the trainable QNN in this environment, so it is not counted twice as another independent family. QSVC is the quantum-kernel family. Data re-uploading is a distinct trainable-circuit architecture. Quantum reservoir computing uses a fixed quantum map and a trained classical readout.

The larger 24-origin expansion was stopped after system available memory dropped below 450 MiB; its interrupted attempt wrote no fold metrics. This report instead extends the already completed nine-origin pilot using exactly matched records and folds. Results are exploratory and do not establish quantum advantage. The 23-feature historical Extra Trees and nested grouped metrics in `model_comparison.csv` are context only because their coverage and protocol differ.

Elapsed extension runtime: 115.32s. Process RSS before/after: 185.1/573.1 MiB. Circuit evaluations, qubit count, depth, fold metrics, per-class scores, and origin-bootstrap intervals are in the adjacent CSV/JSON outputs.

## Runtime and circuit cost

| Model | Train s total | Inference s total | Train circuit executions | Inference circuit executions | Qubits | Depth |
|---|---:|---:|---:|---:|---:|---:|
| vqc | 72.29 | 1.48 | 2880 | 72 | 4 | 2 |
| qsvc | 20.55 | 10.17 | 6912 | 3456 | 4 | 2 |
| extra_trees_4f | 0.21 | 0.02 | 0 | 0 | - | - |
| logistic_4f | 0.01 | 0.00 | 0 | 0 | - | - |
| rbf_svc_4f | 0.00 | 0.00 | 0 | 0 | - | - |
| data_reupload | 74.05 | 1.50 | 2880 | 72 | 4 | 2 |
| quantum_reservoir | 0.56 | 0.26 | 144 | 72 | 4 | 2 |
