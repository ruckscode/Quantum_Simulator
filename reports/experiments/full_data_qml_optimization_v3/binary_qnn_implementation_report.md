# Full-development binary QNN trainer — implementation status

## Status

The dedicated trainer is implemented at `qmutbench_qml/training/run_binary_development.py`; the trainable circuit is in `qmutbench_qml/models/binary_vqc.py`. This is separate from the existing 12-origin three-class exploration. No QNN or classical model was fitted in this session.

## Data and evaluation contract

- Binary target: WOO=1, OPO=1, Survivor=0. The official development task has 578,117 rows and 305 origins.
- The historical five-fold OOF split was independently checked using only numeric `record_id`, `y`, and `fold` NPY arrays; the archive’s object-valued origin array was not loaded. Its labels and IDs match the development OOF CSV, all 305 origins stay within one fold, and no duplicate/missing development IDs were found.
- Exact reference fold counts: fold 1: 114,961, fold 2: 117,775, fold 3: 115,280, fold 4: 114,906, fold 5: 115,195.
- Training and validation use deterministic sampled records from each exact reference fold, capped at 256/class/fold and 32/origin/class. This is a sampled experiment, not a full-row fit or full-held-out-fold score. QNN, Logistic Regression, and ExtraTrees share the same fold-specific records.
- `ReducedFeatureEncoder` fits within each training fold, selects four of the 23 validated metadata/QASM-difference inputs, and transforms validation with the train-fitted encoding/scaler. IDs, origins, paths, folds, and `Killed` are not predictors; suspicious `source_gate_count_match` remains excluded.

## Models and checkpoints

- QNN: four-qubit, depth-two angle re-upload VQC with trainable Rot gates, ring CNOTs, a trainable affine sigmoid readout, analytic PennyLane `default.qubit`, and backpropagated gradients. Defaults: 8 epochs, batch size 32, Adam at 0.02; optimizer, depth, qubits, seed, and budget are configurable.
- A centered finite-difference check validates one circuit-parameter gradient on each fold before fitting. It is implemented but has not been exercised on PennyLane in this session.
- Classical controls: balanced Logistic Regression and ExtraTrees (100 estimators, depth 16), fitted on the same sampled rows/features.
- Fold checkpoints are atomic, single-file JSON records with an embedded SHA-256 checksum. One atomic replace commits the complete checkpoint, eliminating the old two-file checkpoint/sidecar interruption window. They contain IDs, fold, seed/configuration, feature schema/selection, source and feature fingerprints, software versions, predictions/metrics, QNN parameters and loss curve, gradient check, runtime, and memory. Resume accepts only complete matching checkpoints; stale/corrupt files are rejected.
- Metrics include accuracy, macro F1, balanced accuracy, per-class precision/recall/F1/support, confusion matrices, fold mean/SD, Brier score, runtime, RSS, and available RAM. Historical 79.33% accuracy / 79.33% macro F1 / 79.41% balanced accuracy remains context only because this run is sampled and uses a reduced feature schema.

## Validation actually run

- The independent standard-library split audit passed: 578,117 IDs and labels verified, 305 origins, zero origin-fold overlaps. Numeric archive SHA-256: `bfc1affdd40ac1032b65e417548075ce066407297fc603085ba6519403f56367`.
- Python compilation passed for the trainer, VQC, and tests. `--validate-config` passed without loading data or ML libraries.
- Mocked/pure-Python tests: **13 passed**. They cover label mapping, grouped-fold isolation, deterministic sampling, interrupted-run resume, atomic-write interruption, no accidental overwrite, stale configuration and data fingerprints, corrupt/incomplete checkpoints, and checkpoint provenance. Pytest used a workspace temp directory because the default Windows Temp path is sandbox-protected.
- The normal entry point exited at preflight: `MemoryError: available RAM varied across guard probes (724 MiB, 718 MiB, and 477 MiB); the final probe reported 477 MiB, while the estimated startup requirement remained 2048 MiB to preserve the 1024 MiB floor`. The observed amount varies; a later check was 718 MiB. No science imports, dataset load, preprocessing, training, or frozen-test access occurred.
- Sampling audit: each validation sample comes only from the requested held-out fold; training samples come only from the other four folds. Deterministic per-origin/class caps are applied before per-class caps. The pipeline separately checks record-ID and origin separation, verifies both classes are present, and fits the four-feature encoder on the sampled training rows only. All three models share the same fold rows and validation records. Metrics therefore describe the sampled, class-capped development OOF experiment; they are not estimates over every row or the natural class prevalence.
- Current available RAM at this audit: **756.8 MiB**, below the 2,048 MiB startup minimum. No PennyLane import, synthetic circuit/gradient check, real training, or frozen-test access was attempted. The checkpoint-resume mechanism was exercised with mocked fold results, not with a trained QNN checkpoint. No real QNN metric is claimed.

## Current-workspace continuation audit (2026-10-28)

- Rechecked the actual checkout: branch `qmutbench-master-optimization`, HEAD `a2d51be22fd985bbdd10655f7e31dc77b9d19ed6`; existing untracked artifacts were preserved. No Git lock files were present. The `.git` directory has explicit inherited-task SID write/delete deny rules, so changes were not staged or committed.
- Recovery ZIPs were checked without replacing them. The original, `v2`, and `final` archives pass ZIP integrity and their legacy per-member SHA-256 lists match. The audit-v3 archive and top-level archive SHA-256 also match.
- Found and fixed a real inference defect: `BinaryVQC.predict_proba` called `_logits` without the required affine-readout arguments. Inference now requests neutral scale/bias for the expectation, then applies the trained readout once. A pure AST regression test guards this call shape without importing PennyLane.
- The earlier test suite had 14 pure-Python tests. Subsequent epoch-recovery and RAM-gate regression coverage brought the current suite to **18 passed**. PennyLane is installed in `.venv` (0.45.1), as is scikit-learn (1.9.1); this only verifies package metadata, not backend execution.
- RAM has varied during successive audits. The latest guarded campaign preflight measured **649 MiB**, below the **2,048 MiB** startup requirement and 1,024 MiB in-run floor. No PennyLane import, synthetic circuit/gradient run, data loading, real fit, or frozen-test access was performed.
- The prediction-call fix is syntax-checked and tested. Epoch-boundary checkpointing is implemented with pure-Python serialization/recovery tests, but restoration through a live QNN is still unverified. Real circuit output, finite/nonzero gradient, parameter update, loss, determinism, and campaign metrics remain unverified until the memory preflight passes.

## Frozen test and remaining limit

Frozen-test outcomes were not accessed. Its checksum remains unverified because no authoritative local artifact path/hash was recovered. The full development QNN experiment remains unrun until RAM passes the 2 GiB estimated startup gate and stays above 1 GiB while training.

## Commands

Validate static configuration:

```powershell
.\.venv\Scripts\python.exe qmutbench_qml\training\run_binary_development.py --validate-config
```

Run the bounded sampled five-fold development comparison after the RAM guard passes:

```powershell
.\.venv\Scripts\python.exe qmutbench_qml\training\run_binary_development.py --output reports/experiments/qmutbench_binary_qnn_development
```

Resume validated completed folds after interruption:

```powershell
.\.venv\Scripts\python.exe qmutbench_qml\training\run_binary_development.py --output reports/experiments/qmutbench_binary_qnn_development --resume
```
