# Safe recovery and QML campaign status — 2026-10-28

- Branch/HEAD: `qmutbench-master-optimization` / `a2d51be22fd985bbdd10655f7e31dc77b9d19ed6`.
- Existing untracked work was preserved. No reset, clean, or staging was performed. `.git` and `.git/index` have inherited write-deny ACLs; no Git lock/process was active when inspected.

## QMutBench data integrity

- Official metadata contains 723,079 rows; the official checkout and ingestion copy were previously compared value-identically.
- The development corpus contains 578,117 records across 305 origins. Its three-fold OOF record IDs have zero origin overlap. The binary label mapping WOO=1, OPO=1, Survivor=0 agrees with all saved development labels.
- Structural QASM and mutation-difference caches each contain 578,117 records in 58 chunks, with zero parser errors. The prior full cache audit rehashed all 116 chunks; its results remain in `feature_extraction_audit.json`. This turn did not repeat the full rehash.
- The established five-fold incumbent split was independently verified in this turn from safe numeric arrays in the saved NPZ and the development OOF CSV: 578,117 rows, 305 origins, and zero group crossings. See `binary_reference_fold_audit.json`. The NPZ object array was not opened.

## Binary QNN implementation

A dedicated bounded binary trainer is now implemented in `qmutbench_qml/training/run_binary_development.py`; the model is `qmutbench_qml/models/binary_vqc.py`. It uses the established five origin-grouped folds, development-only deterministic samples, training-fold-only feature reduction, a genuine analytic variational QNN, and same-sample Logistic Regression/ExtraTrees controls. Defaults cap each fold at 256 training and 256 validation rows per class, and cap each origin/class at 32. Results must be described as sampled-fold results, never as full-row training.

The trainer has atomic, checksummed per-fold checkpoints and validates dataset/config/feature fingerprints before resume. Tests cover mocked interruption/resume, corrupt and incomplete files, stale configuration, label mapping, grouped folds, and deterministic sampling. The full specification is in `binary_qnn_implementation_report.md`; defaults are in `binary_qnn_default_config.json`.

## Execution and blockers

- Python compilation and static `--validate-config` passed. The eight pure-Python tests passed.
- The entry point stopped before scientific imports/data access with `MemoryError: Only 724 MiB available; estimated peak requires 2048 MiB to preserve the 1024 MiB floor`. A later RAM snapshot was about 718 MiB. A later post-compilation RAM snapshot was 692 MiB; still below the startup estimate. No model training, feature fitting, or frozen-test access occurred. The implemented finite-difference circuit-gradient check and real PennyLane checkpoint resume have not been exercised because preflight did not pass.
- The frozen test was not opened; its checksum is unverified because no authoritative artifact path/hash was found in available configuration/manifests.
- Git staging remains blocked by explicit ACL denies. No ACL changes were made. A checksummed implementation bundle and its SHA-256 manifest are retained in this report directory.

See `binary_qnn_implementation_report.md` for the exact commands to run after RAM reaches the 2 GiB startup estimate.
