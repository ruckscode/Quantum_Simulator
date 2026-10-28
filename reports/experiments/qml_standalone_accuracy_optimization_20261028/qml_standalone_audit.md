# Standalone QML campaign audit

- Recomputed selected-model accuracy, macro F1, and balanced accuracy directly from copied development OOF predictions.
- Confirmed OOF record IDs are unique within each saved artifact.
- Confirmed each saved group appears in exactly one evaluation fold for every indexed prediction artifact.
- Parsed and integrity-checked 95 copied checkpoint artifacts; NPZ members passed ZIP CRC checks and contain expected parameter arrays.
- Replayed 20570 new model OOF predictions directly from checkpoint weights; probabilities matched within 2e-07.
- New QNN candidates were trained with parameter-shift gradients; each completed fold records its finite-difference gradient check.
- Frozen test data was not accessed; final held-out test accuracy remains unavailable.
- A QMutBench third-seed run was blocked before training by the 700 MiB RAM preflight; it is not counted as completed.

See `qml_standalone_audit.json` for file-level checks and recomputation status.
