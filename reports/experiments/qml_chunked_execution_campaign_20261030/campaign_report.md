# QML chunked execution campaign — 2026-10-30

This report reconciles the registered 18-task preprocessing matrix and reuses existing validated artifacts. It does not start a classical comparison or access a frozen test set.

Status: 14 tasks have reusable completed artifacts; 2 are ineligible; 1 is blocked by data/licensing/split gates; 1 full-development task remains partial.

## Live execution status

The continuous runner is active in fold 1 at durable checkpoint 486, with 7,776 optimizer updates. SHA-256: `2e0ae7a8cc585a71de8965108f26d6262f98a2cc2afe7718806c8ee117e2d5f6`. Available RAM at report time: 1939.5 MiB. It runs bounded 128-chunk sessions, checks memory before each session, preserves chunk checkpoints, waits when RAM is too low, and requests Windows to stay awake while active. The disposable next-update test passed; frozen-test access: no.

The five-fold run remains partial. OOF predictions and final task metrics are pending. See `runtime/continuous_resume_status.json` and `runtime/runner_*.log` for live status.

## Other QML work checked

No additional eligible incomplete QNN task was found. The 546-row Qolumbina multi-backend grouped sweep is complete; the Bugs4Q-Robust and QSimBench 20-epoch QNN sweeps both report five completed folds; and the development-only QMutBench QASM feature extraction is verified for 578,117 records across 305 origins with zero parse errors. Two ineligible tasks and the licensing/split-blocked platform workbook task remain correctly excluded. The registered broad classical comparison suite remains out of scope and was not run.

## Incomplete eligible work

QMutBench binary development remains in fold 1 of 5 in `reports/experiments/qmutbench_binary_qnn_full_development_chunk16_20261029`. The verified starting audit records checkpoint 371 and 5,936 optimizer updates. The disposable continuation passed; the real run has advanced to checkpoint 486 with 7,776 updates. No fold is complete and no OOF metrics are available.

The initial controller attempt used the full-development default chunk size of 128 and hit a fingerprint mismatch. The controller was corrected to pass the checkpoint's recorded `chunk_batches=16`; the subsequent disposable resume passed. The original checkpoint hash before the real resume was `928cbbef8dc42afe43059dc8bff016aaa3b2f37370dc3440751d040444589298`. The real run is continuing only from verified chunk boundaries.

Measured full-run memory and duration estimates remain provisional. The latest observed available RAM was 2092.9 MiB; training checkpoints are being committed while the runner remains above its memory pause floor.

## Execution limits

All other eligible QNN tasks already have grouped OOF/checkpoint evidence and are marked for reuse. The original QBugs, LintQ, and platform-comment rows retain their current leakage/label caveats; separately versioned corrected variants remain separate tasks. Bug-Pattern and QChecker lack validated supervised targets/splits. QML platform workbooks remain blocked by unresolved license and split issues. Mute-Alg's full-source task remains blocked by pickle compatibility; its compatibility-selected subset remains ineligible for representative training.

The separate broad classical suite is explicitly outside this execution and was not initiated. QMutBench binary training is the only eligible incomplete QNN task. Its execution estimate is measured from existing progress evidence; no completion time is promised.

## Per-task status

See `task_status_matrix.csv` and `progress_registry.json` for all 18 tasks, classifications, readiness states, and source-matrix references.

## Resume

Re-run `.venv\Scripts\python.exe experiments\run_qml_chunked_execution_controller.py` when available memory is sufficient. Each call requests one checkpointed chunk and retains the existing training configuration.
