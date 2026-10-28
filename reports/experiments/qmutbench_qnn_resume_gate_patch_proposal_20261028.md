# QNN recovery gate patch proposal — 2026-10-28

This proposal addresses the confirmed resume-gate inconsistency without lowering either memory threshold.

## Current defect
At the start of run_campaign, when the output directory is non-empty and --resume is used, the code searches only the fold_checkpoints directory for fold_*.json. This overlooks valid epoch-boundary files named fold_NN_qnn_epoch_NNNN.json in the same directory. The inner training path already calls load_latest_qnn_epoch_checkpoint, so the preflight gate prevents its valid recovery path from being reached.

## Intended narrow fix
Replace the initial boolean check with a read-only helper that considers either:
- a completed-fold JSON file under fold_checkpoints/fold_NN.json, or
- at least one epoch checkpoint with the expected naming convention under fold_checkpoints/fold_NN_qnn_epoch_NNNN.json.

The gate should only determine whether some candidate state exists. It must not deem that state trusted. Existing strict loading must still validate checksums, schema, configuration, data fingerprint, fold identity, epoch continuity, and required numeric state before using it. An epoch-only directory whose files are corrupted, stale, or non-consecutive must be rejected with a clear validation error when loaded. A completed fold checkpoint must retain priority for a completed fold; epoch checkpoints must allow training to resume and finish inference/evaluation when no completed fold file is present.

## Required tests
1. --resume accepts an output tree with a valid epoch-1 checkpoint but no completed-fold file.
2. Resume rejects epoch histories with a gap, duplicate/malformed filename, stale config, wrong fold, wrong fingerprint, or checksum failure.
3. A completed fold result is reused without retraining.
4. If epoch training finishes but the process stops before final fold persistence, a subsequent run restores the final epoch and executes inference/classical controls rather than restarting QNN optimization.
5. A non-empty directory containing unrelated files and no valid QNN/fold checkpoints is not accepted as resumable.
6. The RAM preflight and 1,024 MiB in-run floor remain unchanged; no training starts on a failed preflight.

## Execution limitation
This is a code-review/patch specification, not a claim that the code has already been modified or tests run. The workspace is only accessible here through GitHub APIs; local uncommitted files and the user's Windows interpreter are not directly executable from this chat.
