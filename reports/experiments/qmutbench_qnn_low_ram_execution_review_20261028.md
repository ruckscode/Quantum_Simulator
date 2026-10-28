# Low-RAM execution review — 2026-10-28

## Evidence inspected
- Backup branch: `backup/qmutbench-binary-qnn-20261028`
- Runner blob SHA: `79ebc69137b37b6d6b7d30843ac15c377ba38cb7`
- QNN model blob SHA: `78efe35dacd2fcae84f0b8b7eb863de77105d45d`
- Data loader blob SHA: `11250771f17ae4fc8a003f73662661a66dc298a3`
- The runner and model were read from GitHub. No local machine or local filesystem was executed/modified by this review.

## Key findings
1. The QNN model implements an analytic `default.qubit` circuit, PennyLane/autograd gradient check, optimizer step loop, and calls a checkpoint callback after each completed epoch.
2. The runner uses 5-fold development OOF numeric arrays and validates the OOF CSV row-by-row before sample selection, which avoids loading the object-valued NPZ origin array.
3. The data loader currently does `pd.read_csv(oof_path, usecols=[record_id, origin_program, y_true, fold])` for all 578,117 development rows, converts all record IDs into a Python set, then filters to requested IDs. It scans the full raw metadata CSV and holds matching metadata rows, then chunks the QASM diff CSVs. This is a bounded row-wise source scan but OOF DataFrame/set use is avoidable for a small capped sample.
4. The runner also retains the complete `split["strata"]` lists and `split["fold_by_id"]` / `y_by_id` arrays for the entire campaign. It accumulates predictions and fold metrics for all completed folds in Python lists. This may increase memory over campaign duration but is not the likely sole reason for a sharp RAM drop inside epoch 5.
5. The QNN trainer persists each epoch through the callback, but `run_campaign` only detects an existing `fold_checkpoints/fold_*.json` in its initial resume check. There is a consistency defect: epoch checkpoints alone are not included in `has_checkpoints`, so if a non-empty output directory has epoch files but no completed-fold checkpoint, `--resume` is rejected before calling the epoch restore logic. This should be fixed with a regression test on the backup branch.
6. A second recovery concern: a crash after training has completed all epochs but before the completed fold JSON is atomically written leaves only epoch checkpoints. Resume should accept these and continue through inference/classical controls, preserving the saved QNN state; the runner needs to distinguish epoch checkpoint files from completed fold JSON files.
7. Chunking the OOF CSV and avoiding the whole raw OOF DataFrame/set is a reasonable optimization candidate, but it must preserve exact OOF IDs, labels, folds, origin-group consistency, and source checksum checks. No change was made here because the connector exposed no local test runtime/dataset files to run parity tests.
8. The existing startup gate and 1,024 MiB running floor should not be lowered.

## What could and could not be done here
- Successfully accessed the public repository and reviewed the QNN runner, model, and data loader on the dedicated backup branch.
- Confirmed GitHub repository metadata reports push permission; this does not grant access to the user's local workspace or prove a remote workflow has run.
- No GitHub Actions workflow file was found at the guessed path `.github/workflows/python-tests.yml`; no workflow was launched.
- No tests were run in the user's Windows environment, no local files were changed, and no quantum circuit/gradient/training campaign was executed during this review.
- The reported last RAM reading of 1,345 MiB remains below the 2,048 MiB startup requirement (per user-provided audit), so training should remain blocked.

## Next implementation steps for Codex in the actual workspace
1. Add an `has_resumable_state` predicate that recognizes validated complete-fold JSON or a consecutive set of immutable `fold_NN_qnn_epoch_*.json` checkpoints. Validate the epoch sequence and provenance before deciding resume is allowed.
2. Add tests for: epoch-only output directory accepted on resume; gaps/corruption rejected; completed-fold checkpoint reused; a crash before final fold save can resume from last epoch into inference/evaluation.
3. After tests, optimize loader OOF selection with chunked CSV reading or an indexed lightweight manifest, then compare exact selected row IDs, label values, fold assignments, origin assignments, and hashes against the current path using a small deterministic fixture and a full parity test only where memory permits.
4. Verify memory with process and OS measurements, leave safety guards in place, and do not start campaign until 2,048 MiB startup threshold passes.
