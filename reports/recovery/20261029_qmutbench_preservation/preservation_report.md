# Preservation report

Inventory created: 2026-10-29 (local time context). Repository root: `C:\Users\pcm\OneDrive\Desktop\Quantum_Simulator`.

## Preserved state

The QMutBench full-development run was paused and checkpointed. No matching Python process was detected. All 56 files from the active run directory were copied to this recovery directory with their relative paths preserved; SHA-256 checks matched for all 56 copies. The original run files were not modified. A binary-safe patch snapshot of the current tracked Git diff is also saved here.

A read-only audit validated all 51 checkpoint payloads, checkpoint file hashes against the fold manifest, continuous chunk and batch progress, record counters, gradient checks, parameter-update flags, and the latest campaign progress. It found no issues: 816 Adam updates, 26,112 training records, fold 1 at epoch 0 / batch 816 / next chunk 52, and no completed folds. Frozen-test access remains false. The detailed state is in `training_state.md`; file hashes and copy paths are in `artifact_manifest.csv`.

## Git state

Branch: `qmutbench-master-optimization`, at `a2d51be22fd985bbdd10655f7e31dc77b9d19ed6` (2026-10-28), tracking branch 8 commits ahead of the remote. At inventory time: 31 tracked modified files, 0 staged paths, 46,198 untracked paths, and 117,377 ignored paths. Extensive existing environment and experiment output remains local; nothing was staged, committed, cleaned, or discarded. The current tracked diff snapshot is `tracked_worktree.patch` (228,603 bytes). See `git_state.txt`.

## Resume

Use the existing `.venv` (Python 3.12.10), repository root as working directory, and the exact bounded resume command in `resume_command.txt`. Let the runner's normal memory preflight pass before continuing. The most recent recorded attempt was blocked before training because available memory was below the configured estimate.

## Limits and unresolved risks

WMI denied process command-line inspection; `Get-Process` showed no Python-family process, but that check cannot identify a workload launched under another host process. The C: free-space reading succeeded before the copy. The source fingerprint matches and JSON checkpoint integrity are verified. There are no separate run logs, prediction artifacts, or scheduler state files; the QNN checkpoint state and manifests are complete for the recorded Adam resume point. The long full-development run remains far from completion, with all five folds still pending/completing and no frozen-test evaluation.
