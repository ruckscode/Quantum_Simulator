# QMutBench binary QNN completion audit — 2026-10-28

## Scope and result

This audit follows the supplied master task against the live workspace. It does not claim the binary QNN campaign completed. The requested output directory `reports/experiments/qmutbench_binary_qnn_development` exists but is empty; no four-epoch log or checkpoint was found there. The task's four reported losses are therefore unverified in this workspace.

The existing runner at `qmutbench_qml/training/run_binary_development.py` is a development-only sampled five-fold campaign, not a full 578,117-row QNN fit. It keeps the binary mapping `WOO=1`, `OPO=1`, `Survivor=0`, uses the numeric fields from the incumbent five-fold OOF archive, and has a 2,048 MiB startup gate plus a 1,024 MiB in-run floor. It writes immutable, checksummed, atomic epoch-boundary QNN checkpoints and validates configuration, data fingerprint, fold, history, optimizer accumulation, RNG state, and finite parameter values on resume. It has no batch-position resume because its checkpoint boundaries are completed epochs.

## 1. Verified completed work

- Branch: `qmutbench-master-optimization`; starting and ending HEAD: `a2d51be22fd985bbdd10655f7e31dc77b9d19ed6` (`Optimize standalone QNNs on local grouped folds`). The local remote-tracking ref points to the same commit.
- Tracked working tree was clean at audit start (`git diff` empty). There were 59 untracked status entries, including the QML implementation, tests, data, experiments, and reports. They were preserved and not staged.
- Existing local recovery patch `.tracked-working-tree-recovery.patch` SHA-256: `69D89FA899DF7ADD960345FD197CD061D0F609361E26A4C7D980228BADF708C1`. This is a checksum of that patch only; it does not prove every untracked artifact is included or verify any separate recovery bundle.
- Local remote-tracking branch is present, but `git ls-remote` could not connect to GitHub (`github.com:443`). Current remote contents and whether untracked work is backed up remain unverified.
- Existing campaign report identifies the incumbent binary Extra Trees result: macro F1 0.7933135654, accuracy 0.7933290320, balanced accuracy 0.7940874828, on 578,117 development rows across 305 origins and five grouped folds. These are classical OOF results, not QNN results. The reported production-model hash is `8cd332a309e27ef09cc46c5bf8a8788da35c05aa6ed25ed5de21d26ced076175`.
- Existing `test_manifest.json` records a historical 26-test pass for Bugs4Q validation/integration, and a separate earlier QNN report of 10 passed, 2 fit tests stopped by a 500 MiB guard, and 1 deselected. Neither entry is a fresh run of `tests/test_qmutbench_binary_qnn.py` in this audit.
- Existing multi-source campaign manifests state that the frozen QMutBench test was not accessed. The binary handoff warns that the historical frozen set was previously reused; no new frozen-set inspection was done.

## 2. Partially completed work

- Binary QNN code, matched-control infrastructure, tests, and campaign reports exist locally. The report at `reports/experiments/qmutbench_qml_handoff_20261027.md` preserves grouped-fold and leakage constraints.
- The campaign has other completed and partial QNN work, but its `checkpoint_manifest.json` explicitly claims no QMutBench three-class or Bugs4Q QNN real-data parameter checkpoint. Those tasks are distinct from the binary task.
- The supplied report of four binary QNN epochs (losses 0.706045, 0.694693, 0.686852, 0.679502) could not be reconciled to logs or saved state. Do not resume from those numbers as if verified.
- Checkpoint helper functions have unit coverage in `tests/test_qmutbench_binary_qnn.py`, but this audit did not execute that test file or a real circuit gradient/update smoke test.

## 3. Failed or blocked work

- Live system-memory snapshot through the project venv's `psutil`: total 7,488 MiB, available 989 MiB. This is below the prescribed 1,024 MiB run floor and 2,048 MiB startup threshold. No expensive run was started or retried.
- The Windows WMI memory query was denied. The direct `psutil` measurement succeeded.
- GitHub remote inspection failed because outbound HTTPS was unavailable. No push or commit was attempted.
- The named binary development-run directory is empty. There is no verified interrupted fold or epoch checkpoint to resume.

## 4. Work not attempted

- Fresh binary-QNN test suite, circuit forward/gradient/update smoke, binary campaign, classical controls matched to its sampled features, and any new fold metrics.
- Frozen-test evaluation, feature or threshold tuning, and any campaign using frozen-test outcomes.
- Cleanup of generated caches or datasets, changes to watcher/search exclusions, or edits to training code. Disk occupancy is not a direct measure of physical RAM; no cleanup was justified from this audit.
- Commit or push. The numerous untracked files were not staged indiscriminately.

## 5. Risks and dependencies

- The larger repository contains many untracked data and experiment artifacts. The recovery patch checksum does not establish their backup coverage. Preserve them until a manifest-level backup audit is completed.
- `reports/experiments/qmutbench_multisource_qml_campaign_20261028/resource_usage.json` records earlier resource guards and explicitly says peak process RSS was not sampled. Its 500 MiB floor applies to those reports; it does not replace the binary runner's 1,024 MiB guard.
- The binary runner imports its scientific stack only after preflight. Its estimate is a guard estimate, not a measurement of peak RSS or proof that an 8-epoch campaign will fit.
- The reference archive and local datasets are necessary to generate a run fingerprint and preserve origin folds. The current audit did not recompute or independently verify their hashes.
- No current remote verification means local-only changes may not be backed up. Network access or a verified external artifact store is required for that check.

## 6. Exact next actions

1. Keep the existing untracked files and recovery patch intact. When outbound access is available, verify the remote branch and separately inventory/backup untracked source, reports, and data with hashes before any delivery.
2. On this machine, wait until measured `psutil.virtual_memory().available` is at least 2,048 MiB before running the binary campaign. Continue honoring the runner's 1,024 MiB in-run stop. Do not lower either guard or retry while preflight fails.
3. First run the bounded integrity suite: `python -m pytest tests/test_qmutbench_binary_qnn.py -q`. Then run a tiny real-circuit gradient and parameter-update smoke only if its own safe memory conditions are met. Distinguish real-circuit evidence from mocked checkpoint tests.
4. Start in the existing empty output directory: `python -m qmutbench_qml.training.run_binary_development --output reports/experiments/qmutbench_binary_qnn_development`. Record the preflight snapshot and command. To continue only from validated saved outputs, add `--resume`; do not copy reported losses into a checkpoint.
5. Compare QNN and classical models on identical sampled record IDs, fold assignments, and fold-local four-feature representation. Keep model selection within development origins and report no frozen-test metric unless an independently untouched test protocol is established.
6. Update this report with actual test logs, measured peak or available-memory evidence, checkpoint hashes and resume validation, completed epochs/folds, and valid OOF metrics. Do not call the campaign complete until those artifacts exist.

## Audit commands and outcomes

- `git status --short --branch`, `git rev-parse HEAD`, `git diff --stat`: branch/HEAD as above; tracked diff empty; 59 untracked entries preserved.
- `Get-FileHash -Algorithm SHA256 .tracked-working-tree-recovery.patch`: hash recorded above.
- `git ls-remote --heads origin qmutbench-master-optimization`: failed due outbound HTTPS connection error; local remote-tracking ref matches HEAD.
- `.venv/Scripts/python.exe -c "import psutil; print(psutil.virtual_memory())"` (equivalent snapshot): 7,488 MiB total, 989 MiB available at audit time; only 22 MiB RSS across the two observed `python.exe` processes. The WMI query was denied.
- Checkpoint directory listing: `reports/experiments/qmutbench_binary_qnn_development` is an existing empty directory (zero entries).
- No cleanup, source edits, tests, training, staging, commit, or push were performed in this audit.

## Continuation update — 2026-10-28

### Recovery patch and QNN bundles

- Recomputed `.tracked-working-tree-recovery.patch`: SHA-256 `69d89fa899df7add960345fd197cd061d0f609361e26a4c7d980228badf708c1`, matching the original audit (case-insensitive hex comparison).
- Inspected its Git summary: 43 tracked Quantum Simulator files, 3,605 insertions and 2,416 deletions. It contains neither `qmutbench_qml/` nor `tests/test_qmutbench_binary_qnn.py`; it is not the QNN implementation recovery artifact. `git apply --check` and `git apply --reverse --check` both fail on multiple files, so this historical patch is not directly applicable to the current checkout. It remains untouched.
- Located six QNN implementation/recovery ZIPs plus `safe_recovery_changes_20261028.zip` and their SHA-256 manifests under `reports/experiments/full_data_qml_optimization_v3/`. Recomputed each archive hash, checked ZIP CRC/integrity, and verified every manifest-listed member hash. All seven bundles passed. The latest `binary_qnn_interrupted_recovery_20261028_v5.zip` has SHA-256 `c9ab8e62fc24562e7b89e4eff30674ac313e0fcdb7344509ea427a0dbfac602e`; all 10 manifest-listed members match the current workspace, including the runner, VQC, feature reducer, and test file. `safe_recovery_changes_20261028.zip` also passed with SHA-256 `81b837c6ace4d6b9fada7694e854b3ee0e7d8077785f9a5e461a9ddae5270282`. No recovery artifact was changed.

### Missing run and historical epochs

- A previous same-day audit in `reports/experiments/full_data_qml_optimization_v3/binary_qnn_interrupted_epoch_recovery_audit.md` records that `reports/experiments/qmutbench_binary_qnn_development` existed but was empty when inspected. It had no fold or epoch checkpoint, history, predictions, or run summary. The current directory is also empty (created 2026-10-28 19:21:22 +05:30; zero children). The earlier audit's “absent” wording was an inspection error: `Get-ChildItem` with no output cannot distinguish an absent path from an empty directory. Git does not preserve empty directories, but that does not explain missing checkpoint files here; the directory has no files to preserve.
- That prior audit also records that the earlier trainer wrote a fold checkpoint only after QNN training, inference, and both classical controls. A stop during an epoch or before completing those controls therefore left no resumable state. The current runner adds atomic checkpoints at completed QNN epoch boundaries, but that improvement cannot reconstruct state from the earlier run.
- Searched preserved reports, logs, CSV/JSON artifacts, and QNN/test files for the reported loss values (`0.706045`, `0.694693`, `0.686852`, `0.679502`) and the expected output path. The exact losses appear only in audit/task descriptions, not a saved training log or checkpoint. Treat them as historical user-reported terminal output, not independently verified results. No result was recreated.
- Batch-level checkpointing was not added. The current epoch-boundary payload preserves the trainable parameters, Adam accumulation, RNG state, loss history, and completed epoch. A stop inside an epoch can safely replay that epoch from the last completed boundary; batch-level resume would additionally require persisting and validating the exact shuffled batch order and position, with no current evidence that this complexity is needed for the bounded sampled workload.

### Validation completed in this continuation

Commands and actual results:

- `Get-FileHash -Algorithm SHA256 .tracked-working-tree-recovery.patch`: matched the recorded digest above.
- `git apply --stat -- .tracked-working-tree-recovery.patch`: 43 files; patch content is not QNN-specific. Forward and reverse `git apply --check` both failed; patch was not applied.
- ZIP/member validation script over all six `binary_qnn*.zip` recovery bundles and corresponding manifests: all six archive digests matched; every ZIP integrity check passed; every listed member digest matched.
- Exact focused-test command (pytest plugin autoload disabled; workspace-local temp paths):

  ```powershell
  $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
  $env:TEMP=Join-Path (Get-Location).Path '.pytest-codex-binary-temp'
  $env:TMP=$env:TEMP
  $base=Join-Path (Get-Location).Path '.pytest-codex-binary-base'
  .venv\Scripts\python.exe -m pytest --basetemp=$base tests\test_qmutbench_binary_qnn.py -q
  ```

  Result: **18 passed in 0.67s**. These are bounded integrity tests with mocked checkpoint/resume behavior; they do not prove a real PennyLane gradient or training update.
- AST parse of `run_binary_development.py`, `binary_vqc.py`, `features/reduction.py`, and `test_qmutbench_binary_qnn.py`: all four passed.
- `.venv\Scripts\python.exe qmutbench_qml\training\run_binary_development.py --validate-config`: passed, printed `configuration_validated_no_data_loaded` with the five-fold default configuration and established label mapping.
- Fresh binary-QNN test coverage includes atomic-write interruption, preservation of the last completed epoch, resume from the latest validated boundary, and rejection of corrupted, stale, incomplete, or gapped checkpoints. No source change was needed.
- `git diff --check`: passed. The tracked worktree remains unchanged. The 59 pre-existing untracked entries remain present; the current status count is 60 including this audit report. No unrelated files were staged, committed, or deleted.

### Remote backup check

- Retried `git ls-remote --heads origin qmutbench-master-optimization`: GitHub responded with remote commit `8965d3cfa9d40a53f841e99268d72eedd50fbe56`.
- Fetched that branch for read-only object/tree inspection (no checkout, merge, staging, or commit). `origin/qmutbench-master-optimization` is now eight commits ahead of local HEAD `a2d51be22fd985bbdd10655f7e31dc77b9d19ed6`.
- `git ls-tree -r --name-only origin/qmutbench-master-optimization -- qmutbench_qml tests/test_qmutbench_binary_qnn.py reports/experiments/full_data_qml_optimization_v3` showed only the report directory README; it showed no QNN package, test, or recovery ZIPs. A broader remote tree search likewise found no QNN implementation or test paths. **The QNN work is not verified as backed up to GitHub.** The local v5 ZIP plus its verified manifest is the current recovery copy. The branch was not pulled because that would change the checkout and interact with the preserved untracked workspace.

### Current resource state and restart commands

- Recent `psutil` readings varied between 842 and 1,210 MiB available, with 7,488 MiB total. The latest recorded probe was 1,210 MiB: above the 1,024 MiB in-run floor but below the 2,048 MiB startup requirement. No QNN campaign, PennyLane import, real-circuit forward/gradient test, or frozen-test access was started.
- Start fresh in the existing empty output directory only after the runner's measured preflight passes:

```powershell
.\.venv\Scripts\python.exe -m qmutbench_qml.training.run_binary_development --output reports/experiments/qmutbench_binary_qnn_development
```

- To resume that output after at least one validated checkpoint exists, use:

```powershell
.\.venv\Scripts\python.exe -m qmutbench_qml.training.run_binary_development --output reports/experiments/qmutbench_binary_qnn_development --resume
```

- Keep startup at or above 2,048 MiB and the in-run stop floor at 1,024 MiB. The current blocker is insufficient startup RAM. Valid QNN predictions, gradients, parameter updates, completed epochs/folds, and binary QNN metrics remain unverified.

## Continuation update — backup and recovery tests

### Dedicated remote backup

- Verified all ten members in `binary_qnn_interrupted_recovery_20261028_v5.sha256.json` against the current local workspace: all ten local SHA-256 values match. At the then-current remote branch HEAD `8965d3cfa9d40a53f841e99268d72eedd50fbe56`, all ten paths were absent.
- Created a temporary shallow clone of that remote commit, then committed only the QNN implementation package and supporting files. The backup branch is `backup/qmutbench-binary-qnn-20261028`; its first commit `b785b67d8df6a4214538677cc69db0d9aa009e66` added 21 scoped files (QNN runner/model/feature/data loader, package scaffolding, tests, requirements, implementation/recovery documentation, audit script, and the verified v5 bundle/manifest). No dataset, credentials, or other experiment outputs were included.
- Added two synthetic checkpoint regression cases and an updated test report, plus a refreshed recovery bundle. Follow-up commit: `4f948f39c0262c56eeccee282ea9e7de2d681dbd`. The bundle is `binary_qnn_interrupted_recovery_20261028_v6.zip`, SHA-256 `9d2d8e1bffe99a9a3e074ec9ebdfcb28b666cd41261a52fa8cb69912fd58f986`; its external manifest lists ten source/report files.
- Pushed the dedicated branch. Direct `git ls-remote` verification returned `4f948f39c0262c56eeccee282ea9e7de2d681dbd`, matching the local clone's HEAD. The latest commit changes only four files from the prior backup commit; the full branch adds 23 distinct paths relative to remote base `8965d3c`. No unrelated files were staged.
- Verified all ten v6 source/report members directly from the pushed commit's bundled ZIP against its manifest; archive hash matched and no member was missing or corrupt. Seven remote Git blobs have the exact manifest hash; the other three text files differ only because Git normalized CRLF/LF. Comparing with line endings normalized, all ten match and there are no content mismatches.
- The original workspace branch remains at `a2d51be22fd985bbdd10655f7e31dc77b9d19ed6`, eight commits behind `origin/qmutbench-master-optimization`. It was not switched, merged, staged, or committed. The 59 pre-existing untracked entries remain preserved. The current status count remains 60 including the audit report; the v6 files are inside an already-untracked experiment directory.

### Checkpoint recovery tests and code review

- Added `test_qnn_epoch_checkpoint_rejects_wrong_fold_split_and_configuration` to verify epoch restoration rejects a different fold, a changed data-split fingerprint, and changed configuration.
- Added `test_incomplete_epoch_is_never_written_or_reported_complete` to verify a two-epoch state with only one loss entry is rejected, no epoch-2 checkpoint is written, and epoch 1 remains the latest valid boundary.
- Reran the focused suite with pytest plugin autoload disabled: **20 passed in 0.69s**. Available memory was 1,224 MiB before and 1,238 MiB after. These are synthetic/integrity tests; no PennyLane circuit ran.
- Code inspection confirmed epoch checkpoints are written only after the full epoch loop, with model/readout parameters, optimizer accumulation, RNG state, and completed loss history. The atomic writer flushes and fsyncs a temp file before `os.replace`; resume checks the checksum, schema, fold, epoch, full configuration, data fingerprint, numeric state, optimizer state, and consecutive history. The model checks the 1,024 MiB floor at epoch and batch starts. No checkpoint code change was needed.

### Memory profile and remaining opportunity

- Ran the runner's actual `memory_preflight(DEFAULT_CONFIG)` once after the regression tests. It blocked safely with `Only 1198 MiB available; estimated peak requires 2048 MiB to preserve the 1024 MiB floor`. No scientific stack was imported by that preflight, and no training launch was attempted.
- The default sample bound is 5,120 rows. The guard estimates 1 MiB for selected arrays, 1 MiB for statevector/autodiff work, and 1 MiB for optimizer parameters, then reserves 700 MiB above the 1,024 MiB floor. That gives an estimated 1,727 MiB peak, so the fixed 2,048 MiB startup threshold governs. These are code estimates, not observed training peaks.
- Static allocation review found the numeric fold/label lookup arrays and duplicate-detection bytearray are compact (about 2.1 MiB combined at 723,079 positions). More avoidable memory is in `_reference_rows`, which retains Python lists of all 578,117 IDs grouped by fold/label/origin, and in `load_development_records`, which reads the full 578,117-row OOF CSV into a pandas frame before filtering to the bounded sample and builds a full ID set. The QASM cache itself is read in 50,000-row chunks and only selected rows are retained; circuit batches are capped at 32.
- A chunked OOF validation/selection path could reduce those transient allocations while preserving duplicate/count/origin checks and exact fold alignment. It was identified but not implemented here because its integrity behavior needs dedicated loader fixtures and validation; no speculative data-loading change was made without those tests. Peak RSS during a real QNN fit remains unmeasured.
- A later idle `psutil` snapshot was 1,345 MiB available of 7,488 MiB total, still below the 2,048 MiB startup requirement. The documented start/resume commands above remain valid once the measured preflight passes; the 1,024 MiB in-run floor remains unchanged. No real quantum gradient, update, epoch, fold, or QNN metric is claimed.
