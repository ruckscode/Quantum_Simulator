# QMutBench binary QNN interruption and recovery audit

Date: 2026-10-28

## Observed workspace state

- Checkout: `qmutbench-master-optimization`, HEAD `a2d51be22fd985bbdd10655f7e31dc77b9d19ed6`.
- The requested output directory `reports/experiments/qmutbench_binary_qnn_development` exists but was empty when inspected. It contains no fold checkpoint, epoch checkpoint, history, prediction, or run summary. The report of four completed epochs and their losses came from the user; corresponding raw console output was not available in this workspace to verify independently.
- The prior implementation only wrote a completed-fold checkpoint after QNN training, QNN inference, and both classical controls. Therefore the reported partial epoch/fold cannot be restored from the inspected artifacts. No Python QNN process was present in the accessible process inventory.
- Frozen-test data was not accessed.

## Recovery implementation

- Added immutable atomic JSON checkpoints at each completed QNN epoch under `fold_checkpoints/fold_NN_qnn_epoch_EEEE.json`.
- Each checkpoint carries the fold data/config fingerprint, completed epoch, loss history, trainable weights and readout, NumPy RNG state, Adam first/second moments and step count, elapsed training time, execution count, and peak RSS. A canonical SHA-256 digest protects the payload.
- A resume validates every epoch file, rejects stale/corrupt configurations and histories with gaps, restores optimizer and RNG state, then starts at the next epoch. An interruption inside an epoch leaves the previous complete boundary intact; that incomplete epoch is replayed. Completed fold checkpoints remain the campaign-level result boundary.
- The campaign output policy accepts epoch checkpoints for resume. An empty output directory remains valid for the first guarded invocation.
- Existing memory gates remain: startup requires at least 2,048 MiB and the training path retains the 1,024 MiB floor. Neither threshold was lowered.

## Checks and blockers

- Pure-Python checkpoint, recovery, memory-gate, sampling, and inference-call regression suite: **18 passed**.
- `py_compile` and `--validate-config` passed after the recovery changes.
- The guarded `--resume` invocation stopped before data/model imports with `Only 649 MiB available; estimated peak requires 2048 MiB to preserve the 1024 MiB floor`. No training was started by that invocation.
- Available RAM varied during this audit: 955.4 MiB at initial inspection, 558.8 MiB during a later check, 608.6 MiB at process inventory, and 649 MiB at the campaign preflight. The accessible process inventory contained only small shell/utility Python processes (roughly 75 MiB PowerShell, 18 MiB Python, 7 MiB conhost, and 4 MiB project Python); it did not expose the foreground VS Code process, so the system-wide cause of the RAM drop could not be identified.
- PennyLane (0.45.1) and scikit-learn (1.9.1) are installed according to package metadata. No PennyLane import or circuit execution was attempted because the measured available RAM did not pass preflight. Real forward values, gradients, parameter updates, optimizer restoration through a live QNN, and campaign metrics remain unverified.

## Continuation

The inspected partial run has no recoverable epoch state, so it must start its first fold from initialization once memory passes preflight. Run:

```powershell
.\.venv\Scripts\python.exe -m qmutbench_qml.training.run_binary_development --output reports/experiments/qmutbench_binary_qnn_development --resume
```

After a completed epoch is saved, the same command resumes from the latest validated epoch checkpoint. It does not treat the user-reported four earlier epochs as trained state.
