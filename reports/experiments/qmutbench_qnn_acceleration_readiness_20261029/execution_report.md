# QMutBench QNN acceleration readiness

**Assessment:** BLOCKED — existing CPU workflow remains intact; no accelerator path or CPU implementation change was safe to apply and validate in this session.

**Scope:** inspected the existing full-development command and run state only. No training, checkpoint writes, installs, branch changes, staging, commits, or frozen-test access occurred.

## Preservation and repository state

- Recovery inventory: `reports/recovery/20261029_qmutbench_preservation`. Its 65-row artifact manifest had no mismatches when checked: all 56 copied run files matched their originals, and all recorded source fingerprints matched.
- Active run: 51 checkpoint files, 51 committed chunks, 816 Adam updates, 26,112 processed rows; fold 1, epoch 0, next batch 816, next chunk 52; no completed folds. Checkpoint payloads and file hashes passed the read-only audit. Frozen-test access is false.
- Branch `qmutbench-master-optimization`, HEAD `a2d51be22fd985bbdd10655f7e31dc77b9d19ed6`; 31 tracked files modified, zero staged. Existing changes and untracked work were left as found. This report is the only new project artifact.

## Execution path and workload

The command enters `qmutbench_qml.training.run_binary_development`, then calls `run_full_streaming_campaign`, which reads indexed development data through SQLite, uses bounded feature chunks and temporary NumPy memmaps, and invokes `BinaryVQC.fit`.

The QNN uses PennyLane 0.45.1 `default.qubit`, `interface="autograd"`, `diff_method="backprop"`, and `pennylane.numpy`. Its 4-wire, depth-2 circuit applies angle embedding, per-wire Rot gates, a CNOT ring, and a Pauli-Z expectation. The loss is a sigmoid binary cross-entropy with class weights; PennyLane Adam updates the circuit parameters and readout. The fit loop creates three batch tensors and evaluates the circuit once per example (32 QNode executions per full batch). This path is CPU-only; it does not import or place tensors on a GPU.

The recorded training run is the available CPU baseline: 837.308 seconds for 816 updates (1.026 seconds/update), a maximum checkpointed process RSS of 468.0 MiB, and minimum available system RAM of 1,747.96 MiB. The audit extrapolates about 164.8 hours of QNN optimizer time for 578,136 updates. This is not a repeat benchmark and excludes stream preparation and fold evaluation.

## Hardware and accelerator compatibility

- Windows 10 Home, build 26300; AMD Ryzen 5 5625U with Radeon Graphics; 7,488 MiB total physical memory.
- Display device as reported by Windows: AMD Radeon (TM) Graphics, PCI `VEN_1002&DEV_15E7`, HP subsystem `8A1B103C`. Installed display driver: `30.0.13036.2`, dated 2021-11-09. Windows reports the generic Radeon name; no separate discrete GPU was found in the display-device listing. The approximately 496 MiB dedicated and shared GPU memory values are from the user's Windows GPU panel.
- Python 3.12.10; PennyLane 0.45.1; NumPy 2.5.3; pandas 3.0.6; scikit-learn 1.9.1; Autograd 1.8.0; psutil 7.2.2. Torch, `torch-directml`, JAX, TensorFlow, and PennyLane Lightning plugins are absent from `.venv`.
- The only enumerated QNN simulator is the existing `default.qubit` path. PennyLane's documented GPU path for its Torch interface is CUDA. Its docs also caution that GPU overhead is likely to outweigh benefits below 15 wires; this model has 4 wires. Lightning-GPU requires NVIDIA CUDA/cuQuantum, so it does not match this AMD iGPU.
- DirectML is the plausible Windows route for a DX12 GPU, but it is a PyTorch backend, while this QNN uses Autograd tensors and the existing PennyLane CPU simulator. A Torch/DirectML port would require a new device-placement and complex-statevector/operator validation path. No DirectML runtime or package is installed, and the small integrated GPU shares the same constrained system RAM. I did not install the preview package or claim operator compatibility without running the QNN's complete gate, complex arithmetic, expectation, gradient, loss, and Adam path on the device.
- AMD's current Windows ROCm/PyTorch support matrix lists select Radeon 7000/9000 GPUs and not this Ryzen 5 5625U iGPU; current Windows support is Windows 11. ROCm is therefore not a supported path for this machine. (The display driver is also dated 2021.)

References: [Microsoft: PyTorch with DirectML on Windows](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows), [PennyLane: PyTorch interface and CUDA support](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html), [PennyLane Lightning-GPU device requirements](https://docs.pennylane.ai/projects/lightning/en/latest/lightning_gpu/device.html), [AMD: Windows compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html).

## CPU optimization decision and benchmark limits

The primary runtime hotspot is repeated per-row QNode execution and Autograd reverse-mode simulation. Batching/vectorizing that circuit or changing the model implementation could reduce Python overhead, but `binary_vqc.py` is included in each checkpoint's source fingerprint. A direct edit would change the run fingerprint and make the current 51 checkpoints fail the strict resume-configuration check. A safe opt-in or migration path would need numerical/gradient equivalence tests, deterministic optimizer/RNG continuity tests, checkpoint compatibility coverage, and a matched repeat benchmark before the existing run could use it. No such change was made without being able to complete those validations.

The full-development memory estimate is 2,076 MiB: 1,024 MiB in-run floor + 700 MiB runtime reserve + 350 MiB bounded streaming/preprocessing allowance + rounded 1 MiB circuit-state estimate + rounded 1 MiB optimizer estimate. Available system RAM at the latest read was 1,969 MiB, so the normal preflight would still block (107 MiB below its estimate). The existing 1,024 MiB in-run floor was not changed. Current availability is variable; the earlier test run began below the 2,048 MiB startup threshold.

A GPU-vs-CPU benchmark was not run because there is no compatible installed GPU backend. No post-change CPU benchmark exists because no code optimization was applied. The only measured CPU baseline is the previously recorded 816-update run above. No full 578,117-row run was launched.

## Tests

Command: `.venv\Scripts\python.exe -m pytest tests/test_qmutbench_binary_qnn.py tests/test_qmutbench_streaming.py -q`

Result: **35 passed, 1 skipped, 0 failed** in 4.72 seconds. The skipped real PennyLane chunk-resume/gradient/update smoke test requires the unchanged 2,048 MiB startup reserve. The passing tests cover checkpoint integrity/resume helpers, streaming fixtures, and mocked memory-guard behavior; they do not establish a post-optimization benchmark or GPU compatibility.

## Resume requirements and next action

The existing run remains checkpoint-compatible with its unchanged trainer, model, preprocessing sources, Python 3.12.10 environment, optimizer state, RNG state, and batch order. Do not resume while the preflight blocks. Retry only when available RAM is stably at or above 2,076 MiB and remains at least 1,024 MiB after imports and throughout training; close memory-heavy applications first, then rely on the normal preflight and in-run guards.

Exact bounded command from the repository root, when those conditions are met:

```powershell
.venv\Scripts\python.exe -m qmutbench_qml.training.run_binary_development --mode full-development --output reports/experiments/qmutbench_binary_qnn_full_development_chunk16_20261029 --chunk-batches 16 --max-chunks-per-invocation 50 --resume
```

No speed or memory improvement is claimed. Remaining blockers are the unvalidated DirectML/PennyLane operator path, the checkpoint fingerprint constraint on CPU hot-path edits, and current memory below the normal startup estimate.
