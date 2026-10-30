# Project synchronization report

Date: 2026-10-26

## Repository synchronization

- Repository: `C:/Users/pcm/OneDrive/Desktop/Quantum_Simulator`
- Branch: `qmutbench-master-optimization`
- Common base before synchronization: `96bfbaa7863b4cc4a0badb34858fed0a91aee060`
- Pre-merge local HEAD: `a0920931c369a7e197be79ed496512fcd1c44eaa`
- Fetched remote branch HEAD: `c277ee0afb8e9dae7d87116ebf18955721a55e5a`
- Synchronization: **succeeded**. A normal merge retained the eight local commits and incorporated all ten remote commits after the common base, with no conflicts. Remote HEAD is an ancestor of the resulting local HEAD.
- Current commit: `869aa55034d5016eb4bb4661f5390a7ff38cb8b7` (merge commit).
- Remote: `https://github.com/ruckscode/Quantum_Simulator.git` (fetch and push).
- The requested commits were inspected in the fetched history and incorporated:
  - `c7eeac7e542f1bf9dc6544029a8a06f8779fd3f9` adds checksum, metadata, feature-count, version, and inference-loader checks to the Colab report viewer.
  - `c277ee0afb8e9dae7d87116ebf18955721a55e5a` updates demo bundle-availability messaging and uses the example mutant gate `x`.
- The remote additionally contains eight other commits after the common base; these were incorporated with the two named commits.

## Preserved local state

- Before synchronization the tracked worktree was clean; there were no untracked non-ignored files.
- Eight local-only commits were preserved, including test-fixture/isolation changes and `.gitattributes` work. The merge changed no existing local files; it added the remote demo/viewer files.
- Numerous ignored generated artifacts and caches were present, including `.venv`, QMutBench source data, model bundle files, pytest work directories, and experiment outputs. They were not cleaned or overwritten. The model and metadata are ignored local artifacts, not committed files.
- No unresolved conflicts remain. After creating this report, the only expected worktree change is this new report.

## QMutBench data and model verification

- The committed research report `reports/experiments/qmutbench_final_research_report_20261026.json` records 723,079 metadata rows and 375 original-program groups. It records 578,117 development rows across 305 origins and 144,962 historical frozen-test rows across the remaining 70 origins. The local raw metadata CSV exists at `data/qmutbench_ingestion/raw/official_all_data_20261015.csv` (59,913,911 bytes); the report records SHA-256 `3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778`.
- The model is `qmutbench-mutation-aware-v1`, an Extra Trees classifier with 100 trees and a 23-feature input schema. It consumes QMutBench metadata plus original and mutant OpenQASM 2.0 strings to derive mutation-aware features. The target map is WOO/OPO = detected (1), Survivor = undetected (0).
- Local bundle files exist at `data/qmutbench_ml/mutation_aware_v1/best_model.joblib` and `best_model_metadata.json`. The model SHA-256 is `7532be26b6959e7e62277220f0c0b78e648d723a0be35d2b693d68e1c9ebf085`, matching both the metadata and committed report. The inference loader checks integrity/schema and supports prediction; the focused inference suite passed.
- The report records 78.89% mutation-aware macro F1 on saved three-fold grouped OOF and 79.28% macro F1 on five-fold nested grouped development evaluation. These are separate development results, not frozen-test performance.
- The historical 73.49% frozen-test accuracy is recorded in the legacy outcome-ML report for the selected-v3 model. The 2026-10-26 report explicitly did not load frozen-test labels or circuits; therefore 73.49% is not a verified frozen-test score for the current mutation-aware candidate.
- The 85% target remains unmet. The model predicts historical mutation outcomes; it is not a hardware-noise detector or fault-localization model.
- The 2026-10-26 research report's baseline/feature audit is marked `verified`. The available local evidence supports the counts, split, feature contract, checksum, and reported development metrics above; no compatible independent external validation is recorded.

## Simulator and test verification

- The simulator is a statevector implementation with measurement/reset, traced execution, and hardware-model support. README documentation lists single-qubit gates/rotations, `cx`, `cz`, `swap`, `ccx`, reset, barriers, and measurement. Its stochastic noise behavior covers readout flips, probabilistic gate skipping, T1 amplitude-damping jumps, and remaining T2 dephasing; it is not a general Kraus/depolarizing or real-device model.
- There is no fixed universal simulator qubit ceiling. The code offers an opt-in memory budget and estimates three statevector-sized complex128 arrays, about `48 * 2^n` bytes, excluding process overhead.
- The QMutBench QASM mutation parser handles its declared gate/parameter arities and rejects unsupported gates explicitly; measurement and barrier instructions are excluded from mutation gate/wire features.
- Ran with the project virtual environment: `python -m pytest -q tests/test_qmutbench_inference.py tests/test_qmutbench_qasm_features.py tests/test_simulator.py` — **75 passed in 6.45s**.
- Running the same focused command under the system Python failed during test collection because `psutil` is missing there. The project `.venv` has the needed dependencies and passed. No system-wide packages were installed. Full-dataset retraining and the full repository test suite were not run.

## Starting point for the separate QML experiment

Use branch `qmutbench-master-optimization` at commit `869aa55034d5016eb4bb4661f5390a7ff38cb8b7`, which contains both the eight preserved local commits and the fetched remote branch through `c277ee0afb8e9dae7d87116ebf18955721a55e5a`. The tracked worktree was clean before this report was added; this synchronization report is the sole intended new tracked file. Local ignored research artifacts remain present and untouched. No QML development or training was started.
