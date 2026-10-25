# Autonomous stabilization journal

## 2026-10-14 â€” P0-1 measurement and reset semantics â€” complete

**Requirement.** Physical measurement collapse must follow the sampled quantum outcome even when the classical readout is flipped. Reset must implement the reset channel for arbitrary single-qubit states and entangled inputs under a declared statevector model.

**Inspection.** Read `reports/experiments/comprehensive_project_audit.md`, `comprehensive_project_inventory.json`, and `stabilization_work_plan.json`. Confirmed no autonomous status or journal existed before creation. The canonical implementation in `src/quantum_simulator/quantum_simulator.py` was already modified in the user's working tree; unrelated prior changes were retained. Existing tests already asserted that a forced readout flip leaves the quantum state unchanged.

**Confirmed defects and changes.** `measure` previously replaced physical selected bits with flipped bits before collapsing. It now keeps separate physical and reported outcomes, using the former for collapse and the latter for results/classical bits. `reset_qubit` previously discarded the |1> branch and renormalized the |0> branch (postselection). It now samples the reset channel's `|0><0|` / `|0><1|` Kraus trajectory, maps the target to |0>, and preserves the conditional state of the remaining system. The docstring states that entangled mixed output is represented across seeded pure-state trajectories.

**Independent checks.** Added `tests/test_measurement_reset_semantics.py`. A same-seed ideal measurement identifies the physical result; a forced readout error reports its complement while the state collapses to the physical basis state. Basis and |+> reset cases yield |0>. A Bell pair reset on q0 yields only q0=0 per trajectory, and the average density matrix over 1,024 seeds matches the analytical `|0><0| tensor I/2` result within absolute tolerance 0.06.

**Executed validation.** `.venv/Scripts/python.exe -m pytest tests/test_measurement_reset_semantics.py tests/test_simulator.py -q --basetemp .pytest-tmp-autostab-p0-1-rerun` exited 0: **37 passed in 1.54s**. The first run used 256 trajectories and failed the preset absolute tolerance 0.08 by 0.00594 on one population; the circuit/trajectory invariants passed. Increased the bounded ensemble to 1,024 and reran successfully with tolerance 0.06. `git diff --check -- src/quantum_simulator/quantum_simulator.py` found no whitespace errors; Git emitted its LF-to-CRLF normalization warning.

**Scope and checkpoint.** No dependencies installed, no large jobs or external resources used. This completes P0-1 only; no full simulator or release readiness claim is made. The next eligible phase is P0-2, independently checking gate matrices, operand ordering, and state invariants. P0-3 noise semantics depends on P0-2.

## 2026-10-14 â€” P0-2 gate oracle and invariants â€” complete

**Requirement.** Check supported matrices and local statevector execution for documented operand order, basis indexing, and unitary state transitions against an independent reference.

**Reference availability.** The existing workspace `.venv` already contains Qiskit 2.5.2 and Aer; no package was installed. Added `tests/test_gate_reference_qiskit.py`, which builds circuits with Qiskit's independent named gate methods and `Statevector`, maps the simulator's big-endian logical bit order onto Qiskit qubits, and compares all basis columns. All one-qubit gate families and parameterized families are covered; every two-qubit gate is checked in both operand orders; all supported three-qubit gates are checked over all basis states. Existing gate tests check shapes/unitarity and analytical gate effects.

**Confirmed defects and changes.** The first oracle run exposed two source issues plus one malformed test case. The test had passed three parameters to `u2` rather than its two API parameters; after correcting that case, the remaining mismatch confirmed the simulator's `U2(phi, lambda)` formula was nonstandard. Replaced it with the standard `U(pi/2, phi, lambda)` matrix. Qiskit also showed the local `rccx` lacked the standard Â±i phases in its controlled target-swap subspace; updated the matrix and corresponding existing phase expectation. No other supported gate mismatch remained.

**Executed validation.** Final command `.venv/Scripts/python.exe -m pytest tests/test_gate_reference_qiskit.py tests/test_gate_correctness.py tests/test_gates.py -q --basetemp .pytest-tmp-autostab-p0-2-final` exited 0: **72 passed in 7.10s**, exact statevector absolute tolerance 1e-12. Re-ran `.venv/Scripts/python.exe -m pytest tests/test_measurement_reset_semantics.py tests/test_simulator.py -q --basetemp .pytest-tmp-autostab-p0-1-final`: **37 passed in 1.54s**. `git diff --check` found no whitespace errors in affected tracked source/tests; it reported Git's existing LF-to-CRLF normalization warnings.

**Scope and checkpoint.** The gate oracle covers unitary gate basis transitions and all local columns, not arbitrary randomized circuit families or noise. P0-1 and P0-2 are complete; P0-3 noise channel validation is now in progress. The local hardware model remains a simplified stochastic model until its channel semantics are analytically checked.

## 2026-10-14 â€” P0-3 noise channel contract â€” complete

**Requirement.** Independently check reported readout flips, gate calibration error behavior, and the implemented T1/T2 trajectory semantics against exact or analytical channel expectations.

**Inspection and scope.** Reviewed `HardwareModel`, `_apply_hardware_relaxation`, existing `tests/test_noise_model_validation.py`, and README claims. The implementation supports symmetric classical readout flips, Bernoulli gate skips, T1 amplitude damping trajectories, and T2 residual phase-flip dephasing. It does not implement depolarizing or arbitrary Kraus noise. Updated README to state those concrete semantics and limits; no broader physical-device claim is made.

**Independent checks.** Added `tests/test_noise_channel_reference.py`. The readout test samples 2,000 measurements of |0> at error probability 0.2 and checks a seeded four-standard-error interval while confirming the physical state stays |0>. The T1 test constructs amplitude-damping Kraus matrices independently, then compares their exact density-matrix result for |+> against the ensemble of 4,096 seeded simulator trajectories. Existing focused tests check the T1 endpoint, the T2 coherence factor via H-I-H, and gate-skip endpoint behavior.

**Executed validation.** `.venv/Scripts/python.exe -m pytest tests/test_noise_channel_reference.py tests/test_noise_model_validation.py -q --basetemp .pytest-tmp-autostab-p0-3` exited 0: **7 passed in 1.70s**.

**Limitations and checkpoint.** This validates representative declared stochastic semantics, not calibration accuracy on hardware or every malformed calibration combination. P0-3 is complete within the documented limited-model contract. Next is P0-4 finite-shot anomaly calibration: inspect the saved NO-GO evidence and protocol only; do not tune thresholds or rerun a large campaign.

## 2026-10-14 â€” P0-4 finite-shot calibration â€” blocked by evidence

**Inspection.** Read `finite_shot_blocker_status.json`, `finite_shot_distribution_calibration_diagnostic.md`, the protocol-gap register, and the current confidence-bound helper/tests. The saved diagnostic reproduces 60 registered decisions with 30 pass and 30 fail. The current TVD detector has 100% false positives on negative controls at 256 shots and about 88â€“92% at 1,024 shots; multiple detector/stratum cells still fail at 4,096 shots. The proposed 252-family experiment remains NO-GO with execution unauthorized because identifiability, independent ground truth, family sampling, endpoints, and novelty are unresolved.

**Bounded current-tree check.** The old diagnostic notes a one-sided lower-bound defect in its then-current helper. The present `_cp_lower` implementation and its current regression tests are already corrected: `tests/test_finite_shot_confidence_bounds.py` exited 0 (**4 passed in 0.53s**), and six representative values match `scipy.stats.beta.ppf(0.05, k, n-k+1)` to floating-point precision. The old summary still contains the stale field; the diagnostic's recomputed table is the valid reference. No historical artifact was overwritten.

**Disposition.** No threshold, criterion, registry, or calibration artifact was changed. P0-4 cannot pass from the available evidence; dependent diagnosis validation is held. The separate 252-family run was not started. Continue with independent benchmark truth/provenance review and other safe work.

## 2026-10-14 â€” P1-3 benchmark truth provenance â€” partial

**Static manifest.** Created `reports/experiments/benchmark_provenance_static_audit_20261014.json` without running benchmark cases. It maps 12 QMutBench cases and 42 Bugs4Q source cases to 150 existing source files with SHA-256 hashes; all listed source files exist. It also hashes the result snapshots and current adapter sources, and records support/translation/execution outcomes, truth status, and checkpoint metadata availability.

**Findings.** The QMutBench pilot report has 11/12 supported/executed, while the separate full-results snapshot has 12/12; the RXX case is marked unsupported in the pilot and executable in full results. Both report zero diagnosis-category-eligible labels. Bugs4Q's generic validation report found no dataset cases, while the official-source integration snapshot covers 42 cases (41 parsed, 19 translated/executable/diagnosed, 22 unsupported) and has zero diagnosis-category labels. The QMutBench checkpoint is a bare array without schema/runner/source/runtime fingerprints. Current file hashes cannot prove which source revision generated historical results.

**Disposition.** The static review satisfies the safe manifest action but does not resolve snapshot provenance or establish independent labels. Benchmark accuracy, fault attribution, and localization remain unsupported. Full phase closure depends on diagnosis-identifiability evidence, currently blocked by P0-4. No existing benchmark report, dataset, or checkpoint was modified.

## 2026-10-14 â€” P2-1 resource envelope â€” partial

**Measurement.** The existing environment has `psutil`; WMI system-memory query was denied, so available memory was read via psutil. Created `reports/experiments/resource_envelope_smoke_20261014.json` using one H and one CX at 1, 4, 8, 12, and 14 qubits in Python 3.12.10 on Windows. RSS was sampled during the run; baseline was about 33.0 MiB and sampled peak about 35.9 MiB. The 14-qubit case used a 262,144-byte statevector, held at least 524,288 bytes for the input/output vectors during local gate application, took 0.189 seconds, and had norm error 2.22e-16. psutil reported about 793 MB available memory before the profile.

**Disposition.** This is a small smoke workload, not a reliable safe-width ceiling. No machine-wide cap or allocation guard was inferred from it. README now states the exponential memory formula and lack of a universal guard. P2-1 remains partial pending a defensible workload ceiling/preflight design. Next: statically audit saved ML split, leakage, and calibration evidence; do not train or process the full QMutBench corpus.

## 2026-10-14 â€” P1-2 ML integrity and generalization â€” partial

**Data and split evidence.** Read the grouped-CV audit and manifest. The controlled dataset contains 270 rows, 30 circuit families and 3 classes; the saved 5-fold `StratifiedGroupKFold` has no family overlap. The 270 fold assignments are unique, and 4,050 saved OOF rows cover the 15 fixed feature/model configurations exactly once per run/configuration. The current data fingerprint matches the saved manifest. Two source hashes do not match the current working tree: `src/quantum_simulator/quantum_simulator.py` and `src/quantum_simulator/diagnosis_engine.py`. Those evaluation results predate this repair batch and cannot support claims about current implementation behavior. Existing signal-quality audit documents cross-label duplicate feature vectors and a `gate_mismatch_fraction` construction proxy; calibration fields also encode synthetic hardware profiles. The five production observables exclude direct labels/metadata, but the data remain controlled simulator scenarios rather than independently adjudicated faults.

**Artifact state and confirmed issue.** `src/quantum_simulator/models/observable_logistic_regression.json` exists and was created 2026-10-13, after older integration reports that stated no artifact existed. It contains a 330-row F1/F2-only five-feature fit, marks probabilities uncalibrated, and has no held-out-family deployment validation. The runtime diagnosis still defaults to `UnavailableObservableClassifier`; callers must explicitly inject an adapter. The Phase 5 writer nevertheless labeled an F1/F2 development-only fit `FINAL DEPLOYABLE MODEL`, and a test asserted the default artifact must be absent. Updated future exports to say `DEVELOPMENT CANDIDATE; NOT VALIDATED FOR DEPLOYMENT`, record validation as `NOT_PERFORMED`, and expose `deployment_validated` only when a passing validation marker exists. Updated the stale absent-artifact test to validate the existing model if present. The existing JSON model artifact and historical reports were not overwritten.

**Executed checks.** `.venv/Scripts/python.exe -m pytest tests/test_ml_evaluation_audit.py tests/test_ml_training_artifacts.py tests/test_observable_model_adapter.py tests/test_phase5_model.py -q --basetemp .pytest-tmp-autostab-p1-2-rerun` exited 0: **35 passed in 7.52s**. The trainer test used six synthetic rows in a temporary directory only; no production training, full dataset processing, or benchmark run occurred. Current environment can load the existing model artifact through the explicit portable adapter, but this proves schema/inference mechanics only, not calibration or generalization.

**Disposition.** ML validity remains partial/blocked. Do not interpret the 0.60 cutoff as calibrated confidence. No post-core-fix evaluation of synthetic training artifacts or held-out-family performance has been run. Continue with current-workspace deterministic replay and dependency/version recording; a pristine checkout remains separate work.

## 2026-10-14 â€” P2-2 Windows reproducibility â€” partial

**Executed validation.** The cohesive core regression command `.venv/Scripts/python.exe -m pytest tests/test_gate_reference_qiskit.py tests/test_gate_correctness.py tests/test_gates.py tests/test_simulator.py tests/test_measurement_reset_semantics.py tests/test_noise_model_validation.py tests/test_noise_channel_reference.py tests/test_local_package_import.py -q --basetemp .pytest-tmp-autostab-final-core` exited 0: **120 passed in 12.34s**. This includes same-seed replay and distinct-seed sampling checks. Runtime versions are Python 3.12.10, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, pytest 9.1.1, Qiskit 2.5.2, Qiskit Aer 0.17.2, and psutil 7.2.2. No package installation or full-suite run occurred.

**Limitation.** A clean commit-level run was not established because the shared workspace contains extensive pre-existing uncommitted user changes. P2-2 remains partial; P2-1 also lacks a universal pre-allocation resource guard.

## 2026-10-14 â€” P3-2 documentation and claims â€” in progress

**Changes so far.** README now records the limited noise-channel contract, exponential statevector memory, current 120-test core outcome, finite-shot detector blocker, and actual Phase 5 model-artifact status (an uncalibrated development candidate not loaded by default). Historical reports remain untouched. Next reconcile current documentation/status claims with the release-gate checkpoint without claiming overall readiness.

## 2026-10-14 — P2-2 Windows reproducibility — partial

**Executed validation.** The cohesive core regression command `.venv/Scripts/python.exe -m pytest tests/test_gate_reference_qiskit.py tests/test_gate_correctness.py tests/test_gates.py tests/test_simulator.py tests/test_measurement_reset_semantics.py tests/test_noise_model_validation.py tests/test_noise_channel_reference.py tests/test_local_package_import.py -q --basetemp .pytest-tmp-autostab-final-core` exited 0: **120 passed in 12.34s**. Runtime versions are Python 3.12.10, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, pytest 9.1.1, Qiskit 2.5.2, Qiskit Aer 0.17.2, and psutil 7.2.2. No package installation or full-suite run occurred.

**Limitation.** A clean commit-level run was not established because the shared workspace contains extensive pre-existing uncommitted user changes. P2-2 remains partial; P2-1 also lacks a universal pre-allocation resource guard.

## 2026-10-14 — P3-2 documentation and claims — partial

README now records the limited noise-channel contract, exponential statevector memory formula, observed core test result, finite-shot detector blocker, and actual Phase 5 model-artifact status (an uncalibrated development candidate not loaded by default). Historical reports remain untouched. Current public claims were reconciled to available evidence. Overall readiness is not claimed because diagnosis/finite-shot truth and benchmark provenance remain unresolved.

## 2026-10-14 — P0-5 randomized reference and input validation — complete

**Independent differential validation.** Added a fixed-seed test that composes 32 circuits (8 per width, 1–4 qubits, depth 24) from parameterized one-qubit gates, asymmetric controlled gates, and 3-qubit gates. Qiskit `Statevector.from_instruction` is used as the independent oracle, with logical-to-physical bit reversal explicit. `.venv/Scripts/python.exe -m pytest tests/test_gate_reference_qiskit.py -q --basetemp .pytest-tmp-autostab-random-circuits` passed: **33 passed in 3.22s**.

**Confirmed input defects and repairs.** The canonical simulator used to coerce floats, numeric strings and booleans into integer qubit indices and did not reject repeated operands in its low-level multi-qubit path. Constructor and operand validation now reject booleans/lossy coercions, accept integral NumPy scalar indices, and reject duplicate operands before state mutation. Added regressions for all cases.

**Affected regression.** `.venv/Scripts/python.exe -m pytest tests/test_gate_reference_qiskit.py tests/test_gate_correctness.py tests/test_gates.py tests/test_simulator.py tests/test_measurement_reset_semantics.py tests/test_noise_model_validation.py tests/test_noise_channel_reference.py tests/test_local_package_import.py -q --basetemp .pytest-tmp-autostab-latest-core` passed: **129 passed in 19.72s**.

**Disposition.** Randomized small-circuit state evolution now has direct external-reference coverage, and malformed qubit operands fail early. This does not close full randomized/property testing, input validation for every public API, resource preflight, diagnosis, or final release gates. Existing unrelated workspace edits remain preserved.

## 2026-10-14 — P0-6 unitary and finite-parameter guards — complete

**Confirmed defects.** Parameterized gate factories accepted NaN/infinity, and the public custom-matrix gate path accepted non-finite, malformed-rank, or non-unitary matrices. These could silently corrupt normalization or propagate invalid values. The simulator now rejects non-finite parameters and validates custom gate matrix rank, dimensions, finiteness, and unitarity before applying it.

**Regression evidence.** Added tests for NaN, ±infinity, non-unitary matrices, NaN matrix entries, and incorrect matrix rank; all assert that rejection leaves the original state intact. The cohesive command `.venv/Scripts/python.exe -m pytest tests/test_gate_reference_qiskit.py tests/test_gate_correctness.py tests/test_gates.py tests/test_simulator.py tests/test_measurement_reset_semantics.py tests/test_noise_model_validation.py tests/test_noise_channel_reference.py tests/test_local_package_import.py -q --basetemp .pytest-tmp-autostab-final-core-2` passed: **135 passed in 11.25s**.

**Disposition.** P0-6 is complete for current gate entry points. Full suite, resource guard, independent diagnosis truth, ML generalization, and benchmark provenance remain unresolved or blocked; current worktree is not release-ready.

## 2026-10-14 — P0-7 shot replay and count validation — complete

**Confirmed defects.** `run_shots` accepted non-integral/boolean shot counts until a later runtime failure, and it reused the caller-provided iterable for every shot. A generator was consumed after shot one, so later repetitions could silently execute an empty circuit. It now requires a positive integral count and materializes the instruction iterable once before replay.

**Validation.** Added tests for zero, negative, fractional and boolean shot counts, plus fixed-seed equivalence between a list circuit and a generator circuit. `tests/test_simulator.py` passed **53 tests in 2.71s**. The cohesive core command `.venv/Scripts/python.exe -m pytest tests/test_gate_reference_qiskit.py tests/test_gate_correctness.py tests/test_gates.py tests/test_simulator.py tests/test_measurement_reset_semantics.py tests/test_noise_model_validation.py tests/test_noise_channel_reference.py tests/test_local_package_import.py -q --basetemp .pytest-tmp-autostab-final-core-3` passed **140 tests in 11.07s**.

**Disposition.** Seeded shot replay is consistent for iterable and generator inputs. This does not verify durable artifact recovery or the entire experiment pipeline. The full suite and release gates remain open/blocked as recorded.

## 2026-10-14 — P2-3 atomic experiment artifact writes — complete

**Confirmed reliability issue.** `experiments/run_experiment.py` wrote result and reproducibility JSON directly to final destinations. Interruption or a Windows/OneDrive file lock during a write could leave a truncated artifact. Added `_atomic_write_text`: it writes and fsyncs a temporary sibling, then atomically replaces the destination; replacement failure cleans the temporary file and leaves an existing destination intact. Both experiment result and batch reproducibility report writers use it.

**Validation.** Added direct tests for complete JSON replacement and simulated `PermissionError` during replacement. The latter confirms prior destination contents remain intact and no temporary sibling remains. `.venv/Scripts/python.exe -m pytest tests/test_experiment_runner.py -q --basetemp .pytest-tmp-autostab-atomic-writes` passed: **11 passed in 2.12s**.

**Disposition.** The tested local filesystem semantics are robust to a replacement failure. This does not prove OneDrive cloud synchronization durability or multi-process coordination; no concurrency lock was added. Existing reports were not touched by these tests.

**Integrated checkpoint.** After P2-3, the combined targeted regression covering the 140-test core set plus experiment-runner artifact tests passed **151 tests in 14.15s**: `.venv/Scripts/python.exe -m pytest tests/test_gate_reference_qiskit.py tests/test_gate_correctness.py tests/test_gates.py tests/test_simulator.py tests/test_measurement_reset_semantics.py tests/test_noise_model_validation.py tests/test_noise_channel_reference.py tests/test_local_package_import.py tests/test_experiment_runner.py -q --basetemp .pytest-tmp-autostab-integrated-final`.

## 2026-10-14 — P2-4 source fingerprints in reproducibility reports — complete

**Gap and change.** Existing batch reproducibility reports captured Python, NumPy and platform versions but did not identify the source revision. New reports now include SHA-256 fingerprints for `pyproject.toml`, the experiment runner, and the canonical simulator, gate, diagnosis, and hardware-model source files. Historical reports were not changed.

**Validation.** Added a batch-run test that verifies deterministic repeated output, recomputes the simulator source hash independently, and checks the saved report has the same fingerprint map. `tests/test_experiment_runner.py` passed **12 tests in 2.68s**. The integrated core + experiment-runner command passed **152 tests in 14.49s**.

**Disposition.** Future local reports identify the exact key implementation files and environment versions. This is not a clean checkout/install reproduction, and it does not fingerprint datasets, optional dependencies, or the entire repository tree.

## 2026-10-14 — P2-5 benchmark checkpoint interruption safety — complete

**Confirmed recovery risk.** The QMutBench resume checkpoint had schema/source compatibility checks but wrote directly to the final JSON path. A failed write could truncate the only checkpoint and force recomputation. `_write_qmut_checkpoint` now writes and fsyncs a temporary sibling before atomic replacement; a failed replacement preserves the prior checkpoint and cleans up the temporary file.

**Validation.** Added fixture-only tests for complete checkpoint persistence and simulated Windows replacement failure. `.venv/Scripts/python.exe -m pytest tests/test_full_benchmark_runner_checkpoint.py -q --basetemp .pytest-tmp-autostab-checkpoint-atomic` passed **6 tests in 0.44s**. The integrated simulator, experiment-runner, and checkpoint regression passed **158 tests in 15.00s**. No benchmark dataset or case was processed.

**Disposition.** Local checkpoint replacement recovery is tested. This does not establish end-to-end interruption recovery during a full benchmark run, and the full dataset was not run.

## 2026-10-14 — P2-6 benchmark report write safety — complete

**Change.** Extended the benchmark runner's atomic writer to its per-benchmark JSON, CSV, and combined JSON artifacts. The resumable checkpoint remains the recovery source; final report files are individually replaced atomically. Multi-file transaction semantics are not claimed.

**Validation.** Added a fixture-only test for JSON/CSV report writing and temporary cleanup. `.venv/Scripts/python.exe -m pytest tests/test_full_benchmark_runner_checkpoint.py -q --basetemp .pytest-tmp-autostab-benchmark-report-writes` passed **7 tests in 0.38s**. The integrated simulator, experiment runner, and benchmark writer/checkpoint suite passed **159 tests in 10.85s**. No dataset discovery or benchmark case execution occurred in these tests.

**Disposition.** Each report file is protected from partial direct writes; if interrupted between artifacts, the checkpoint is the source for regeneration. End-to-end process interruption and large-run memory/disk behavior remain untested.

## 2026-10-14 — P2-7 unsupported and malformed operation disposition — verified

**Inspection.** The execution adapters return structured unsupported/execution-failure rows and do not pass failed cases into diagnosis. Existing fixture tests cover unsupported and malformed QMutBench cases separately from broad dataset execution.

**Validation.** The narrow QMutBench test `test_unsupported_and_malformed_cases_are_not_forced_into_diagnosis` passed **1 test in 0.21s**. Experiment runner unsupported/malformed-result tests plus QMutBench execution-failure handling passed **3 tests in 0.31s**. The broad 12-case and full dataset runs were not invoked.

**Disposition.** No additional code change was justified. Unsupported operations remain explicit and are not scored as diagnosis successes; full adapter coverage and larger regressions remain separate.

## 2026-10-14 — P2-1 opt-in resource preflight — partial

**Design decision.** The saved profile measures only through 14 qubits and reports roughly 793 MB available at that time; this does not justify a default safe ceiling. Added `QuantumSimulator.estimate_working_array_bytes(n)` using three complex128 statevector-sized blocks (48 bytes per amplitude) to model the principal state/output/probability arrays. `QuantumSimulator` accepts optional `memory_budget_bytes` and rejects an over-budget or platform-unaddressable statevector before allocating. `run_shots` propagates the configured budget to each trajectory, and `QuantumCircuit.run` exposes it. Default behavior remains uncapped because no portable machine budget is established. Documentation explicitly says the estimate excludes Python/NumPy/runtime overhead and is not a total-process memory guarantee.

**Validation.** Tests cover exact estimate, budget boundary, invalid budgets, pre-allocation over-budget rejection, addressability rejection, shot propagation, and circuit API propagation. Simulator and circuit tests passed **106 tests in 4.05s**. The integrated core + circuit + experiment/checkpoint regression passed **213 tests in 16.13s**.

**Disposition.** This is an opt-in modeled-array guard, not a universal safe resource envelope. P2-1 remains partial pending workload profiling, overhead characterization and a defensible default policy, if one is possible. No high-width allocations or large-shot profile were run.

## 2026-10-14 — P2-1 resource preflight edge validation — partial

**Adaptive profile disposition.** A planned 15–18-qubit H/CX RSS profile did not allocate or execute a circuit: psutil reported 532,836,352 bytes available (about 508 MiB), below the predeclared 512 MiB stop floor. The new versioned `resource_envelope_profile_20261014.json` records the stop reason and empty profile. This is an environment stop, not a failed simulation or a performance datapoint.

**Estimator defect found and fixed.** Addressability validation previously computed `1 << n_qubits` before checking the platform limit; an extreme width could make the guard itself allocate an enormous Python integer. The estimator now compares width against the platform-derived maximum before shifting. Regressions include width 59 and `10**12`; the latter fails immediately with `MemoryError`. The focused estimator tests passed **6 tests in 0.69s**; simulator plus circuit API suites passed **107 tests in 2.92s**.

**Disposition.** The opt-in budget prevents requests whose modeled principal arrays exceed the caller's selected budget and now safely rejects unaddressable widths. No machine-specific ceiling or total-process guarantee is supported. The adaptive RSS sweep must resume only after available memory rises above the stop floor.

## 2026-10-14 — P2-8 Phase 5 artifact temporary-file recovery — complete

**Confirmed recovery issue.** The development-model exporter used a fixed `model.json.tmp` sibling. If a prior process stopped leaving that file behind, the next valid training export failed even though the final destination was absent. The exporter now uses a uniquely named sibling temporary file, flushes/fsyncs it, and retains its atomic no-overwrite hard-link behavior. It leaves unrelated stale files untouched.

**Validation.** Added a tiny six-row synthetic fixture test with a pre-existing fixed-name stale temp; export succeeded, the artifact round-tripped, and the stale file remained unchanged. `.venv/Scripts/python.exe -m pytest tests/test_phase5_model.py tests/test_ml_training_artifacts.py tests/test_observable_model_adapter.py -q --basetemp .pytest-tmp-autostab-ml-atomic-temp` passed **32 tests in 6.42s**. No production rows or model artifact were read or written by the test.

**Disposition.** The fixed-name stale-temp failure is repaired without deleting the existing file. This does not validate ML calibration/generalization or deployment; the produced model status remains development-only.

## 2026-10-14 — P2-1 adaptive RSS profile resumed — partial

**Execution.** After available memory recovered above the 512 MiB stop floor, ran one H and one nonadjacent CX at widths 15–18 with the same seed and 10 ms RSS sampling. Source and measurements are recorded in `reports/experiments/resource_envelope_profile_20261014_rerun.json`. Times were 0.571, 0.846, 1.463, and 2.838 seconds. The maximum sampled process RSS was 43,757,568 bytes (about 41.7 MiB); norm absolute error was 2.22e-16 at every width. No shots or dataset cases were run.

**Interpretation.** Host available memory fluctuated substantially during the run (about 0.98–1.07 GB in the per-case samples), so this profile is not a controlled machine-wide resource guarantee. Polling can miss transients, and process RSS includes unrelated/runtime baseline while the array estimate excludes overhead. The earlier stopped attempt remains separately preserved in `resource_envelope_profile_20261014.json`.

**Disposition.** The observed 15–18 workload completes on this session, but neither a safe maximum width nor a total-memory guarantee follows. P2-1 remains partial; retain opt-in budget and continue only bounded profiling if useful.

## 2026-10-14 ? P2-9 grouped-CV source compatibility ? partial

**Audit.** Recomputed the dataset source fingerprint using the training pipeline's shared fingerprint function and hashed all nine sources listed in the saved evaluation manifest. The 270-report dataset fingerprint still matches exactly. Three source hashes differ: `experiments/run_experiment.py`, `src/quantum_simulator/diagnosis_engine.py`, and `src/quantum_simulator/quantum_simulator.py`. The latter two include changes made during this stabilization.

**Disposition.** The saved grouped-CV result remains byte-compatible with its input dataset, but is not source-compatible with the current implementation. It is historical evidence only and cannot validate the repaired simulator or diagnosis path. No fitting or report regeneration occurred. Details: `reports/experiments/ml_source_compatibility_audit_20261014.json`. ML validity remains partial/blocked pending current-source validation, probability calibration, independent grouped holdout and trustworthy diagnosis labels.

## 2026-10-14 ? Diagnosis regression after source repairs ? partial

**Validation.** Current diagnosis engine, software fault-localizer, and hardware fault-localizer suites passed **71 tests in 0.70s** using `.pytest-tmp-autostab-diagnosis-followup`.

**Disposition.** This confirms the covered unit behavior after current source changes, but does not change the diagnosis gate: the registered finite-shot comparator still fails its control error requirements, the proposed campaign remains NO-GO, and independent labeled diagnosis validity is not established.


## 2026-10-14 ? Project-wide regression findings and fixes

**Current-tree defect found.** Full-suite triage reached Bugs4Q case 42 and exposed `_canonicalize_circuit` treating `cp` and other two-qubit parameterized operations as one-qubit gates. The second qubit was cast to a float, then rejected by the simulator. Canonicalization now keeps two qubit operands for `cp`, `crx`, `cry`, `crz`, `rxx`, and `rzz`; six parameterized regression cases were added.

**Discovery scalability defect found.** Repository-root Bugs4Q discovery used an unrestricted recursive walk and stalled while traversing unrelated large dataset trees. The search now scans conventional Bugs4Q-named root and `data/` directories only. Added a test verifying unrelated trees are not traversed.

**Scientific test expectation corrected.** Two end-to-end assertions expected readout-only noise to produce an internal pre-measurement state-trace deviation. Readout changes reported classical values after collapse; internal state remains the same. Tests now assert no state-trace localization for pure readout and use a skipped-gate case for hardware-induced state-trace deviation. This preserves hardware attribution and avoids fabricated program evidence.

**Validation.** The focused diagnosis, end-to-end, Bugs4Q validation, and 42-case Bugs4Q integration suite passed **78 tests in 19.83s**. A full test run collected 641 items, then was stopped after it entered `test_bounded_recipe_comparison_120_templates_and_finite_shot_subset` and remained active beyond 40 seconds. This finite-shot comparison covers 120 templates across two recipes and a multi-shot sampled subset; it was not allowed to proceed while the registered calibration design remains NO-GO. No broad suite pass is claimed.

**Checkpoint.** The canonical project backlog now contains the newly discovered defects and a partial project-wide regression task. Core regression evidence updated; benchmark, diagnosis validity, ML deployment/generalization, clean-environment, resource-bound, and live-hardware gates remain open or externally blocked.


## 2026-10-14 ? Final project-wide routine validation checkpoint

**Dataset discovery.** QMutBench root discovery had the same unbounded repository scan pattern as Bugs4Q. It now limits scans to conventional `data/qmutbench`, direct QMutBench roots, and named fixture roots. A fixture confirms `data/qmutbench_full` is not traversed. The targeted QMutBench suite passed **24 tests in 10.05s**, including its representative 12-case QASM integration test (8.21s). No full mutant corpus was processed.

**Current-tree suite.** Command: `.venv/Scripts/python.exe -m pytest tests -q --tb=short -k "not bounded_recipe_comparison_120_templates_and_finite_shot_subset and not independent_calibration_fixture_and_permutation_comparison" --durations=10 --basetemp .pwb-final2`. Outcome: **640 passed, 1 failed, 2 deselected in 206.89s**. The one failure is `tests/test_reference_diagnosis_generalization.py::test_heldout_generalization_outcomes_and_targets`: 7 of 45 frozen detection/classification/abstention outcomes differ on the current tree (three QFT-like subtle program-fault seeds, two ring-cluster correct controls, and two equivalent mutants). Source and hardware target-localization assertions passed. No threshold, held-out target, label, split, or expected result was changed.

**Test semantics.** Updated controlled hardware assertions to reflect the documented physics: readout flips affect reported classical bits after collapse, so they produce no pre-measurement state trace candidate. Gate skip/coherence cases retain trace localization, classified as neutral program evidence. `tests/test_stage3_controlled_diagnosis_evaluation.py` passed **23 tests in 12.65s**.

**README reconciliation.** Corrected the `diagnose_circuit(reference_circuit=...)` contract and documented the actual routine test outcome and the two excluded calibration comparison tests.

**Verdict.** Not ready for broad diagnosis, benchmark accuracy, ML deployment/generalization, publication novelty, or real-hardware claims. The registered finite-shot study remains NO-GO; current held-out diagnosis acceptance fails; benchmark provenance/independent truth and ML source/calibration evidence remain incomplete; live hardware is not authorized. The canonical status, backlog, and report are current checkpoints.


## 2026-10-14 ? Held-out mismatch root cause and ML hash reconciliation

**Reproduction.** Re-ran the frozen 45-case evaluator and independently computed exact statevector output distributions for each of the seven mismatches. The three QFT-like program faults have exact TVD 0.00786, 0.00126, and 0.01130, all below the fixed 0.05 threshold; sampled TVD is 0, 0.01953, and 0.01953 against the 0.0625 guard at 256 shots. The frozen expected-positive outcomes cannot be supported by this criterion for these cases. The four ring-cluster correct/equivalent cases have exact TVD 0 but sampled TVD 0.08594 or 0.125 at the registered seeds, exceeding the unchanged guard. Equivalent CZ reordering leaves the exact distribution unchanged while syntactic localization reports source candidates; the false-positive distribution decision drives the observed category. Source target hits remain true and no hardware target miss occurred.

**Disposition.** No implementation defect in these seven decisions was demonstrated. The failure reflects low sensitivity below the threshold and poor finite-shot specificity above the guard, consistent with the registered calibration NO-GO. Added `test_frozen_heldout_mismatches_are_explained_by_distribution_resolution`; it passed 1 test in 6.62s. The original 7/45 acceptance test and its frozen fixture remain unchanged and failing. No threshold, split, label or outcome tuning. Evidence: `reports/experiments/heldout_diagnosis_root_cause_20261014.json`.

**ML source audit.** Recomputed all nine current source hashes. The three evaluated sources still differ from saved manifest hashes. The original audit's recorded current diagnosis-engine hash was stale against the live file; it is retained unchanged as historical evidence, and the verified refresh is `reports/experiments/ml_source_compatibility_audit_20261014_rerun.json`. The source changes map to atomic/provenance handling, diagnosis semantics, and simulator measurement/reset/noise/gate semantics. Dataset bytes still match; grouped CV was not rerun and no training occurred. Added verified backlog item P2-13 for audit snapshot freshness.

**Gate state.** Diagnosis, finite-shot calibration, ML generalization/deployment, benchmark truth, and hardware gates remain blocked/deferred. Continue with independent bounded backlog work; no full QMutBench processing, large training, campaign, or live hardware job was launched.


## 2026-10-14 ? Isolated clean-environment validation

**Environment.** Created `.pwb-clean-env-20261014` with Python 3.12.10 and installed the editable project plus `.[dev]` declared dependencies. The offline cache lacked setuptools, so dependency installation used the approved network escalation. `pip check` reported no broken requirements; core imports succeeded.

**Routine suite.** The same bounded suite/exclusions completed **604 passed, 1 failed, 4 skipped, 2 deselected in 149.78s**. The single failure is the unchanged held-out acceptance (the same seven outcomes). The two calibration comparison tests remain excluded under the existing NO-GO; no QMutBench full corpus, training, or hardware work ran.

**Coverage difference.** Clean collection contained 610 test items versus 643 in the primary environment after adding the forensic regression. The 33 missing items are all in `tests/test_gate_reference_qiskit.py`; Qiskit is installed only in the primary environment and is not declared in project metadata. Record this as an optional reference-oracle coverage gap; the primary environment's Qiskit differential evidence remains available. No dependency metadata was changed. P2-2 clean-environment status remains partial because the acceptance suite fails and the clean environment omits that optional oracle suite.

**Disposition.** A clean install is reproducible, the known held-out blocker reproduces, and clean dependency coverage differs explicitly. The project is not ready.


## 2026-10-14 ? Bounded resource profile continuation and checkpoint

Profiled one H and one nonadjacent CX at 19 and 20 qubits under the existing 512 MiB available-memory stop floor and 5% modeled-array cap. Both completed with norm error 1.11e-16; elapsed time was 6.205s and 11.669s, with sampled RSS peaks of 52,260,864 B and 69,079,040 B. Available host memory fell to 419,151,872 B after width 20, below the stop floor, so profiling stopped. Results: `reports/experiments/resource_envelope_profile_20261014_checkpoint2.json`. This does not establish a safe maximum or portable resource guarantee.

Saved the isolated environment package inventory and routine suite result in `reports/experiments/clean_environment_validation_20261014.json`. The only clean-suite failure remains the seven-case held-out acceptance failure; optional Qiskit oracle coverage is not present in declared project dependencies. No further calibration, full dataset, training, or hardware work was run.


## 2026-10-14 ? Current claim reconciliation

Updated the README's current status to distinguish the earlier 639-pass primary-environment checkpoint from the clean isolated run and to link current evidence. A read-only review found that `project_finalization_audit.md` (Oct 12) has a stale statement that `diagnose_circuit` lacks a reference-circuit argument and a FINALIZE verdict that predates current held-out/source blockers. The Oct 12 final research report contains historical ML results and is not current-source validation. Those documents were preserved unchanged and scoped in `project_claims_reconciliation_20261014.md`. The documentation gate remains partial because the entire historical archive has not been reissued for a specific publication bundle.


## 2026-10-14 - Method feasibility, ML provenance, optional oracle, and final suite

**Diagnosis-method feasibility.** Added `diagnosis_method_feasibility_design_20261014.md`, synthesizing the existing finite-shot no-go, identifiability and candidate-selection evidence without repeating the seven-case investigation. Output anomaly detection is testable only under a calibrated declared sampling model; cause attribution from output counts is impossible for observationally equivalent mechanisms. Source diffs can support syntactic candidates, and hardware telemetry must be independent to support physical cause claims. The proposal separates objectives, requires family-level independent splits, blinded truth, joint power/coverage/type-I criteria and explicit resource bounds. No campaign was executed and existing criteria remain frozen.

**ML pipeline audit.** Rechecked all 270 historical rows across 30 families: balanced labels by construction, unique run IDs, no duplicate report fingerprints, finite production features, five `StratifiedGroupKFold` folds with no family overlap. A fixed Logistic Regression evaluation was run twice on the current tree in memory and matched saved OOF predictions/probabilities exactly (maximum delta 0; accuracy/balanced accuracy 0.563, macro F1 0.540, Brier 0.558, log loss 0.908). The data rows and dataset manifest lack generation source hashes and labels are synthetic by construction; this result is repeatability on the same rows, not current-source generation, independent validation, calibration, or deployment evidence. Reports: `ml_current_pipeline_audit_20261014.md/.json`.

**Provenance repair.** Added a scoped source SHA-256 map to each newly generated experiment result, covering project metadata, config, dataset generator, runner, simulator, gates, diagnosis and hardware model. Historical data was not backfilled or overwritten. The post-repair compatibility snapshot identifies the original three implementation hash deltas and one later path-only `ml_evaluation.py` delta against saved evaluation hashes. The dataset byte fingerprint still matches. Evidence: `ml_source_compatibility_audit_20261014_post_provenance.json`.

**New bounded defect.** Calling ML preflight with a valid relative report root raised `ValueError` because report paths were resolved against an absolute project root. `load_dataset` now resolves inputs and emits repository-relative paths for in-repository records, absolute paths for external roots. Regression suite: ML evaluation/preflight 8 passed; combined experiment-runner/ML tests 20 passed.

**Qiskit oracle strategy.** Confirmed Qiskit is an optional test oracle, not a runtime or required core-validation dependency. Added an `oracle` project extra (`qiskit>=2.5,<3`), metadata regression and explicit README install/test instructions. No package was installed and no environment was altered. In the pre-existing primary environment, the Qiskit reference suite passed 33 tests with Qiskit 2.5.2. In the existing clean environment, combined oracle/core gate validation passed 37 tests with 4 optional skips.

**Benchmark evidence boundary.** Revisited the case/source audit: QMutBench's 12 supported/unsupported fixture records are integration evidence with zero compatible diagnosis labels; Bugs4Q source and bug-type metadata do not provide diagnosis category or independent localization truth. Simulator-derived observations, benchmark case execution, external benchmark truth and real hardware evidence remain separately identified. No full QMutBench run, large training, or hardware job was launched.

**Final repository review and validation.** `git diff --check` reported no whitespace errors; only line-ending normalization warnings appeared for modified Windows files. An initial full routine test attempt used a long pytest temporary root and four calibration-resume tests hit Windows path-length errors. Reran with a short workspace-local root: **642 passed, 1 failed, 2 deselected in 168.94s**. The sole failure is the known frozen held-out acceptance test (7/45 outcomes); no thresholds or expected results changed. The two registered campaign-like finite-shot comparisons remain deselected under NO-GO. This is not a ready verdict.

**Checkpoint and reopened dependencies.** Backlog now contains 34 tasks: 20 verified, 6 partial, 6 blocked, 2 deferred. Qiskit strategy, relative-path handling, report-level source provenance implementation and diagnosis design are recorded as evidence-backed tasks. ML generalization remains blocked by historical row provenance, synthetic labels, calibration and no external holdout; benchmark truth remains blocked; resource envelope remains partial; hardware remains deferred pending authorization/credentials. Current status and completion report were updated after final tests.


## 2026-10-14 - Final post-manifest-provenance suite

After the dataset generator was updated to write scoped source hashes and Python/NumPy runtime into future manifests, the ML compatibility and grouped-evaluation snapshots were refreshed. Five of nine source hashes differ from the saved grouped-evaluation manifest: the original experiment runner, diagnosis engine and simulator changes, plus the later ML relative-path and dataset-manifest provenance repairs. The saved dataset bytes still match; historical rows and manifest have no generator hashes and were not modified. The fixed grouped evaluation again repeated exactly and matched the saved OOF output.

Final bounded command: `.venv/Scripts/python.exe -m pytest tests -q --tb=short -k "not bounded_recipe_comparison_120_templates_and_finite_shot_subset and not independent_calibration_fixture_and_permutation_comparison" --durations=10 --basetemp .pwbH14`. Result: **644 passed, 1 failed, 2 deselected in 154.36s (647 collected)**. Sole failure: the same frozen 7/45 held-out diagnosis acceptance test. The two campaign-like finite-shot comparisons remain excluded under NO-GO. No further source changes followed this run; subsequent edits updated canonical records only.

Final backlog: 35 tasks (20 verified, 7 partial, 6 blocked, 2 deferred). Future-data provenance, Qiskit optional-oracle contract, relative ML path support and diagnosis feasibility design are in the backlog with limitations. Scientific/benchmark truth and live hardware evidence remain unavailable. Project remains **NOT READY**.


## 2026-10-14 ? Final optional integration and release review

Reconciled the prior IBM readiness report with the current adapter. Transpiler layout extraction rejects absent/invalid/unresolved routing maps before submission, remaps diagnosis operations into physical coordinates, and fixture tests enforce the 900-second default calibration freshness boundary. Added an explicit optional `ibm` dependency extra and README commands for isolated install and offline adapter tests; no dependencies were installed and no IBM service was contacted. IBM's install guide documents Runtime as a separately installed client (https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit-runtime).

Focused check: `tests/test_experiment_runner.py::test_ibm_runtime_is_an_explicit_optional_integration_extra`, Qiskit extra metadata test, and `tests/test_ibm_validation.py`: **45 passed in 3.41s**. Current adapter-only test run: 43 passed in 2.26s. Local mock readiness is recorded in `hardware_adapter_local_readiness_20261014.md`. Live hardware remains blocked by no intended backend name, credentials/authorization, and absent service evidence.

Fresh bounded repository suite: **645 passed, 1 failed, 2 deselected in 147.80s (648 collected)**. The sole failure is unchanged: `test_heldout_generalization_outcomes_and_targets`, with the same seven frozen outcome mismatches. The two campaign-like calibration comparisons remain deselected under NO-GO. No held-out investigation, threshold tuning, acceptance edit, expensive campaign, large training, or hardware job was performed.

Backlog now has **36 tasks: 21 verified, 7 partial, 6 blocked, 2 deferred**. Resource task explicitly records that no portable safe maximum or total-process resource cap is established from the bounded 20-qubit session profile. Status, backlog, and completion report reflect the final run. Overall verdict remains **NOT READY** for scientific validity, broad diagnosis, ML deployment, benchmark accuracy, publication readiness, or hardware validation.


## 2026-10-14 ? Final backlog closure and ML evidence gate

Reconciled all remaining partial, blocked, and deferred tasks against current code/evidence. P2-18 is now verified for the future-manifest scope its acceptance defines; historical traceability remains a separate unresolved limitation. Added P2-20 after discovering the standard ML training preflight could proceed despite missing generator/per-run provenance. The loader now preserves null provenance for rejection rather than crashing. Standard training stops before row loading/fitting when manifest source/runtime identity is missing, row maps are absent or inconsistent, or generator hashes do not match current code. On the preserved 270-row historical dataset, ready=false with three explicit provenance blockers; no fit/prediction/export/write occurred. The full source audit now records six of nine saved evaluation-source hashes changed. Historical evaluation artifacts and dataset bytes were preserved.

ML focused regressions: **29 passed in 3.06s** (loader, preflight, provenance, artifacts); a broader focused ML plus IBM adapter run passed **72 tests in 4.69s**. Existing isolated clean environment, with no optional installs, passed **45 IBM adapter and oracle/IBM metadata tests in 0.72s**. Diagnosis review materials now specify family-frame construction, lineage grouping, independent truth/adjudication, competing explanations, separate endpoints, conditional power limits, resources, and reviewer requirements. No target frame, independent labels, or reviewer is available, so study remains NO-GO.

Final bounded full suite: **648 passed, 1 failed, 2 deselected (651 collected, 140.45s)**. Sole failure remains the frozen held-out acceptance with 7/45 mismatches. The two campaign-like finite-shot tests remain excluded under NO-GO. No seven-case re-investigation, threshold tuning, full benchmark, large training, environment installation, credential inspection, or hardware contact occurred.

Final backlog reconciliation: **37 tasks: 23 verified, 6 partial, 6 blocked, 2 deferred**. Partial/blocked/deferred rows now record the current blocker class and exact prerequisite. No remaining critical/high local defect is identified; scientific and external gates remain open. Overall verdict **NOT READY** for broad diagnosis, ML deployment, benchmark accuracy, publication, or real hardware.


Final rerun after requiring exact Python/NumPy dataset-runtime identity: **648 passed, 1 failed, 2 deselected in 151.10s (651 collected)**. The sole failure is unchanged held-out acceptance (7/45); all seven case evidence remains the previously recorded root-cause report and was not reinspected. Expanded ML provenance suite: **29 passed in 7.05s**. Updated status/backlog/report now use this final runtime and result.


[2026-10-14] ML data foundation and lineage-safe closure
- Compared all nine evaluator-source paths in the saved grouped-CV manifest to the current tree. The prior release-final snapshot has six mismatches; after this lineage implementation, a new immutable snapshot has seven. The six original file changes are explained, but historical row-to-source attribution remains unresolved because no per-row hashes or dataset generation provenance exist. Dataset bytes still match the saved fingerprint.
- Audited features: program-mismatch feature reveals synthetic PROGRAM_FAULT exactly (90/90 vs 0/180); calibration candidates come from configured synthetic hardware profiles. Both are excluded from production. No duplicate report fingerprints; exploratory circuit sequence comparison identifies related recipe siblings.
- Added `circuit_lineage_id` to future generator configs, rows and manifest. Current ML preflight requires it; split and fold validators prohibit lineage overlap. Historical rows without lineage remain ineligible.
- Specified a prospective root-separated train/development, probability-calibration and untouched-test protocol with class-wise metrics, proper scoring, lineage-cluster bootstrap intervals and pre-study acceptance/power requirements. Protocol remains design-only; no independent labels or adequately sized external frame.
- Validation: 28 focused tests passed in 5.30s. Current preflight: ready=false, 0 eligible / 270 ineligible, no fit or writes. No package install, full generation, training or hardware use.
- Backlog: 38 total (24 verified, 6 partial, 6 blocked, 2 deferred). ML deployment/generalization remains blocked; overall verdict NOT READY.

[2026-10-14] Validation rerun after report lineage propagation assertion: 28 focused tests passed in 5.35s; git diff --check passed with line-ending normalization warnings only. README and claims reconciliation now point to the post-lineage ML audit.

[2026-10-14] Final focused rerun after enforcing lineage metadata exclusion from predictor features: 29 passed in 5.17s. Post-lineage source compatibility remains 7/9 changed; dataset byte fingerprint matches; current preflight remains not ready (0 eligible, 270 ineligible).

[2026-10-14] Expanded source-provenance closure: included transitive statistics/localization modules, package entry/circuit files and root import shims in future report/dataset source hashes. Focused regression: 29 passed in 5.88s. Refreshed source audit still shows 7/9 evaluator-source hash differences; historical data bytes match but current preflight blocks all 270 rows.


## 2026-10-15 ? Independent ML data pilot and final bounded regression

**Pilot.** Added `experiments/ml_data_pilot.py` and generated `reports/experiments/ml_data_foundation_pilot_20261014_final/`: 9 simulator-only runs, 3 families and 3 recipe lineages, 32 shots/run. The manifest captures scoped execution/generation/extraction/validation hashes, Git HEAD and tracked-worktree status digest, all run configurations and a config digest, per-run seeds/lineage, exact Python implementation/build and NumPy versions, platform, artifact SHA-256 and deterministic run fingerprints. Observation predictors and synthetic scenario labels are separate; raw diagnosis reports are labeled internal-only. `gate_mismatch_fraction`, configured calibration profiles, labels, IDs and diagnostic fields are excluded from predictor rows. The structural/provenance validator passes; `fit_eligible=false` because all nine labels are synthetic and none has independent adjudication.

**Protocol and historical source audit.** Added `ml_independent_evaluation_protocol_20261014.md` covering population-specific family frames, independent adjudication, lineage-disjoint training/calibration/sealed test, classwise and calibration/abstention metrics, root-cluster uncertainty, and family-level power inputs. Missing frame, prevalence, cluster sizes/ICC, independent reviewers, target use thresholds and reviewed resource cap preclude a numeric power claim or new evaluation. Refreshed saved-evaluation comparison: 7/9 hashes differ at the file level; all seven row-generation attributions remain unresolved because historical reports/manifest lack source identity. Dataset bytes still match. Current preflight rejects all 270 historical rows, no training/prediction/output writes.

**Validation.** Focused pilot/ML provenance/lineage/artifact suite: 54 passed in 35.52s. Final `.venv` bounded project suite: 659 passed, 1 failed, 2 deselected (662 collected) in 192.07s; the sole failure is unchanged frozen held-out diagnosis acceptance with 7/45 mismatches. Two finite-shot campaign-like tests stayed excluded. A system-Python run had 623 passed, same held-out failure, 4 skipped and 2 deselected. An intermediate longer temp root hit a Windows path-length error; the final short `.pwb17` run passed the affected recovery test. The ML artifact-fold tests initially exposed a lineage fixture omission; their rows were updated with explicit roots and regressions now reject absent/crossed lineage; focused tests pass. No held-out expectations/thresholds changed; no full dataset, model fitting, package installation or hardware execution.

**Disposition.** Added backlog P2-23 as verified engineering; counts are 39 tasks: 25 verified, 6 partial, 6 blocked, 2 deferred. ML generalization/deployment, independent diagnosis validity and overall project release remain blocked/not ready.


Exact-current-tree rerun after adding stale-config and internal-report tampering regressions: focused ML/pilot/provenance/lineage suite passed 55 tests in 39.77s. Final bounded `.venv` suite passed 660, failed only on the frozen 7/45 held-out diagnosis acceptance, and deselected two finite-shot NO-GO tests (663 collected; 185.31s). This supersedes the prior interim totals. Backlog remains 39 tasks: 25 verified, 6 partial, 6 blocked, 2 deferred. No criteria changed and no held-out root-cause work repeated.


## 2026-10-15 ? Post-pilot blocker closure

Reconciled the canonical 39-task backlog against the task records. Its top-level verified count was stale at 24 while the task records and checkpoint both reported 25; corrected the summary to 25 verified, 6 partial, 6 blocked and 2 deferred. No task was closed by intention alone. The two P3 hardware entries cover the general release gate and IBM integration scope; both remain deferred pending a named backend, credentials, explicit authorization and possible paid quota.

No new independent evidence supports changing the frozen diagnosis method. The recorded forensic evidence already classifies the seven mismatches as method/observation-resolution limitations; no retuning, threshold changes, acceptance edits, or repeated forensic analysis was performed.

The bounded post-pilot integration suite passed **344 tests in 69.10s**, covering gate oracles, simulator/circuit, measurement/reset, noise, experiment and checkpoint recovery, dataset provenance, ML pilot/preflight/artifacts, and benchmark/IBM adapters. The existing nine-run `ml_data_foundation_pilot_20261015_v3` was revalidated read-only: provenance and schema pass, 9/9 records, 3 families/lineages, zero near-duplicate pairs, zero independent adjudications, and fitting remains blocked.

Fresh bounded repository run: **660 passed, 1 failed, 2 deselected (663 collected) in 179.93s**. The only failure remains `test_heldout_generalization_outcomes_and_targets` with the same frozen 7/45 outcome mismatch. The two finite-shot campaign-like tests remain deselected under NO-GO. No new method investigation, training, full dataset generation, dependency installation, full benchmark, or hardware job occurred.

Backlog remains **39 tasks: 25 verified, 6 partial, 6 blocked, 2 deferred**. No new locally actionable defect was found in the reviewed integration paths. Overall project remains **NOT READY** for broad diagnosis, ML generalization/deployment, benchmark accuracy, publication readiness, or hardware validation.


## 2026-10-15 ? Final project-wide evidence closure

Reconciled the latest 39 backlog records, dependencies, status summaries and release claims. Task counts agree: **25 verified, 6 partial, 6 blocked, 2 deferred**. Corrected stale current figures in the completion report (older suite and 37-task count) and claims table (six versus seven current evaluator hash differences); prior dated test and audit history remains recorded. Added a capability-by-capability evidence disposition to the existing completion report.

The one requested bounded repository run collected **663 tests: 660 passed, 1 failed, 0 skipped, 2 deselected in 168.65s**. The sole failure remains the frozen held-out acceptance (`test_heldout_generalization_outcomes_and_targets`, 7/45). No tests or thresholds were weakened. The two finite-shot campaign comparisons remain excluded by the registered NO-GO.

No concrete locally actionable defect was independently demonstrated, so no code changes were made. Local engineering closure is supported for the bounded tested simulator, recovery, provenance, and offline integration scopes. Research closure remains blocked by independent truth, family sampling and protocol review; hardware is deferred for backend identity, credentials and authorization. Project remains NOT READY for broad diagnosis, ML generalization/deployment, benchmark accuracy, publication readiness or hardware validation.


## 2026-10-15 ? Independent diagnosis protocol milestone

Inspected the current diagnosis interfaces, finite-shot statistic and existing identifiability, feasibility, research-gap and power records. The interface keeps distribution statistics, program-operation candidates and hardware-model candidates in separate result structures, but the current TVD rule is a fixed threshold/guard, and localizer candidates are not calibrated causal probabilities. Existing records establish that output-only cause attribution fails for observationally equivalent mechanisms; simulator labels cannot establish physical hardware truth.

Updated the existing `diagnosis_method_feasibility_design_20261014.md` with a target claim, candidate two-layer method (not implemented as a detector), independent truth/adjudication requirements, a family-separated train/calibration/untouched-test design, separate detection/cause/localization/abstention endpoints, uncertainty/multiplicity/missingness rules, and explicit power/resource/authorization blockers. The proposed count permutation or multinomial calibration remains conditional on sampling assumptions and requires independent review. No numeric performance floor or sample size was invented. Existing 7/45 and 30/60 no-go findings remain unchanged; conditional power examples are not promoted to a study design. Technical method references were added for the exchangeability conditions of two-sample permutation tests and sparse multinomial calibration.

Added the standard-library `experiments/diagnosis_evaluation_protocol.py` record guard and `tests/test_diagnosis_evaluation_protocol.py`. The validator checks observable/adjudicator separation, raw count and shot-total consistency, source/runtime/protocol provenance, unique case IDs, exact family fingerprints, family/lineage-disjoint partitions, non-collapsed ambiguous/equivalent cause sets, and confirmatory prerequisite blockers. It explicitly reports only structural prerequisites, not independent label validity, sample sufficiency, power or method performance.

Initial test-authoring run: **40 passed, 3 failed** because three test assertions expected incorrect error wording. Corrected only those assertions; final focused command covering the new protocol contract and existing pilot/lineage/artifact guards passed **46 tests in 34.01s**. No simulator, diagnosis algorithm, frozen acceptance, historical artifact, dependency or environment was changed. No full repository run followed this protocol-only addition; the last full run before it was 660 passed, one frozen 7/45 failure, zero skipped, two deselected (663 collected; 168.65s).

Added P2-24 as verified engineering. Backlog is **40 tasks: 26 verified, 6 partial, 6 blocked, 2 deferred**. The protocol milestone is prepared but confirmatory work remains blocked by missing target population/frame, independent adjudicators, independent review, use-specific acceptance decisions, family-level power assumptions, and a reviewed resource/stop plan. No study is authorized or executed.


### Final protocol endpoint check (2026-10-15)

Separated general cause-disposition/abstention eligibility from exact single-cause attribution eligibility in the record validator. Ambiguous and observationally equivalent test cases remain valid for the declared disposition/abstention endpoints but block a protocol that requires exact single-cause truth for those same cases. Final focused rerun: **47 passed in 35.93s**. The full repository suite was not rerun after these protocol-only additions; its previous exact counts are retained with that scope note.

## 2026-10-15 - Development-only diagnosis method advancement

Audited evidence lineage before selecting a candidate. The 3,600-record finite-shot results are saved synthetic evidence from one circuit recipe with repeated templates/conditions. Calibration cells motivate a narrow healthy-control false-alarm limitation: fixed TVD flagged 120/120 controls at 256 shots and 104-110/120 at 1,024; the recorded permutation comparator flagged 2-8/120. The prior untouched-validation split has already been examined and cannot serve as fresh validation. It also shows the sensitivity trade-off on subtle effects (permutation 14/120 at 256 and 36/120 at 1,024 on calibration). The three-family reference-generalization fixture is an engineering fixture, not independent development truth. The frozen 45-case outcomes were not used.

Added experiments/diagnosis_candidate_methods.py with a seeded count-based pooled-label TVD permutation p-value function; it is isolated from the default detector and does not attribute or localize faults. Added seven small contract tests for deterministic seeding, identical counts, and invalid input rejection. Focused command: .venv/Scripts/python.exe -m pytest tests/test_diagnosis_candidate_methods.py -q --tb=short --basetemp .pytest-tmp-diagnosis-candidate-20261015; result: 7 passed in 0.74s. No campaign was rerun. Candidate is implemented but not evaluated as a diagnosis method.

Updated P1-4 evidence and canonical status/report with the data boundary and a prospective bounded development-only design requiring new family-disjoint development and validation families. Existing validation results cannot be reused; independent truth, target frame, review, acceptance/power/resource decisions, and separate authorization remain prerequisites for confirmatory work. Backlog stays 40 tasks: 26 verified, 6 partial, 6 blocked, 2 deferred; no existing task was closed by contract tests alone. The bounded repository suite was not rerun because the new helper is isolated and its focused tests passed; the last suite remains 660 passed, one frozen 7/45 failure, zero skipped, two deselected (663 collected), predating this helper.

## 2026-10-15 - Family-disjoint development validation checkpoint

Inspected finite-shot, reference-generalization, controlled-fault, full-dataset, generator, runner, and resource-profile artifacts. The finite-shot registry specifies a single paired-CNOT/phase-tag recipe and its completed run manifest records the recipe and output counts, not a complete per-template family/descendant fingerprint index. The three-family reference fixture describes three synthetic families but is used by the frozen outcome test. Controlled-fault records contain some family IDs but no complete ancestry-root inventory. The repository therefore cannot certify that a proposed validation family is unrelated to every previously inspected template or descendant.

Created reports/experiments/diagnosis_development_validation_plan_20261015.json before any data generation. It specifies the intended development/validation split, synthetic healthy and three parameter-shift strata, shot/repetition conditions, fixed baseline/candidate rules, provisional directional pilot categories, uncertainty reporting, and hard local circuit/shot/runtime/memory/output caps. Its status is blocked_before_generation. No circuits or counts were generated; no threshold was selected; no study, campaign, hardware job, package install, or model training ran.

Implemented experiments/diagnosis_development_study.py with a fail-closed complete-inventory check, prior-root/ancestor/topology exclusion, partition checks, deterministic condition-manifest seed derivation, and budget reservations that fail atomically. Added tests/test_diagnosis_development_study.py for incomplete/excluded lineage, cross-partition roots, deterministic manifests, and resource stop behavior. Combined focused candidate/preflight/protocol tests passed 34 in 0.80s. The final short-path bounded repository suite completed: 694 passed, one unchanged frozen 7/45 failure, zero skipped, and two NO-GO campaign tests deselected (697 collected; 205.14s). A first long-basetemp run had one extra Windows path-length failure; the isolated recovery test then passed with short basetemp and the final full rerun had no path-length failure. The final suite includes all new protocol/candidate/preflight test modules; the sole failure remains the frozen diagnosis acceptance.

The diagnosis feasibility/backlog task remains partial. Local safeguard contracts are implemented, but the experiment is blocked until a complete prior-family/descendant inventory is assembled and independently reconciled, then the planned family frame is verified disjoint. The frozen 7/45 diagnosis failure and finite-shot no-go remain unchanged.

Final follow-up after descendant-root guard: added a regression rejecting a validation family whose ancestor list names another proposed family root, even when IDs and fingerprints differ. Focused candidate/preflight/protocol suite passed 35 tests in 0.85s. The exact-tree bounded repository suite passed 695, failed only the frozen 7/45 diagnosis acceptance, skipped 0, and deselected 2 NO-GO campaign tests (698 collected; 187.28s). The initial long-path failure and its short-path recovery remain recorded above. Study disposition remains blocked before generation; no pilot circuits or shots were executed.

Full-dataset family review: experiments/full_dataset.py assigns 30 nominal circuit_family_id values while cycling only six circuit_lineage_id recipe roots. Those 30 IDs cannot be treated as 30 independent validation families; all six roots are excluded from the proposed pilot pending the complete prior inventory.

## Prior lineage inventory and materialized family frame (2026-10-15)

Added a deterministic, stdlib-only provenance scanner and frame builder. Inventory: `reports/experiments/diagnosis_prior_lineage_inventory_20261015.json`, SHA-256 `0338cb2d6ee8fbebb3afa1882cb8457dd4a010461dc1f8a23ff4969dd195ca32`. It covers 5,431 files in `experiments`, `tests/fixtures`, and `reports/experiments`, including 5,366 circuit-like records. Five files have exact duplicate content; 5,186 circuit-like records have unknown/unreconciled lineage. All discovered circuit records are conservatively excluded as prior evidence. Canonical mutable status/plan/journal/backlog/completion controls are listed as excluded from the scanner inputs to avoid self-referential hashes.

The versioned deterministic frame is `reports/experiments/diagnosis_family_sampling_frame_20261015.json`, SHA-256 `6a96efd133d1e1f2fb9d19299b3206101499f22083493781afc1a4bc5c841e9a`. It contains zero eligible roots, zero selections, and an 8-development/8-validation shortfall. The preflight validates document hashes, sampler version, source inventory binding, completeness, and partition conditions; it remains blocked because completeness cannot be certified and no fresh roots are proven. No generation, simulation, shots, threshold selection, training, or hardware work occurred.

Focused candidate/protocol/lineage suite: 40 passed in 0.73s. Final bounded repository suite: 699 passed, 1 failed, 0 skipped, 2 deselected (702 collected; 165.26s). The sole failure remains the unchanged frozen diagnosis acceptance test with 7/45 mismatches. The two registered campaign-like finite-shot tests remain deselected. No expectations were changed and no study budget was consumed.

Final scan reconciliation correction (2026-10-15): explicit cross-artifact family/root relationships are preserved as six review flags; the scanner does not decide whether they represent legitimate shared lineage or contradictory provenance. The final inventory/frame hashes are `0338cb2d6ee8fbebb3afa1882cb8457dd4a010461dc1f8a23ff4969dd195ca32` and `6a96efd133d1e1f2fb9d19299b3206101499f22083493781afc1a4bc5c841e9a`. Added the review-flag regression. Final focused run: 40 passed in 0.73s. Final bounded suite after all code changes: 700 passed, 1 frozen 7/45 failure, 0 skipped, 2 deselected (703 collected; 228.74s). Prior counts above are retained as historical checkpoints.

Final fail-closed completion check (2026-10-15): tightened materialized-study preflight so a `complete: true` field alone cannot authorize eligibility; any future complete inventory must have zero unknown/conflicting lineage, no missing scan roots, and a hash-bound reconciliation attestation naming at least two independent reviewers and zero unresolved records. Regression coverage rejects a forged/self-declared completeness bit. Updated inventory/frame hashes after this source and test change: `8c952df9165b11b94fe68f07aede5a5d6a7a340f2e1ed10a11e0156cae5bec17` / `f808b8b19c477df0675b26f01a05f0392bb6f49e8560842c8624d60ee9cdb719`. Focused suite: 41 passed in 0.85s. Final broad suite: 701 passed, 1 failed, 0 skipped, 2 deselected (704 collected; 223.99s); sole failure is unchanged frozen diagnosis acceptance, 7/45. The pilot remains blocked and no circuit/shot budget was consumed.

Expanded prior-output inventory checkpoint (2026-10-15): discovery now includes 108 primary, data/ML, and retained prior temp-output roots under fixed caps (20,000 files, 128 MiB total, 25 MiB per file). The bounded scan reached 20,000 artifacts / 124,638,914 bytes and skipped five oversized files, so it remains explicitly truncated and incomplete. It found 5,366 circuit-like records; 5,186 have unknown lineage; six cross-artifact family/root flags remain unadjudicated. Three recipe roots occur in saved rows; the generator source declares six possible `design_recipe_01`–`design_recipe_06` roots, but only three appear in scanned saved rows. Zero families qualify for the new frame. Final inventory/frame hashes: `ec39aeccbac38407d7cb9aa7891cd388d681b056332a0aa4197dfd74400f5c37` / `ccef18762d9c87412e0f88e6d2b84ebdbbfd8b9d4edbc3ff0d22c69c03c0031b`.

Final focused lineage suite: 46 passed in 0.62s. Final broad suite after the root index, retained-output discovery, and resource-cap checks: 706 passed, 1 failed, 0 skipped, 2 deselected of 709 collected (248.38s). The only failure is still the frozen held-out test at 7/45. No experiment generation or execution occurred.

Inventory count reconciliation: once retained temp-output roots were included, the deduplicated content-link count rose from the earlier primary-tree scan's five to 457 exact duplicate-content artifacts across the capped 20,000-file scan (19,543 distinct content hashes among scanned files). All duplicate paths remain recorded with `duplicate_of`; no evidence files were deleted. Six family/root relationship flags remain review-only. Latest summary also records 1,671 family-bearing artifacts without an explicit root. Inventory/frame hashes and study disposition are unchanged; frame remains zero eligible roots.

Final exact-tree regression (2026-10-15): the bounded repository suite completed with 706 passed, 1 failed, 0 skipped, and 2 deselected out of 709 collected (217.46s). The only failure remains the frozen diagnosis acceptance test with 7/45 mismatches. Focused lineage/candidate/protocol tests passed 46/46 in 0.62s. The bounded inventory and zero-eligible-root frame are hash-valid and deterministic; no circuit, shot, threshold, model, or hardware work occurred.


## 2026-10-15 ? ML feature eligibility and fail-closed entry points

Current-tree audit found two fitting/export routes that did not enforce the strict dataset preflight: fallback training and observable-model export. Both now stop before feature loading, fitting or output when preflight fails. The historical 270-row dataset remains `ready=false`, 0 eligible: per-row source hashes and generation provenance are missing and the generator-source fingerprint is incompatible. No fitting, model export, historical metric reproduction or dataset regeneration occurred.

Feature-schema review confirms the five output-distribution observables as the primary predictor set. Configuration-derived features remain secondary calibration-only variables. `gate_mismatch_fraction` is restricted to an explicitly diagnostic leakage-ablation schema and rejected from unreviewed schemas. Detailed feature evidence and limitations: `reports/experiments/ml_feature_eligibility_20261015.md`.

Focused validation (`tests/test_ml_fallback.py`, `tests/test_ml_training_artifacts.py`, `tests/test_ml_preflight.py`, `tests/test_ml_data_pilot.py`): **40 passed in 50.33s**. Final bounded repository suite: **709 passed, 1 failed, 0 skipped, 2 deselected (712 collected; 217.11s)**. The only failure remains the unchanged frozen diagnosis acceptance, 7/45; no expectations or thresholds changed. The two registered finite-shot tests remained deselected under NO-GO. Backlog remains 40 tasks: 26 verified, 6 partial, 6 blocked, 2 deferred. ML generalization is still blocked by independent family data, adjudicated labels and external review.


## 2026-10-15 ? Historical ML record recovery and generalization infrastructure

**Data recovery.** Rebuilt the existing `ml_historical_record_recovery_20261015.json` table in memory and confirmed exact equality across all 270 records. The bounded reader processed 3,765,323 report bytes (64 MiB maximum), with 270 unique run IDs and 30 nominal family IDs. Saved reports preserve IDs, circuit/configuration values, seeds, shots, Python/NumPy versions, a generic simulator string and synthetic scenario labels. All 270 lack per-run source hashes and recipe ancestry; the manifest has no `generation_provenance`; independent adjudications and real-hardware records are both zero. Recomputed hashes authenticate present saved bytes only. No provenance was inferred from current source.

**Other candidate sources.** QMutBench local ML features describe a separate mutant/original behavioral-deviation task, not three-way diagnosis. The source hash expected by its manifest differs from the local 60,636,991-byte CSV hash. Its local Git source is a single grafted commit. Bugs4Q has a local 40-commit extracted tree, but existing project audit found 42 official-source cases, only 19 translated/executed and zero metric-eligible diagnosis cases; no matching ML feature/independent cause-label corpus was found. The nine-row pilot, Phase 5 rows and finite-shot results are synthetic. None supply eligible independent truth for primary ML generalization.

**Implementation.** `ml_training.py` now requires exact registered feature schemas (primary five-observable schema by default), includes majority and stratified dummy baselines plus fixed Logistic Regression, Linear SVM, Decision Tree and Random Forest candidates, caps RF/permutation work at one worker, validates exhaustive train/calibration/test splits by family and lineage, detects cross-root duplicate circuit fingerprints, and reports deterministic lineage-cluster bootstrap intervals only when class support permits. Evaluation manifests explicitly mark synthetic-only labels and disallow independent-generalization/hardware claims. New outputs target `grouped_cv_audit_v3`; nonempty destinations are refused before data load/fit, preserving v1/v2. No production fit occurred. Added bounded recovery reader and regression tests.

**Validation.** Focused recovery/evaluation/preflight/fallback suite: **41 passed in 10.52s**. Latest bounded repository suite: **719 passed, 1 failed, 0 skipped, 2 deselected of 722 collected (216.10s)**. The only failure is the unchanged frozen diagnosis acceptance at 7/45. The two registered NO-GO finite-shot campaign tests remain deselected. Current saved-evaluator comparison remains seven source hashes changed of nine; current strict preflight is `ready=false`, 0 eligible of 270, and fitting/prediction/export/writes are all false. Five oversized inventory files were streamed in 1 MiB chunks (315,670,588 bytes total under a 320 MiB cap); the 20,000-file/108-root inventory remains truncated.

Backlog grew to **42 tasks: 28 verified, 6 partial, 6 blocked, and 2 deferred**. Research readiness, power and acceptance remain externally dependent. No model was fitted, no full data generation or benchmark run occurred, and no frozen diagnosis expectation was changed.


### Final evidence-status table and rerun ? 2026-10-15

Preserved the existing recovery table and wrote a new v2 table that explicitly classifies row fields as recorded, inferred from saved metadata, recomputed from present bytes, or missing. The v2 file rebuilds deterministically and matches exactly. Focused recovery/evaluation/preflight/fallback suite: **41 passed in 15.39s**. Final bounded repository suite: **719 passed, 1 failed, 0 skipped, 2 deselected of 722 collected (219.00s)**; the only failure is still the frozen 7/45 diagnosis acceptance. Backlog remains **42 tasks: 28 verified, 6 partial, 6 blocked, 2 deferred**. No model fit or dataset regeneration occurred.


### Newly recorded external ML integration blocker ? 2026-10-15

The current ML report loader accepts only simulator scenario-construction labels. The diagnosis record validator keeps adjudicator truth separate, but the ML data path does not yet ingest/join such records. Adding a manifest declaration alone cannot verify reviewer independence. Added backlog P1-4, blocked on an externally reviewed, blinded adjudication schema and provenance-complete observation data with independent lineage roots. No integration is declared complete without those decisions and examples. Current backlog: **43 tasks ? 28 verified, 6 partial, 7 blocked, 2 deferred**.

## 2026-10-15 - ML evaluation engineering and diagnosis error analysis

Reviewed the current ML feature schemas, evaluator, artifact tests, integration decision path, saved diagnosis root-cause JSON, and existing independent-study protocol. The primary predictor remains the five inference-time distribution observables. `gate_mismatch_fraction` remains confined to the diagnostic leakage ablation; calibration configuration features remain excluded from primary evaluation. The evaluator reports fixed model comparisons and class-wise/confusion metrics on common outer folds, plus deterministic lineage-cluster intervals when class support permits. It does not emit probability calibration metrics because no eligible independent calibration data exist. Probabilities/confidence from an injected classifier do not have a validated calibration basis; the default backend remains unavailable.

**Fixes.** The grouped-CV main entry point previously relied on the splitter before fitting and only explicitly verified family/lineage isolation later while constructing result artifacts. It now validates every fold before model construction or fitting; a synthetic, deliberately family-leaking splitter is rejected before `models()` is called. Separately, an existing equivalent-circuit fixture established that `diagnose_execution` and its ML caller could treat a syntactic diff as behavioral program-fault attribution despite identical observed distributions. Both paths now retain localization candidates and abstain on that behavioral attribution. They do not infer a semantic cause from source differences alone. No held-out case was used to select this fix.

Saved case-level diagnosis evidence remains: three QFT-like expected-positive cases have exact TVD below the fixed threshold; two healthy ring controls exceed the finite-shot guard at registered seeds; and two distribution-equivalent ring reorderings also cross the guard, with their syntactic differences insufficient to justify behavioral attribution. Localization target checks pass. No seven-case forensic rerun or threshold change was made. The unchanged 45-case acceptance still fails at 7/45.

Focused tests: **92 passed in 13.72s**. Final bounded repository suite: **722 passed, 1 failed, 0 skipped, 2 deselected (725 collected; 224.68s)**. The sole failure is the unchanged frozen diagnosis acceptance; the two finite-shot campaign tests remain deselected under NO-GO. No model fitting, large data generation, package installation, benchmark campaign, or hardware job occurred.

Added verified backlog task `P2-27-equivalent-program-diff-abstention`; total is **44: 29 verified, 6 partial, 7 blocked, 2 deferred**. Independent family frame, reviewed blind adjudication, accepted use-specific criteria, defensible family-cluster power/resource assumptions and new eligible observations remain prerequisites. Current ML generalization/calibration and diagnosis research performance remain unmeasured. Project verdict remains **NOT READY** for broad scientific diagnosis, ML deployment, benchmark accuracy, publication readiness, or real hardware.
Final-tree verification addendum - 2026-10-15: after making raw estimator probability semantics explicit in the evaluation manifest, the focused diagnosis/ML suite passed **92 tests in 16.89s**. The bounded repository suite collected **725: 722 passed, 1 failed, 0 skipped, 2 deselected in 221.49s**. Its sole failure remains the unchanged frozen 7/45 diagnosis acceptance; the two finite-shot NO-GO tests remain deselected. This supersedes the immediately preceding timing/result paragraph while preserving that earlier run as checkpoint evidence. No model was fit.
Validation development note: the first two focused invocations each had **89 passed, 2 failed** during regression-assertion refinement. The failures exposed an outdated expectation that a syntactic diff establishes behavioral PROGRAM_FAULT despite identical distributions, an unsupported hardware-candidate assumption, and brittle summary wording. The corrected regressions require abstention while retaining localization and check the stable behavior contract. The final focused suite passed 92 tests; no frozen acceptance criteria or held-out expected results were edited.

## 2026-10-15 - Independent diagnosis protocol package and fail-closed review/provenance gates

Prepared `reports/experiments/independent_diagnosis_protocol_package_20261015.json` by consolidating the existing feasibility design into a machine-readable review package. It specifies the research question and limits; family-plus-lineage independence; complete observation provenance; blinded, separate adjudication; disjoint train/calibration/untouched-test partitions; endpoints, uncertainty, abstention and cost inputs; power-analysis inputs; resource/recovery gates; and explicit REQUIRED INPUT / NOT YET APPROVED / BLOCKED statuses. It does not claim calibration, approval, power, independent truth, or study execution.

Extended `experiments/diagnosis_evaluation_protocol.py` to block confirmatory readiness for missing generation ID, source revision, ancestry, circuit/configuration hashes, seed, noise configuration, runtime versions, or reproduction-record hash. Protocol-embedded approval strings and review IDs are no longer sufficient: a caller must pass a protocol-bound attestation produced by an external trust-policy verifier. The local structural validator does not verify signatures or reviewer independence. Regression coverage includes malformed/missing provenance and self-attested approval.

Focused suite: 54 passed in 0.82s (`tests/test_diagnosis_evaluation_protocol.py`, `tests/test_diagnosis_candidate_methods.py`, `tests/test_diagnosis_development_study.py`). No study, simulation shots, threshold selection, model fit, package installation, or hardware job was run. The previously recorded outcomes and interpretations remain separate: the frozen 45-case gate still fails 7/45; no threshold or expectation changed. Its cause-attribution limits are explicit: observed counts can establish neither source causation nor hardware cause where causes are observationally equivalent. The existing equivalent-circuit decision-path defect fix remains an independently demonstrated engineering correction, not method-performance evidence.

Research remains BLOCKED on a complete reconciled family frame (current inventory incomplete with zero eligible roots), independent blinded adjudicators, external review and protocol-bound attestation, predeclared use-specific criteria, defensible family-level power/resource assumptions, and separate execution authorization. Backlog remains 44 tasks; protocol-record engineering task P2-24 remains verified within structural scope. Broad validation result will be appended after the final bounded repository run.


Final validation for the 2026-10-15 protocol-package phase: focused protocol/candidate/development-preflight suite passed 54/54 in 0.82s. First broad run (733 collected) had 729 passed, two failed and two deselected: frozen diagnosis acceptance at 7/45 plus a Windows path-length failure in a recovery test. That recovery test passed alone 1/1 with a short base directory. Final short-basetemp broad rerun completed with **730 passed, 1 failed, 0 skipped, 2 deselected (733 collected; 222.36s)**; its sole failure is the unchanged frozen diagnosis acceptance. The two NO-GO finite-shot tests remain excluded. `git diff --check` reported no whitespace errors; existing line-ending normalization warnings remain. Backlog reconciles to 44 tasks: 29 verified, 6 partial, 7 blocked, 2 deferred.


## 2026-10-15 - Saved provenance reconciliation and independent adjudication gate

Implemented `experiments/diagnosis_evidence_gates.py`. The deterministic reconciler consumes the persisted inventory, sampling frame, and v2 270-row recovery table; it performs no new filesystem discovery and enforces the recorded 20,000-file / 128-MiB evidence and 1,000-row / 64-MiB recovery caps. `reports/experiments/diagnosis_sampling_frame_reconciliation_20261015.json` links candidate roots and recovery rows to captured/current hashes and records directly verified, inferred, missing, conflicting, and externally attested categories without upgrading inferred facts. Counts: 6 candidate roots (3 fingerprint-conflicted, 3 source-inferred); six inventory conflicts (including 3 ambiguous family-to-root mappings); 5,186 unknown-lineage circuit records; 457 exact duplicate artifacts across the bounded inventory (one duplicate relationship directly attached to these root evidence paths); 30 nominal historical ML family IDs / 270 rows lacking recipe roots and per-run source hashes; zero frame candidates and zero eligible independent families. Inventory remains truncated/incomplete.

Added privacy-aware independent-adjudication records with opaque externally issued reviewer tokens, protocol version/hash binding, immutable evidence hashes, blinded separate judgments, explicit behavioral-program/hardware/equivalent/no-change/insufficient/unresolved outcomes, preserved disagreements, and a separately attested resolver for resolved disagreements. Self-declared independence/blinding, raw reviewer identity fields, simulator/scenario/mutation labels, system predictions, repeated executions counted as separate family cases, and invalid frame-to-train/calibration/test lineage mappings cannot pass. The final study preflight combines reconciled family/frame evidence, per-case source/generation/execution provenance, independent adjudication, protocol review, use-specific criteria/power/resource hashes, and strict family/lineage splits. Attestation authenticity remains an external trust boundary; no attestation was created.

Focused suite: **64 passed in 3.32s**. No ML fitting, dataset generation, experiment, package install, or hardware job occurred. The frozen 7/45 result and both finite-shot NO-GO exclusions remain unchanged.


## Final bounded validation - provenance/adjudication phase (2026-10-15)

Focused reconciliation, adjudication, study-preflight, and protocol tests: **64 passed in 3.32s**. Final bounded repository suite: **747 passed, 1 failed, 0 skipped, 2 deselected (750 collected; 226.33s)**. The sole failure is the unchanged frozen diagnosis acceptance at 7/45; both finite-shot NO-GO tests remained deselected. No ML fitting, dataset regeneration, experiment, package installation, or hardware job occurred. The canonical backlog now contains **46 tasks: 31 verified, 6 partial, 7 blocked, 2 deferred**. External family reconciliation, independent adjudication/review, criteria/power/resource decisions, and authorization remain blockers.


## Hash-bound provenance conflict review ledger (2026-10-15)

Added a resumable custodian-review ledger to `experiments/diagnosis_evidence_gates.py`. Each note binds to the SHA-256 of one exact saved inventory conflict and requires an opaque reviewer token, rationale, and stable evidence IDs/digests. Stale conflict IDs, duplicate reviews, and malformed evidence are rejected. Local classifications remain reviewer assessments; no review resolves a source conflict. `EXTERNALLY_ATTESTED` is recognized only when a matching attestation is supplied through the external trust-verifier boundary. The CLI accepts an optional `--reviews` JSON input and continues to refuse output overwrite.

Focused evidence-gate tests: **21 passed in 2.78s**. Bounded repository suite: **752 passed, 1 failed, 0 skipped, 2 deselected (755 collected; 232.80s)**. The sole failure remains the unchanged frozen 7/45 diagnosis acceptance; the two finite-shot NO-GO tests remain deselected. No external evidence, ML fitting, dataset generation, experiment, package installation, or hardware job occurred. Backlog: **47 tasks, 32 verified, 6 partial, 7 blocked, 2 deferred**. Inventory completeness, external adjudication, and study readiness remain blocked.


## Simulator and observation-boundary correctness (2026-10-15)

A noise-model review found that non-finite T1/T2 values and gate/edge durations passed the nonnegative validator. `HardwareModel` now requires finite nonnegative values. Its `default_gate_error` and `default_gate_duration` constructor parameters previously had no effect; they are now validated, used when no matching gate-specific calibration exists, and included in calibration snapshots. Tests cover gate skipping, duration-driven T1 relaxation, and snapshot roundtrip.

The statistics boundary also allowed NaN mapping entries through normalization, which could turn TVD into NaN and make the `>` anomaly comparison false. Mapping and sequence inputs now reject negative, Boolean, and non-finite entries; invalid thresholds, colliding stringified keys, and multidimensional arrays are rejected. Max-scaled normalization avoids overflow for very large finite inputs. Diagnosis-level regression confirms malformed observations cannot become a confident no-anomaly result.

Focused command: `.venv/Scripts/python.exe -m pytest tests/test_statistics.py tests/test_noise_model_validation.py tests/test_hardware.py tests/test_simulator.py tests/test_diagnosis_engine.py tests/test_ml_diagnosis_integration.py -q --tb=short --basetemp .t6`; **149 passed in 4.45s**. Final bounded command: `.venv/Scripts/python.exe -m pytest tests -q --tb=short -k "not bounded_recipe_comparison_120_templates_and_finite_shot_subset and not independent_calibration_fixture_and_permutation_comparison" --basetemp .t9`; **768 passed, 1 failed, 0 skipped, 2 deselected (771 collected; 224.48s)**. The only failure is the unchanged frozen 7/45 diagnosis acceptance; both registered finite-shot NO-GO tests remain deselected. No ML fitting, dataset generation, large benchmark, installation, hardware job, or claim update occurred.


### Seeded trace parity check - 2026-10-15

Added a deterministic integration fixture comparing traced and untraced execution with the same seed and hardware model containing default gate error/duration, T1/T2, and per-qubit readout noise. Final statevectors match to 1e-12. `tests/test_traced_hardware_metadata.py`: **6 passed in 0.82s**. Final bounded suite: **768 passed, 1 unchanged frozen 7/45 failure, 0 skipped, 2 NO-GO tests deselected (771 collected; 224.48s)**. This verifies seeded instrumentation parity for the tested path only.


## ML inference contract validation - 2026-10-15

Reject invalid confidence cutoffs (NaN, infinity, booleans, nonnumeric values, and out-of-range values), malformed classifier probability values, and missing classifier feature schemas. Focused suite: 129 passed in 24.10s. Bounded suite: 778 passed, 1 unchanged frozen 7/45 diagnosis failure, 0 skipped, 2 NO-GO deselections (781 collected; 223.55s). No model fitting or evaluation occurred; these are software contract checks.


## Diagnosis evidence gate - 2026-10-15

Removed the inference that a direct circuit-operation mismatch proves a behavioral anomaly when execution distributions are absent. The source-level candidate remains available; anomaly status is unknown and the diagnosis abstains. Focused diagnosis/pipeline tests: 85 passed in 33.18s. Bounded suite: 778 passed, 1 unchanged frozen 7/45 diagnosis failure, 0 skipped, 2 finite-shot NO-GO deselections (781 collected; 224.61s).


## Causal evidence gating - 2026-10-15

Separated source-localization candidates from candidates supported by modeled output impact. Program attribution now requires source-only counterfactual terminal outcomes to differ and the changed qubits to reach measured outputs; measurement mapping changes remain explicit output changes. An unrelated source mismatch plus anomaly abstains, and source plus relevant hardware evidence remains ambiguous. Uncertain candidate reports no longer say PROGRAM FAULT DETECTED. Focused suite: 90 passed in 32.39s. Bounded suite: 783 passed, 1 frozen failure (same 7/45 cases), 0 skipped, 2 NO-GO deselected (786 collected; 225.90s).


Final causal-gating rerun: focused diagnosis/pipeline suite 90 passed in 28.50s. Bounded suite 783 passed, 1 unchanged frozen diagnosis test failure with the same 7/45 mismatches, zero skipped, and two finite-shot NO-GO tests deselected (786 collected; 226.82s). A generic calibration timestamp validity check was tried and removed: the software experiment path intentionally uses synthetic-config, and the check created four regressions. No generic calibration age policy was introduced; IBM preflight retains its separate explicit freshness contract.


## Generic calibration provenance and freshness contract - 2026-10-15

`HardwareModel` no longer substitutes construction time for a missing calibration timestamp. Snapshots preserve the underlying timestamp and its `reported`/`synthetic`/`unknown` provenance separately from snapshot `created_at`; JSON `to_dict()`/`from_dict()` roundtrip this distinction. Reported timestamps are caller-supplied metadata, not independently authenticated. `assess_calibration_freshness()` handles timezone offsets, rejects naive/invalid/future times as current evidence, and reports `FRESH`/`STALE` only under a caller-provided positive age limit. Generic diagnosis excludes reported calibration candidates from causal attribution when that policy does not establish freshness; synthetic fixtures including `synthetic-config` remain eligible as simulator configuration. IBM preflight?s separate 900-second backend contract is unchanged.

Focused hardware/diagnosis/IBM preflight tests: **108 passed**. The requested long temporary path caused one Windows temporary-directory failure in the first final suite attempt; the short-path rerun completed with **793 passed, 1 failed, 0 skipped, 2 deselected (796 collected; 221.97s)**. The sole failure is the unchanged frozen 7/45 diagnosis acceptance; both finite-shot NO-GO tests remained deselected. No training, hardware job, installation, or large experiment occurred. Generic freshness has no default maximum age and cannot authenticate acquisition provenance; no physical-hardware validity claim follows.


## Generic calibration provenance/freshness and bounded follow-up - 2026-10-15

The generic `HardwareModel` freshness contract now distinguishes reported acquisition time, synthetic configuration, unknown legacy provenance, and snapshot creation time. An explicit caller maximum age is required to label a reported calibration fresh or stale; stale/unknown/invalid/future calibration cannot support current hardware attribution under that policy. JSON snapshot roundtrip preserves provenance, and synthetic-config remains allowed. Focused hardware/diagnosis/IBM preflight suite: 109 passed. Final bounded suite: 794 passed, one unchanged frozen 7/45 failure, zero skipped, two finite-shot NO-GO deselections (797 collected; 227.40s). A first longer-basetemp run hit a Windows temporary-directory setup error; the short-path run had none.

Two existing safe backlog items were also revalidated without code changes: two-qubit parameterized-gate canonicalization (8 passed) and bounded Bugs4Q discovery (1 passed, 11 deselected). The latter's default temp-root attempt was denied by Windows; its workspace-local rerun passed. No training, hardware execution, installation, or large benchmark occurred.


Safe discovery-boundary follow-up (2026-10-15): current-tree Bugs4Q conventional-root scan test passed (1 passed, 11 deselected) and QMutBench conventional-root scan test passed (1 passed, 23 deselected). Both implementations were already present; stale backlog priority markers were reconciled. No benchmark data were regenerated or broadly scanned.
