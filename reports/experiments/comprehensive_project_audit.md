# Comprehensive Evidence-Based Project Audit

**Audit date:** 2026-10-14  
**Scope:** Read-only source, test, report, dataset, configuration, and artifact inspection. No tests, simulations, model training, benchmarks, hardware jobs, or full-scale processing were run for this audit.  
**Project state caveat:** The workspace contains a large pre-existing uncommitted change set. This audit did not edit source, existing reports, datasets, or model artifacts.

## Executive summary

The repository has a broad, actively developed state-vector simulator and a diagnosis/research pipeline, with source, focused tests, multiple generations of experiment records, benchmark adapters, and saved ML evaluations. The evidence is uneven: implementation is extensive; unit tests and historical local executions are documented; independent mathematical validation of the current complete implementation is not established. Some gate checks use Qiskit Aer when available, but these are optional test-source references and no current run was performed here. No result establishes physical-device behavior.

The current code has two source-supported correctness defects requiring immediate targeted checks: readout-error flips are reused as the state-collapse condition (`src/quantum_simulator/quantum_simulator.py::measure`), and reset projects away the qubit's one branch and renormalizes rather than implementing a reset channel (`reset_qubit`). For entangled inputs, exact reset of a subsystem generally creates a mixed state, which a pure state-vector representation cannot express without trajectories or a density-matrix representation. These issues mean simulator correctness is not ready to freeze. The hardware model is a simplified stochastic simulator, not a general noise-channel implementation.

Diagnosis logic, abstention, source comparison, calibration scoring, and trace collection are implemented and have extensive focused tests and synthetic experiment records. Their demonstrated accuracy is limited to controlled simulator-generated conditions; several reports identify calibration-only classifications, family concentration, output equivalence, historical report inconsistencies, and incomplete provenance. The binary and multiclass ML work has saved grouped cross-validation results on synthetic scenario data, but no genuine production inference model is integrated. The trained-artifact integration report says the runtime backend is absent; Phase 5.8 says the proposed learner artifact still needs training/export. The ML numbers do not support real-hardware generalization.

**Recommended next action:** Freeze no behavior yet. Add and run a minimal analytical verification set for measurement collapse under readout flips and reset on basis, superposition, and entangled inputs; use the results to define whether the core needs trajectory semantics or density matrices before proceeding to broad simulator validation.

## Architecture and component inventory

| Responsibility | Main implementation | Evidence present | Current assessment |
|---|---|---|---|
| Public API and circuit builder | `src/quantum_simulator/__init__.py`, `quantum_circuit.py`; compatibility modules at root | API tests, README examples, package import test | Implemented; tested in historical reports; current full-tree run unverified |
| Gate definitions and execution | `gates.py`, `quantum_simulator.py`; local matrices for 1–3 qubits, rotations, aliases, barriers, reset, measurement | `tests/test_gates.py`, `test_gate_correctness.py`, `test_simulator.py`; some optional Qiskit Aer oracle code | Implemented; test source exists. Independent validation of all gate conventions/current tree unverified |
| State, indexing, measurement and shots | Complex state vector; bit-string qubit indexing; seeded RNG; repeated circuit sampling in `run_shots` | Simulator and gate tests; saved seeded experiment results; finite-shot calibration fixtures/results | Implemented; mathematical edge cases remain, including defects below; sampling calibration concerns documented |
| Noise and hardware model | `hardware_model.py`, `quantum_simulator.py`; probabilistic gate skips, readout flips, T1/T2-like trajectories, calibration/connectivity metadata | `tests/test_noise_model_validation.py`, hardware tests, synthetic Phase 1/2 and finite-shot records | Implemented as a simplified model; general channel validity and physical calibration fidelity not established |
| Statistics and finite-shot detection | `statistics.py`; TVD/JS/Hellinger/entropy and finite-shot guard; reports also contain permutation comparison | `tests/test_statistics.py`; `reports/experiments/finite_shot_distribution_calibration_*`; calibration run JSON | Implemented; current guard has documented poor control behavior for tested controls; replacement protocol no-go/unresolved |
| Trace and program localization | `fault_localizer.py`, `run_circuit_traced`; alignment, operation differences, trace deviation | localization and trace integration tests; Phase 4 reliability audit | Implemented; trace is simulator-derived structural evidence, not causal proof |
| Hardware diagnosis and abstention | `diagnosis_engine.py`, `hardware_fault_localizer.py`; relevance filtering, supported-operation checks, ambiguity/insufficient evidence | many diagnosis tests; Phase 1/2 saved scenarios and Phase 4 report | Implemented/tested in saved runs; synthetic-only; category can reflect calibration condition even without output anomaly |
| ML feature, labels, split, training/evaluation | `ml_training.py`, `ml_evaluation.py`, `experiments/full_dataset.py`, `qmutbench_ml_*` | `ml_results/grouped_cv_audit_v2/` metrics, OOF predictions, fold assignments, manifest, audit; fallback results | Synthetic grouped evaluation exists; current generalization is limited; no production trained inference artifact integrated |
| ML runtime integration | `ml_diagnosis.py`, `observable_model_adapter.py`, `src/quantum_simulator/models/` | `reports/ml_diagnosis_integration_report.md`; adapter tests; model JSON artifact | Adapter/mock contract implemented; genuine trained runtime model absent per integration report and Phase 5.8 |
| External datasets | `bugs4q_validation.py`, `bugs4q_integration.py`, `qmutbench_validation.py`, `qmutbench_ml_pipeline.py`; `data/` | benchmark result JSON/CSV, QMutBench dataset inspection, smoke manifests | Adapters and data present; successful integration is not diagnosis accuracy; QMutBench labels lack diagnosis-compatible independent truth in cited reports; Bugs4Q material incomplete in prior status |
| External simulator / hardware integration | `ibm_validation.py`, `check_ibm.py` | `reports/ibm_validation_report.json`, IBM adapter tests, live-preflight reports | Adapter and offline fixtures implemented; no live IBM execution in reviewed evidence; integration optional |
| Experiments and saved results | `experiments/`, `reports/experiments/`, `reports/benchmarks/`, `ml_results/` | phase reports, CSV/JSON results, preregistration, calibration manifests | Large; mixed generations and inconsistent/stale reports require artifact-level provenance checks |
| Runtime/dependencies/resources | `pyproject.toml`, `README.md`, `.gitignore` | Python >=3.10; NumPy, pandas, scikit-learn; pytest optional; previous Windows temp/DLL notes | Exponential state-vector storage; full-matrix-like scalar gate loop; OneDrive/Windows temp and sklearn DLL issues noted in existing reports |

Root compatibility files (for example `quantum_simulator.py`, `gates.py`, and `diagnosis_engine.py`) coexist with the canonical `src/quantum_simulator/` package. This duplication raises import-path and drift risk; `tests/test_local_package_import.py` exists, but this audit did not execute it.

## Evidence-strength assessment

The statuses below apply to the current inspected source plus identifiable saved evidence. “Tested” means relevant tests or saved runs/reports exist; it does not mean they were rerun against the current working tree. “Independently validated” requires a distinct analytical derivation or external implementation actually evidenced for the relevant behavior.

| Component | Implemented | Tested evidence | Independent validation | Status / limitation |
|---|---|---|---|---|
| Gate library and state-vector execution | Yes | Gate/simulator test modules; multiple reports cite focused passes; optional Aer-based oracle code in `test_gate_correctness.py` | Partial test-source intent only; no current execution proof, oracle coverage, or pinned environment result reviewed | **UNVERIFIED** for whole implementation; small unit-tested subsets reported. Gate correctness cannot be inferred from successful execution |
| Measurement and shot sampling | Yes | Seed reproducibility and Bell/H sampling tests; seeded experiment artifacts | No full distributional reference validation established | **TESTED**, with source-supported readout-collapse defect; statistical calibration remains unverified |
| Reset | Yes | Reset-related tests exist in simulator tests | No independent channel or entangled-subsystem proof identified | **KNOWN DEFECT/LIMITATION**: source implements projection/postselection-like behavior; pure state-vector model cannot represent general reduced mixed state |
| Noise model | Yes | Noise-model tests and controlled simulated runs | No independent channel implementation comparison or real-device calibration validation established | **KNOWN DEFECT/LIMITATION** as simplified stochastic model; exact semantics require targeted validation |
| Finite-shot anomaly detection | Yes | Unit tests, calibration fixtures and a large saved calibration run | Permutation comparator is a distinct statistical method in test/report artifacts, but current cutoff not independently calibrated | **KNOWN DEFECT/LIMITATION**: report diagnostics show high false-positive rates for matching controls under the current TVD guard; later proposed study marked no-go |
| Program/hardware diagnosis | Yes | Many tests and local synthetic scenario reports | No independent real fault adjudication or physical-hardware comparison | **TESTED** on controlled local cases; accuracy/causal attribution beyond those cases **UNVERIFIED** |
| Diagnosis abstention/equivalence | Yes | Reference/generalization and diagnosis tests; identifiability reports | Analytical equivalence examples in local artifacts, no external truth needed for those specific cases | **TESTED** for selected cases; generalized identifiability **UNVERIFIED** |
| ML evaluation | Yes | Grouped CV artifacts: 270 cases, 30 families, 5-fold family-group split; metrics/OOF outputs and leakage audit | Split methodology is explicit, but test set remains generated by same simulator and deterministic family/intervention process | **TESTED** on synthetic benchmark only; real-world generalization **UNVERIFIED** |
| ML runtime model | Adapter implemented | Mock adapter tests and report | No genuine fitted model in connected runtime | **KNOWN LIMITATION**: production artifact absent; probabilities/calibration not deployment-validated |
| Dataset adapters | Yes | QMutBench/Bugs4Q tests and saved integration results | Ground truth incomplete/incompatible for definitive accuracy on reviewed benchmark snapshots | **TESTED** for parsing/integration; diagnosis validity **UNVERIFIED** |
| IBM integration | Yes, optional | Offline fixture report and adapter test sources | No live run/current external-system evidence | **UNVERIFIED** for live hardware; no hardware claim supported |

Saved ML metrics in `ml_results/grouped_cv_audit_v2/PRODUCTION_FIVE_OBSERVABLES__logistic_regression__metrics.json` report 270 evaluated cases and macro-F1 0.540; its family-grouped split is documented in `evaluation_manifest.json`. The same metrics show weak fault-class recall (0.344 hardware, 0.422 program) despite 0.922 NORMAL recall. These are generated data results, not device performance. The v2 audit warns that expanded schemas using program mismatch or calibration can encode scenario construction shortcuts.

`reports/experiments/phase4_diagnosis_reliability_audit/REPORT.md` documents Phase 1/2 simulated diagnosis outcomes, including a corrected Phase 1 false positive, a stale Phase 1.3 report, concentrated errors by family, and cases where a hardware category occurs without an output anomaly. `reports/experiments/finite_shot_blocker_status.json` labels the proposed 252-family combined study **NO-GO**, citing unresolved novelty, condition definition, ground truth/leakage protocol, and observational equivalence. These reports are evidence about those saved runs/designs, not proof of general simulator correctness.

The existing finalization audit and README refer to historical test counts with differing scopes and provenance. The repository contains changed source and tests, and `git status --short` shows a very large uncommitted set. Therefore none of the historical pass counts is treated as a current-tree test result.

## Known defects versus hypotheses

### Source-supported defects and limitations

1. **Readout error changes collapse branch.** In `QuantumSimulator.measure`, the sampled true measured bits are flipped for readout, then `selected_bits` is replaced with the reported bits and used to select amplitudes for collapse. A classical readout error should change the reported classical result while the quantum state collapses according to the pre-flip measurement outcome. This can corrupt subsequent gates/measurements. This is a direct source-level mathematical mismatch; exact scope should be confirmed with a targeted analytical test.
2. **Reset is not a general reset channel.** `reset_qubit` keeps only basis amplitudes whose target bit is zero and renormalizes; it discards the one branch. For input |1⟩ this fallback sets |0⟩, but for superposition it postselects; for an entangled state it loses the branch mixture/coherence information. This is not the general CPTP reset behavior. A trajectory reset or density-matrix support is required for general semantics.
3. **Readout/gate/coherence noise is a limited model.** Gate calibration error is modeled as skipping a gate; T1/T2 is a per-operation stochastic pure-state evolution; this is not a universal channel simulator. This is an explicit model-scope limitation, not evidence that every implemented branch is mathematically wrong.
4. **Resource growth.** A dense state vector stores (2^n) complex amplitudes (at least 16·(2^n) bytes before overhead). `_apply_local_gate` iterates over every basis index and local output row, so the simple implementation can become costly. No tested qubit ceiling or memory guard is specified in `pyproject.toml`/README.
5. **ML deployment evidence missing.** The integration report says no genuine trained runtime model is integrated; Phase 5.8 calls for training/export. Saved cross-validation must not be interpreted as deployment-ready calibrated probability or generalization.

### Unverified hypotheses requiring targeted checks

- Operand ordering for every non-symmetric two-/three-qubit gate, controlled rotations, QASM-imported circuits, and mixed measurement/classical-bit schemas may differ between APIs or references. Existing tests are evidence for selected gates only; do not generalize without checks.
- State normalization under every branch of reset, noise, partial measurement, and malformed imported calibration merits invariant checks. No broad invariant sweep was run.
- `run_shots` constructs a simulator and executes instructions independently per shot; cost may dominate as shot count/circuit width rises. Measure only after correctness semantics are frozen.
- Feature leakage may remain through artifacts beyond the audited named five-feature schema or through source-family overlap in future data. v2’s manifest helps, but new-family and independent-label validation is still needed.
- Diagnosis categories can conflate evidence of a bad calibration with evidence that it caused an observed output change. Phase 4 explicitly observes hardware categories with `anomaly_detected=false`; consumers need separate interpretation.
- Hash/provenance integrity is mixed: grouped ML results have source fingerprints, while prior benchmark reports/checkpoints include conflicting snapshots or insufficient compatibility metadata.
- Windows OneDrive execution may create temporary-file, path-locking, DLL policy, or large-artifact synchronization problems. Existing reports document pytest temporary-directory access failures and sklearn predictor DLL blocking; they are environmental observations, not universal platform failures.

## Cross-component risks

- **Gate convention and basis order:** The simulator formats basis indices as bit strings and treats qubit index 0 as the leftmost displayed bit; gate matrices encode operand-order assumptions. Validate against analytical basis transitions and an independent implementation for each gate family.
- **Measurement and noise:** Classical readout faults must not change quantum collapse; reset needs explicit trajectory/mixed-state semantics. Until resolved, diagnosis and ML distributions generated through those paths may inherit incorrect labels/features.
- **Finite-shot inference:** The current scalar TVD guard has saved evidence of excessive false positives in matching controls. Shot count, seed, multiple outcomes, and uncertainty must be modeled/calibrated before using anomaly flags as labels or ML inputs.
- **Attribution identifiability:** Distinct program and hardware causes can yield identical observations. A source mismatch or bad calibration is evidence about a candidate condition, not necessarily the cause of observed outputs. Preserve ambiguous/insufficient outcomes.
- **ML data validity:** Family-grouping helps prevent direct family overlap but cannot overcome one deterministic simulator/generator, scenario-label coupling, limited mechanisms, or uncalibrated scores. Synthetic CV is not real-hardware validation.
- **Unsupported operations:** QASM/dataset adapters can parse more than the simulator supports; skipped/unsupported semantics must be surfaced in denominators and never silently treated as normal results.
- **Engineering/provenance:** Root compatibility modules and `src/` package can drift. Historical outputs have inconsistent snapshots; the dirty workspace makes clean reproduction and release tagging essential.

## Prioritized stabilization plan

Detailed costs, dependencies, resource estimates, acceptance criteria, and freeze-blocking fields are machine-readable in `stabilization_work_plan.json`. Estimates are focused human hours, ±50% initially; they exclude unplanned implementation rewrites and external approvals.

| Priority | Work | Why / exit gate |
|---|---|---|
| P0 | Measurement and reset semantics; analytical minimal suite | Correct probability/state semantics are the prerequisite for all downstream evidence. Blocks core freeze. |
| P0 | Gate/indexing oracle suite and state invariants | Establish supported gate conventions, dimensions, unitarity, norm, and selected independent agreement. Blocks core freeze. |
| P0 | Noise-channel semantics and shot-calibration redesign | State model and null behavior must be defensible before diagnosis results can be trusted. Blocks noise/diagnosis freeze. |
| P1 | Trace, localization and attribution validity | Assess target localization, equivalence, abstention and unsupported paths separately from simulator correctness. Blocks diagnosis freeze. |
| P1 | Reproducible ML audit and independent-family holdout | Keep labels, groups and runtime features separated; calibrate/abstain and use untouched generated families. Blocks ML claim freeze. |
| P1 | Benchmark truth/provenance and adapter support matrix | Parsing/integration must not be mistaken for accuracy; resolve dataset/ground-truth gaps. Blocks benchmark claims. |
| P2 | Performance/resource envelope and memory guards | Characterize only after correctness; set explicit supported-size/shot constraints. Blocks release engineering, not core math gate. |
| P2 | Clean Windows/OneDrive reproducibility and artifact manifest | Re-run from a clean pinned environment and commit/hash exact outputs after source freeze. Blocks release gate. |
| P3 | Independent live hardware validation | Only after core and noise model pass; requires backend, calibration provenance, controls and authorization. Blocks hardware claims, not simulator-only release. |
| P3 | Documentation/claims reconciliation | Align README/research/finalization claims with current frozen code and artifact evidence. Blocks public release claims. |

### Initial focused-hour estimate

Approximately **95 focused hours (plausible range 65–145)** across the ordered plan in the companion JSON. This covers targeted correctness work, independent checks, diagnosis/ML protocol, engineering reproducibility, and a scoped hardware-validation design; it does not include waiting for hardware access, implementing a density-matrix backend if required, literature review beyond the existing audit, or large benchmark execution. The range is broad because reset/noise architecture and hardware availability may change scope. Estimated compute for early gates is small: analytical 1–4 qubit cases and short seeded samples on a workstation; broad hardware/benchmark phases are not costed as run-ready until protocol closure.

## Release gates

- **Simulator correctness:** no release freeze until all supported gates pass analytical basis/state tests and independent cross-checks; readout noise leaves post-measurement quantum state correct; reset semantics are explicitly represented and verified for superposed and entangled inputs; norm/probability invariants hold; unsupported operations fail visibly.
- **Diagnosis validity:** simulator core frozen; finite-shot false-positive/false-negative behavior characterized with prespecified controls and uncertainty; attribution evaluated against independent scenario truth; observationally equivalent causes abstain or remain explicitly ambiguous; localization scored independently of anomaly detection.
- **ML validity:** data and artifacts fingerprinted; no family/source leakage; labels are independently declared and input-excluded; untouched families and fault modes evaluated; probability calibration, abstention, baselines, per-class metrics and uncertainty reported; no physical generalization claim absent hardware data.
- **Engineering stability:** clean environment install/import; supported Python/dependency matrix; workspace-local temp handling on Windows; deterministic seeds and provenance; test run recorded against exact commit/tree; resource limits and supported operations documented; root/package import behavior reconciled.

## Single recommended next task

Create and execute the focused analytical measurement/reset verification described in P0-1, beginning with readout flip after measuring |+⟩ followed by a gate, and reset tests on |1⟩, |+⟩, and a Bell state. Record the expected post-measurement or post-reset state/distribution and decide whether reset semantics require trajectories or density matrices. Do not start broad tests or later roadmap experiments before this gate is resolved.

## Inspection limitations

This audit inspected representative core files, test modules, high-value reports, saved ML metrics/manifests, directory inventories, dependency metadata, and git status. It did not read every source/data/artifact file; did not execute Python, tests, simulations, or scripts; did not validate JSON schemas programmatically; did not access the internet or IBM; and did not inspect all full-scale output rows. The deliverables are a bounded evidence map, not a fresh execution validation.
