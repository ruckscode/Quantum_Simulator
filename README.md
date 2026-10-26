# Hardware-Aware Data-Driven Testing and Fault Localization for Quantum Programs

> **Research tracks:** This repository contains (1) a quantum simulator, noise models, and diagnosis research; (2) historical QMutBench mutation-outcome prediction; and (3) future hardware-aware fault-localization work. QMutBench outcome prediction is a separate task from simulator diagnosis and hardware fault localization.

## Start here

- Reusable simulator and diagnosis package: `src/quantum_simulator/` (compatibility entry points remain at the repository root).
- QMutBench feature extraction and inference: `qmutbench_ml/`; reproducible experiment entry points: `experiments/`.
- Automated checks: `tests/`; research reports, metrics, and manifests: `reports/` and `reports/experiments/`.
- Small QASM examples are tracked in `data/qmutbench/`. Full datasets, the selected model bundle, and generated ML intermediates are local-only under `data/` and are excluded by `.gitignore`.

The selected historical QMutBench model is **`qmutbench-mutation-aware-v1`** (100-tree Extra Trees, 23 features, inference threshold 0.5). Its fixed-model grouped development macro F1 is **79.34%**; five-fold nested grouped-selected development macro F1 is **79.28%**. The 85% target remains unmet. These are development results, not independent external validation; no compatible independent external dataset has been verified. See the [final QMutBench research report](reports/experiments/qmutbench_final_research_report_20261026.json) and [inference guide](qmutbench_ml/README.md). The local model bundle must be supplied to run inference; this repository does not retrain it during inference.

## Setup and common checks

From the repository root, install the declared dependencies and run tests:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

The [QMutBench inference guide](qmutbench_ml/README.md) documents single-record and batch prediction. The simulator quick start and focused suites are below. The optional IBM integration and Qiskit oracle dependencies are documented in their sections; neither is needed for core use.

## 1. Project Overview

This project provides a statevector quantum simulator, a circuit-building API, distribution comparison, and an evidence-based diagnosis workflow. It compares expected and observed execution behavior, applies a configured finite-shot tolerance, and combines circuit-level and hardware-calibration evidence to identify a likely program fault or hardware anomaly. When the evidence cannot distinguish plausible causes, the diagnosis can remain ambiguous or report insufficient evidence.

The core package is `quantum_simulator` under `src/quantum_simulator`. IBM Quantum support lives in the separate, optional `ibm_validation.py` integration.

## 2. Problem Statement

Quantum program outcomes can differ from their expected distributions because of program defects, hardware behavior, sampling variation, or incomplete evidence. A distribution difference alone does not identify its cause. This project brings together simulation, statistical comparison, program-operation comparison, and hardware calibration evidence to support testing and fault localization without claiming certainty when the evidence is insufficient.

## 3. Objectives

- Represent and execute supported quantum circuits with the local simulator.
- Compare expected and observed measurement distributions using statistical distance metrics and a finite-shot tolerance.
- Localize differences between expected and observed circuit operations.
- Evaluate hardware calibration evidence relevant to the circuit.
- Preserve evidence and report `NO_ANOMALY`, `PROGRAM_FAULT`, `HARDWARE_ANOMALY`, `AMBIGUOUS`, or `INSUFFICIENT_EVIDENCE` as supported by the available inputs.
- Keep the core simulator independent of IBM credentials, an IBM account, a specific IBM backend, and live IBM connectivity.

## 4. Architecture

```text
Quantum Program
      |
      v
Circuit Representation
      |
      v
Ideal / hardware-aware execution
      |
      v
Statistical comparison
      |
      v
Anomaly detection
      |
      v
Program / hardware localization
      |
      v
Diagnosis
```

`diagnose_circuit()` runs the supplied circuit ideally and, when a `HardwareModel` is supplied, runs it with that model. The resulting distributions and circuit/calibration context are passed into the existing diagnosis components. The returned diagnosis includes evidence and missing-evidence details.

## 5. Core Components

- `QuantumCircuit`: public builder for gates, measurement, and local execution.
- `QuantumSimulator`: statevector execution, supported gate operations, measurement, and shot sampling.
- `HardwareModel`: backend-independent representation of qubit properties, gate calibrations, and connectivity.
- `statistics.py`: distribution and count comparisons, including total variation distance and finite-shot tolerance.
- `fault_localizer.py`: comparison of expected and observed operations and program-fault candidates.
- `hardware_fault_localizer.py`: calibration-backed hardware candidates relevant to the circuit.
- `diagnosis_engine.py`: integrated diagnosis through `diagnose_circuit()` and `diagnose_execution()`.
- `validation_framework.py`, `demo.py`, and benchmark adapters: controlled local validation, demonstration, and dataset reporting.
- `ibm_validation.py`: optional IBM backend discovery, execution, calibration inspection, and calibration conversion.

### Calibration provenance and freshness

`HardwareModel.calibration_timestamp` records a caller-reported calibration acquisition timestamp; it is never filled with the model-construction time. A new model with no timestamp defaults to `synthetic` configuration, while imported legacy snapshots with no provenance default to `unknown`; callers can explicitly set `calibration_timestamp_provenance="unknown"` when acquisition provenance is missing. The provenance values are `reported`, `synthetic`, or `unknown`. `CalibrationSnapshot.timestamp` and `timestamp_provenance` preserve those acquisition fields, while `created_at` records when that snapshot object was created. JSON-compatible snapshots can be round-tripped with `to_dict()` and `from_dict()`.

Generic freshness is caller-policy driven. `assess_calibration_freshness()` computes age only for a reported, timezone-aware timestamp. It returns `FRESH` or `STALE` only when the caller supplies a positive `max_age_seconds`; otherwise it reports that freshness is unassessed. Missing/unknown provenance, invalid or timezone-naive timestamps, and future timestamps do not establish freshness. `synthetic-config` remains a synthetic fixture marker and is not rejected for lacking a real acquisition time. Diagnosis excludes reported calibration candidates from causal attribution when an explicitly supplied age policy does not establish freshness. The generic model has no universal age limit or mechanism to authenticate timestamp provenance.

IBM preflight keeps its separate backend-specific freshness contract and configured age limit. Neither generic software tests nor IBM preflight software checks establish physical-hardware validity.

## 6. Installation

The project requires Python 3.10 or newer. The package metadata declares NumPy, pandas, and scikit-learn; pytest is in the optional development extra. Core statevector execution uses NumPy, while the repository's ML and evaluation tools use the additional declared libraries.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

To install only the declared runtime dependencies from the repository root:

```powershell
python -m pip install -e .
```

Qiskit 2.5 or newer within the 2.x line is an optional test oracle, not a runtime dependency. Install it with the test extra to run the independent gate-reference suite:

```powershell
python -m pip install -e ".[dev,oracle]"
python -m pytest tests/test_gate_reference_qiskit.py -q
```

The ordinary test extra remains `python -m pip install -e ".[dev]"`. Without Qiskit, `tests/test_gate_reference_qiskit.py` is skipped by `pytest.importorskip`; the rest of the routine suite remains available. The core simulator does not require Qiskit, `qiskit-ibm-runtime`, IBM credentials, network access, or an IBM account.

## 7. Basic QuantumCircuit Usage

```python
from quantum_simulator import QuantumCircuit

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

result = circuit.run(shots=1000, seed=42)
print(result.get_counts())
```

`QuantumCircuit` supports the gates and special operations implemented in the package, including single-qubit gates and rotations, `cx`, `cz`, `swap`, `ccx`, `reset`, barriers, and measurement. See `src/quantum_simulator/quantum_circuit.py` for the current builder methods.

## Run a Circuit and Generate Its Diagnosis Report

Use `run_diagnosis.py` when you want to run your own circuit and get a readable diagnosis report plus a machine-readable JSON report. Create an input file such as `my_circuit.json`:

```json
{
  "name": "Bell state",
  "circuit": [
    ["h", 0],
    ["cx", 0, 1],
    ["measure", [0, 1]]
  ],
  "shots": 1000,
  "seed": 42,
  "threshold": 0.05
}
```

From the repository root, run:

```powershell
python run_diagnosis.py my_circuit.json
```

By default this writes `reports/diagnosis_report.json` and `reports/diagnosis_report.txt`. Use `--output path/to/report.json` to choose the JSON destination; the text report uses the same path with a `.txt` extension. You can also provide `"reference_circuit": [...] ` to compare the submitted circuit against a reference. An optional `hardware_model` object accepts `n_qubits`, optional `qubits` entries (for example `{"qubit": 1, "readout_error": 0.1}`), `gates` entries (gate name, qubits, error and duration), and `edges` as qubit pairs or calibration objects.

The report contains the diagnosis category, anomaly status, total-variation distance, effective threshold, program/hardware candidates, evidence, and missing evidence. Without a reference, the circuit is compared against its ideal execution as supported by the API. Without a hardware model, the report cannot attribute an anomaly to physical hardware. This command uses the local simulator only; it does not execute on IBM hardware.

## 8. Running Simulations

`QuantumCircuit.run()` returns a result with a `get_counts()` method. The simulator can also be used directly:

```python
from quantum_simulator import QuantumSimulator

simulator = QuantumSimulator(2, seed=42)
counts = simulator.run_shots(
    [("h", 0), ("cx", 0, 1), ("measure", [0, 1])],
    shots=1000,
)
print(counts)
```

Run the offline demonstration from the repository root with:

```powershell
.\.venv\Scripts\python.exe demo.py
```

The demo uses controlled local scenarios and does not claim its observations came from physical hardware.

## 9. Diagnosis Workflow

The public `diagnose_circuit()` function accepts either a `QuantumCircuit` or a supported instruction sequence. It returns a `DiagnosisResult` with category, anomaly status, statistical result, localization candidates, evidence, and missing evidence.

```python
from quantum_simulator import diagnose_circuit

circuit = [("h", 0), ("cx", 0, 1), ("measure", [0, 1])]
diagnosis = diagnose_circuit(circuit, shots=1000, seed=42, threshold=0.05)

print(diagnosis.category)
print(diagnosis.summary)
print(diagnosis.evidence)
```

When no hardware model is supplied, the ideal distribution is compared against the same ideal execution; hardware calibration evidence is unavailable. Supplying a model enables the model-aware execution and calibration analysis:

```python
from quantum_simulator import diagnose_circuit
from quantum_simulator.hardware_model import HardwareModel

hardware = HardwareModel(2)
hardware.set_qubit_properties(1, readout_error=0.5)
circuit = [("h", 0), ("cx", 0, 1), ("measure", [0, 1])]

diagnosis = diagnose_circuit(
    circuit,
    hardware_model=hardware,
    shots=1000,
    seed=42,
)
print(diagnosis.category)
```

The finite-shot tolerance is applied by the statistical comparison. An anomaly can be detected without enough evidence to attribute it to a specific cause.

## 10. Program Fault Localization

The program localizer compares expected and observed instruction sequences and can report operation-level candidates, such as changed, inserted, or removed operations. The integrated `diagnose_execution()` workflow accepts expected and observed distributions and optional expected and observed circuits:

```python
from quantum_simulator.diagnosis_engine import diagnose_execution

result = diagnose_execution(
    expected_distribution={"0": 0.0, "1": 1.0},
    observed_distribution={"0": 1.0, "1": 0.0},
    expected_circuit=[("x", 0), ("measure", 0)],
    observed_circuit=[("h", 0), ("measure", 0)],
    shots=1000,
)
print(result.category)
print(result.program_evidence_candidates)
```

This lower-level workflow accepts supplied execution distributions. The high-level `diagnose_circuit()` API treats `circuit` as the submitted/observed program and accepts an optional `reference_circuit`:

```python
from quantum_simulator import diagnose_circuit

result = diagnose_circuit(
    submitted_circuit,
    reference_circuit=expected_circuit,
    hardware_model=hardware,
    shots=1000,
    seed=42,
)
```

With a reference, the expected distribution is sampled from the reference without a hardware model; the submitted circuit is sampled with the supplied model, if any. The model is also passed to diagnosis as calibration evidence. The two circuits must use compatible measurement-operation structures, output widths, and classical-bit destinations. Without a reference, the existing same-circuit ideal-versus-hardware behavior is preserved. A source-level mismatch can be localized even when measured distributions agree; that is not evidence the mismatch changes behavior. A defect that leaves all supplied program, measurement, and calibration observations indistinguishable cannot be identified by this comparison.

## 11. Hardware Fault Localization

Hardware localization evaluates calibration metrics and connectivity in relation to the supplied circuit. The integrated diagnosis can report relevant qubit, edge, or gate candidates when calibration evidence crosses the configured threshold. Calibration evidence alone does not guarantee that a hardware anomaly caused an observed difference; the diagnosis may be ambiguous or report insufficient evidence.

## 12. HardwareModel

`HardwareModel` is the simulator’s generic hardware representation. It stores qubit count, coupling information, backend name and calibration timestamp, qubit T1/T2 and readout error, and available gate/edge properties. It has no IBM dependency. Values left at model defaults are generic model values, not measurements of a real backend. `default_gate_error` and `default_gate_duration` apply when no matching gate-specific calibration exists and are preserved in calibration snapshots; time constants and durations must be finite and non-negative.

```python
from quantum_simulator.hardware_model import HardwareModel

hardware = HardwareModel(
    2,
    coupling_map=[(0, 1)],
    backend_name="example-backend",
)
hardware.set_qubit_properties(1, readout_error=0.02, t1=120.0, t2=80.0)
hardware.add_gate_calibration("cx", qubits=(0, 1), error=0.01, duration=0.2)
```

The units supplied to `HardwareModel` are used as values; callers should keep units consistent with the calibration data they provide.

The simulator uses a limited stochastic trajectory model: `readout_error` flips reported classical bits symmetrically after physical state collapse; a calibrated gate `error` is the probability of skipping that gate; T1 applies amplitude-damping jumps, and T2 applies the remaining phase-flip dephasing after the T1 coherence contribution. These trajectories reproduce the corresponding channel behavior across repeated seeded runs. They do not implement depolarizing noise or arbitrary Kraus channels, and they are not a general physical noise model or evidence of real-device fidelity.

Statevector memory grows exponentially: a complex128 state stores 16*2^n bytes for n qubits. The simulator estimates three statevector-sized blocks for its principal arrays (state, output and measurement temporaries), excluding Python, NumPy allocator and other process overhead. Callers can inspect QuantumSimulator.estimate_working_array_bytes(n) or pass memory_budget_bytes to QuantumSimulator or QuantumCircuit.run() to reject a request before allocation when that modeled estimate exceeds their chosen budget. The budget is opt-in and is not a guarantee on total process memory; there is no universal configured qubit ceiling. A bounded 1-18-qubit smoke profile is recorded in reports/experiments/resource_envelope_profile_20261014_rerun.json; fluctuating host availability, sampled RSS and limited workloads do not establish a safe machine-wide maximum.

## 13. Optional IBM Quantum Integration

IBM Quantum is an **optional external integration**. The core simulator remains usable without IBM credentials, an IBM account, a particular IBM backend, live backend access, or an internet connection.

When IBM integration is used, the caller explicitly selects a backend and authenticates through external IBM configuration or credentials. The integration retrieves calibration information dynamically from that selected backend, can submit an IBM execution, and can analyze returned results with the project’s statistics and diagnosis components. The calibration capture can also be converted into a `HardwareModel`. Calibration values and available fields can change over time; some backend fields may be unavailable.

The adapter is in `ibm_validation.py`. Its offline validation path uses local fixtures and does not submit an IBM job. The optional IBM dependency set is installed with `python -m pip install -e ".[ibm]"` in an isolated environment; it pins the locally fixture-tested Qiskit 2.5 / Runtime 0.50 compatibility line. Run offline adapter tests with `python -m pytest tests/test_ibm_validation.py -q`; these do not authenticate or contact IBM. Live integration requires valid external access, network connectivity, and an explicitly selected backend. No backend is a permanent or default project configuration. Before submission, the adapter requires a current calibration timestamp and a usable logical-to-physical transpiler map; unresolved routing or stale/missing calibration blocks submission. Those checks have fixture coverage only, not live backend validation.

The repository contains an IBM validation report for an offline fixture. It is not a live hardware result. Any earlier backend experiment, including one performed on `ibm_marrakesh`, is a historical validation snapshot only and is not needed to run the project.

## 14. Bugs4Q / QMutBench Validation

The repository includes separate adapters for Bugs4Q and QMutBench data. They normalize supported dataset inputs, run supported cases through project functionality, and produce benchmark-specific reports. Mutation metadata is not used as a diagnosis label. Benchmark metrics are not claimed without suitable independent, reliable ground truth.

The checked-in QMutBench data contains a small set of QASM source/mutant examples. The Bugs4Q adapter and integration tests are present; external/official dataset material may not be available in every checkout. Synthetic test fixtures and controlled local scenarios are not benchmark ground truth.

### QMutBench outcome ML experiment

The selected model is `qmutbench-mutation-aware-v1` (100-tree Extra Trees, 23 features, threshold 0.5), documented in the [inference guide](qmutbench_ml/README.md). Current results are 79.34% fixed-model grouped development macro F1 and 79.28% five-fold nested grouped-selected development macro F1. The 85% target remains unmet, and no compatible independent external dataset has been verified. These metrics are development evidence for predicting the historical QMutBench mutation outcome only; they do not establish hardware diagnosis, source fault localization, or independent external generalization. The model bundle and full training data are local-only under `data/qmutbench_ml/mutation_aware_v1/` and are not included in Git. Older model comparisons and results remain in the linked historical reports and experiment records.

Run benchmark reporting from the repository root with:

```powershell
.\.venv\Scripts\python.exe -m benchmark_validation --root .
```

## 15. Validation Results

The repository contains historical full-suite counts from different runs (including 280, 346, and 358 passed). Their source revisions and scopes are not recorded consistently, so none is presented here as current. The 2026-10-14 bounded primary-environment checkpoint collected 651 tests and completed with 648 passed, one frozen held-out diagnosis acceptance failure, and two finite-shot calibration comparison tests deselected under the current NO-GO/resource policy. The latest exact-current-tree result is recorded in the addendum below. The suite uses a short workspace-local pytest base directory because deeply nested temporary paths can exceed Windows path limits. The latest isolated clean-environment result is recorded immediately below.

The post-push review on 2026-10-25 reconfirmed that the frozen seven-case held-out acceptance remains failing while its forensic companion passes. The current diagnosis abstains on all seven: source-level targets are localized for the three sub-threshold QFT-like mutations, while the four ring cases are finite-shot false alarms with no supported cause attribution. The review also corrects a stale historical statement that equivalent-ring syntactic candidates caused `PROGRAM_FAULT`; current diagnosis requires behavioral counterfactual evidence. See the [dated forensic addendum](reports/experiments/postpush_heldout_forensic_20261025.md) and the [final research report erratum](reports/FINAL_RESEARCH_REPORT.md#16-post-push-forensic-erratum-2026-10-25). Existing calibration comparisons remain exploratory, single-recipe synthetic evidence; they do not establish cross-family or hardware performance.

Saved benchmark artifacts contain distinct scopes and conflicting snapshots. The historical combined validation report records 8 selected Bugs4Q cases as 3 `NO_ANOMALY` and 5 `PROGRAM_FAULT`; the separate focused integration JSON lists the same eight case IDs as 2 and 6, and generation metadata does not establish which code snapshot explains the difference. A separate full-source report records 42 Bugs4Q cases (41 parsed, 19 translated/executed/diagnosed, 22 unsupported, and 1 parsing error); its diagnoses are not accuracy labels. For QMutBench, the pilot validation report records 12 cases (11 supported, executed, and diagnosed; 1 unsupported; all 12 missing diagnosis-compatible ground truth). The separate full-results report and historical checkpoint record all 12 as supported and diagnosed, including the `rxx` case that the pilot validation report marks unsupported. The old checkpoint has no compatibility metadata, so that disagreement cannot be attributed to a known runner or data version. The current runner reuses cached rows only when checkpoint metadata matches, including the schema, result-producing source files, dataset cases, execution settings, and runtime versions; its checkpoint records which case IDs were reused or executed fresh. These are integration results, not accuracy or fault-attribution guarantees. The reports do not establish benchmark accuracy, precision, recall, F1, or mutation score.

ML fitting is gated by `python ml_training.py --preflight-only`. The preflight requires dataset generation source/runtime identity, matching per-report source hashes, and a circuit lineage ID; outer folds group by lineage so related recipe siblings cannot cross partitions. The historical 270-row dataset lacks generation provenance, per-report source hashes, and lineage, so the read-only preflight inspects and rejects all rows before fitting or training-output writes. Its synthetic scenario labels are not independent truth. The saved evaluation had six of nine source hashes different at the release-final checkpoint; a later lineage-aware current-tree audit records seven of nine. Historical grouped-CV results remain preserved but were not rerun under the current provenance and lineage requirements, so they are not current-source or independent validation. See the [ML data foundation audit](reports/experiments/ml_data_foundation_audit_20261014.md), [current preflight](reports/experiments/ml_data_foundation_preflight_20261014.json), and [latest source compatibility snapshot](reports/experiments/ml_source_compatibility_audit_20261014_lineage_final.json).

A separate bounded ML pipeline pilot is available through `python -m experiments.ml_data_pilot --output-dir <new-empty-directory>`. The preserved current pilot at [`ml_data_foundation_pilot_20261014_final`](reports/experiments/ml_data_foundation_pilot_20261014_final/pilot_manifest.json) contains nine 32-shot local-simulator cases from three distinct recipe lineages, writes observation predictors and synthetic scenario labels to separate artifacts, records source/config/runtime/seed/lineage and Git checkout identity, and refuses to overwrite a nonempty destination. `python -m pytest tests/test_ml_data_pilot.py -q` checks integrity and leakage rejection. The pilot is deliberately marked synthetic, unresolved for independent attribution, and ineligible for fitting. The review draft [`ml_independent_evaluation_protocol_20261014.md`](reports/experiments/ml_independent_evaluation_protocol_20261014.md) defines the family-level sampling, adjudication, partitions, metrics and power inputs needed before independent evaluation; it makes no numeric performance or power claim.

### Replay-derived trace comparison

`replay_trace_features.py` replays each completed report in `reports/experiments/full_dataset/` with its recorded seed and hardware configuration, then writes a replay-derived feature table and provenance audit under `reports/experiments/replay_derived_trace_features/`. Reproduce it with `python replay_trace_features.py`; compare the five trace features with the five distribution-summary predictors using `python ml_replay_trace_experiment.py`. The features are maximum, mean, and median per-operation probability difference, the largest deviation's share of total deviation, and the fraction of operations above the fixed 0.05 threshold. Empty, malformed, non-finite, or materially out-of-range traces fail with an explicit error; values within `1e-12` of the theoretical `[0, 1]` range are clamped for floating-point roundoff.

The experiment uses five-fold `StratifiedGroupKFold` splits grouped by `circuit_family_id`, and reports accuracy, macro-F1, pooled confusion matrices, and per-class recall. `ground_truth_label` is the target only; run, family, repetition, and seed metadata are used for joining or grouping. These are fresh simulation replays, not original-run traces or physical-hardware measurements. The labels describe synthetic scenarios, so the results are descriptive and do not establish independent fault-attribution performance. The input report tree is a local research dataset and is not included in the repository; its report paths and hashes are recorded in the generated feature table.

An isolated Python 3.12.10 install from the declared project and dev dependencies was also checked on 2026-10-14. Its bounded routine suite completed with 604 passed, 1 failed, 4 skipped, and 2 deselected; the failure is the same frozen 7-of-45 held-out diagnosis acceptance failure described below. That environment was created without the now-declared optional `oracle` extra, so it did not collect Qiskit oracle cases. A clean-environment gate subset passed 37 tests with 4 optional skips; the environment was not modified. The IBM fixture suite plus both oracle/IBM metadata checks also passed 45 tests in that environment without installing optional packages. See `reports/experiments/clean_environment_validation_20261014.json` for the original suite inventory and command.

## 16. Project Structure

```text
src/quantum_simulator/    packaged simulator, circuit, hardware, statistics, diagnosis
tests/                    simulator, diagnosis, integration, and validation tests
experiments/              controlled local experiment configurations and runner
reports/                  historical demo, experiment, benchmark, and IBM adapter reports
data/qmutbench/           included QMutBench QASM examples
demo.py                   offline diagnostic demonstration
ibm_validation.py          optional IBM integration and offline validation
benchmark_validation.py   combined benchmark report runner
bugs4q_validation.py      Bugs4Q dataset adapter
qmutbench_validation.py  QMutBench dataset adapter
```

Top-level compatibility modules are retained alongside the `src/quantum_simulator` package for repository-local imports and existing tools.

## 17. Current Limitations

- Hardware-aware execution and localization are limited to the behavior implemented by `HardwareModel` and the simulator; this is not a general physical noise simulator.
- IBM integration depends on external IBM access when used. Backend availability and calibration data can change, and some calibration fields may be unavailable.
- A detected distribution difference does not by itself establish a cause. Diagnosis can remain ambiguous or report insufficient evidence when multiple causes are plausible or evidence is incomplete.
- `diagnose_circuit()` accepts an optional `reference_circuit`. Its outcomes remain governed by finite-shot statistical evidence, so a direct source mismatch can be localized while cause classification abstains when measured behavior does not support it.
- The benchmark adapters do not claim accuracy, precision, recall, F1, mutation score, or guaranteed fault attribution without independent reliable ground truth.

## 18. Future Work

Potential future work includes extending hardware execution/noise modeling and evaluating localization against independent benchmark ground truth. These are future possibilities, not current capabilities.

## 19. Testing

Run the bounded suite from the repository root with a short workspace-local pytest base directory. The finite-shot 120-template recipe comparison and the independent finite-shot permutation comparison are excluded while the registered calibration study remains NO-GO:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --tb=short -k "not bounded_recipe_comparison_120_templates_and_finite_shot_subset and not independent_calibration_fixture_and_permutation_comparison" --basetemp .pytest-tmp-final
```

The 2026-10-14 run completed with 648 passed, one frozen held-out acceptance failure, and two deselected. The excluded tests execute additional finite-shot calibration comparisons; running them requires a separately justified study while the registered calibration design remains NO-GO.

If the environment’s default pytest temporary directory is inaccessible, configure pytest to use a writable temporary directory. For IBM adapter fixtures, use an isolated environment and `python -m pip install -e ".[ibm,dev]"`, then run `python -m pytest tests/test_ibm_validation.py -q`; these tests do not require credentials or a live IBM job.

## 20. License / Repository Information

No license declaration or repository URL is provided in the project metadata, so this README does not state a license or external repository address.

## Current Project Status

Current status as of 2026-10-15: the frozen held-out diagnosis acceptance still differs in 7 of 45 detection/classification/abstention outcomes; source and hardware localization target checks pass. Exact-versus-sampled TVD analysis found that three expected positive cases have exact TVD below the unchanged 0.05 threshold, while four exact-equality ring controls exceed the unchanged 0.0625 finite-shot guard at their registered seeds. The failure is retained; thresholds and held-out expectations were not retuned. The Qiskit gate oracle is available through the optional `oracle` extra and passes in the primary environment; clean environments without it skip those checks. Two finite-shot comparison tests remain deselected under the registered NO-GO decision. The principal-array memory budget is opt-in and does not guarantee total process memory safety. A Phase 5 F1/F2 five-feature development artifact exists, but its probabilities are uncalibrated and held-out generalization/deployment validation is not established; it is not loaded by default. An earlier 270-row grouped evaluation matched its historical OOF file on the same synthetic rows, but it predates the strict provenance and lineage guard and is not current-source or independent evidence. The current data foundation audit rejects all 270 historical rows before fitting. Historical benchmark snapshots have unresolved provenance, and there is no live hardware validation. See the [ML data foundation audit](reports/experiments/ml_data_foundation_audit_20261014.md), [current claim reconciliation](reports/experiments/project_claims_reconciliation_20261014.md), and [project completion status](reports/experiments/project_wide_completion_report.md).

**2026-10-15 post-pilot validation:** focused simulator, gate, measurement/reset, noise, circuit, recovery, data-provenance, ML pilot/preflight, and optional-adapter regression passed 344 tests in 69.10s. The bounded project `.venv` suite collected 663 tests: 660 passed, one failed, zero skipped, and two campaign-like finite-shot cases were deselected (168.65s). The only failure remains the unchanged frozen 7/45 held-out diagnosis acceptance. The existing nine-row ML pilot is provenance/structure valid but its labels are synthetic and it is blocked from model fitting; no ML generalization claim is supported. No new held-out forensic analysis or tuning was performed. See the current status, completion report and claims reconciliation for the complete limits.


**2026-10-15 diagnosis research protocol:** a development-only, evidence-separated evaluation design and fail-closed case-record validator are documented in [the existing diagnosis feasibility design](reports/experiments/diagnosis_method_feasibility_design_20261014.md). Focused protocol/pilot/lineage/artifact tests passed 47 cases in 35.93s. The last full repository suite predates this protocol-only addition and remains 660 passed, one frozen 7/45 diagnosis acceptance failure, zero skipped, and two finite-shot comparisons deselected (663 collected). No detector study or research-performance claim is made.


**2026-10-15 ML safeguards update:** the current bounded suite collected 712 tests: 709 passed, one failed, zero skipped, and two NO-GO finite-shot cases were deselected (217.11s). The unchanged failure is the frozen 7/45 diagnosis acceptance. Focused current ML tests passed 40/40. Strict provenance preflight now guards fallback training and model export; the historical 270-row dataset remains ineligible (zero eligible rows), and no model was trained. The primary five observable predictors are documented separately from calibration-only inputs and the diagnostic `gate_mismatch_fraction` ablation in [the feature eligibility review](reports/experiments/ml_feature_eligibility_20261015.md). This supports data-integrity safeguards only, not ML generalization or deployment readiness.


**2026-10-15 ML recovery checkpoint:** current bounded tests collected 722: 719 passed, one frozen diagnosis acceptance failure (7/45), zero skipped, and two NO-GO finite-shot tests deselected (216.10s). Focused ML recovery/evaluation tests passed 41/41. A row-level audit confirms all 270 historical reports lack per-run source hashes and recipe ancestry; the dataset manifest lacks generation provenance, and there are zero independent adjudications. No model was fit. New comparison infrastructure includes strict registered feature schemas, family/lineage train-calibration-test validation, dummy baselines, fixed model comparators, family-cluster intervals, synthetic-only report scope and non-overwriting v3 outputs. See [the recovery and generalization checkpoint](reports/experiments/ml_generalization_recovery_checkpoint_20261015.md). ML generalization remains unestablished.


**Final 2026-10-15 ML checkpoint:** the record-level recovery table v2 classifies each row field as recorded, inferred, recomputed from current bytes, or missing; the prior table is preserved. Focused tests: 41 passed in 15.39s. Final bounded repository suite: 719 passed, 1 unchanged frozen diagnosis failure at 7/45, 0 skipped and 2 deselected (722 collected; 219.00s). Backlog: 42 tasks (28 verified, 6 partial, 6 blocked, 2 deferred). No ML model was fit; independent-family data and adjudicated truth remain unavailable.


**Prior ML integration checkpoint (2026-10-15):** the training path remained synthetic-label only, and independent adjudication ingestion was blocked pending an approved schema and independent records (P1-4). At that checkpoint the backlog had 43 tasks; the latest reconciled total is recorded below.

**2026-10-15 ML evaluation and diagnosis update:** a family-leaking evaluation fold is now rejected before model construction or fitting. Diagnosis retains source localization but abstains on behavioral program-fault attribution when observed distributions match; an injected classifier cannot convert that source diff into behavioral proof. Focused tests passed 92/92. The bounded suite collected 725 tests: 722 passed, the unchanged frozen 7/45 diagnosis acceptance failed, zero skipped, and two finite-shot NO-GO tests were deselected. The canonical backlog is 44 tasks (29 verified, 6 partial, 7 blocked, 2 deferred). No model was trained, no calibration/generalization result is available, and the independent-data/review blockers remain.

**2026-10-15 independent diagnosis protocol package:** `reports/experiments/independent_diagnosis_protocol_package_20261015.json` lists the population, lineage, provenance, truth, partition, metric, power, resource, and authorization inputs still required. The protocol validator checks complete record structure and rejects self-attested approval fields; it cannot verify an external reviewer signature or establish study approval. Focused protocol/candidate/preflight tests passed 54 tests. Confirmatory diagnosis research remains blocked, and the frozen 7/45 acceptance failure and two finite-shot NO-GO deselections are unchanged.


**Final 2026-10-15 protocol-package validation:** focused protocol/candidate/preflight tests passed 54/54. The bounded repository suite collected 733 tests: 730 passed, one unchanged frozen 7/45 diagnosis acceptance failed, zero skipped, and two NO-GO finite-shot tests were deselected (222.36s; short base directory). One longer-path run had a Windows temporary-path failure; the recovery test passed under the short path used for the final run. The protocol package remains unapproved and blocked on independent evidence and review.


**2026-10-15 provenance gate update:** `experiments/diagnosis_evidence_gates.py` produces the bounded evidence-linked reconciliation at `reports/experiments/diagnosis_sampling_frame_reconciliation_20261015.json` and supplies the confirmatory preflight. Current saved evidence has six candidate roots, unresolved lineage/fingerprint conflicts, 5,186 unknown-lineage circuit records, 270 synthetic historical rows without recipe ancestry or per-run source hashes, and zero eligible independent families. Adjudication requires opaque reviewer tokens and externally verified protocol-bound blinding/independence attestations; the local validator cannot authenticate those attestations. Focused tests passed 64. The study remains blocked and no external review or adjudication is claimed.


## Final bounded validation - provenance/adjudication phase (2026-10-15)

Focused reconciliation, adjudication, study-preflight, and protocol tests: **64 passed in 3.32s**. Final bounded repository suite: **747 passed, 1 failed, 0 skipped, 2 deselected (750 collected; 226.33s)**. The sole failure is the unchanged frozen diagnosis acceptance at 7/45; both finite-shot NO-GO tests remained deselected. No ML fitting, dataset regeneration, experiment, package installation, or hardware job occurred. The canonical backlog now contains **46 tasks: 31 verified, 6 partial, 7 blocked, 2 deferred**. External family reconciliation, independent adjudication/review, criteria/power/resource decisions, and authorization remain blockers.


**2026-10-15 provenance review tooling:** inventory conflicts can now be reviewed individually through a deterministic, hash-bound ledger with opaque reviewer tokens, rationales, and evidence references. Local review notes do not resolve conflicts; external verification remains required for externally attested status. Focused tests passed 21/21. The bounded suite collected 755 tests: 752 passed, the unchanged frozen diagnosis acceptance failed at 7/45, and two finite-shot NO-GO tests remained deselected. No scientific performance claim changed; see the current completion report and stabilization journal.


**2026-10-15 simulator correctness update:** hardware calibration time constants/durations reject NaN and infinity; configured default gate error/duration now affect simulation and round-trip through calibration snapshots. Distribution inputs reject negative/non-finite values and invalid thresholds; normalization is stable for very large finite values, and malformed observation shapes fail explicitly. Focused suite passed 149 tests. Final bounded suite collected 771 tests: 768 passed, the unchanged frozen diagnosis acceptance failed at 7/45, and two finite-shot NO-GO tests remained deselected. A seeded traced-versus-untraced hardware execution fixture also matches final states with gate, relaxation, and readout noise. These are deterministic software-correctness checks, not scientific performance or physical-hardware evidence.


2026-10-15 ML inference contract: classifier output validation rejects non-finite or malformed confidence/probability values and classifiers without the approved feature schema. Focused integrated tests passed 129; the bounded suite passed 778 with the unchanged frozen 7/45 diagnosis failure and both finite-shot NO-GO tests deselected (781 collected). This verifies software contract handling only; it does not establish ML accuracy, calibration, or generalization.


2026-10-15 diagnosis evidence safeguard: absent expected/observed distributions, direct source mismatches remain localizable but no longer produce behavioral anomaly or program-fault attribution; diagnosis abstains. Focused suite passed 85 tests. The bounded suite passed 778, with the unchanged frozen 7/45 failure and two finite-shot NO-GO tests deselected (781 collected).


2026-10-15 causal evidence gate: source localization is now reported separately from modeled measured-output impact. Program attribution requires the source-only counterfactual to change outcomes on measured/reachable qubits; unrelated source mismatches abstain, and simultaneous program/hardware evidence remains ambiguous. The focused suite passed 90 tests; bounded suite passed 783 with the unchanged frozen 7/45 failure and two finite-shot NO-GO tests deselected (786 collected).


Final 2026-10-15 causal-gating validation: focused diagnosis/pipeline suite passed 90 tests. Bounded suite passed 783, with the unchanged frozen 7/45 diagnosis failure and two finite-shot NO-GO tests deselected (786 collected). Generic HardwareModel calibration freshness has no defined max-age policy; this does not support physical-hardware validity claims.


## Calibration provenance validation - 2026-10-15

The generic calibration contract distinguishes caller-reported acquisition time from snapshot creation time, preserves `reported`/`synthetic`/`unknown` provenance through JSON snapshots, and requires a caller-supplied maximum age before reporting freshness. Diagnosis excludes reported calibration candidates from causal attribution if an explicit policy cannot establish freshness; unknown and invalid timestamps do not become current evidence. Synthetic simulator configurations, including `synthetic-config`, remain valid fixtures. Focused hardware/diagnosis/IBM preflight tests passed 109; the final bounded suite passed 794 with the unchanged frozen 7/45 failure, zero skipped, and both finite-shot NO-GO tests deselected (797 collected). No physical-hardware validity claim follows.

## Finalization report

The authoritative current-tree research and software finalization report is [final_research_report.md](final_research_report.md). It records the bounded validation result, frozen 7/45 failure and case explanations, claim boundaries, evidence limitations, and release disposition. The separate [independent research handoff](reports/independent_research_handoff_20261015.md) defines a blocked, planning-only follow-up protocol and its provenance gates. Historical status entries above remain dated snapshots; use the report's finalization section for the current milestone verdict.


## Synthetic-by-default and optional live IBM calibration

The diagnosis runner now uses synthetic hardware calibration automatically when no IBM Quantum token is configured. If a token is present, `hardware_source: "auto"` fetches live calibration metadata from the selected IBM backend. The circuit itself is still executed locally by this simulator; this does **not** submit jobs to IBM hardware.

Install the optional provider in Colab:

```python
%pip install qiskit-ibm-runtime
```

Set the token as a secret/environment variable; do not put it in a JSON file or commit it:

```python
import os
from getpass import getpass
os.environ["QISKIT_IBM_TOKEN"] = getpass("IBM Quantum API token: ")
os.environ["QISKIT_IBM_BACKEND"] = "ibm_kingston"  # choose a backend available to your account
# Optional for accounts that require an instance:
# os.environ["QISKIT_IBM_INSTANCE"] = "your-instance-crn"
```

Input JSON can omit hardware settings entirely for automatic mode. With no token, the runner creates repeatable synthetic calibration. With a token, it attempts live IBM calibration and reports an error if authentication/backend lookup fails rather than silently labelling synthetic data as real.

To force synthetic data even if a token is configured, add `"hardware_source": "synthetic"`. To require real IBM data, use `"hardware_source": "ibm"`.

Example circuit input:

```json
{
  "name": "Bell hardware-noise diagnostic",
  "hardware_source": "auto",
  "backend_name": "ibm_kingston",
  "circuit": [
    ["h", 0],
    ["cx", 0, 1],
    ["measure", [0, 1]]
  ],
  "reference_circuit": [
    ["h", 0],
    ["cx", 0, 1],
    ["measure", [0, 1]]
  ],
  "shots": 1000,
  "seed": 42,
  "threshold": 0.05
}
```

The imported real calibration is limited to the circuit-sized prefix of the backend, with logical qubit `q` mapped to physical qubit `q` (0-based). Review this mapping before interpreting results; remapping and hardware-aware transpilation are not performed. Synthetic calibration is labelled as synthetic and must not be presented as a measurement of a real backend.
