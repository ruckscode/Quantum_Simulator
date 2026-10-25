# Phase 3 Hardware Validation Readiness

## Readiness decision

**Real-hardware validation is not ready to produce defensible qubit-level diagnosis results.** The repository contains an explicit IBM Runtime adapter and calibration parser, but the checked-in validation report and tests provide offline or injected-fixture evidence only. The execution path does not carry the transpiler's logical-to-physical layout into diagnosis, does not reject stale calibration, and does not verify that the calibration snapshot corresponds closely enough to the execution. No job was submitted for this audit.

The next step is a code-and-fixture readiness pass for layout-aware mapping and calibration freshness, followed by a read-only backend calibration capture. Actual execution requires separate explicit authorization after those gates pass.

## Integration audit

| Area | What the code does | Evidence and status |
|---|---|---|
| Backend access | `ibm_validation.py::inspect_ibm_backend_calibration` asks a supplied or Runtime service for the named backend. `execute_on_ibm_backend` requires an explicit token and backend name, obtains a backend, captures calibration, transpiles, submits with Runtime `Sampler`, and parses counts. | Implemented, but no live backend retrieval/execution was performed. `reports/ibm_validation_report.json` says `OFFLINE_TESTED`, origin `OFFLINE_FIXTURE`, capture source `OFFLINE_FIXTURE`, `real_hardware_executed: false`, and no job ID. |
| Calibration fields | `capture_backend_calibration` reads backend name, qubit count, coupling map, status, `properties().last_update_date`, per-qubit T1/T2/readout error, and gate error/duration parameters. Absent values are listed in `unavailable_fields` and represented as null. `hardware_model_from_ibm_calibration` retains the captured calibration timestamp and maps the captured indices and calibrations into `HardwareModel`. | Field extraction and unit conversion have fixture tests. `retrieval_status="AVAILABLE"` only requires a properties object and qubit count; it does not mean every field is present. Consumers must inspect `unavailable_fields` and `errors`. |
| Physical/logical mapping | Calibration indices and coupling edges are loaded using backend qubit indices. Transpilation records operation counts, depth, text, and gate-name mapping notes. | **Blocker:** the transpiled layout is not serialized in `transpilation_info` or applied to the circuit passed to diagnosis. Diagnosis receives the original plan's logical qubit indices with the backend-indexed calibration model. Thus a localized logical qubit/coupling can be compared with the wrong physical calibration entry after routing/layout. |
| Execution/result parsing | The adapter extracts `sampler_result[0].data.c.get_counts()`, converts counts to probabilities, and includes shots, job ID, and a result timestamp. | Implemented but only fixture/offline paths were exercised here. Bit/classical-register interpretation, layout remapping, and real result compatibility remain unverified against a live backend. |
| Snapshot/execution association | Calibration `timestamp` comes from backend properties `last_update_date`; the result stores this snapshot alongside job ID and `execution_timestamp`. | Partial. `execution_timestamp` is assigned when `_make_result` is built after execution, not as a before/after execution interval. No maximum snapshot age is checked, and the execution record does not prove calibration was fresh at submission. The internal `HardwareModel` created by `capture_backend_calibration` is stamped with local capture time, whereas the serialized capture timestamp is the backend properties update time. |
| Missing/stale/unavailable data | Missing calibration metrics are explicit nulls/unavailable fields; absent Runtime dependencies/credentials block execution; exceptions return structured failure states. | Missing data are not numerically fabricated by the converter. Staleness is not distinguished from current calibration and does not force uncertainty. Missing calibration fields can coexist with `retrieval_status="AVAILABLE"`; diagnosis must not interpret that status as complete calibration. |
| Calibration vs observed anomaly | Diagnosis receives ideal and measured distributions plus a calibration-backed hardware model. It stores statistical anomaly fields separately from hardware candidates. | The Phase 2 RZZ cases demonstrate calibration-backed hardware suspicion despite no distribution detector flag. This is simulator evidence only. On hardware, discrepancies could also reflect mapping, measurement parsing, drift, compilation, or sampling; diagnosis output alone is not ground truth. |

### Evidence provenance

- `ibm_validation.py` contains both real adapter functions and offline fixture functions. `run_offline_validation` synthesizes an `OFFLINE_FIXTURE` calibration capture and local counts; it does not contact IBM.
- `tests/test_ibm_validation.py::BackendFixture` and its service fixtures manually construct Qiskit-like backend properties. The calibration conversion tests therefore establish parser behavior on known fixture shapes, not that the current IBM service returns the same shape or that any values were retrieved in this workspace.
- `reports/ibm_validation_report.json` is explicitly an offline result. No current report inspected here demonstrates real IBM calibration capture or a physical execution.
- `experiments/run_experiment.py::_build_hardware_model` takes manually supplied `ExperimentConfig.hardware_parameters` and stamps them `synthetic-config`. `run_experiment` runs `QuantumSimulator`/`SimulatorHardwareModel`; these profiles and results are not backend observations.
- Phase 2 artifacts under `reports/experiments/independent_generalization_phase2/` are local seeded simulator runs with constructed hardware profiles. They are not calibration snapshots from a QPU.

## Phase 2 subtle-anomaly review

The saved Phase 2 protocol uses 512 shots and a 0.05 distribution threshold. The scenario ground truth is assigned by the manifest before each run. Results below describe only the three fixed seeds in the saved simulator experiment.

| Case | Output-distribution detector | Diagnosis/localization | Interpretation |
|---|---|---|---|
| Readout error 0.04 on q3 (`hardware_readout_q3_subthreshold`) | TVD by seed: 0.037109, 0.054688, 0.025391. Detector flags: false, true, false. | Predicted `NORMAL`, `AMBIGUOUS / INSUFFICIENT_EVIDENCE`, `NORMAL`; no hardware candidate was justified/localized. | Two detector misses plus one sampled distribution crossing the threshold. The profile's 0.04 readout error is below the localizer's 0.05 calibration cutoff, so the observed variation does not support q3 localization. The two NORMAL outputs under scenario-defined hardware ground truth are classification misses; the ambiguous output is an abstention. |
| Readout error 0.055 on q3 (`hardware_readout_q3_near_threshold`) | TVDs are 0.044922, 0.064453, 0.048828. Detector flags: false, true, false. | All three are classified `HARDWARE_ANOMALY` with q3 as a calibration-backed candidate. | The two runs below the distribution threshold are detector misses, not diagnosis-classification errors: the calibration profile still supports suspicion and q3 localization. The above-threshold run has both signal types. This does not establish that a real backend's calibration value has the same meaning or fidelity. |
| RZZ gate error 0.055 on q0-q2 (`hardware_rzz_error_near_threshold`) | TVDs are 0.025058, 0.037690, 0.025971; anomaly flag is false for all. | All three are `HARDWARE_ANOMALY`; gate calibration candidate `(0, 2)` is retained and localization is marked correct against the constructed target. | Correct calibration-based suspicion with no output-distribution detection. It demonstrates separable flags in the implementation, not independently adjudicated hardware fault localization. |

Phase 2's scenario reports and aggregate are historical evidence and were not changed. The three-seed results are sampling-sensitive and share scenario families; they do not calibrate a threshold for a QPU.

## Proposed hardware validation protocol

### Preconditions and evidence package

1. Resolve and test transpiler layout handling end to end. Record logical-to-physical qubit mapping, final transpiled circuit, physical coupling edges, classical-bit mapping, and how measured count keys are interpreted. Diagnosis and localization must consume indices in the same coordinate system as calibration.
2. Add a pre-execution and post-execution backend snapshot with backend name/version, backend status, calibration update timestamp, capture timestamp, job ID, and execution start/end timestamps. Declare a maximum calibration age before running. Missing required calibration or an exceeded age limit makes the run ineligible for hardware attribution; it must not be filled with defaults.
3. Preserve raw counts, returned shot count, result metadata, submitted circuit, transpiled circuit, layout, backend identity, and snapshot payload/hash. Record SDK/runtime versions and source revision. Do not label synthetic/manual profiles as backend captures.
4. Pre-register the circuit, shots, repeat schedule, expected ideal distribution, statistical test/interval, calibration age limit, fault-attribution rules, unsupported-operation handling, stopping criteria, and independent adjudication procedure before observing diagnosis output.

### Execution stages

1. **Read-only calibration preflight:** select an explicitly authorized backend; retrieve calibration and connectivity without submitting a job. Verify backend availability, required metrics, timestamp age, supported instructions, qubit capacity, and a route for a small test circuit. Abort on missing/stale data, inconsistent topology, or unavailable result-parsing/layout evidence.
2. **Healthy baseline:** use a small supported circuit with known behavior, such as prepare/measure a single qubit in computational basis plus a two-qubit Bell circuit only if the backend coupling and transpilation support are confirmed. Run an ideal/no-noise simulator reference and repeated hardware executions at a predeclared shot count. Do not call a particular hardware calibration “healthy” solely because diagnosis returns NORMAL; define baseline acceptance using independent backend status/calibration and repeated output distributions.
3. **Known program behavior:** execute a second small circuit whose output probabilities can be calculated analytically and independently checked. Keep circuit parameters and shot counts fixed across repeats. If a program-fault control is included, make it a separately specified circuit and label it from the submitted source/transpiled circuit comparison, not the diagnosis prediction.
4. **Repeatability and sampling:** collect a predeclared number of independent jobs/runs (minimum three repeats per circuit is a pilot, not a generalization sample). Keep shot count constant within a matched comparison. Report counts and multinomial confidence intervals or a preselected distributional test; do not classify one finite-shot deviation as hardware fault without its registered error control.
5. **Near-time calibration:** capture immediately before and after the execution series. Reject or separately analyze the series if the backend calibration timestamp changes, snapshot age exceeds the predeclared limit, backend identity/status changes, or layout/topology cannot be reconciled.

### Attribution and stopping rules

- **Program fault:** require an independently recorded source-vs-intended circuit mismatch or known injected program mutation, mapped through transpilation, with hardware/calibration evidence insufficient to explain the difference. Diagnosis prediction is not the label.
- **Hardware suspicion:** require consistent backend identity/layout mapping, calibration evidence on the physical qubit/gate/coupler used, and repeated output deviation beyond the predeclared sampling criterion. Calibration-only suspicion and output-distribution detection must be reported as separate outcomes.
- **Ambiguous/insufficient:** use this outcome when program and hardware explanations overlap, mappings/calibration are uncertain, or measured deviations do not clear the predeclared evidence threshold. Do not force a cause.
- **Stop before submission:** unsupported gate or measurement behavior; missing/stale calibration; incomplete physical layout; inability to parse raw results; backend not operational; unexpected compilation change; or authorization/cost uncertainty.
- **Stop after a run:** job/result metadata mismatch; counts inconsistent with returned shots; calibration changes during the series; unexpected transpilation/layout; or a safety/cost boundary. Preserve partial outputs and do not automatically rerun.

### Acceptance criteria for a first pilot

- All submitted operations are supported, and the source, transpiled form, physical layout, calibration indices, and classical result-bit mapping are auditable.
- Pre/post snapshots satisfy the registered age rule and are attached to every job record.
- Every job has backend, job ID, timestamp interval, requested/returned shots, raw counts, and immutable circuit/configuration provenance.
- Repeated healthy baselines are compatible with the predeclared simulator distribution within the selected statistical uncertainty procedure; this is a bounded pilot criterion, not proof of backend-wide health.
- Fault attribution is independently adjudicated from circuit and calibration records; the diagnosis output is scored only after labels are fixed.
- Report detection, diagnosis, localization, abstention, unsupported runs, and execution failures separately. Make no physical-hardware generalization claim from a single backend or pilot.

## Focused verification

- `pytest -q tests/test_ibm_validation.py --basetemp=.pytest-tmp-batch-20261012/phase3-readiness`: collection failed (`ModuleNotFoundError: ibm_validation`); no tests ran.
- `python -m pytest -q tests/test_ibm_validation.py --basetemp=.pytest-tmp-batch-20261012/phase3-readiness`: **21 passed, 2 failed** in 0.27s. The failures are `test_backend_discovery_parses_injected_offline_fixture_without_claiming_live_discovery` and `test_offline_execution_uses_deterministic_fixture_and_never_claims_ibm`; both expect `ACCESS_CONFIGURED`/`NOT_EXECUTED`, while this environment reports `BLOCKED` because Qiskit Runtime and/or credentials are unavailable. No live service or job was invoked by the test cases. Remaining adapter tests passed.
- `python -m pytest -q tests/test_hardware_fault_localizer.py --basetemp=.pytest-tmp-batch-20261012/phase3-localizer`: **18 passed** in 0.12s.
- `git diff --check`: run after adding this report; result recorded in the completion message.

## Remaining limitations

No real backend calibration was retrieved and no quantum job was submitted. Hardware field shape, timestamp behavior, qubit mapping after transpilation, returned count conventions, calibration drift, and diagnosis validity on physical results remain unverified. The focused IBM adapter suite has two environment-sensitive failures. Phase 2 readout and RZZ outcomes are constructed simulator evidence only and cannot establish physical-hardware accuracy. A read-only backend preflight and a mapping/freshness fix with mocked layout tests should precede any separately authorized execution.
