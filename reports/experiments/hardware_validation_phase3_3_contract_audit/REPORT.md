# Phase 3.3 — Hardware Integration Contract Audit

## Scope and evidence

This audit follows `ibm_validation.py` from read-only preflight through backend calibration capture, transpiler layout handling, sampler-result parsing, diagnosis, and report serialization. Contract verification used local fixtures and mocks only. No IBM account configuration was read, no IBM service was contacted, and no circuit or job was submitted.

Relevant implementation and tests:

- `ibm_validation.py`: `preflight_ibm_backend`, `capture_backend_calibration`, `calibration_freshness`, `_extract_transpilation_layout`, `_map_plan_to_physical`, `execute_on_ibm_backend`, `counts_to_probabilities`, `_make_result`, and `build_ibm_validation_report`.
- `src/quantum_simulator/diagnosis_engine.py`: calibration-gap reporting and selection of calibration-supported hardware candidates.
- `src/quantum_simulator/hardware_fault_localizer.py`: physical-index candidate lookup from the supplied hardware model.
- `tests/test_ibm_validation.py`, `tests/test_hardware_fault_localizer.py`, and `tests/test_diagnosis_engine.py`.

## Data-flow findings

1. Preflight fetches a named backend, captures its identity, status, topology, properties timestamp, retrieval time, and calibration values. It reports individual checks and a deterministic ID for the captured snapshot. The same ID, backend name, source, and timestamps are now included in the calibration assessment. Caller-injected services are labelled injected and cannot claim authenticated live evidence.
2. `AVAILABLE` means backend properties and qubit count were retrieved; it does not mean every calibration field is present. Completeness, unavailable values, and parse errors are reported separately. Stale, missing, invalid, future, or timezone-naive calibration timestamps block preflight and prevent execution.
3. The transpiler’s logical-to-physical mapping is checked for width, uniqueness, and backend bounds, then applied to every operation operand, including multi-qubit gates and measurement. Missing layouts, final routing permutations, SWAP routing, and invalid indices do not produce a guessed physical map. Calibration is indexed using backend physical qubit IDs.
4. Diagnosis uses the captured backend hardware model and the mapped physical circuit. The diagnosis engine reports absent gate and coupling calibration as missing evidence and only promotes calibration-supported, circuit-relevant candidates to hardware evidence. An anomaly without supporting cause evidence remains insufficient/ambiguous rather than being attributed to hardware.
5. Offline fixture runs serialize `real_ibm_status=NOT_EXECUTED`. A pre-execution safety failure is `BLOCKED`; result/diagnosis failures are `FAILED`. `AVAILABLE` is a retrieval status, not an execution or validation-success status.

## Contract violations found and fixes

- **Shot/result integrity:** the count parser previously normalized counts without requiring their sum to equal requested shots, accepted mixed-width outcomes, and did not require binary outcomes. A partial result could therefore reach diagnosis as if it were complete. It now rejects zero/invalid shots, non-binary or mixed-width bitstrings, and totals that differ from the requested shots. The adapter checks returned measurement width against the expected distribution and returns `FAILED` without diagnosis when parsing or compatibility checks fail.
- **Execution interval:** the adapter previously did not validate its start/end timestamps before diagnosis. It now requires both timestamps, ISO-8601 parsing, timezone offsets, and a non-reversed interval before diagnosis.
- **Backend identity:** execution previously did not stop if the selected service returned a backend whose reported name differed from the requested name. It now returns `BLOCKED` before transpilation/submission and records the mismatch. Calibration snapshot IDs are kept in sync between the capture and its assessment.
- **Calibration topology indices:** coupling-map conversion previously coerced values with `int()`, which could silently truncate a malformed fractional index. It now validates indices without coercion and reports malformed topology as incomplete calibration.
- **Backend status and shot arguments:** an unknown operational status could pass the execution guard, and Python booleans could pass integer shot validation. Unknown/non-operational status now blocks before transpilation, and booleans are rejected as shot counts.

Missing multi-qubit gate and coupling calibration is explicitly surfaced as missing evidence. The focused diagnosis case remains `AMBIGUOUS / INSUFFICIENT_EVIDENCE` under the existing serialized enum value and does not produce hardware evidence candidates.

## Verification

Command:

```powershell
python -m pytest -q tests/test_ibm_validation.py tests/test_hardware_fault_localizer.py tests/test_diagnosis_engine.py --basetemp=.pytest-tmp-batch-20261012/phase3_3_contract_final_3
git diff --check
```

Result: **88 passed in 0.53 seconds**. `git diff --check` passed; Git emitted line-ending conversion warnings for existing modified files. The tests cover fixture-only successful preflight, explicit blocked outcomes, mismatched backend identity, snapshot association, stale/incomplete calibration, layout and physical-index handling, missing multi-qubit calibration, malformed/partial shot results, incompatible bitstring widths, timestamp intervals, and prevention of transpilation/execution on read-only preflight or pre-execution blockers.

## Remaining blockers and live-execution criteria

Local dependency inspection reported `BLOCKED`: `qiskit` and `qiskit-ibm-runtime` are not installed. A saved account configuration was detected, but it was neither opened nor used. No target backend was selected and no live authentication or calibration evidence exists. The prior offline fixture status remains `NOT_EXECUTED` for IBM hardware.

Before any real execution is considered, a separately authorized, live read-only preflight must establish all of the following for the explicitly selected backend:

- Required IBM dependencies are installed and authentication succeeds using credentials explicitly supplied for that request.
- The returned backend identity exactly matches the requested backend and reports operational status.
- Calibration has a parseable timezone-aware backend timestamp no more than 900 seconds old, a separately recorded retrieval time, complete fields needed for the planned circuit, and valid physical qubit/coupling indices.
- Transpilation exposes a complete, consistent logical-to-physical layout; any unresolved routing or SWAP mapping is a stop condition.
- The execution result schema, returned shot total, measured width, and execution time interval validate before diagnosis; any discrepancy yields an explicit failure without a definitive diagnosis.
- The assessment, result, and report carry one matching backend identity and calibration snapshot ID. Preflight must confirm that it submitted no job and performed no transpilation.

Live calibration freshness can change between preflight and execution; it must be checked again immediately before any separately authorized execution. Mocked tests do not establish IBM service compatibility, live backend access, or physical-hardware diagnosis accuracy.

## Decision

**NOT READY** for live backend execution. After dependency/access availability is addressed, the next step is a separately authorized read-only preflight against one explicitly named backend. This audit did not perform that preflight against IBM.
