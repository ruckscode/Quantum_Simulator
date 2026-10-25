# Phase 3.5 — Live Read-Only IBM Preflight

## Result: BLOCKED

No live preflight was attempted because the repository does not unambiguously identify an intended backend. The caller must supply the backend name before IBM access can be considered. No backend was guessed and backend discovery was not used as a fallback.

## Backend selection evidence

- `experiment_config.py`: `ExperimentConfig.backend_name` defaults to `None`; it does not select an IBM backend.
- `README.md` §13: says no backend is a permanent or default project configuration.
- `README.md` §13 describes `ibm_marrakesh` only as a historical validation snapshot, not the intended current backend.
- `ibm_validation.py::preflight_ibm_backend` requires an explicit `backend_name` argument.

These sources do not support choosing a backend for this preflight. No saved credentials or token values were inspected.

## Check outcomes

| Check | Outcome | Evidence / reason |
| --- | --- | --- |
| Intended backend selected | **BLOCKED** | No unambiguous backend in project configuration; backend name must be supplied. |
| Dependency/Runtime API check for this preflight | **NOT RUN** | Stopped at backend-selection blocker. Phase 3.4 previously recorded Qiskit 2.5.2 and IBM Runtime 0.50.0 in `.venv`; this was not rechecked in this batch. |
| Authentication | **NOT ATTEMPTED** | No credential access or authentication attempt was authorized before a backend was selected. |
| Live backend identity and status | **NOT RETRIEVED** | No IBM service contact. |
| Calibration retrieval timestamp and backend calibration timestamp | **NOT RETRIEVED** | No IBM service contact. |
| Calibration age against 900-second limit | **NOT VERIFIABLE** | No live calibration timestamp. |
| Required gate/readout/qubit calibration fields | **NOT VERIFIED** | No live calibration snapshot. |
| Physical qubit indices and coupling topology | **NOT VERIFIED** | No live backend topology. |
| Live assessment completeness | **NOT VERIFIABLE** | Requires metadata from the explicitly selected backend. |

## Safety boundary

This report contains no live IBM evidence. No credentials were read, printed, copied, or used. No authentication, backend request, calibration retrieval, transpilation, circuit execution, sampler call, job creation, or job submission occurred. The adapter's separately documented execution path is `execute_on_ibm_backend()`; it is not part of this read-only preflight attempt.

## Required next action

Supply the exact intended IBM backend name. After that, a separately authorized preflight can use the existing project `.venv`, inspect only credentials already configured for the intended account through the adapter's authorized path, and stop at preflight. It must not invoke `execute_on_ibm_backend()`.

**Final result: BLOCKED — backend name required.**
