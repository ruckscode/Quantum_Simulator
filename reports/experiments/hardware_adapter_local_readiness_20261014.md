# IBM adapter local readiness follow-up ? 2026-10-14

## Scope and evidence class

This is a source and fixture review only. It makes no IBM service contact and establishes no physical-device result. The previously recorded phase 3 readiness audit accurately described its then-current code state; its identified mapping/freshness gaps are superseded by this follow-up. The backend-selection blocker in phase 3.5 remains current.

## Current adapter checks

- `transpile_for_backend` extracts `logical_to_physical`; missing, invalid, or routing-permutation layouts yield a non-ready status. The execution path returns before submission unless the map is usable.
- A plan is remapped into backend physical coordinates before diagnosis/localization. Fixture coverage checks a nontrivial permutation, unused backend qubits, a two-qubit operation, and invalid mappings.
- Calibration freshness requires a timezone-aware backend update timestamp and checks the configured limit (default 900 seconds). The execution path blocks on stale or missing calibration before submission. Offset normalization and boundary behavior have fixture tests.
- `tests/test_ibm_validation.py`: **43 passed in 2.26s** on the current tree. The fixtures do not exercise the IBM service.

## Dependency and clean-environment contract

`qiskit-ibm-runtime` is not a core dependency. `pyproject.toml` now defines an `ibm` extra pinned to the tested Qiskit 2.5 / Runtime 0.50 compatibility line. README instructions install it in an isolated environment and run fixture tests without authentication. The current primary environment was not modified. IBM's current install guide treats Runtime as a separately installed client: [IBM Quantum Compute client installation](https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit-runtime).

## Remaining hardware prerequisites

No intended backend is configured, and no credential or account was inspected. A future read-only live preflight requires the user to supply the exact backend name and explicitly authorize use of already-configured credentials. Any execution requires separate explicit authorization. No job was submitted.
