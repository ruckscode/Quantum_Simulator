# Null-confirmation driver implementation report (2026-10-16)

## Implementation

Created `experiments/finite_shot_null_confirmation.py` as a separate prospective driver. The original finite-shot runner and its fixed 12,000-pair cap were not changed. The driver reuses the registered `stream_seed` and `pair_outcome` behavior, the existing exact Clopper-Pearson bound helpers, and `compare_distributions`; it enforces the preregistered 1,024-shot, four-outcome null, 38,228-pair workload, fixed 0.05 threshold, one worker, and exact pass boundary (at most 2,217 flags).

Before sampling, it validates the preregistration structure and fixed values, registered source hashes, driver/binding hashes supplied as external pins, configuration digest, complete seed ledger, seed uniqueness and no overlap with prior 1,024-shot streams, resource plan, output path, runtime versions, and pending review release. Pair-cap checks occur before each pair; CPU and traced-memory limits are checked during processing; projected and final serialized output sizes are bounded; output is create-only. It requires both `--execute-sampling` and a released review record before any sample call. The current review release remains pending, and this task did not execute the driver.

A full ordered seed ledger was derived from the registered hash policy without generating multinomial counts: 76,456 stream seeds, all unique, zero overlap with prior 1,024-shot streams. Ordered seed bytes SHA-256: `222e5522204fd09f56f860c4a474130b2357d34997c9393f0d47521953f341ad`.

## Hash binding

- Preregistration SHA-256: `bd50e03c195b0b58c589364a2872eb17ab21e68534ee54d8e249917b59da8e20`
- Detector statistics module SHA-256: `2d90d01a024aa527a4dd7805a09ba0216e35afceb3e718d979e9ed5399212d82`
- Reference finite-shot runner SHA-256: `a40a1d81a0fd1a7aaadcb187535484849a80c14f6b6165ed32501974a9e63a9f`
- New driver SHA-256: `3617a5e4b965c30d6160c2c789a197da3d285f32c689cef4007c016ba010e35a`
- Local driver-binding record SHA-256: `d8e68545665c8e9f6806c811248fb731127c8d1aaf259853d619cfe220ba2296`

The registered prospective JSON was not edited. Its source-binding limitation is addressed by the separate local binding record. The command line additionally requires the caller to supply the expected driver and binding hashes. The binding explicitly records `independent_review_release.status=pending`; no independent review or attestation is claimed. Its future evidence-file hash check verifies file integrity, not reviewer identity or independence. The runtime check requires Python 3.12.10 and NumPy 2.5.3, as recorded in the prospective design; actual execution must record these versions and the NumPy bit generator.

## Tests and integrity checks

Focused command:

```text
.venv\\Scripts\\python.exe -m pytest tests/test_finite_shot_null_confirmation.py -q --tb=short --basetemp .t-null-confirmation-focus-final3-20261016
12 passed
```

Tests cover registered configuration and hashes, mismatched prereg/source/driver hashes, exact pair and resource limits, the 2,217/2,218 decision boundary, repeatable seed-ledger derivation, and fail-closed sampling/review gates. They do not call `pair_outcome` or generate multinomial data. Lightweight JSON, source-hash, seed-ledger, decision-boundary, and prior-result integrity checks passed. `git diff --check` passed with existing line-ending warnings on previously modified files.

## Files created

- `experiments/finite_shot_null_confirmation.py` ? separate driver; SHA-256 listed above.
- `tests/test_finite_shot_null_confirmation.py` ? focused analytical and preflight tests.
- `reports/experiments/finite_shot_null_confirmation_driver_binding_20261016.json` ? local hashes/configuration/seed binding; release remains pending and no independent attestation is represented.
- `reports/experiments/finite_shot_null_confirmation_implementation_20261016.md` ? this report.

No prior preregistration, result, report, audit, detector, threshold, diagnosis fixture, or frozen outcome was modified. The designated confirmation result file does not exist. **No confirmation samples were generated and the statistical experiment was not executed.**
