# Finite-shot null confirmation trust gate

**Status: execution remains blocked.** The driver now requires a detached Ed25519 approval matched to an externally provisioned reviewer key. No reviewer identity, key, signature, or approval was created for this work. The 38,228-pair confirmation was not run.

## Trust model

The trusted root is fixed outside the checkout at:

- `C:\ProgramData\QuantumSimulator\finite-shot-null-confirmation\trusted_reviewer.json`
- `C:\ProgramData\QuantumSimulator\finite-shot-null-confirmation\approval.json`

The execution caller cannot choose these paths through the CLI or `execute_confirmation` arguments. The trust directory must be provisioned by an administrator and have Windows ACLs that allow the execution identity to read the records but not replace, edit, or delete the trusted reviewer record. The operator must inspect the effective ACLs (for example with `icacls`) and confirm that the execution identity is not an administrator and has no write grant through direct or group membership. The code does not claim to establish those OS ACLs itself; this is an explicit external trust assumption. The production files are currently absent, so verification fails closed.

The trusted reviewer JSON has schema `finite-shot-trusted-reviewer-v1` and contains `reviewer_id`, `reviewer_independence_verified: true`, the raw Ed25519 public key in hex, and `key_id` equal to SHA-256 of the raw public-key bytes. The administrator must verify the reviewer’s real identity and independence through an out-of-band institutional or organizational process before installing that mapping. The field is meaningful only because the trust directory is protected; it is not self-authenticating evidence.

The signed approval JSON has schema `finite-shot-independent-approval-v1` and contains the same reviewer and key IDs, review timestamp, `decision: approved`, a nonempty independence basis, a review scope covering the preregistration, driver, detector/statistical dependencies, configuration, seed ledger, and resource controls, the complete artifact manifest, and `signature_base64`. The reviewer signs the UTF-8 canonical JSON of every approval field except `signature_base64`, serialized with sorted keys and compact separators, using Ed25519. Verification checks the reviewer-to-key mapping, key fingerprint, signature, approval status, required scope, and exact current manifest equality. Missing, malformed, stale, altered, or differently signed records are rejected before any sampling call.

## Protected artifact manifest

`artifact_manifest()` in `experiments/finite_shot_null_confirmation.py` hashes and signs:

- preregistration JSON;
- null-confirmation driver;
- local driver-binding record;
- `src/quantum_simulator/statistics.py` detector implementation;
- `experiments/finite_shot_synthetic_characterization.py` sampling, seed, and confidence-bound helpers;
- `pyproject.toml` dependency declaration;
- canonical registered configuration digest;
- complete ordered seed-ledger digest;
- Python, NumPy, and cryptography versions and the declared NumPy default bit generator.

The local binding is still checked for internal source/configuration/seed consistency, but is explicitly not authorization. Caller-supplied expected hashes have been removed from the function API and command line. The detached approval is the trusted statement over the complete manifest; changing a listed artifact or runtime invalidates it. The registered distribution, 1,024-shot setting, detector threshold, 38,228-pair workload, and one-sided 5% Clopper–Pearson decision rule were not changed.

Current source/manifest verification passed locally. Current driver SHA-256: `311d3082ad9a8e0450848fb3527273301946f09f8c5047e318437b75c8cd3c26`. The preregistration, detector, statistical helper, project dependency, configuration, and seed-ledger digests are recomputed at verification time; no signed approval exists for this driver digest. The seed ledger remains 76,456 unique streams with zero overlap and digest `222e5522204fd09f56f860c4a474130b2357d34997c9393f0d47521953f341ad`.

## Security checks

Focused tests verified:

- absent or malformed trust/approval records fail closed;
- pending/rejected status, wrong reviewer identity, wrong public key, malformed/tampered signature, and incomplete/stale manifests fail closed;
- edits to the preregistration, driver, binding, detector module, statistical reference runner, dependency declaration, runtime version, configuration, or seed-ledger commitment invalidate verification;
- the CLI/API no longer accepts caller replacement hashes;
- pending review blocks the execution path before `pair_outcome` can be called;
- a test-only generated key signs and verifies only when its matching separately supplied trusted public-key record is used;
- existing fixed workload, 2,217/2,218 decision boundary, and deterministic seed-ledger checks still pass without generating samples.

Test-generated keys are disposable fixtures and make no assertion about any real reviewer or independent review.

## External prerequisites before any execution approval

1. An independent reviewer must review the final frozen driver and all manifest-listed artifacts, verify the statistical design and execution protections, and state their scope and independence basis.
2. A trusted administrator must authenticate the reviewer identity and independence, register the reviewer public key in the fixed ProgramData trust directory, and enforce/read back ACLs that prevent the execution identity from modifying the trust key or approval.
3. The reviewer must sign an approval generated from the final current artifact manifest. Any subsequent change to the driver, preregistration, binding, detector/helper files, dependency declaration, configuration, seed ledger, or recorded runtime requires a new review/signature.
4. The operator must independently confirm the trust-root ACL and key-to-identity mapping before invoking the explicit execution command.

Until these external steps are complete, the absent trust records cause the driver to reject execution. This implementation does not claim independent review has occurred and does not authorize a run.

## Validation and changed files

Focused command: `.venv\Scripts\python.exe -m pytest tests/test_finite_shot_null_confirmation.py -q --tb=short --basetemp .pytest-finite-shot-trust-20261015-e` — **8 passed**. Lightweight JSON parsing, internal binding checks, artifact/seed manifest calculation, and fail-closed production approval check passed. The confirmation output path remains absent. `git diff --check` passed; Git reported pre-existing LF/CRLF warnings in other modified files.

Changed files:

- `experiments/finite_shot_null_confirmation.py` — signed external trust and manifest verifier; removed caller-supplied pins.
- `tests/test_finite_shot_null_confirmation.py` — focused authorization and tamper tests; no sampling.
- `reports/experiments/finite_shot_null_confirmation_driver_binding_20261016.json` — refreshed local source hashes and clarified that the binding is not authorization; review remains pending.
- `pyproject.toml` — declared the already-present `cryptography` verification dependency; no packages were installed.
- `reports/experiments/finite_shot_null_confirmation_trust_gate_20261016.md` — this trust model and validation record.

The prospective preregistration and all prior results, audits, fixtures, and research artifacts remain untouched. No confirmation samples were generated and no experiment, training, ingestion, hardware, or QML job was run.
