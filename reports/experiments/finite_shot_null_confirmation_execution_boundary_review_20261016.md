# Finite-shot null-confirmation execution-boundary review

**Decision: BLOCK EXTERNAL TRUST PROVISIONING.** The Windows junction path check is now exercised by a passing test. The sampler bypass, atomic binding of verified artifacts to executed code, and independently verified trust-root permissions remain blockers. No confirmation samples were generated; the experiment was not run.

## Findings

| Area | Status | Evidence and effect |
|---|---|---|
| Sampler bypass / actual execution boundary | **BLOCKER** | `experiments/finite_shot_null_confirmation.py` imports the generic `pair_outcome` directly at module load (lines 20–31), then `execute_confirmation` calls it in the registered pair loop (line 512). The same public function remains directly callable from `experiments/finite_shot_synthetic_characterization.py:52–54`; its `run` function also calls it (`:124–149`). The test `test_public_pair_helper_reaches_rng_without_confirmation_authorization` (`tests/test_finite_shot_null_confirmation.py:365–377`) intercepts the RNG before any draw and proves that direct invocation reaches sample generation without the confirmation authorization gate. Renaming or wrapping only the driver's import would not close the direct import path. Modifying or gating the generic helper would change the source whose digest is pinned by the earlier registration, so there is no defensible code-only correction within the preservation constraints. **Smallest defensible architectural correction:** execute the confirmation in a separately versioned, isolated sampler/runtime under an execution identity that cannot import or access the legacy public helper; bind its immutable bundle to a protected launcher and retain the existing preregistration unchanged. |
| Verified artifact versus executed code / import and TOCTOU | **BLOCKER** | The driver imports `cryptography`, NumPy, and the reference runner before it can verify bindings (`experiments/finite_shot_null_confirmation.py:18–32`). `verify_hash_bindings` hashes files and then `verify_loaded_code_origins` compares module identity and `co_filename` (`:191–235`); this does not establish that loaded bytecode equals the bytes hashed from disk or prevent in-process replacement. `sha256_file` path-checks with `lstat` and subsequently reads by path (`:62–90`), leaving a replacement race. The confirmation loop later invokes the already-imported function at `:512`. Hash/path checks are useful integrity checks but do not provide atomic execution binding. **Prerequisite:** a protected bootstrap must verify and execute an immutable snapshot before importing any experiment code, constrain imports to that snapshot, and run under a separately protected identity. |
| Trust-root ACL and identity | **BLOCKER** | The trust root is fixed at `C:\ProgramData\QuantumSimulator\finite-shot-null-confirmation` (`experiments/finite_shot_null_confirmation.py:42–47`). The active identity was `DESKTOP-M0FQML9\CodexSandboxOffline`; `Test-Path` reported the trust root absent, and querying the parent ACL failed because `C:\ProgramData\QuantumSimulator` does not exist. The code rejects reparse paths and missing trust records, but does not inspect effective Windows ACLs or establish that this identity cannot create, replace, or delete trust material. **External prerequisite:** an administrator must provision the root, configure and inspect effective ACLs (including inheritance and group grants) for the actual execution identity, and independently authenticate the reviewer key and identity. A configuration field or absent path cannot prove this prerequisite. |
| Filesystem reparse-point test | **PASS for directory junction detection; symlink test remains skipped** | Added `test_junctioned_trust_directory_is_rejected_when_platform_supports_junctions` in `tests/test_finite_shot_null_confirmation.py`. It creates a disposable Windows directory junction using `mklink /J` and asserts `_assert_directory_no_reparse` rejects it. The test passed in this environment. The existing symlink-record test still skips because this identity cannot create symlinks. The junction test exercises the Windows reparse-point property relevant to the current path check, but does not establish ACL protection or close path races. |
| Registered design and hash integrity | **PASS** | The preregistration and binding parse and validate. `verify_hash_bindings` passed; the current driver SHA-256 is `01b3610d63bc4824532cfdeaa05a09856c747ecc9ac24973e8a2e0ee343b83ab`, matching `driver_binding_20261016.json`. No science parameter, earlier preregistration, prior result, or frozen diagnosis artifact was changed. |

## Validation

Focused command:

```text
.venv\Scripts\python.exe -m pytest tests/test_finite_shot_null_confirmation.py -q --tb=short --basetemp .pytest-execution-boundary-20261015-a
```

Result: **15 passed, 1 skipped**. The passing junction test exercises the reparse-point rejection; the single skip is the separate symlink-creation test. Authorization tests use test-only keys and intercept sampling before any multinomial draw. The interruption test stops at preflight. No confirmation samples were generated.

Lightweight checks: strict preregistration/binding load, driver/hash binding, loaded-origin check, and `git diff --check` passed. `git diff --check` emitted existing LF/CRLF warnings for unrelated working-tree files. The production trust root is absent, so no effective ACL audit was possible. The designated result file remains absent.

## External prerequisites and disposition

1. Build and review the isolated immutable sampler/runtime described above; the current generic helper remains callable without the confirmation gate.
2. Provision a protected execution identity and launcher that verify artifact bytes before import and prevent modification or redirection between verification and execution.
3. Have an administrator provision the trust root and verify effective ACLs for that execution identity. Independently verify the reviewer identity and key through the project's external review process.
4. Run the filesystem tests in the supported deployment environment as well; local junction coverage does not replace deployment ACL testing.

The registered experiment remains fixed at 38,228 pairs, 1,024 shots, the registered null distribution, detector and threshold, and one-sided 5% Clopper–Pearson rule with the 2,217-flag cutoff. This review does not approve execution.

## Files changed in this task

- `tests/test_finite_shot_null_confirmation.py` — adds an actual Windows junction/reparse-point test as an alternative filesystem-security exercise.
- `reports/experiments/finite_shot_null_confirmation_execution_boundary_review_20261016.md` — records evidence, unresolved blockers, validation, and the decision.

No experiment was run and no confirmation samples were generated.

**BLOCK EXTERNAL TRUST PROVISIONING**
