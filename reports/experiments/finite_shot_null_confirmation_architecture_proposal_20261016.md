# Finite-shot null-confirmation execution-boundary architecture proposal

**Purpose:** propose a bounded architecture for closing the two software execution-boundary blockers and specify the external controls required for trust-root protection. This is a planning document only. It does not authorize provisioning or execution.

## Current call graph and bypass

The relevant paths in the current source are:

```text
python experiments/finite_shot_null_confirmation.py [--execute-sampling]
  -> main()                                      finite_shot_null_confirmation.py:547
  -> execute_confirmation(... execute_sampling) :456
       -> fixed-path and preregistration checks
       -> verify_hash_bindings()                 :191
       -> verify_signed_approval()               :275
       -> consume_approval_run()                  :338
       -> seed_ledger_summary()                   (pure deterministic seed derivation)
       -> for each of 38,228 ledger rows:
            pair_outcome(...)                     :512
              -> default_rng(...).multinomial     finite_shot_synthetic_characterization.py:52–54
              -> compare_distributions(...)       src/quantum_simulator/statistics.py
```

The generic runner has a second route:

```text
python experiments/finite_shot_synthetic_characterization.py
  -> main()                                       finite_shot_synthetic_characterization.py:191, 200
  -> run(...)                                     :124
  -> pair_outcome(...)                           :148
```

Any Python caller can also import `experiments.finite_shot_synthetic_characterization.pair_outcome` and call it directly. The focused adversarial test does this at `tests/test_finite_shot_null_confirmation.py:366–377`, intercepting `default_rng` to prove that the RNG is reached without producing a draw. The confirmation module itself imports the function at module import time (`finite_shot_null_confirmation.py:20–32`). The `execute_sampling` flag and detached approval guard `execute_confirmation`; they do not guard the public helper or the separate legacy runner.

`compare_distributions` is a deterministic detector once its count vectors are supplied; it is not itself a sample-generation path. Sampling occurs in the two `default_rng(...).multinomial(...)` expressions in `pair_outcome`. Seed generation is deterministic and pure in `stream_seed`; it must remain available to analytical tests without opening a sampling capability.

The current hash checks are not an execution boundary. Imports occur before verification. `verify_loaded_code_origins` checks object identity and `co_filename`, not equality between loaded code and the exact verified bytes. `sha256_file` separately checks path metadata and opens by path, so a replaceable file may change between verification and import/use. `-I` alone would not make the bundle immutable or bind a later import to bytes already checked.

## Architecture options

### Option 1 — Dedicated Windows service with a protected, immutable worker bundle (recommended)

Run a small service under a dedicated non-interactive service identity. It exposes only a narrowly defined local IPC operation for the fixed confirmation experiment. The service authenticates the caller, validates the detached approval and its one-time run ID, verifies the final bundle manifest, and launches or hosts one fixed worker. Callers receive no shell, Python prompt, module-import interface, arbitrary path, command, or caller-selected configuration. The worker imports the sampler only after the service has authorized that specific run.

The bundle is versioned and placed in an administrator-owned directory. The service identity has read/execute rights to the code and dependencies, but no write/delete rights to the bundle, its parent directories, or the trust key/approval. The separate run-consumption/output store has only the minimum rights needed by the service; the requestor cannot edit or remove its records. The service rejects reparse points in every path component. Administrator-only updates create a new versioned directory, verify it, and atomically switch a service configuration pointer before any new approval is obtained. No in-place update is permitted.

The bootstrap/service must verify a manifest that covers the preregistration, new worker/sampler code, detector and confidence-bound helpers, every imported project module, Python executable/runtime, standard library used by the worker, NumPy and its native extensions, cryptography verification code, dependent DLLs, and a locked dependency inventory. `pyproject.toml` alone is not a runtime lock. The reviewer approval binds the experiment ID, bundle-manifest digest, fixed configuration and seed-ledger digest, trust-root/key identity, and unique run ID. The service verifies the bytes before loading them; administrator-only, non-replaceable paths prevent a write/rename between verification and import. The service constrains `sys.path` to the verified bundle, runs with isolated settings (`-I`, `-S` where the chosen runtime permits), disables user site/startup hooks, starts outside the checkout, and verifies loaded module origins/digests. Native-library search paths must also be fixed to the bundle/runtime.

**Advantages:** closes direct Python import/CLI bypasses for ordinary requestors because they cannot run as the service identity or access the protected worker bundle; gives one authorization and replay boundary; supports auditable request logs and fixed operation parameters.

**Costs and limits:** requires administrator deployment, service/IPC code, identity and ACL review, and a protected runtime. A local administrator or compromise of the service identity can still bypass policy; those principals must be in the trusted computing base. A compromised service process is outside application-level guarantees. Test-only Python monkeypatching does not substitute for deployment tests under the actual service token.

### Option 2 — Signed self-contained executable or package in a read-only virtual machine/container image

Package the fixed worker and dependency closure in a signed image. A host-side privileged launcher verifies image signature/digest and starts a fresh isolated VM or container with the bundle mounted read-only. Give the guest/container only the approval and an output channel; do not expose an interactive shell or arbitrary process launch to the requestor. Persist one-time run consumption in a protected host-side store.

**Advantages:** clean runtime boundary, reproducible dependency environment, and a read-only image naturally separates worker code from writable results. A VM gives stronger separation from host Python import state than an in-process Python module.

**Costs and limits:** more operational machinery (image signing, host launcher, Hyper-V/Windows Sandbox or a managed container platform), potentially difficult persistence for the consumed-run ledger and output, and still requires host ACLs and identity management. A plain self-contained executable started by the same interactive user is insufficient: the user could invoke the sampler or replace a writable executable, so the privileged fixed launcher and isolation remain necessary.

### Option 3 — ACL-protected virtual environment plus a script or scheduled task

Use a read-only venv and a fixed signed script launched through Task Scheduler. This is smaller initially but weaker unless ordinary callers cannot run arbitrary commands as the task identity, cannot edit task arguments or actions, cannot access/import the worker, and cannot alter the venv or any ancestor. `-I` does not remove the generic helper route when users can run Python themselves. A scheduled task with a mutable action or user-controlled arguments merely moves the bypass.

**Decision:** do not use this as the final boundary without the same protected service identity, fixed launcher, ACL, runtime closure and adversarial evidence required by Option 1. Its only advantage is reduced service implementation work; it does not by itself resolve the reported blockers.

## Recommended narrow design

Adopt Option 1, with a service-controlled one-shot worker and a new, separately versioned worker bundle. Keep the existing finite-shot runner, preregistration, saved results, reports, and frozen diagnosis fixtures untouched. The new worker may use the exact existing `pair_outcome`, `stream_seed`, and detector logic copied or packaged byte-for-byte into the protected bundle, but the ordinary execution identity must not have a route to run that module. Statistical helpers such as `compare_distributions`, Clopper–Pearson bounds, seed derivation, and decision evaluation stay importable/testable as pure logic; only the two RNG/multinomial operations and the end-to-end sampling operation reside behind the worker capability boundary.

Do not treat a leading underscore, an unexported function, an environment variable, `if __name__ == '__main__'`, or a Python flag as an authorization control. Python has no robust in-process private-function boundary against a caller that can import the module or run arbitrary code in the same interpreter. The enforceable boundary is the OS-controlled service process and IPC contract: only the service can access/execute the worker bundle under its restricted identity, and the service exposes no arbitrary code or sampler RPC.

For verify-to-execute binding, use versioned, immutable deployment paths and least-privilege ACLs. The verifier and worker must consume the same bundle version identifier and manifest digest. No update process may modify an active bundle. If code is staged at runtime, stage to a unique new protected directory, verify the full staged tree, then mark it immutable to the worker identity before loading it. Do not verify a checkout path and later import from another path. Keep Python and native dependencies inside a controlled runtime bundle; no checkout, current directory, user site-packages, `PYTHONPATH`, `.pth`, `sitecustomize`, or unpinned DLL search path may participate.

Approval and run state remain separate: an independent reviewer signs the canonical manifest for the exact bundle, preregistration, detector, sampling configuration, fixed seed ledger, resource limits, trust-root/key identity, and a fresh UUID run ID. The service atomically consumes the run ID in a service-protected ledger before entering the worker. A failed or interrupted run is terminal; recovery requires a new reviewer-approved run and an explicitly non-overlapping seed allocation. Existing seed values and the current registered run are not to be silently reused.

## Enforceability split

| Guarantee | Application/service can enforce | OS/deployment prerequisite |
|---|---|---|
| Fixed experiment parameters, validation, signature checking, canonical payload, run-ID replay check, resource limits, result schema | Yes, within the service process | Service code, configuration and process must be protected from the requestor. |
| No unauthorized sampling through direct imports or helper calls | Only if sampling code is unavailable to requestors and reachable only through the fixed service worker | Dedicated service identity with no interactive/batch logon; protected bundle ACLs; fixed IPC; no arbitrary Python execution as the service identity. |
| Hash of approved source equals source imported/executed | Verify complete bundle before import, constrain module and DLL resolution | Bundle and all ancestors must be non-writable/non-replaceable to worker/requestor from verification through execution; protected runtime and launch policy. |
| Trusted reviewer key and independence | Verify signature against configured public key | Independent organization/reviewer identity verification; administrator provisions the authenticated key; execution identity cannot replace it. |
| No approval replay or duplicate seed run | Consume run ID before sampling and refuse resume | Protected durable run ledger; requestor cannot delete/restore it; backup/rollback policy must not restore an older ledger silently. |
| Effective Windows rights and path integrity | Detect some missing/reparse/malformed paths and fail closed | Administrator establishes ownership/DACL/inheritance/group membership, service identity, service-control/IPC ACLs, code signing or application-control policy, and change control. Code cannot certify its own ACL trust. |

## Staged implementation and adversarial test plan

1. **Freeze a worker specification.** Record byte-for-byte source lineage for the detector and sampler, exact import/dependency closure, runtime build, fixed configuration, current preregistration digest, and seed-ledger digest. Keep the statistical behavior and all registered values unchanged. Produce a new bundle identifier; do not edit the prior preregistration or results.
2. **Build the isolated worker and service interface.** The service accepts only a fixed operation and signed run approval; reject user paths, configuration, module names, commands, and arbitrary payloads. Place RNG calls only in this worker path. Keep pure detector/statistical functions separately unit-testable without sampling.
3. **Provision the deployment boundary.** An administrator creates the service identity, protected versioned bundle directory, read-only trust material, separate run ledger, and output directory. Configure the IPC ACL and service-control permissions. Use a restricted token/job object or equivalent so the worker cannot spawn an interactive shell or break out of the service job. The service identity must not be usable for interactive or arbitrary scheduled logon.
4. **Verify artifact binding.** Independently sign the full bundle manifest, including executable/runtime and all Python/native dependencies. The launcher verifies it before importing the worker. Load only from the protected bundle. Add deployment checks proving the service identity cannot alter, replace, rename, delete, junction, or redirect bundle and trust files. Any change results in a new bundle digest and new approval.
5. **Unit/adversarial tests, no confirmation workload.** Use generated test-only keys and a spy sampler that fails if invoked. Verify pending/missing/invalid approval denies requests before the spy; direct `pair_outcome`/legacy-runner import is unavailable to the worker requestor; no alternate CLI, module import, RPC, task argument, or helper reaches the spy. Test approval replay and interruption terminality. Change each manifest-covered file (driver, sampler, detector, helper, dependency, runtime, configuration, seed ledger) after signing and verify worker process creation/import never occurs. Test import shadowing (`cwd`, `PYTHONPATH`, user site, startup hooks, malicious module/DLL), path/reparse manipulation, rename/swap races during verification and launch, stale bundle IDs, and unauthorized service-control/IPC access. Instrument tests to prove no `default_rng`/`multinomial` call occurs on rejection.
6. **Deployment acceptance review.** Run ACL and service-account checks on the actual target host as the provisioned execution identity, not only in a developer pytest fixture. Review effective rights on the trust root, bundle, every parent, run ledger, output, service binary/configuration, Python runtime, and DLL directories. Preserve signed deployment/ACL evidence and independent reviewer attestation. Only after the independent reviewers accept that evidence may a separate execution authorization be considered.

The negative security tests should not run the 38,228-pair workload. A separate, explicitly authorized later run would remain the only activity that generates the registered confirmation samples.

## Windows external prerequisites

Before provisioning can be approved, an administrator must:

- Create the production trust root and bundle directories under protected system locations; set owner and explicit DACLs; disable unintended inheritance; enumerate direct and group-derived access for the actual service and operator identities; verify write, delete, delete-child, ownership, and ACL-change rights are denied where required.
- Establish a dedicated service identity and confirm that requestors cannot log on as it, launch arbitrary programs under it, change service binary/arguments, replace service configuration, debug/inject into the service, or bypass its IPC ACL. Restrict service start/stop/control to administrators/deployment automation.
- Protect every parent directory and the Python/native runtime path against reparse-point substitution and rename/replacement. Verify both ACLs and effective behavior using the actual service token; a `Get-Acl` listing or a JSON `protected: true` value is not sufficient.
- Authenticate the independent reviewer outside this repository, verify their identity/independence, provision only their public key, and retain the signed approval over the final manifest. Do not create a reviewer identity or production key in this planning task.
- Establish immutable deployment/versioning, audit-log retention, run-ledger backup/rollback controls, and a response process for interrupted runs. The worker must refuse to resume a consumed run.

These are external requirements, not claims about the current machine. The current report says `C:\ProgramData\QuantumSimulator\finite-shot-null-confirmation` is absent and no effective ACL could be inspected. Therefore provisioning remains blocked now.

## Scientific reproducibility and limits

The proposal changes where the registered algorithm can execute, not what it computes. Preserve 38,228 independent count-pairs; 1,024 shots; both registered null distributions; the existing detector and threshold; the one-sided 5% Clopper–Pearson endpoint; and the 2,217-flag pass boundary. Keep the registered seed protocol and ledger exactly as prescribed unless a new run is prospectively approved with distinct seeds. Record bundle and manifest digests, approval and run IDs, runtime/dependency hashes, seed-ledger digest, exact pair count, flag count, resource use, and result digest. The protected worker must continue to use one reference and one submitted multinomial vector per pair with the registered per-stream deterministic seeds.

This architecture can make execution reproducible and auditable. It cannot prove that a reviewer is independent merely from a valid signature, establish correctness of the statistical design, protect against a local administrator or compromised service host, or generalize a synthetic multinomial experiment to hardware or the separate three-way diagnosis problem.

## Go/no-go criterion

**Current status: NO-GO.** Do not provision external trust or run the experiment until all of the following are evidenced on the actual deployment host:

1. The only reachable route to confirmation sampling is the fixed service operation; unprivileged direct imports, legacy entry points, arbitrary Python, and alternate IPC/CLI routes are denied or unavailable.
2. A signature over the exact complete runtime bundle is verified before import; hostile shadow imports and verification-to-use replacement attempts fail before any RNG call.
3. The actual service identity cannot modify or redirect the trust root, active bundle, service launcher/configuration, or replay ledger in ways that defeat approval; administrators have documented effective ACL/service rights and tested them using that identity.
4. A separately authenticated independent reviewer has signed the final manifest and exact run ID; consumed/aborted approvals cannot be replayed or resumed.
5. All adversarial rejection tests pass without sampling, and the registered scientific configuration/hash checks are unchanged.

Failure or lack of evidence for any item is a no-go. Passing this architecture review only supports implementing and externally evaluating the proposed boundary; it is not execution approval.

## Files changed

- `reports/experiments/finite_shot_null_confirmation_architecture_proposal_20261016.md` — this planning proposal.

No code was modified. No samples were generated, and the experiment was not run.
