# Finite-shot null-confirmation control feasibility review

**Decision: NO-GO.** A fixed one-shot Windows scheduled task is technically capable of providing the required official-run boundary, but only with administrator-created task, identity, storage and trust configuration plus target-host acceptance evidence. The current repository/environment does not establish any of those external controls. Several details in the control plan need to be made concrete before implementation: the task account needs the correct batch-logon right; a task DACL must allow triggering without modification; a complete runtime closure must be fixed; anti-rollback semantics need a selected store; and the host-result attestation must be distinguished from proof of loaded code.

This review covers the official registered result only. It cannot prevent a user with the readable development checkout and Python from calling the public legacy helper or independently reproducing its computation. That output must remain nonofficial and fail the official-result verifier.

## 1. Scheduled-task identity and Windows boundaries

**Feasible in principle; not provisioned or verified here.** Task Scheduler supports a task principal that defines the account/security context, and a task has an ACL that controls who may read, update, delete and run it. Microsoft documents both the task principal and task ACL controls, including `IRegisteredTask::SetSecurityDescriptor` ([Tasks](https://learn.microsoft.com/en-us/windows/win32/taskschd/tasks), [Security Contexts for Tasks](https://learn.microsoft.com/en-us/windows/win32/taskschd/security-contexts-for-running-tasks)). However, the default task creator can read, update, delete and run their task; an administrator must register the task and set its DACL so the operator can only trigger the approved action, not read/update/delete or change principal, action, arguments, working directory, triggers or settings. Administrators and SYSTEM retain control, so they remain trusted deployment principals.

Task Scheduler can start an action under a dedicated account, but this requires a carefully selected logon type and rights. Microsoft states that `TASK_LOGON_PASSWORD` and `TASK_LOGON_S4U` tasks require the task user to have **Log on as a batch job** ([Security Contexts for Tasks](https://learn.microsoft.com/en-us/windows/win32/taskschd/security-contexts-for-running-tasks)). Thus “deny interactive/arbitrary logon” cannot mean denying batch logon entirely: the account must receive only the required task logon right, while interactive logon and unrelated use of the account are denied and tested. Exact feasibility depends on the chosen Windows edition/domain policy, task principal/logon type, and actual token; none is specified or present in this environment.

The task identity can be denied write/change rights to the worker bundle, trust key, task definition and host-attestation key by administrator-owned security descriptors on each object and its ancestors. Windows evaluates access using the caller's token and the object's security descriptor; ACLs may inherit from parent directories, and rights include write, delete, `WRITE_DAC` and `WRITE_OWNER` ([File Security and Access Rights](https://learn.microsoft.com/en-us/windows/win32/fileio/file-security-and-access-rights)). An administrator must therefore check ownership, explicit and inherited ACEs, group membership, privileges, and all parent directories. Microsoft `AccessCheck` evaluates a security descriptor against a token; Sysinternals AccessChk can report effective permissions for a named account ([AccessCheck](https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-accesscheck), [AccessChk](https://learn.microsoft.com/en-us/sysinternals/downloads/accesschk)). These tools/evidence do not exist for the absent ProgramData root reported earlier.

**Required administrator boundaries:**

- Register the fixed task as administrator; write down its principal, logon type, action executable, arguments, working directory, environment assumptions, settings and task-folder/task-file DACL.
- Let an operator start only the fixed action. Prove they cannot change/delete/read the task definition, choose parameters, alter the task principal, use the task identity for another action, or modify/stop deployment controls.
- Use a dedicated least-privilege task identity that is not an administrator. Permit only the batch-logon right required by the selected task logon type; deny interactive and unrelated logon and do not grant credentials for arbitrary use. Verify the task token's groups/privileges.
- Keep the task identity from owning protected files or directories. Allow only read/execute on the immutable code/runtime bundle and reviewer trust pin; permit the minimum required writes to a separate run ledger, output and audit location. Protect every ancestor from rename/reparse/ACL changes by the operator/task identity.
- Protect the host-attestation key so the operator cannot use it; restrict its use to the controlled completion path. Administrators and the trusted OS remain in the trust base.

The plan's statement “requestors may trigger the task but cannot change it” is not guaranteed by creating a task. It is an administrator-set DACL requirement with an actual effective-access and adversarial verification result.

## 2. Worker integrity and verification-to-execution

**Hashing followed by ordinary imports is not sufficient.** The current driver imports cryptography, NumPy and `pair_outcome` before authorization (`experiments/finite_shot_null_confirmation.py:23–35`). `verify_hash_bindings()` hashes files and calls `verify_loaded_code_origins()` (`:191–220`), while the origin function checks module identity and `co_filename` (`:223–235`); neither verifies the bytes of the live code objects. `sha256_file()` checks a path and then separately reads it (`:69–71`). The current signed artifact manifest hashes the driver, binding, detector source, legacy runner, `pyproject.toml`, configuration and seed ledger, and records runtime version strings (`artifact_manifest`, `:237–267`); it does not cover the full interpreter, standard library, all transitive Python/native dependencies or DLL bytes. `pyproject.toml` is not a dependency lock. A race-free import guarantee is therefore not present.

**Defensible alternative for the official task:** administrator stages a unique versioned bundle in an admin-owned, non-reparse directory; the bundle and all ancestors are read-only/non-replaceable to both operator and task identity before verification starts. The fixed task points to that exact version and starts a minimal bootstrap from the protected path. The bootstrap verifies a signed complete manifest before importing worker code; it launches the interpreter and worker only from that same protected bundle. The runtime must disable environment and user-site import hooks, use a clean working directory, constrain `sys.path` and native DLL search to manifested paths, and record/verify loaded module and DLL origins. No development checkout is mounted or importable in the task environment. Subsequent import-by-path is safe only under the enforced assumption that no untrusted identity can write, rename, replace or redirect the checked files until execution ends. A separately signed bundle manifest plus Windows ACLs, not a hash check alone, closes the check/use window.

The complete manifest must cover, at minimum: launcher/bootstrap; worker and its sampler; detector and all statistical/seed/confidence helpers; preregistration and fixed config; all project imports; exact Python executable/build and relevant standard library; NumPy and every native extension; cryptography and its native dependencies; dependent DLLs and their resolution paths; runtime settings; seed-ledger digest; and the result-attestation component. Any OS/runtime file loaded outside this closure must either be a separately pinned trusted platform component or block approval. Application code can reject an incomplete inventory; only deployment control can make the inventory immutable and keep imports on it.

**Current feasibility:** manifest generation and tamper-rejection logic can be developed/tested locally with fixture bundles. Actual import closure, Windows DLL resolution, code-integrity/ACL protection and the no-substitution guarantee require the target Windows image and task identity. The proposed phrase “verify loaded module/dll origins” is not yet a complete design: it needs exact inventory, hash/identity measurement, resolution rules, and evidence produced by the actual bootstrap/worker.

## 3. Trust anchor and reviewer independence

The current driver has a fixed trust path under `C:\ProgramData\QuantumSimulator\finite-shot-null-confirmation` (`experiments/finite_shot_null_confirmation.py:42–47`) and verifies Ed25519 signatures in `verify_signed_approval()` (`:275–335`). This establishes cryptographic validity only relative to the public key the process read. The binding record is explicitly local consistency, with review status pending; it is not a trust anchor or authorization. The execution-boundary review says the production path is absent and its ACL could not be read. No task setup, public-key pin or reviewer attestation exists locally.

Before implementation deployment, an administrator must pin the reviewer public key or certificate in a machine-controlled trust location and authenticate the human/key mapping outside the execution identity. The task identity may read the pin but may not write it or its parent. The signed approval is untrusted input: the verifier checks canonical bytes/signature, reviewer/key IDs, experiment ID, complete bundle digest, preregistration/configuration, seed ledger, resource limits, trust-root identity and fresh run ID. A duplicate-key/malformed/tampered payload or wrong key fails before worker start. The independent reviewer keeps the private signing key externally and signs only after reviewing the final manifest and boundary evidence.

A valid Ed25519 signature proves that the corresponding private key signed the payload. It does **not** prove reviewer identity, institutional authorization, independence, completeness of review, or security of the public-key file. Those require an external organizational identity/independence check and an administrator-protected trust mapping. A production approval cannot be produced from the current state; fixture keys in tests are not evidence.

## 4. Durable anti-replay and run consumption

Current mechanics: `verify_signed_approval()` looks for `<run_id>.consumed` before signature completion (`experiments/finite_shot_null_confirmation.py:299–335`), then `execute_confirmation()` calls `consume_approval_run()` before seed preflight (`:482`). The latter uses exclusive file creation (`:338–351`), so on a stable local filesystem one concurrent caller should win that create and the other should see an existing marker. That narrow property is useful but not a full durable anti-replay system: there is no concurrency test, no restart/restore test, no fsync/transaction guarantee, no anti-rollback policy, and no ACL evidence. The marker's trust root is absent. The current interruption test stops in preflight and retries in the same process/test directory (`tests/test_finite_shot_null_confirmation.py:293–311`); it does not exercise process restart or power-loss recovery.

For a fixed task, all launches must share one protected state store with a unique constraint on the signed run ID and an atomic transition such as `approved → consumed/running` committed durably before any RNG call. A repeated or simultaneous launch must lose that transaction before worker entry. Store `complete` only after a full output is durably written and verified; any exception/termination after consumption is terminal `aborted` (or terminal `running` treated as aborted), never resumable. Record a tombstone even when no draw occurred. If a backup/snapshot can restore the store to before consumption, it permits replay; either prevent/detect rollback with a separately protected monotonic/remote anchor or prohibit such restore for this run and document that host restore ambiguity means NO-GO. A local database transaction cannot by itself detect a whole-volume rollback.

Local tests can exercise transaction APIs with temporary databases, concurrent processes, injected crashes, uniqueness constraints and “no RNG before commit.” The target Windows task identity must then verify actual storage semantics, ACLs, restart survival, backup/restore policy, concurrency and durability. No such evidence is currently available.

## 5. Official-result attestation feasibility

The plan correctly notes that the current result JSON and successful process exit are insufficient. The official output needs a detached attestation signed by a protected host key over a canonical envelope containing:

- exact result-file digest and canonical serialization/version;
- experiment ID and preregistration SHA-256;
- complete worker/runtime manifest digest and bundle version;
- reviewer approval digest, reviewer key ID and decision;
- trust-root identity and one-time run ID;
- seed allocation/ledger digest and registered seed protocol;
- completion state, exact pair count (38,228), shots (1,024), flags, endpoint and resource-accounting digest;
- task identity/task identifier, runtime identity, and completion-ledger record identifier.

The host key should be protected by a Windows machine cryptographic provider with key ACLs that exclude the operator and permit use only from the controlled task path. The verification public key is provisioned separately. The task must write only a sidecar after the full result and completion record are durable; partial outputs have no completion signature. The output verifier rejects unknown keys, mismatched digests, a result without completed status, wrong counts/registration/run ID, or duplicate/untrusted approval.

**Important limit:** a host signature alone does not prove which code executed. If arbitrary code can run under an identity authorized to use the key, it can sign fabricated results. Therefore key access is meaningful only with the locked task action, protected worker/runtime, task identity isolation, and a restricted signing operation. A software key available to the same arbitrary Python account is not adequate. The current repository has no host key, signature implementation, verified key ACL, output verifier or attested result. Whether an administrator-managed CNG key suffices depends on the exact OS provider, key ACL and process identity configuration; if the threat model includes compromise of the task process or local administrator, a stronger measured/remote attestation design would be required and is not specified here.

## 6. Scheduled task versus protected service

| Dimension | Fixed one-shot scheduled task | Protected service |
|---|---|---|
| Smallest single-run design | Best fit if an admin can register a fixed action under a dedicated account and set a task DACL that grants the caller only run/trigger access. No persistent broker or custom request listener. | Larger than needed for one run; requires a long-running process and IPC server. |
| Execution identity | Task principal provides the run security context. Must have the required batch logon type/privilege and no interactive/general-purpose logon. | Service account/SID provides process context; requires service DACL, startup/recovery configuration and service-control policy. |
| Action/argument integrity | Administrator-fixed executable, arguments, working directory and settings protected by task DACL. Operator may only trigger; task history helps audit but does not attest output. | Fixed internal dispatch plus protected service configuration; request parser/IPC is a larger attack surface. |
| Trust/bundle/ACL | Same external trust root and immutable bundle requirements. A task does not make files immutable automatically. | Same requirements; additionally protect service binary, configuration, IPC endpoint and update path. |
| Replay/concurrency | Scheduler “multiple instances” policy can reduce accidental duplicates, but is not a replacement for atomic run-ID consumption; other invocations/restores remain possible. | Service can serialize requests but still requires a durable atomic ledger across restarts and rollback protection. |
| Maintenance/verifiability | Simpler to review one task action, principal, task DACL and history; still needs target-token ACL tests and deployment evidence. | More ongoing code, logging, IPC hardening, patching and incident handling. More appropriate for recurring/multi-client controlled jobs. |

**Recommendation:** do not implement a persistent service unless a fixed task cannot satisfy trigger-only permissions, identity isolation, controlled request flow, or replay requirements on the target host. Task Scheduler supports principals and task DACLs, but those features must be configured and verified; task creation alone does not prove them. Microsoft documents that the task creator normally can read/update/delete/run their task and that the task DACL defines those rights ([Security Contexts for Tasks](https://learn.microsoft.com/en-us/windows/win32/taskschd/security-contexts-for-running-tasks)). The smallest defensible plan is a fixed admin-registered task plus protected bootstrap/worker bundle, external trust store, durable run ledger, and host attestation. If the task DACL or launch environment cannot make the task a fixed, non-parameterized operation, NO-GO; then evaluate a service as a separately specified design.

## 7. Controls feasible now versus external prerequisites

| Control | Feasible locally without elevation | Requires Windows administrator / separate identity | Requires reviewer/external trust |
|---|---|---|---|
| Approval parser/signature logic | Yes: malformed, duplicate-key, stale, wrong-key, altered signature and manifest tests with ephemeral test keys. | Protect verifier/bootstrap and trust pin; exercise actual trust paths under task identity. | Verify real person and independence; sign the final manifest/run ID. |
| Task request/launch contract | Mock fixed-input validation, no user paths/commands, and spy-RNG fail-before-run logic. | Create task principal/logon type and task DACL; test that operators can trigger only the fixed action, cannot modify it or launch arbitrary code as task identity. | Reviewer reviews final task/boundary evidence. |
| Bundle manifest/import closure | Fixture hash inventory, mutation tests and controlled import-shadow tests. | Pin actual Python/runtime/native dependency paths; deny replacement on all files/ancestors; measure actual loaded modules/DLLs. | Approve the final exact bundle manifest. |
| Trust-root ACL | Test fail-closed behavior against temp fixtures; current Windows junction test exercises reparse rejection. | Provision absent ProgramData root; inspect DACL/inheritance/owner/groups/effective rights using actual tokens; attempt prohibited writes. | Authenticate reviewer key mapping. |
| Run consumption/replay | Unit/temporary-store transactions, concurrency test, crash injection and retry-before-RNG behavior. | Prove local storage durability, restart handling, ACLs and backup/rollback protection under actual identity. | Approval binds unique run ID; any new allocation requires new review. |
| Result attestation | Test canonical envelope/signature verification with a disposable host test key; reject forged results. | Provision and ACL-protect machine signing key and output/audit store; test key access from task and denial elsewhere. | Pin the attestation public key for independent verification; reviewer approval remains separate. |

No unit test can establish actual Windows effective permissions, that the task identity is noninteractive, that the task action cannot be altered, that a machine key is restricted as intended, or that a restored disk cannot replay an old run. These require retained target-host evidence.

## 8. Interruption and seed reuse

- **Resume the same run:** prohibited. Consume the run ID durably before the first random draw; after interruption or uncertain state, treat the run as spent and do not resume from checkpoint.
- **Restart the same work using the current seed ledger:** not an independent rerun. `stream_seed()` deterministically derives the same 76,456 stream seeds from the registered protocol/condition/replicate/role (`experiments/finite_shot_synthetic_characterization.py:45–55`). A fresh run ID or approval does not create new evidence. Do not rerun and describe it as independent.
- **New independent seed allocation:** not allowed by current preregistration. It requires a prospective amendment/new preregistration specifying the new seed derivation, ledger hash, uniqueness/prior-overlap checks, sample size, analysis and decision procedure; independent review and a new signed approval before any sampling.
- **Partial/failed run:** keep a protected terminal `aborted` record and any partial data separately marked nonofficial; never sign it as complete, pool it with a retry, or claim 38,228 pairs. If host restart/rollback makes state uncertain, NO-GO.

## 9. Explicit acceptance and go/no-go

### Negative acceptance matrix

Every rejection case must assert task/worker entry and RNG count are zero. Local fixture tests are necessary but do not replace deployment checks.

| Attack | Local test | Required deployment test |
|---|---|---|
| Tampered or pending approval | Change decision/signed bytes; reject before worker/RNG. | Submit tampered input through the installed task interface; verify logged denial. |
| Untrusted reviewer key / changed trust pin | Sign with wrong ephemeral key; verifier rejects. | Verify operator/task token cannot replace machine trust pin; independently document reviewer identity/key mapping. |
| Duplicate-key approval document | Duplicate decision/key/run ID JSON fields; strict parser rejects before interpretation. | Same denial through the protected task's actual input path. |
| Modified worker/runtime/dependency | Mutate each fixture bundle item after approval; pre-import verification rejects. | Attempt write/rename/delete/reparse as operator and task identity while task is active; OS denies all protected mutations. |
| Import or DLL substitution | Hostile cwd/env/user-site/startup/module/DLL fixtures; bootstrap rejects or loads only exact manifested paths. | Repeat under task token with actual runtime and search paths; preserve loaded-file/hash log. |
| Replay of consumed run ID | Reuse transaction fixture; only first commit accepted. | Restart task/host and replay; old ID rejected; test restoration/rollback policy. |
| Concurrent start | Two local processes contend on one fixture transaction; exactly one consumes; loser makes zero RNG calls. | Trigger same task/run concurrently; verify Task Scheduler policy plus ledger yields one worker entry. |
| Interrupted run | Inject termination before and during a spy worker; run becomes terminal and retry is rejected. | Kill/reboot during execution and restart; no resume or duplicate worker; partial artifact has no attestation. |
| Forged result/sidecar | Modify result, invent metadata, or sign with wrong host fixture key; verifier rejects. | Attempt to write/sign through output/key path as operator; verify actual key ACL and signed completion channel. |
| Direct legacy helper invocation | Existing test demonstrates direct helper reaches RNG; classify output as untrusted. Ensure official verifier refuses any unattested file. | Confirm production acceptance consumes only verified signed official artifacts; if literal prevention is required, execute on a host without accessible legacy checkout and restrict arbitrary code. |
| Task alteration/arbitrary action | Mock task request accepts no caller args/path/module/command. | As operator, attempt to read/update/delete task and replace executable/arguments/principal; task DACL denies all except the authorized trigger operation. |

### GO criteria

Change from NO-GO to **GO for a separate execution-authorization review only** after:

1. The official-result claim and legacy-helper scope are explicitly accepted; direct checkout helper output cannot pass the official verifier.
2. An administrator-provisioned fixed Task Scheduler action uses a dedicated least-privilege identity, correct batch-logon configuration, task DACL, protected executable/arguments and no operator-controlled parameters; effective access is verified with actual tokens.
3. The reviewer public key is pinned by an administrator outside requestor/task write control; reviewer identity and independence are verified externally; final approval signs the exact experiment, preregistration, bundle, configuration, seed ledger, resource controls and unique run ID.
4. The full runtime/dependency bundle is complete, immutable to operator/task identity and verified before import; actual module/DLL inventory is confined to and matches that bundle; substitution tests fail before RNG.
5. A protected durable ledger atomically consumes run IDs before RNG and survives concurrency, process/host restart and the documented backup/restore policy; interrupted IDs are terminal.
6. A protected host key signs only a completed result envelope; independent verification binds exact result bytes, run/approval/bundle/registration/seed ledger and complete outcome. Forged/partial files fail.
7. All local negative tests pass without registered sampling, and every OS-level test passes on the deployment host under the actual operator/task identities. Retain administrator and reviewer evidence.
8. Registered science remains unchanged: 38,228 pairs, 1,024 shots, null distribution, detector/threshold, one-sided 5% Clopper–Pearson rule and 2,217-flag cutoff. A new seed allocation requires prospective review.

Any missing evidence means **NO-GO**. Passing this feasibility review does not authorize provisioning or execution.

## Sources

- Microsoft, [Tasks](https://learn.microsoft.com/en-us/windows/win32/taskschd/tasks) — task principals, actions, and setting task/folder ACLs.
- Microsoft, [Security Contexts for Tasks](https://learn.microsoft.com/en-us/windows/win32/taskschd/security-contexts-for-running-tasks) — task run context, batch-logon requirement for password/S4U task logon, and task DACL permissions.
- Microsoft, [File Security and Access Rights](https://learn.microsoft.com/en-us/windows/win32/fileio/file-security-and-access-rights) — token/security descriptor checks, inherited ACLs, file/directory rights.
- Microsoft, [AccessCheck](https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-accesscheck) and [Sysinternals AccessChk](https://learn.microsoft.com/en-us/sysinternals/downloads/accesschk) — checking requested rights against tokens/security descriptors and reporting effective permissions.

## Files inspected and created

Inspected:

- `reports/experiments/finite_shot_null_confirmation_control_implementation_plan_20261016.md`
- `reports/experiments/finite_shot_null_confirmation_execution_design_decision_20261016.md`
- `reports/experiments/finite_shot_null_confirmation_implementation_readiness_20261016.md`
- `reports/experiments/finite_shot_null_confirmation_trust_gate_20261016.md`
- `reports/experiments/finite_shot_null_confirmation_security_remediation_20261016.md`
- `reports/experiments/finite_shot_null_confirmation_execution_boundary_review_20261016.md`
- `experiments/finite_shot_null_confirmation.py`
- `experiments/finite_shot_synthetic_characterization.py`
- `src/quantum_simulator/statistics.py`
- `tests/test_finite_shot_null_confirmation.py`
- `reports/experiments/finite_shot_null_confirmation_driver_binding_20261016.json`
- `reports/experiments/finite_shot_null_confirmation_preregistration_20261016.json`

Created:

- `reports/experiments/finite_shot_null_confirmation_control_feasibility_review_20261016.md` — this feasibility decision.

No code, tests, preregistration, approval, seed ledger, result, or prior report was changed. No Windows identity, task, ACL, service or key was provisioned. No dependencies were installed and no experiment or sampling was run.

**Remaining external blockers:** the ProgramData trust root is absent; no target task identity or task DACL exists; no immutable runtime bundle or verified import closure exists; there is no qualified durable anti-rollback ledger, protected host-attestation key or official-result verifier; reviewer identity/key are not independently authenticated; and target-host tests/evidence have not been produced.

## 10. Control acceptance records and readiness checklists

The table below makes the proposed guarantee, actual mechanism, evidence basis, test boundary, failure response and acceptance criterion explicit for each control. “Local test” means a software/fixture check only; it never substitutes for Windows administrator evidence.

| Control and proposed guarantee | Actual enforcement mechanism and current evidence | Local testability | External prerequisite | Failure mode and acceptance criterion |
|---|---|---|---|---|
| **Protected trust root and isolated task identity:** the requestor can only trigger a fixed action; the task account cannot alter trust, worker, task definition, dependencies or attestation key. | Administrator-owned trust and bundle paths plus ancestors; separate least-privilege task principal; task/folder DACL grants trigger without update/delete; task account receives required batch logon only. Current driver expects a ProgramData trust path (`experiments/finite_shot_null_confirmation.py:42-47`), but it is absent/unverified. | Mock request validation and verify no caller-controlled command/path; test fail-closed behavior for missing/malformed trust. | Admin configures task, account, DACL, logon rights, ownership and inherited ACLs; tests with actual operator/task tokens and denied mutations. | **Reject/NO-GO** if task or trust can be altered or arbitrary code can run as task identity. **Accept** only with task definition/DACL, effective access and token/privilege evidence plus denied write/replace attempts. |
| **Immutable worker/dependency bundle and loaded-code binding:** only the reviewed bytes and dependencies execute. | Admin-staged immutable versioned bundle; protected bootstrap verifies a complete manifest then launches from that same bundle with constrained import/DLL resolution. Existing `verify_hash_bindings`/`verify_loaded_code_origins` (`finite_shot_null_confirmation.py:191-235`) do not bind live code bytes; `artifact_manifest` (`:237-267`) is incomplete. | Fixture mutation, import-shadow, hostile environment/cwd and symlink/reparse tests; compare expected/observed fixture closure. | Admin seals full interpreter, standard library, Python/native dependencies, DLLs and bundle ancestors; target-host run records actual loads and denies replacement. | **Reject/NO-GO** on any unmanifested load, writable ancestor, substitution race or incomplete inventory. **Accept** only if actual loaded file identities/hashes equal the approved closure and substitution attempts fail before sampler entry. |
| **Reviewer trust and approval:** approval is from an independently identified reviewer and binds the exact proposed run. | Strict canonical payload + signature against an administrator-pinned key, binding experiment, preregistration, bundle/config, seed allocation, trust root and unique run ID. Current `verify_signed_approval` (`:275-335`) can prove key possession only; reports record pending review and no production key. | Disposable test-key tests for invalid signature, wrong key, duplicate JSON keys, altered manifest and replay; must never call keys “reviewer credentials.” | External organization verifies reviewer identity/authority/independence; admin pins public key outside task/operator write control; reviewer signs final exact manifest and run approval. | **Reject/NO-GO** for unknown key, pending status, ambiguity, mismatch or unverified identity. **Accept** only when signature verifies against independently provisioned key and separately retained evidence verifies reviewer identity and review scope. |
| **Durable anti-replay and interruption:** a run ID is consumed once; partial runs cannot resume or masquerade as complete. | Protected atomic durable store commits consumption before any random draw; terminal states `running`, `complete`, `aborted`; no resume; rollback prevented/detected. Existing driver uses marker existence and exclusive file creation (`:299-351`), called before preflight at `:482`; existing test covers same-process preflight failure only (`tests/test_finite_shot_null_confirmation.py:293-311`). | Temporary database/file-store transaction, concurrent claimant, process-kill injection, restart, retry and “no RNG before commit” tests. | Admin selects durable store, ACLs and backup/restore policy; actual concurrent task, host restart, power-loss and rollback tests. Add external monotonic/remote anchor if whole-volume rollback is in scope. | **Reject/NO-GO** if state is uncertain, replayable, resumable or can roll back undetected. **Accept** when one claimant consumes, all others fail before sampling, interruption remains terminal after restart, and restore policy is enforced/tested. |
| **Official result and host attestation:** verifier can authenticate completed output without trusting a process-writable JSON. | Canonical result envelope signed only after durable completion by a protected host key; bind result digest, experiment/preregistration, full bundle, approval/key, trust root, run ID, seed digest, counts/outcome/resource accounting and ledger record. Current driver writes unsigned JSON (`:523-543`); no signer/verifier exists. | Disposable host test key; altered, forged, incomplete and mismatched envelope tests. | Admin provisions key with restricted use, protects output/audit store and supplies public verification material; independent verifier checks signed bindings. | **Reject/NO-GO** if key is usable by arbitrary task-account code, any field is unbound, or partial/forged output verifies. **Accept** only when an independent verifier validates signature and every digest/ID/count against durable completion evidence. |

### Before implementation

- [ ] Admin confirms target Windows policy and a task logon mode that grants required batch-logon rights while disallowing general-purpose interactive use.
- [ ] Admin/security owner approves an operator/task/admin permission matrix for task, trust, bundle, runtime, ledger, output and attestation paths, including every parent and reparse-point policy.
- [ ] Choose the durable anti-replay store and state its restart, crash, backup and rollback guarantees; identify a separately controlled monotonic anchor if required.
- [ ] Freeze a technically complete worker/runtime/import/DLL inventory and bootstrap design; identify trusted OS components outside that manifest.
- [ ] Specify the result envelope, when completion is signed, the protected signer interface and a separate verification procedure.
- [ ] Explicitly classify outputs from the accessible development checkout and legacy helper as nonofficial; only the protected task plus verifiable attestation can create official evidence.
- [ ] Approve a no-sampling deployment probe plan for actual task-token permissions, immutability, task DACL, import closure and key access.

### Before experiment execution

- [ ] Fixed admin-created task, dedicated principal, trigger-only task DACL, correct logon policy and protected fixed action are installed; actual token and effective-access evidence is retained.
- [ ] Reviewer key is admin-pinned outside execution-identity control; reviewer identity, authority and independence are externally authenticated; the reviewer has signed the exact experiment, preregistration, immutable bundle, configuration, seed ledger, resource cap and unique run ID.
- [ ] Complete worker/runtime manifest has independent review; observed actual Python/native module loads match; operator/task cannot replace files or redirect paths, including while the task is running.
- [ ] Atomic run consumption is durable before RNG and survives tested concurrency/restart/interruption; no resume is possible; backup/rollback behavior is proved safe or independently anchored.
- [ ] Host-attestation key is protected; completion is signed only after result and ledger are durable; independent verifier rejects altered, partial, forged, stale or mismatched records.
- [ ] Local negative tests pass and all OS-level tests pass on the actual host under actual operator/task identities. Unit tests alone do not prove ACL effectiveness, isolation or host attestation.
- [ ] Independent review confirms the registered science is unchanged: 38,228 pairs, 1,024 shots, registered null, detector/threshold, one-sided 5% Clopper–Pearson rule and 2,217-flag boundary. Any new seed allocation has prospective amendment and approval.

**Recommendation:** prefer the fixed one-shot scheduled task over a persistent service for this single bounded run, conditional on administrator evidence that its task DACL and identity create a fixed, trigger-only boundary. A service is warranted only if that boundary cannot meet the stated controls and a separate service design is reviewed. Current status remains **NO-GO**; neither this report nor local tests authorize trust provisioning or experiment execution.
