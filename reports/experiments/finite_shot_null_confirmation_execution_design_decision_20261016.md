# Finite-shot null-confirmation execution design decision

**Decision: conditional design recommendation only; current execution is NO-GO.** For this one bounded run, prefer a fixed, administrator-controlled one-shot Windows scheduled task over a persistent service, but only if the task, worker identity, trust root, bundle, replay ledger, and official-result attestation meet the acceptance criteria below. A task configured as an ordinary user script is not sufficient. If administrators cannot establish equivalent isolation through Task Scheduler, use a protected service. Neither design is ready to provision today.

This review is planning only. It does not authorize trust provisioning or experiment execution.

## Security claim and what counts as official

The enforceable claim should be narrowly stated:

> An **official confirmation result** is a complete result produced by the fixed, approved worker after the protected launcher has verified an independently signed approval for the exact worker bundle, registered configuration and seed ledger, and atomically consumed the unique run ID. The result is accepted only with a verifiable host execution attestation tying its bytes and provenance to that run.

The current JSON result is not sufficient evidence of this claim by itself. `execute_confirmation()` writes result fields such as experiment ID, run ID, preregistration hash, driver hash, seed-ledger summary, endpoint, and resource counts (`experiments/finite_shot_null_confirmation.py:523–543`), but it does not sign the result or an execution record. Its default output is under the repository's reports tree (`:40`), and create-only mode prevents overwrite by that particular code path, not by another writer. Therefore a file at that path, or a JSON field naming a run ID, does not prove that the approved worker generated it.

Require a detached **host execution attestation** (for example, a signature made by a non-exportable machine key available only to the protected worker identity) over the completed result digest plus experiment ID, preregistration digest, bundle digest, approval digest, consumed run ID, seed-ledger digest, actual pair count, and completion status. The attestation verifier's public key must itself be pinned outside the output directory. The signing operation must be available only after successful completion and must not be exposed as a general signing API. Preserve the statistical result schema or add a separate signed sidecar; do not modify the registered acceptance rule. If such a key and protected audit store are not provisioned, the official result cannot yet be distinguished from an arbitrary local file.

The legacy `pair_outcome` helper and old characterization runner remain useful for their own prior experiment and are not authorized evidence for this confirmation. Their output cannot establish the exact signed bundle, protected run ID, full confirmation workload, or official completion attestation. The direct helper may still be called by a user who has access to the development checkout; no service or scheduled task can prevent arbitrary sampling or reimplementation on an unrestricted machine. The security boundary protects the canonical official run and acceptance evidence, not every possible computation by every user.

## Evidence from the current code and reports

### Sampling and execution paths

The driver imports `pair_outcome` before executing any authorization logic (`experiments/finite_shot_null_confirmation.py:23–35`). `execute_confirmation()` checks an explicit flag, fixed paths, preregistration/binding, and signed approval (`:456–480`), consumes the run ID (`:482`), derives the seed ledger, and then calls `pair_outcome()` for each ledger row (`:491–513`). The helper creates two multinomial count vectors with two `default_rng(...).multinomial(...)` calls (`experiments/finite_shot_synthetic_characterization.py:52–55`) and calls the detector. The detector is deterministic for those counts.

There are independent unprotected paths: the old `run()` calls the helper in its loop (`experiments/finite_shot_synthetic_characterization.py:124–149`) and the old module's `main()` calls `run()` (`:191–200`). Any caller can import and call `pair_outcome` directly. The test `test_public_pair_helper_reaches_rng_without_confirmation_authorization` in `tests/test_finite_shot_null_confirmation.py:366–377` demonstrates this by replacing the RNG with a sentinel and invoking the helper. Thus the existing CLI flag and approval gate protect only the driver's `execute_confirmation()` path.

The driver’s artifact checks also do not bind bytes atomically to code already loaded. Imports happen at `:23–35`; `verify_hash_bindings()` hashes source paths (`:191–220`); `verify_loaded_code_origins()` checks object identity and `co_filename` (`:223–235`), not loaded bytecode identity. `sha256_file()` path-checks and later opens by path (`:69–71`), leaving a path substitution race. The current manifest (`artifact_manifest()`, `:237–267`) records hashes for the driver, binding, detector file, legacy runner, `pyproject.toml`, configuration and seed ledger, plus runtime version/platform strings. It does not cover the exact Python executable and standard library bytes, every transitive Python/native dependency, or loaded DLLs.

### Authorization and replay evidence

`verify_signed_approval()` in `experiments/finite_shot_null_confirmation.py:275–335` parses the fixed trust-key and approval records, checks reviewer/key/trust-root/run IDs and review fields, compares a current manifest, then verifies the Ed25519 signature. Duplicate JSON keys are rejected by `strict_json_file()` / `_unique_object()` (`:109–120`). Cryptographic verification proves the signature matches the key supplied to the verifier; it does not authenticate the human behind that key or protect the key file. The trust root at `C:\ProgramData\QuantumSimulator\finite-shot-null-confirmation` is currently absent according to the execution-boundary review; effective ACLs have not been inspected.

`verify_signed_approval()` checks for a consumed marker (`:305–311`); `consume_approval_run()` creates it with exclusive file mode (`:338–351`) before seed preflight and sampling (`:482`). The test `test_interrupted_preflight_consumes_authorization_and_cannot_resume` confirms that a preflight interruption cannot reuse the same marker in the same temporary trust directory (`tests/test_finite_shot_null_confirmation.py:293–311`). These tests do not establish crash durability, restart behavior, concurrent trigger safety, anti-rollback behavior, or protection against deletion/restoration of the marker. Those guarantees depend on deployment storage and ACLs.

The driver-binding record says `status: implementation_hash_bound`, while `independent_review_release.status` is `pending`; its note explicitly says it is local consistency only, not authorization. The prospective preregistration fixes 38,228 pairs, 1,024 shots, the null distributions, seed condition and seed ledger (`reports/experiments/finite_shot_null_confirmation_preregistration_20261016.json`). The registered ledger has 76,456 unique streams, zero overlap with prior 1,024-shot streams, and digest `222e5522204fd09f56f860c4a474130b2357d34997c9393f0d47521953f341ad`. Its acceptance remains the one-sided 5% Clopper–Pearson rule and 2,217-flag pass boundary. None of these scientific settings should change in this infrastructure work.

The execution-boundary review reports the focused test result as 15 passed and 1 skipped; the junction test passed and the symlink-creation test skipped. The security remediation and trust-gate reports both state that the production trust root and reviewer approval are not provisioned. The implementation-readiness review correctly identifies that neither a service nor a task prevents access to the old helper while the ordinary user can still run the checkout.

## Threat model and proposed boundary

```text
Independent reviewer (private key remains external)
       | signs exact approval: experiment + bundle + config + seed ledger + run ID
       v
Administrator-protected trust pin and approval store ----> fixed one-shot task
       ^                                                     |
       | actual ACL/key identity evidence                    | launches fixed action only
       |                                                     v
Administrator-protected immutable bundle <--- verify before import --- isolated worker identity
       |                                                               | consume run ID durably
       |                                                               | sample only after authorization
       |                                                               v
       +-------------------------- bundle digest <--- result + signed host attestation
                                                                      |
Ordinary requestor: may request/trigger fixed task, but cannot choose code, args,
paths, config, trust key, output path, or run state; cannot execute as worker identity.
```

The task/service boundary does **not** stop a separate user process from calling the legacy helper in a still-accessible developer checkout. It must ensure that such output cannot be accepted as official. If policy means literally “no call to any sampler using these parameters anywhere,” the protected host must not expose the checkout/helper or a general-purpose Python environment to that identity; even then, software cannot prevent a knowledgeable person from independently reproducing the algorithm on another machine. The auditable and meaningful target is the official confirmation artifact.

## Scheduled task versus protected service

| Requirement | Fixed one-shot scheduled task | Protected Windows service |
|---|---|---|
| Threat model | One preconfigured operation. Operator may trigger it but cannot alter action, args, identity, working directory or files. Task Scheduler launches the worker under a dedicated non-interactive identity. | Long-running privileged broker. Clients submit a narrow request over authenticated local IPC; service validates and launches/hosts the worker. |
| Execution identity | Dedicated task identity; deny interactive and arbitrary logon. Only Task Scheduler can use it. Protect task definition and credential/configuration. | Dedicated service identity/SID; deny interactive logon; restrict service start/stop/config/debug rights and IPC ACL. |
| Trust-anchor protection | Administrator-protected machine trust store, readable by task identity but not writable; approval can be fixed in protected storage or accepted as untrusted input and verified against that pin. | Same trust store requirements; service reads it and must not allow caller-selected trust paths/keys. |
| Worker/dependency integrity | Versioned, immutable bundle accessible only to task identity and administrators; verify full bundle before import. Fixed action points to one version and no caller args. | Same bundle requirement. Service adds an extra protected executable/configuration and service-update path to the integrity closure. |
| Import control | Clean task working directory; controlled Python runtime; no checkout mount, user site, `PYTHONPATH`, `.pth`, `sitecustomize`, unpinned DLL paths, or alternate imports. | Same controls, plus service IPC must not expose Python, module, helper, path, callback or arbitrary command requests. |
| One-time run/replay | Durable transaction in a protected ledger; same run ID cannot be consumed twice even under concurrent task starts. Interrupted run is terminal. | Same ledger. Service can serialize requests naturally, but it still needs durable state across process/host restart. |
| Recovery | Task restart sees consumed ID and fails. No resume. A partially completed run is abandoned. | Service restart likewise fails on consumed ID. No resume. Persistent service availability is not a reason to reuse seeds. |
| Complexity and maintenance | Small fixed action and no server/IPC implementation. Task SDDL, action, identity, trigger and update controls still need administrator setup and deployment testing. Best fit for this single run if these properties are demonstrable. | More code, endpoint hardening, lifecycle, logging, patching and availability work. Better fit if multiple future operations or centrally managed requests are expected. |
| Verifiability | Easier to inspect a single action, task ACL and execution history; still requires testing effective access under actual operator/task identities. | More complex; must inspect service DACL, IPC security, executable/configuration ACLs, account privileges, child process controls and logs. |

**Recommendation:** use a fixed one-shot scheduled task for the single bounded experiment, conditional on proving that only the fixed task action can run the worker under the task identity and that the operator cannot modify or substitute any component. It is smaller than a persistent service without weakening the required trust boundary if those claims pass deployment tests. If task ACLs cannot be configured so the user can only trigger but not alter the task, or if a narrow request API and serialized multi-run support become required, use a service. A normal user-owned scheduled task, script, venv, or Python CLI is not an equivalent option.

## End-to-end approval, code and output trust

1. **Trust anchor:** Administrator provisions the reviewer public-key/certificate pin in a LocalMachine trust store or protected configuration path whose owner, DACL, inheritance and parent paths are controlled. The task identity is read-only; the operator cannot change the key, key-to-reviewer mapping, task action or trust configuration. The reviewer private key remains outside the host. An administrator authenticates reviewer identity/independence out of band. No trust file/JSON field can attest to its own protection.
2. **Approval:** A canonical signed approval binds experiment ID, unchanged preregistration digest, full worker-bundle digest, detector/configuration/resource settings, fixed seed-ledger digest, trust-pin identity and a fresh UUID run ID. It is rejected for malformed or duplicate-key JSON, invalid signature, wrong/untrusted key, stale bundle, changed config, pending decision or consumed run. The binding record's pending field is never used as authorization.
3. **Actual code:** Administrator stages a versioned worker bundle in a protected directory, including all source modules and the exact interpreter/runtime closure: Python binary and relevant standard library, NumPy and native extensions, cryptography and native libraries, detector, seed/confidence helpers, all imported project code and DLL dependencies. A dependency lock with artifact hashes is required; version strings and `pyproject.toml` alone do not suffice. Before import, the fixed launcher verifies the bundle manifest and signature. It launches from the protected bundle itself, not from the checkout. It constrains module/DLL resolution to the bundle/runtime and disables environment/user import hooks. The task identity cannot create, replace, rename, delete, reparse or change ACLs on any bundle component or ancestor between verification and import/use. This OS property closes the check-to-use race; an application hash check cannot establish it alone.
4. **Sampler access:** The worker has no external sampler RPC, Python prompt, command field, arbitrary module argument, or user-supplied callback. Authorization happens before importing/using sampling logic, and a valid unique run is durably consumed before the worker reaches RNG. Direct calls to the historical helper by a developer outside this protected task remain possible but produce no trusted official result.
5. **Replay state:** Use a transactional, durable single-use run ledger protected from the requestor and from replacement by the task identity except through the fixed consume operation. Consume before any RNG call, flush/commit durably, and reject all later starts for that run ID. Store/backup rollback must not silently restore a pre-consumption ledger. If anti-rollback cannot be evidenced, execution remains blocked.
6. **Official result:** Write output to a protected or controlled result channel and create a separate host attestation over the exact result digest and run provenance. Verify that attestation against a separately pinned host public key and match it to the reviewer approval, bundle, fixed ledger, 38,228 actual count and completed status. Keep partial/unattested files nonofficial. A result file written by the current Python code without this attestation is not proof of execution under the trust boundary.

## Replay and interruption policy

| Operation | Policy under the current preregistration |
|---|---|
| Resume the same authorized run after interruption | **Prohibited.** The current marker design already treats the run ID as consumed before seed preflight; deployment must preserve this fail-closed rule durably across restart. Do not continue from a checkpoint or reuse the approval. |
| Restart from the beginning with the same fixed seed ledger under a new approval | **Not an independent confirmation and not an allowed recovery.** The seed ledger deterministically generates the same streams. A new run ID does not make those streams new evidence. Do not rerun and pool or relabel it as an independent sample. |
| Independent rerun with a new seed allocation | **Not authorized by the current preregistration.** Requires a new prospective preregistration or formally reviewed amendment specifying nonoverlapping seed derivation, seed-ledger digest, sample-size and analysis policy; independent review and new approval. Do not change the 38,228-pair result or acceptance criterion retrospectively. |
| Failure before any RNG after run consumption | The run is still spent; abandon it. This is conservative but avoids ambiguous state. |
| Crash after all draws but before complete, attested output | Treat the run as incomplete/nonofficial unless a separately specified, auditable commit protocol can prove completion without rerunning or modifying observations. The current driver has no such protocol. |

## Locally testable guarantees and external prerequisites

### Testable without administrator privileges

- Strict JSON parsing, duplicate-key rejection, signature verification using disposable test keys, wrong-key/tampered/pending/stale approval rejection.
- Exact registered configuration, threshold, cutoff, count cap and seed-ledger calculations using deterministic/analytical fixtures only.
- Injected spy RNG proves every rejection path exits before RNG; do not call multinomial in these tests.
- Fixed worker/task request schema rejects caller-selected commands, paths, modules, output paths, callbacks, settings or replacement hashes.
- Bundle-manifest mutation tests prove signed stale artifacts are rejected before worker import; hostile import-path tests can exercise the bootstrap's declared allow-list.
- Same-process single-use/replay logic and simulated interruption behavior against a fake transactional ledger.
- Static call-graph checks identify all RNG/multinomial entry points in the protected bundle and ensure no alternate CLI/RPC path.

These tests establish application logic only. Current focused tests already cover test-key signature mechanics, duplicate keys, tampering/pending state, the direct legacy-helper bypass (by interception), deterministic ledger and preflight interruption. They do **not** cover host restart, concurrent launches, durable ledger rollback, service/task identity isolation, complete dependency integrity, or an official-result signature. The latest execution-boundary report records 15 passed and 1 skipped; this review did not rerun them.

### Requires external Windows administrator/deployment evidence

- Provision the actual dedicated task/service identity; ensure no interactive or arbitrary logon; inspect effective account groups and privileges.
- Provision and independently verify ACLs for the task/service action/configuration, trust pin, bundle, all parent directories, Python/runtime/DLLs, result attestor key, durable run ledger, logs and output. Confirm denial of write, delete, delete-child, rename, ownership/ACL changes and reparse substitution for the operator and worker where required.
- Verify the requestor can trigger only the fixed action and cannot change arguments, executable, working directory, environment, task principal, service binary, configuration, IPC policy, or lifecycle permissions.
- Test mutation/swap attempts under the actual identities while the launcher verifies and loads the bundle. A junction unit test or `Get-Acl` listing from the developer account is not evidence of effective ACL protection.
- Establish durable transaction semantics, flush behavior, concurrent-start exclusion, and backup/restore anti-rollback for the run ledger on the target storage.
- Provision a protected host result-attestation key and independently pin its public key, or provide another independently verifiable protected audit mechanism.
- Independently authenticate reviewer identity and independence, pin the public key under administrator control, and receive a valid signature over the final manifest/run ID. The current binding record remains `pending`; no reviewer approval exists.

The previous execution-boundary review reports the trust root absent and ACL query impossible. These prerequisites are presently unmet.

## Ordered implementation stages and acceptance criteria

1. **Fix the claim and result format.** Define official status as a valid signed host attestation plus matching reviewer approval and run metadata. Decide whether legacy sampling outside the official workflow is merely untrusted or prohibited. Do not overclaim the latter while the checkout remains accessible. Acceptance: written threat scope identifies trusted operators, admins, service/task identity and out-of-scope local admins; official output has a verifier specification.
2. **Choose a one-shot launcher deployment.** Implement the fixed scheduled task first only if an administrator can make the action immutable to requestors and provide a dedicated noninteractive identity. Keep an architecture fallback to a service if task permissions cannot enforce the same contract. Acceptance: unprivileged caller can only trigger the fixed action; cannot execute arbitrary code as that identity, edit task details or provide task parameters.
3. **Build and freeze the worker bundle.** Keep existing sampler/statistical behavior byte-for-byte where possible; cover the complete import and native dependency closure. The worker bundle excludes the development checkout and has an explicit manifest. Acceptance: bundle digest is independently reviewable, imports resolve only to listed immutable paths, and the registered configuration/seed/decision hashes still match.
4. **Implement pre-import authorization and durable consumption.** Verify approval against the separately pinned trust anchor, verify exact bundle, consume run ID transactionally before RNG, then run the fixed workload. Acceptance: all invalid approvals, altered bundles, import/DLL attacks, duplicate run IDs, concurrent triggers, interruptions and restarts fail before any new draw; the old run cannot resume.
5. **Add output attestation and audit evidence.** Sign only a complete result from the fixed worker; bind the output digest to approval, bundle, preregistration, seed ledger and complete count. Acceptance: altered/forged/local result JSON fails external verification; partial results never appear official.
6. **Local security qualification.** Run unit/adversarial tests with fake keys and a spy sampler, no confirmation samples. Acceptance: tests listed in the checklist pass; static inventory finds no unguarded path inside the protected runtime; all altered artifacts reject before import/RNG.
7. **Deployment qualification and independent review.** Administrator provisions and tests ACLs/service-task restrictions and durable anti-rollback on the target Windows host; independent reviewer signs final manifest. Acceptance: preserve the actual identity, ACL/effective-access, task/service, runtime, ledger, reviewer-key and attestation-verification evidence. A local unit test cannot satisfy this stage.
8. **Separate execution authorization.** Only after all prior gates pass may a distinct authorization decision consider execution. It must remain within the unchanged scientific registration; any new seed allocation requires its own prospective review. This proposal does not provide that authorization.

## Final go/no-go checklist

Current state: **NO-GO**. Do not provision trust or execute until every item is evidenced:

- [ ] Official-result trust claim and legacy-helper limitation are explicitly accepted; no claim is made that arbitrary local sampling is impossible.
- [ ] Only the fixed scheduled task (or equivalently protected service) can launch the official worker; no user-controlled parameters, arbitrary command, Python import/RPC, or alternate official-output path exists.
- [ ] Reviewer trust anchor is independently provisioned and cannot be replaced by requestor/worker; reviewer identity and independence are externally verified.
- [ ] Approval signature binds experiment ID, exact complete worker/runtime bundle, preregistration/configuration, fixed seed-ledger digest, trust-root identity and unique run ID.
- [ ] Complete Python/native import closure is verified before import and protected against replacement for the entire run; adversarial import, DLL and path-swap tests fail before RNG.
- [ ] Durable protected ledger rejects replay after concurrent invocation, interruption, restart and restore; consumed runs are never resumed.
- [ ] An official result is externally verifiable through a protected host attestation and binds exact result bytes and full run provenance.
- [ ] All negative tests pass without confirmation samples; actual-host ACL/identity checks pass under the deployed identities.
- [ ] 38,228 pairs, 1,024 shots, registered null, detector/threshold, Clopper–Pearson rule and 2,217 cutoff remain unchanged; no new seed allocation is used without a new prospective registration.

Missing or failed evidence on any item means NO-GO. Passing the checklist would only permit a separate review of execution authorization; it does not itself launch the experiment.

## Files inspected and created

Inspected:

- `reports/experiments/finite_shot_null_confirmation_implementation_readiness_20261016.md`
- `reports/experiments/finite_shot_null_confirmation_architecture_proposal_20261016.md`
- `reports/experiments/finite_shot_null_confirmation_execution_boundary_review_20261016.md`
- `reports/experiments/finite_shot_null_confirmation_security_remediation_20261016.md`
- `reports/experiments/finite_shot_null_confirmation_trust_gate_20261016.md`
- `experiments/finite_shot_null_confirmation.py`
- `experiments/finite_shot_synthetic_characterization.py`
- `src/quantum_simulator/statistics.py` (detector dependency)
- `tests/test_finite_shot_null_confirmation.py`
- `reports/experiments/finite_shot_null_confirmation_driver_binding_20261016.json`
- `reports/experiments/finite_shot_null_confirmation_preregistration_20261016.json`

Created:

- `reports/experiments/finite_shot_null_confirmation_execution_design_decision_20261016.md` — this decision and implementation plan.

No code, test, preregistration, seed ledger, approval artifact, result, or prior report was modified. No dependencies were installed. No confirmation samples were generated and the experiment was not run. **Remaining blockers:** missing administrator-provisioned Windows trust root and verified effective ACLs; no dedicated execution identity/task/service; no immutable complete runtime bundle/import boundary; no durable anti-rollback run ledger qualified under deployment identity; no host result-attestation key/mechanism; and no independently authenticated reviewer key/signature.
