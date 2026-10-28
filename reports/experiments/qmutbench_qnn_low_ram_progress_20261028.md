# QNN low-RAM progress log — 2026-10-28

## Direct changes committed to the dedicated backup branch

Branch: `backup/qmutbench-binary-qnn-20261028`

1. Resume-gate fix: commit `c287964ca01b2d56ef74706dd5896787d68f8b53`
   - Added `has_resume_candidates(output)`.
   - The output preflight now recognizes a completed fold checkpoint or a correctly named per-epoch QNN checkpoint.
   - This is only candidate detection; existing loaders remain responsible for checksum, schema, configuration, fingerprint, and continuity validation.
2. Resume-gate regression tests: commit `33119b59366db9bbfa814f47ca28ab0fad8ba925`
   - Added `tests/test_qmutbench_resume_gate.py` covering epoch-only candidates, valid completed-fold candidates, and rejecting out-of-range fold names/unrelated files.
3. Lightweight CI workflow: commit `396a2f6ddbb0c4f9b856b5190299d51d5c2630ff`
   - Added `.github/workflows/qmutbench-binary-qnn-integrity.yml`.
   - Installs pytest and psutil on a GitHub-hosted runner and runs the existing integrity tests plus new resume-gate tests.
   - GitHub combined status currently reports no status entries for this commit; therefore CI execution/result is not yet verified.

## Limits and next technical priorities

- These changes are on the remote backup branch only. They have not been applied to the user's local workspace in this chat.
- No local Windows tests, quantum circuit, gradient check, parameter update, or QNN training was executed here.
- The last user-reported available RAM was 1,345 MiB, below the runner's 2,048 MiB startup gate. The full campaign remains blocked on the local machine until memory optimization or a safe alternate execution environment is available.
- The QNN run directory was previously confirmed empty; the historical four epoch losses remain unverified and are not counted as saved training progress.
- The next optimization should be evidence-driven: profile per-batch and per-epoch process RSS, inspect PennyLane/autograd retention, then implement bounded-memory changes with regression tests. The OOF/metadata loader can also be optimized, but any change must preserve exact record IDs, labels, folds, origin grouping, source checksums, and frozen-test isolation.
- Do not lower the 2,048 MiB startup threshold or 1,024 MiB in-run floor merely to force a run to start. If the local system cannot satisfy the guard after tested optimizations, use a remote execution environment with enough memory and the required development data.
