# Recovery notes

- Original run files remain in place and were not changed. A byte-for-byte copy of all 56 run-directory files is under `artifacts/reports/experiments/qmutbench_binary_qnn_full_development_chunk16_20261029/`, preserving the original relative path. SHA-256 was computed for each original and copy; all 56 pairs matched.
- `tracked_worktree.patch` is a binary-safe snapshot of tracked changes relative to HEAD `a2d51be22fd985bbdd10655f7e31dc77b9d19ed6`. It does not contain untracked files. Before applying it, confirm the base commit and run `git apply --check` from the repository root; do not apply over divergent work.
- A prior `.tracked-working-tree-recovery.patch` (380,444 bytes; 2026-10-25) exists but predates the current 2026-10-29 tracked diff and is not a substitute for the new patch.
- No separate `reports/recovery` backup existed before this inventory. The C: volume reported about 44.1 GB available before copying.
- The repository also has extensive untracked and ignored generated work. Git reported 31 tracked modified files, 0 staged paths, 46,198 untracked paths, and 117,377 ignored paths. `.qbugs1_env` is an untracked environment tree; `.venv` is ignored. Generated output and other untracked work remain in place and were not staged, cleaned, or copied here. See `git_state.txt` for status and root summaries.
- The process check found no Python process, but WMI command-line enumeration was denied. If other process types can launch Python workloads indirectly, confirm the machine is idle before resuming.
- The original run has no separate log or predictions to copy. No frozen-test rows were accessed according to the run progress records.
- No data set was copied. The recovery inventory contains the run state and the code/config metadata needed to inspect resumption.
