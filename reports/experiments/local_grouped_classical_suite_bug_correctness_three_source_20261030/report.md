# Three-source pooled correctness grouped classical audit

Status: **VERIFIED**; 1 tasks; 20 model/task summaries; 100 fold/model metrics.

Each task's 20 OOF files were checked for exact unique ID coverage, label/group agreement, group-exclusive five-fold assignments, and aggregate plus per-fold metric replay.
This pooled task combines 1,620 records / 810 source-prefixed issue groups with a common correct/fixed (0) versus buggy (1) target. The 20-model winner is selected on these CV folds, so results are exploratory.

| Task | Rows | Groups | Models | Runtime (s) | Sampled peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| bug_correctness_three_source_pooled | 1620 | 810 | 20 | 73.17 | 170.0 |
