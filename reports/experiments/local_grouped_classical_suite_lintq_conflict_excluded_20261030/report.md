# Conflict-excluded LintQ grouped classical suite audit

Status: **VERIFIED**; 1 tasks; 20 model/task summaries; 100 fold/model metrics.

Each task's 20 OOF files were checked for exact unique ID coverage, label/group agreement, group-exclusive five-fold assignments, and aggregate plus per-fold metric replay.
This corrected LintQ input excludes the whole contradictory exact-context component (3 records), keeps 287 records / 216 groups, and uses GroupKFold by source group. Class imbalance remains, so interpret accuracy alongside macro-F1 and balanced accuracy. This is not a matched QNN comparison.

| Task | Rows | Groups | Models | Runtime (s) | Sampled peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| lintq_context_v2_conflict_excluded | 287 | 216 | 20 | 59.69 | 155.8 |
