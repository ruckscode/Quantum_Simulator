# Nine-source grouped classical suite audit

Status: **VERIFIED**; 9 tasks; 180 model/task summaries; 900 fold/model metrics.

Each task's 20 OOF files were checked for exact unique ID coverage, label/group agreement, group-exclusive five-fold assignments, and aggregate plus per-fold metric replay.
The suite uses independent GroupKFold assignments and is not a matched QNN comparison. LintQ and QSimBench remain exploratory under their documented source caveats.

| Task | Rows | Groups | Models | Runtime (s) | Sampled peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| bugs4q_core | 62 | 31 | 20 | 74.80 | 152.6 |
| bugs4q_na | 62 | 31 | 20 | 86.88 | 153.6 |
| bugs4q_pooled | 124 | 62 | 20 | 80.83 | 146.5 |
| qbugs_qasm_v2 | 1496 | 748 | 20 | 95.72 | 150.8 |
| quantum_platform_comment_v2 | 319 | 304 | 20 | 83.81 | 148.1 |
| qolumbina_ideal | 182 | 26 | 20 | 72.93 | 147.3 |
| lintq_context_v2_exploratory | 290 | 217 | 20 | 78.08 | 147.4 |
| bugs4q_robust_execution | 2220 | 37 | 20 | 96.85 | 155.0 |
| qsimbench_exploratory | 300 | 150 | 20 | 78.14 | 150.5 |
