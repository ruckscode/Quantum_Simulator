# Local grouped classical suite audit

Status: **VERIFIED**. Five tasks; 20 models per task; 100 aggregate rows; 500 fold/model rows.

These are separate classical screens using training-only TF-IDF, SVD, scaling, and sampling. Their folds use GroupKFold by the source issue/program group. Results are not a matched QNN comparison because the classical representation differs from the pilot's hashed features.

| Task | Rows | Groups | Runtime (s) | Max sampled RSS (MiB) |
|---|---:|---:|---:|---:|
| bugs4q_core | 62 | 31 | 61.33 | 152.2 |
| bugs4q_na | 62 | 31 | 62.22 | 153.1 |
| bugs4q_pooled | 124 | 62 | 63.07 | 149.7 |
| qolumbina_ideal | 182 | 26 | 64.07 | 145.6 |
| bugs4q_robust_execution | 2220 | 37 | 77.82 | 151.5 |

All 20 OOF prediction files per task were checked for complete unique IDs, target/group agreement, group-exclusive folds, and aggregate metric replay.
