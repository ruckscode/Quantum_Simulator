# QMutBench development learning curve (2026-10-25)

## Scope and safeguards

This is a development-only, grouped learning-curve experiment for predicting the historical binary QMutBench oracle outcome. It uses the hash-pinned 723,079-row metadata CSV (`3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778`) and the existing label map (`WOO`/`OPO` = Detected; `Survivor` = Not detected). The registered inputs remain the existing seven pre-outcome fields: `algorithm`, `operation`, `gate`, `dominant_state`, `qubits`, `gates`, and `Position_int`. `Killed`, `Origin_program`, and `Unnamed: 0` are excluded from model inputs. Numeric median imputation and categorical most-frequent imputation plus one-hot encoding are fitted inside each fold pipeline; unseen categories are ignored safely.

The existing deterministic five-way `StratifiedGroupKFold` selection was reconstructed first: **70 whole origins / 144,962 records remain frozen for test, and the other 305 origins / 578,117 records are development-only**. All three learning-curve validation folds were created once over those 305 development origins using seed `20261026` (`SEED + 1`) and reused for every candidate and regime. The validation sides contain 91, 101, and 113 origins (194,269; 191,737; and 192,111 records). The held-out test origins appear in no training, validation, sampling, preprocessing, or model selection stage. The final diagnostic used the frozen test only once for the single candidate selected from development validation during this run. Since the same frozen test has been examined in earlier experiments, this remains exploratory evidence.

## Why selected-v3 scored lower

The original full-data model used all 578,117 development records from 305 origins for fitting (40-tree Random Forest, depth 18, leaf size 30, square-root feature sampling, `balanced_subsample`). The selected-v3 run used 125,013 records from 86 origins (100-tree Extra Trees, depth 18, leaf size 30). On selected-v3's own three folds, its candidate fits used 83,351 / 83,183 / 83,492 records from 60 / 56 / 56 origins, validating on 41,662 / 41,830 / 41,521 records from 26 / 19 / 30 origins. Its reported 67.92% group-CV accuracy / 67.70% macro F1 came from folds split only within those 86 origins. Those CV results are **not directly comparable** to the original 73.41% full-data group-CV accuracy / 73.40% macro F1 because their validation origins and split composition differed. Their frozen-test scores (77.65% baseline; 73.49% v3) use the same test definition, but are historical, repeatedly inspected exploratory measurements.

The fixed-fold comparison isolates the data-regime difference. Selected-v3 supplied only 49–66 fitting origins per fold, versus 81–96 for representative-200k, 119–141 for representative-300k, and 192–214 with all development data. Selected-v3 retained the development class balance closely (51.84% Detected versus 51.84% in development) and all 28 algorithms, 3 operations, 19 gates, and 2 dominant-state values. Its selected origins had a smaller median size (867 versus 1,247 records across development) and narrower numeric distributions: median qubits 13 versus 14, median gates 210 versus 178 (while its upper quartile was 330 versus 330), and median mutation position 44 versus 47. The evidence supports reduced origin and record coverage as a substantial explanation for its weaker fixed-fold performance; it does not establish that coverage is the only cause.

The new deterministic nested selector round-robins development-only origins across algorithm-group, per-origin detected-prevalence quintile, and origin-size quartile strata, with seeded order within cells. It improved origin coverage as the nested pool grew. Its row-level class ratio is not perfectly representative at smaller sizes: Detected is 56.47% in representative-125k, 58.46% in representative-200k, and 57.10% in representative-300k, versus 51.84% for all development rows. This is an observed limitation of selecting indivisible complete origins; the all-development reference restores the population balance. Target prevalence was used only to stratify development origins, never as a model feature and never using frozen-test outcomes.

## Data regimes and common-fold validation

“Pool” counts refer to complete development origins before subtracting the fixed fold's validation origins. Fold training counts below are the actual records/origins used to fit; the validation side is fixed across all regimes. The representative prefixes are nested by origin membership.

| Regime | Pool records | Pool origins | Detected share | Fit records by folds 1 / 2 / 3 | Fit origins by folds 1 / 2 / 3 |
|---|---:|---:|---:|---|---|
| selected-v3 | 125,013 | 86 | 51.84% | 91,657 / 90,131 / 68,238 | 66 / 57 / 49 |
| Representative 125k | 127,402 | 81 | 56.47% | 75,005 / 88,220 / 91,579 | 54 / 58 / 50 |
| Representative 200k | 201,066 | 135 | 58.46% | 133,890 / 135,429 / 132,813 | 96 / 93 / 81 |
| Representative 300k | 305,066 | 192 | 57.10% | 217,177 / 188,127 / 204,828 | 141 / 124 / 119 |
| All development | 578,117 | 305 | 51.84% | 383,848 / 386,380 / 386,006 | 214 / 204 / 192 |

Fixed validation records/origins by fold are **194,269 / 91**, **191,737 / 101**, and **192,111 / 113**. The manifest stores exact train/validation record IDs and origin lists in membership checkpoints, with SHA-256 hashes. No training origin overlaps its fold's validation origins or the frozen test.

Three candidates were included where resource limits allowed: the original Random Forest, correctly wired Extra Trees, and scikit-learn HistGradientBoosting with dense one-hot preprocessing. HistGradientBoosting was limited to fit partitions of at most 150,000 records; it was skipped for representative-300k and all-development regimes. No XGBoost, LightGBM, or CatBoost installation was present. Mean metrics below average the three fixed validation folds; `Macro F1` was the selection metric.

| Regime | Candidate | Accuracy | Balanced accuracy | Macro F1 | Total fit seconds | Max process RSS |
|---|---|---:|---:|---:|---:|---:|
| selected-v3 | Random Forest | 69.29% | 69.26% | 69.26% | 12.7 | 303.6 MB |
| selected-v3 | Extra Trees | 70.10% | 70.21% | 70.09% | 20.8 | 312.9 MB |
| selected-v3 | HistGradientBoosting | 69.35% | 69.19% | 69.14% | 12.2 | 379.5 MB |
| Representative 125k | Random Forest | 70.30% | 70.50% | 70.25% | 11.7 | 308.7 MB |
| Representative 125k | Extra Trees | 71.26% | 71.43% | 71.21% | 21.2 | 302.7 MB |
| Representative 125k | HistGradientBoosting | 69.91% | 69.74% | 69.63% | 11.8 | 373.8 MB |
| Representative 200k | Random Forest | 71.65% | 71.88% | 71.56% | 21.7 | 283.0 MB |
| Representative 200k | Extra Trees | 72.43% | 72.66% | 72.35% | 36.9 | 271.8 MB |
| Representative 200k | HistGradientBoosting | 71.11% | 70.80% | 70.69% | 17.4 | 339.9 MB |
| Representative 300k | Random Forest | 72.45% | 72.61% | 72.42% | 36.2 | 320.4 MB |
| Representative 300k | Extra Trees | 73.32% | 73.58% | 73.26% | 60.6 | 318.4 MB |
| All development | Random Forest | 73.41% | 73.57% | 73.40% | 105.6 | 307.0 MB |
| All development | Extra Trees | 74.27% | 74.47% | 74.25% | 177.3 | 301.3 MB |

Fold-level accuracy / balanced accuracy / macro F1:

| Regime and candidate | Fold 1 | Fold 2 | Fold 3 |
|---|---|---|---|
| selected-v3 Random Forest | 70.67 / 70.62 / 70.63% | 69.86 / 69.86 / 69.84% | 67.33 / 67.31 / 67.30% |
| selected-v3 Extra Trees | 71.75 / 71.81 / 71.75% | 71.03 / 71.16 / 71.03% | 67.52 / 67.66 / 67.51% |
| selected-v3 HistGradientBoosting | 70.70 / 70.76 / 70.70% | 69.91 / 69.72 / 69.73% | 67.44 / 67.09 / 66.99% |
| Representative 125k Random Forest | 70.56 / 70.76 / 70.54% | 68.96 / 69.01 / 68.95% | 71.38 / 71.72 / 71.27% |
| Representative 125k Extra Trees | 71.11 / 71.34 / 71.06% | 70.21 / 70.18 / 70.18% | 72.46 / 72.78 / 72.38% |
| Representative 125k HistGradientBoosting | 70.78 / 70.86 / 70.78% | 68.59 / 68.15 / 67.89% | 70.36 / 70.21 / 70.22% |
| Representative 200k Random Forest | 72.23 / 72.56 / 72.12% | 70.50 / 70.50 / 70.48% | 72.22 / 72.59 / 72.07% |
| Representative 200k Extra Trees | 73.11 / 73.45 / 72.99% | 71.44 / 71.43 / 71.42% | 72.75 / 73.09 / 72.65% |
| Representative 200k HistGradientBoosting | 69.92 / 69.64 / 69.60% | 70.68 / 70.19 / 69.88% | 72.73 / 72.57 / 72.59% |
| Representative 300k Random Forest | 72.51 / 72.63 / 72.51% | 72.04 / 72.12 / 72.04% | 72.79 / 73.09 / 72.71% |
| Representative 300k Extra Trees | 74.15 / 74.38 / 74.12% | 72.59 / 72.78 / 72.58% | 73.21 / 73.58 / 73.08% |
| All development Random Forest | 73.31 / 73.44 / 73.31% | 72.30 / 72.42 / 72.30% | 74.63 / 74.85 / 74.60% |
| All development Extra Trees | 74.67 / 74.80 / 74.67% | 74.22 / 74.41 / 74.20% | 73.92 / 74.19 / 73.88% |

The all-development pool improved over selected-v3 on the **same fixed folds** by 4.14 percentage points of macro F1 for the original Random Forest and 4.16 points for base Extra Trees. The representative learning curve also rises with sample size, although the 125k/200k group sets skew toward detected-heavy origins. This supports adding development origin coverage rather than concentrating on a few rare profiles.

## Limited tuning and selected model

Development CV selected all-development Extra Trees as the strongest base family. Two bounded candidates were then compared on the same three development folds. The unweighted comparison held model size/depth/leaf/features constant except class weighting; the deeper candidate changed depth and leaf size as a targeted capacity/generalization check. The tuning run stopped after measured full-data costs made broader search disproportionate: the deeper candidate took about 136 seconds per fold. An interrupted partial Random Forest tuning fold was preserved in the local archive but excluded from selection.

| Candidate | Parameters | Accuracy | Balanced accuracy | Macro F1 | Fit time |
|---|---|---:|---:|---:|---:|
| Base Extra Trees | 100 trees, depth 18, leaf 30, sqrt features, balanced_subsample | 74.27% | 74.47% | 74.25% | 177 s |
| Extra Trees, unweighted | 100 trees, depth 18, leaf 30, sqrt features, no class weight | 74.55% | 74.61% | 74.54% | 167 s |
| **Extra Trees, deeper/smaller leaves** | **100 trees, depth 22, leaf 15, sqrt features, balanced_subsample** | **75.49%** | **75.64%** | **75.48%** | **407 s** |

Tuned fold accuracy / balanced accuracy / macro F1 was **75.35 / 75.46 / 75.35%**, **75.59 / 75.74 / 75.59%**, and **75.52 / 75.74 / 75.50%**. Fold confusion matrices (actual rows Survivor/Detected; predicted columns Survivor/Detected) were `[[73,262,20,283],[27,604,73,120]]`, `[[73,765,18,678],[28,119,71,175]]`, and `[[75,281,17,144],[29,883,69,803]]`.

Across the three validation folds, mean per-class F1 was about **75.50% for Survivor** and **75.13% for Detected**. The final single fitted pipeline was then trained on all 578,117 development records from all 305 development origins. Fit time was 244.5 seconds; peak process RSS was 486 MB.

## One exploratory frozen-test diagnostic

After development-only selection was complete, the selected Extra Trees pipeline was evaluated once on the existing frozen test (70 origins, 144,962 records; zero origin overlap):

| Accuracy | Balanced accuracy | Macro F1 |
|---:|---:|---:|
| **78.84%** | **78.93%** | **78.84%** |

Per-class results:

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Survivor | 69,886 | 76.39% | 81.21% | 78.73% |
| Detected (WOO/OPO) | 75,076 | 81.42% | 76.64% | 78.96% |

Confusion matrix, actual rows / predicted columns `[Survivor, Detected]`: `[[56,753, 13,133], [17,536, 57,540]]`. Group-bootstrap 95% intervals were 76.13–81.66% accuracy, 75.94–81.76% balanced accuracy, and 75.95–81.59% macro F1. Mean/median per-origin accuracy was 81.24% / 83.18% (range 58.36–98.57%).

The point estimate is 1.19 percentage points above the historical full-data Random Forest's 77.65% frozen-test accuracy, but this is not a fresh confirmation: the test has been examined repeatedly in earlier work. The new result is explicitly marked `exploratory_diagnostic_only`, was not used for candidate/feature/parameter selection, and must not be interpreted as independently validated improvement. **The 85% goal remains unmet**: the best grouped-validation accuracy is 75.49%, and even this exploratory test point estimate is below 85%.

## Compute, artifacts, and resume

The experiment preflight observed 7.85 GB total RAM, about 961 MB available after the source table load, and 35.9 GB free disk. Peak RSS was 486 MB during the final full-development fit; dense HistGradientBoosting reached about 380 MB on eligible smaller regimes. Summed estimator fit time across completed base candidates, tuning, and final fit was **1,364 seconds (22.7 minutes)**. The final resume/diagnostic invocation took about 290 seconds; that invocation reused completed fit checkpoints rather than repeating them. There were no failed completed stages. HistGradientBoosting resource skips on 300k/all-development are recorded in the manifest.

All generated outputs are ignored under `data/qmutbench_ml/learning_curve_v1/`: atomic manifest and metrics, exact membership NPZs per regime/fold, checksum-verified fold result JSONs, tuning results, development-selected model, one frozen-test prediction file, and its diagnostic. Resume or inspect status with:

```powershell
.\.venv\Scripts\python.exe -m experiments.qmutbench_learning_curve run --output-dir data/qmutbench_ml/learning_curve_v1
.\.venv\Scripts\python.exe -m experiments.qmutbench_learning_curve status --output-dir data/qmutbench_ml/learning_curve_v1
```

The run identity binds the verified source hash, runner hash, split/group membership, selected-v3 manifest, seed, feature schema, estimator configurations, and preprocessing. Completed folds are reused only when checksums and configuration match. On resume, corrupted/missing fold result JSONs are recomputed; a valid prediction file is reused if test diagnostics need recovery.

## Limitations and verification

`Origin_program` grouping does not establish independent MQT recipe-root ancestry. The task predicts historical oracle outcomes from metadata, not simulator behavior, source localization, or hardware diagnosis. The metadata source has 375 origins, and only 305 are development origins under the preserved split. The frozen test has repeated historical use and cannot serve as a new confirmatory set. Wider circuit-semantic features require source/mutant lineage not included in the local full metadata CSV.

Focused tests: **15 passed** across `tests/test_qmutbench_learning_curve.py`, `tests/test_qmutbench_selected_pipeline.py`, and `tests/test_qmutbench_outcome_ml.py`. They cover identical fixed folds, nested subsets, zero group leakage, feature leakage guards, candidate wiring/unknown categories, interruption and checkpoint recovery, checksum repair, and result consistency. The simulator, diagnosis behavior, frozen tests, labels, original baseline artifacts, selected-v3 artifacts, and historical reports were not modified.
