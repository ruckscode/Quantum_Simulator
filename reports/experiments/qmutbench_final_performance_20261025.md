# QMutBench final performance optimization

## Directly measured results

Dataset SHA-256 `3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778`; 578,117 development records across 305 origins. All comparisons use the same saved three `StratifiedGroupKFold` splits (91, 101, and 113 validation origins). The frozen historical test was not used.

| Model | Macro F1 | Balanced accuracy | Accuracy | ROC-AUC | Average precision | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Extra Trees, 7 features | 0.7548 | 0.7564 | 0.7549 | 0.8491 | 0.8666 | 0.1607 |
| Extra Trees + position fraction, 8 model features | 0.7682 | 0.7697 | 0.7683 | 0.8625 | 0.8767 | 0.1558 |

At threshold 0.5, macro F1 improved by +1.34 pp; balanced accuracy by +1.33 pp; accuracy by +1.34 pp. Mean paired fold macro-F1 gain was +1.35 pp (sample SD 0.43 pp), with 3/3 folds improved.

| Fold | Baseline macro F1 | Candidate macro F1 | Change | Baseline BA | Candidate BA |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7535 | 0.7621 | +0.86 pp | 0.7546 | 0.7633 |
| 2 | 0.7559 | 0.7727 | +1.69 pp | 0.7574 | 0.7741 |
| 3 | 0.7550 | 0.7699 | +1.49 pp | 0.7574 | 0.7718 |

At threshold 0.5, detected-class recall rose from 0.7144 to 0.7309; Survivor recall rose from 0.7985 to 0.8085. False negatives fell from 85,606 to 80,664; false positives fell from 56,105 to 53,311.

The model improved macro F1 on 248/305 origins at threshold 0.5. Mean origin macro F1 changed by +2.74 pp; median change was +1.21 pp. There remain weak origins and category slices; the JSON companion has supported slice comparisons and high-confidence error counts.

## Feature audit and controlled screen

All original selected features have zero missing rows. Algorithm (28 categories) and gate (19) had the strongest average training-fold Cramer's V associations in the audit; Position_int (493 values) had the weakest. These are descriptive, univariate associations computed from training-side samples, not evidence of causal effects or standalone model gain.

The local QMutBench README defines `position` as a circuit-percent band. Position_int ranges beyond gate count on some records, so no ratio to gate count was assumed. `position_fraction` is the deterministic midpoint of the documented percentage band divided by 100. It is available from the source metadata and uses no target or origin identifier.

| 60-tree screen variant | Fold-mean macro F1 at 0.5 | Nested threshold macro F1 |
|---|---:|---:|
| algorithm_gate | 0.7228 | 0.7221 |
| algorithm_operation | 0.7247 | 0.7203 |
| baseline | 0.7318 | 0.7308 |
| gate_operation | 0.7335 | 0.7270 |
| position_bin | 0.7418 | 0.7369 |
| position_fraction | 0.7406 | 0.7409 |

The screen trained on the representative whole-origin selection (201,066 records) and evaluated all rows of each saved held fold. The numeric position fraction had the strongest nested screen score; its categorical-bin encoding was also positive. Algorithm-operation and algorithm-gate interactions were negative; gate-operation was a small non-nested gain but declined under nested thresholds. Only position fraction advanced to the full-development 100-tree comparison.

Prior checked candidates remain negative: no-class-weight Extra Trees was effectively tied; unweighted Random Forest and the smaller selected-v3 subset were lower than the full-data baseline. Existing 200k feature ablations showed algorithm and gate are important; numeric-only features were weak.

The schema audit records target-free cardinality and missingness for available features. `Origin_program` is used only for grouping and diagnostics; `Unnamed: 0` and `Killed` are excluded. One-hot encoding is fitted inside each training fold; no target encoding or resampling was used.

## Thresholds and calibration

Nested grouped threshold selection fitted each threshold on the other two folds, then evaluated it on the held fold. The candidate thresholds were 0.495, 0.460, and 0.480. Candidate nested macro F1 averaged 0.7679, compared with 0.7557 for the baseline's nested thresholds. For the candidate, the nested selected thresholds averaged slightly below using 0.5 directly (0.7682); therefore deployment keeps 0.5. The all-OOF selected threshold 0.475 scores 0.7697 on those same OOF predictions and is explicitly treated as selection-biased.

Brier score improved from 0.1607 to 0.1558; ROC-AUC and average precision also improved. Probabilities remain uncalibrated; reliability bins and ECE summaries are reported in JSON.

## Interpretation and limits

The improvement meets the predeclared +0.5 pp macro-F1 criterion on comparable full-development OOF, improves balanced accuracy and both class recalls, wins in all three grouped folds, and improves on most origins. It is a defensible development improvement, not proof of performance on new external programs. The 85% target was not reached.

This remains historical mutation-outcome prediction. It does not establish hardware-noise detection, hardware-aware fault localization, or quantum-simulator diagnosis.

## Reproduction and artifacts

Run the feature screen with `python -m experiments.qmutbench_performance_phase`, full grouped finalist with `python -m experiments.qmutbench_performance_phase --full-position-finalist`, full-development model fit with `python -m experiments.qmutbench_performance_phase --fit-final-model`, and audit with `python -m experiments.qmutbench_feature_audit`. Generated model and prediction files remain under ignored `data/qmutbench_ml/`.

Selected model: `data/qmutbench_ml/position_fraction_v1/best_model.joblib` (SHA-256 `641a18b67448f910ea0126c914e7e5f17bc946a3fa940b0486c045b845f8ee45`). Inference API: `qmutbench_ml.QMutBenchOutcomeModel`; input includes source `position`, transformed using the saved mapping. Machine-readable results: `reports/experiments/qmutbench_final_performance_20261025.json`.
