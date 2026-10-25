# Finite-shot distribution calibration: diagnostic report

**Analysis date:** 2026-10-13  
**Source run:** `reports/experiments/finite_shot_distribution_calibration_run/`  
**Frozen registry:** `tests/fixtures/finite_shot_distribution_calibration.json`

## Executive findings

- All 3,600 planned output records validate against the deterministic ledger: 0 missing, 0 invalid, and 0 unexpected result JSON files. The manifest and summary both say `completed`; the summary reports 12,902,400 shot-circuit executions.
- The 60 registered acceptance decisions reproduce exactly from the saved outputs: **30 pass and 30 fail**. Calibration contributes 16/30 passes and untouched validation 14/30; the current detector contributes 19/30 and the permutation detector 11/30. The acceptance table below gives every target and its exact count margin.
- The current TVD detector has severe measured false-positive behavior on negative controls at 256 and 1,024 shots: 100% and 87.9-91.7% pooled over the two controls. At 4,096 shots it falls to 0.8-2.1% pooled. This pattern is consistent with finite-sample TVD noise overwhelming the current guard at lower shot counts.
- The permutation detector is much better behaved on controls (2-8 false positives per 120 in each control cell), but it misses many subtle faults at 256/1,024 shots. Sensitivity rises at 4,096 shots; the untouched validation subtle cell reaches 111/120 (92.5%), three detections short of the registered minimum 114.
- **Confidence-interval implementation defect:** the saved summary's `sensitivity_lower_95_one_sided` values are incorrect in all 36 positive-stratum rows. The helper in `experiments/finite_shot_calibration_plan.py` uses the 95th percentile of the upper-tail probability where a one-sided 95% lower Clopper-Pearson bound requires the 5th percentile. Correct bounds are recomputed in this report. This defect does not change the 30/60 pass/fail decisions, which are based on the separately registered integer count cutoffs.

## Run integrity and consistency checks

| Check | Result |
|---|---:|
| Ledger configurations | 3,600 |
| Valid outputs / missing / invalid | 3,600 / 0 / 0 |
| Unexpected result JSON files | 0 |
| Summary target rows / registry target rows | 60 / 60 |
| Recomputed target count or pass mismatches | 0 |
| Recomputed summary TP/FP/TN/abstention count mismatches | 0 |
| Summary sensitivity lower-bound mismatches | 36 of 36 positive rows |
| Summary false-positive upper-bound mismatches | 0 of 24 negative rows |

The ledger has 120 configurations per split x shot count x scenario. The two controls are labeled `NO_ANOMALY`; subtle, moderate, and large are labeled `ANOMALY`. Labels match scenario and stratum on every stored record. Every distribution has the registered 16-outcome support, nonnegative probabilities summing to one, and probabilities quantized to its declared shot count (as enforced during reconciliation). The diagnosis inputs exclude expected labels.

The registry's integer sensitivity minima are consistent with its one-sided exact 95% rule: minimum counts 70, 104, 114, and 119 give lower bounds at or above 50%, 80%, 90%, and 95%, respectively; one fewer detection falls below the corresponding target. The negative-control cutoff of at most one false positive gives a one-sided 95% exact upper FPR bound of 3.892%; two false positives give 5.153%, above the registered 5% ceiling.

## Exact pass/fail table for all 60 registered targets

For anomaly strata, the margin is observed true positives minus the registered minimum. For control strata, it is the registered maximum minus observed false positives. Positive margin passes; negative margin fails. `L95` and `U95` are one-sided exact Clopper-Pearson bounds recomputed from the output counts (not copied from the defective saved sensitivity lower-bound field).

| Split | Shots | Scenario / stratum | Detector | Observed | Registered criterion | Margin | Exact 95% bound | Result | Failure reason |
|---|---:|---|---|---:|---|---:|---|---|---|
| calibration | 256 | identical_control | current_detected | FP 120/120 (100.00%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -119 | U95=100.00% | FAIL | 120/120 exceeds the maximum by 119; U95 100.00% is above 5.00% |
| calibration | 256 | identical_control | permutation_detected | FP 5/120 (4.17%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -4 | U95=8.56% | FAIL | 5/120 exceeds the maximum by 4; U95 8.56% is above 5.00% |
| calibration | 256 | measurement_equivalent_control | current_detected | FP 120/120 (100.00%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -119 | U95=100.00% | FAIL | 120/120 exceeds the maximum by 119; U95 100.00% is above 5.00% |
| calibration | 256 | measurement_equivalent_control | permutation_detected | FP 2/120 (1.67%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -1 | U95=5.15% | FAIL | 2/120 exceeds the maximum by 1; U95 5.15% is above 5.00% |
| calibration | 256 | subtle | current_detected | TP 120/120 (100.00%) | TP >= 70/120 (sensitivity target 50.00%) | +50 | L95=97.53% | PASS | none |
| calibration | 256 | subtle | permutation_detected | TP 14/120 (11.67%) | TP >= 70/120 (sensitivity target 50.00%) | -56 | L95=7.19% | FAIL | 14/120 is 56 below the minimum; L95 7.19% is below 50.00% |
| calibration | 256 | moderate | current_detected | TP 120/120 (100.00%) | TP >= 104/120 (sensitivity target 80.00%) | +16 | L95=97.53% | PASS | none |
| calibration | 256 | moderate | permutation_detected | TP 83/120 (69.17%) | TP >= 104/120 (sensitivity target 80.00%) | -21 | L95=61.51% | FAIL | 83/120 is 21 below the minimum; L95 61.51% is below 80.00% |
| calibration | 256 | large | current_detected | TP 120/120 (100.00%) | TP >= 119/120 (sensitivity target 95.00%) | +1 | L95=97.53% | PASS | none |
| calibration | 256 | large | permutation_detected | TP 120/120 (100.00%) | TP >= 119/120 (sensitivity target 95.00%) | +1 | L95=97.53% | PASS | none |
| calibration | 1024 | identical_control | current_detected | FP 104/120 (86.67%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -103 | U95=91.45% | FAIL | 104/120 exceeds the maximum by 103; U95 91.45% is above 5.00% |
| calibration | 1024 | identical_control | permutation_detected | FP 6/120 (5.00%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -5 | U95=9.63% | FAIL | 6/120 exceeds the maximum by 5; U95 9.63% is above 5.00% |
| calibration | 1024 | measurement_equivalent_control | current_detected | FP 107/120 (89.17%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -106 | U95=93.47% | FAIL | 107/120 exceeds the maximum by 106; U95 93.47% is above 5.00% |
| calibration | 1024 | measurement_equivalent_control | permutation_detected | FP 5/120 (4.17%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -4 | U95=8.56% | FAIL | 5/120 exceeds the maximum by 4; U95 8.56% is above 5.00% |
| calibration | 1024 | subtle | current_detected | TP 119/120 (99.17%) | TP >= 104/120 (sensitivity target 80.00%) | +15 | L95=96.11% | PASS | none |
| calibration | 1024 | subtle | permutation_detected | TP 36/120 (30.00%) | TP >= 104/120 (sensitivity target 80.00%) | -68 | L95=23.14% | FAIL | 36/120 is 68 below the minimum; L95 23.14% is below 80.00% |
| calibration | 1024 | moderate | current_detected | TP 120/120 (100.00%) | TP >= 114/120 (sensitivity target 90.00%) | +6 | L95=97.53% | PASS | none |
| calibration | 1024 | moderate | permutation_detected | TP 120/120 (100.00%) | TP >= 114/120 (sensitivity target 90.00%) | +6 | L95=97.53% | PASS | none |
| calibration | 1024 | large | current_detected | TP 120/120 (100.00%) | TP >= 119/120 (sensitivity target 95.00%) | +1 | L95=97.53% | PASS | none |
| calibration | 1024 | large | permutation_detected | TP 120/120 (100.00%) | TP >= 119/120 (sensitivity target 95.00%) | +1 | L95=97.53% | PASS | none |
| calibration | 4096 | identical_control | current_detected | FP 1/120 (0.83%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | +0 | U95=3.89% | PASS | none |
| calibration | 4096 | identical_control | permutation_detected | FP 5/120 (4.17%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -4 | U95=8.56% | FAIL | 5/120 exceeds the maximum by 4; U95 8.56% is above 5.00% |
| calibration | 4096 | measurement_equivalent_control | current_detected | FP 1/120 (0.83%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | +0 | U95=3.89% | PASS | none |
| calibration | 4096 | measurement_equivalent_control | permutation_detected | FP 8/120 (6.67%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -7 | U95=11.71% | FAIL | 8/120 exceeds the maximum by 7; U95 11.71% is above 5.00% |
| calibration | 4096 | subtle | current_detected | TP 113/120 (94.17%) | TP >= 114/120 (sensitivity target 90.00%) | -1 | L95=89.32% | FAIL | 113/120 is 1 below the minimum; L95 89.32% is below 90.00% |
| calibration | 4096 | subtle | permutation_detected | TP 119/120 (99.17%) | TP >= 114/120 (sensitivity target 90.00%) | +5 | L95=96.11% | PASS | none |
| calibration | 4096 | moderate | current_detected | TP 120/120 (100.00%) | TP >= 119/120 (sensitivity target 95.00%) | +1 | L95=97.53% | PASS | none |
| calibration | 4096 | moderate | permutation_detected | TP 120/120 (100.00%) | TP >= 119/120 (sensitivity target 95.00%) | +1 | L95=97.53% | PASS | none |
| calibration | 4096 | large | current_detected | TP 120/120 (100.00%) | TP >= 119/120 (sensitivity target 95.00%) | +1 | L95=97.53% | PASS | none |
| calibration | 4096 | large | permutation_detected | TP 120/120 (100.00%) | TP >= 119/120 (sensitivity target 95.00%) | +1 | L95=97.53% | PASS | none |
| untouched_validation | 256 | identical_control | current_detected | FP 120/120 (100.00%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -119 | U95=100.00% | FAIL | 120/120 exceeds the maximum by 119; U95 100.00% is above 5.00% |
| untouched_validation | 256 | identical_control | permutation_detected | FP 3/120 (2.50%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -2 | U95=6.33% | FAIL | 3/120 exceeds the maximum by 2; U95 6.33% is above 5.00% |
| untouched_validation | 256 | measurement_equivalent_control | current_detected | FP 120/120 (100.00%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -119 | U95=100.00% | FAIL | 120/120 exceeds the maximum by 119; U95 100.00% is above 5.00% |
| untouched_validation | 256 | measurement_equivalent_control | permutation_detected | FP 7/120 (5.83%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -6 | U95=10.68% | FAIL | 7/120 exceeds the maximum by 6; U95 10.68% is above 5.00% |
| untouched_validation | 256 | subtle | current_detected | TP 120/120 (100.00%) | TP >= 70/120 (sensitivity target 50.00%) | +50 | L95=97.53% | PASS | none |
| untouched_validation | 256 | subtle | permutation_detected | TP 15/120 (12.50%) | TP >= 70/120 (sensitivity target 50.00%) | -55 | L95=7.87% | FAIL | 15/120 is 55 below the minimum; L95 7.87% is below 50.00% |
| untouched_validation | 256 | moderate | current_detected | TP 120/120 (100.00%) | TP >= 104/120 (sensitivity target 80.00%) | +16 | L95=97.53% | PASS | none |
| untouched_validation | 256 | moderate | permutation_detected | TP 93/120 (77.50%) | TP >= 104/120 (sensitivity target 80.00%) | -11 | L95=70.33% | FAIL | 93/120 is 11 below the minimum; L95 70.33% is below 80.00% |
| untouched_validation | 256 | large | current_detected | TP 120/120 (100.00%) | TP >= 119/120 (sensitivity target 95.00%) | +1 | L95=97.53% | PASS | none |
| untouched_validation | 256 | large | permutation_detected | TP 120/120 (100.00%) | TP >= 119/120 (sensitivity target 95.00%) | +1 | L95=97.53% | PASS | none |
| untouched_validation | 1024 | identical_control | current_detected | FP 108/120 (90.00%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -107 | U95=94.13% | FAIL | 108/120 exceeds the maximum by 107; U95 94.13% is above 5.00% |
| untouched_validation | 1024 | identical_control | permutation_detected | FP 4/120 (3.33%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -3 | U95=7.47% | FAIL | 4/120 exceeds the maximum by 3; U95 7.47% is above 5.00% |
| untouched_validation | 1024 | measurement_equivalent_control | current_detected | FP 110/120 (91.67%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -109 | U95=95.41% | FAIL | 110/120 exceeds the maximum by 109; U95 95.41% is above 5.00% |
| untouched_validation | 1024 | measurement_equivalent_control | permutation_detected | FP 8/120 (6.67%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -7 | U95=11.71% | FAIL | 8/120 exceeds the maximum by 7; U95 11.71% is above 5.00% |
| untouched_validation | 1024 | subtle | current_detected | TP 119/120 (99.17%) | TP >= 104/120 (sensitivity target 80.00%) | +15 | L95=96.11% | PASS | none |
| untouched_validation | 1024 | subtle | permutation_detected | TP 45/120 (37.50%) | TP >= 104/120 (sensitivity target 80.00%) | -59 | L95=30.11% | FAIL | 45/120 is 59 below the minimum; L95 30.11% is below 80.00% |
| untouched_validation | 1024 | moderate | current_detected | TP 120/120 (100.00%) | TP >= 114/120 (sensitivity target 90.00%) | +6 | L95=97.53% | PASS | none |
| untouched_validation | 1024 | moderate | permutation_detected | TP 120/120 (100.00%) | TP >= 114/120 (sensitivity target 90.00%) | +6 | L95=97.53% | PASS | none |
| untouched_validation | 1024 | large | current_detected | TP 120/120 (100.00%) | TP >= 119/120 (sensitivity target 95.00%) | +1 | L95=97.53% | PASS | none |
| untouched_validation | 1024 | large | permutation_detected | TP 120/120 (100.00%) | TP >= 119/120 (sensitivity target 95.00%) | +1 | L95=97.53% | PASS | none |
| untouched_validation | 4096 | identical_control | current_detected | FP 4/120 (3.33%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -3 | U95=7.47% | FAIL | 4/120 exceeds the maximum by 3; U95 7.47% is above 5.00% |
| untouched_validation | 4096 | identical_control | permutation_detected | FP 7/120 (5.83%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -6 | U95=10.68% | FAIL | 7/120 exceeds the maximum by 6; U95 10.68% is above 5.00% |
| untouched_validation | 4096 | measurement_equivalent_control | current_detected | FP 1/120 (0.83%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | +0 | U95=3.89% | PASS | none |
| untouched_validation | 4096 | measurement_equivalent_control | permutation_detected | FP 4/120 (3.33%) | FP <= 1/120 (FPR <= 5%; one-sided exact 95% upper bound) | -3 | U95=7.47% | FAIL | 4/120 exceeds the maximum by 3; U95 7.47% is above 5.00% |
| untouched_validation | 4096 | subtle | current_detected | TP 103/120 (85.83%) | TP >= 114/120 (sensitivity target 90.00%) | -11 | L95=79.51% | FAIL | 103/120 is 11 below the minimum; L95 79.51% is below 90.00% |
| untouched_validation | 4096 | subtle | permutation_detected | TP 111/120 (92.50%) | TP >= 114/120 (sensitivity target 90.00%) | -3 | L95=87.28% | FAIL | 111/120 is 3 below the minimum; L95 87.28% is below 90.00% |
| untouched_validation | 4096 | moderate | current_detected | TP 120/120 (100.00%) | TP >= 119/120 (sensitivity target 95.00%) | +1 | L95=97.53% | PASS | none |
| untouched_validation | 4096 | moderate | permutation_detected | TP 120/120 (100.00%) | TP >= 119/120 (sensitivity target 95.00%) | +1 | L95=97.53% | PASS | none |
| untouched_validation | 4096 | large | current_detected | TP 120/120 (100.00%) | TP >= 119/120 (sensitivity target 95.00%) | +1 | L95=97.53% | PASS | none |
| untouched_validation | 4096 | large | permutation_detected | TP 120/120 (100.00%) | TP >= 119/120 (sensitivity target 95.00%) | +1 | L95=97.53% | PASS | none |

### Pass totals

| Group | Passed / registered |
|---|---:|
| Calibration split | 16 / 30 |
| Untouched validation split | 14 / 30 |
| Current TVD detector | 19 / 30 |
| Permutation detector | 11 / 30 |
| Overall | 30 / 60 |

## Split x shot x scenario breakdown

Rates are measured from 120 saved configurations in each cell. The detection column is sensitivity for ANOMALY strata and false-positive rate for NO_ANOMALY controls. Brackets are equal-tailed exact 95% Clopper-Pearson intervals. Abstention is the saved diagnosis `abstained` boolean and has the same value for both detector columns.

| Split | Shots | Scenario / stratum | Label class / rate | Current detector | Permutation detector | Abstention rate |
|---|---:|---|---|---|---|---|
| calibration | 256 | identical_control | NO_ANOMALY / false-positive rate | 120/120 (100.00%) [96.97%, 100.00%] | 5/120 (4.17%) [1.37%, 9.46%] | 120/120 (100.00%) [96.97%, 100.00%] |
| calibration | 256 | measurement_equivalent_control | NO_ANOMALY / false-positive rate | 120/120 (100.00%) [96.97%, 100.00%] | 2/120 (1.67%) [0.20%, 5.89%] | 0/120 (0.00%) [0.00%, 3.03%] |
| calibration | 256 | subtle | ANOMALY / sensitivity | 120/120 (100.00%) [96.97%, 100.00%] | 14/120 (11.67%) [6.53%, 18.80%] | 0/120 (0.00%) [0.00%, 3.03%] |
| calibration | 256 | moderate | ANOMALY / sensitivity | 120/120 (100.00%) [96.97%, 100.00%] | 83/120 (69.17%) [60.09%, 77.27%] | 0/120 (0.00%) [0.00%, 3.03%] |
| calibration | 256 | large | ANOMALY / sensitivity | 120/120 (100.00%) [96.97%, 100.00%] | 120/120 (100.00%) [96.97%, 100.00%] | 0/120 (0.00%) [0.00%, 3.03%] |
| calibration | 1024 | identical_control | NO_ANOMALY / false-positive rate | 104/120 (86.67%) [79.25%, 92.18%] | 6/120 (5.00%) [1.86%, 10.57%] | 104/120 (86.67%) [79.25%, 92.18%] |
| calibration | 1024 | measurement_equivalent_control | NO_ANOMALY / false-positive rate | 107/120 (89.17%) [82.19%, 94.10%] | 5/120 (4.17%) [1.37%, 9.46%] | 13/120 (10.83%) [5.90%, 17.81%] |
| calibration | 1024 | subtle | ANOMALY / sensitivity | 119/120 (99.17%) [95.44%, 99.98%] | 36/120 (30.00%) [21.98%, 39.04%] | 1/120 (0.83%) [0.02%, 4.56%] |
| calibration | 1024 | moderate | ANOMALY / sensitivity | 120/120 (100.00%) [96.97%, 100.00%] | 120/120 (100.00%) [96.97%, 100.00%] | 0/120 (0.00%) [0.00%, 3.03%] |
| calibration | 1024 | large | ANOMALY / sensitivity | 120/120 (100.00%) [96.97%, 100.00%] | 120/120 (100.00%) [96.97%, 100.00%] | 0/120 (0.00%) [0.00%, 3.03%] |
| calibration | 4096 | identical_control | NO_ANOMALY / false-positive rate | 1/120 (0.83%) [0.02%, 4.56%] | 5/120 (4.17%) [1.37%, 9.46%] | 1/120 (0.83%) [0.02%, 4.56%] |
| calibration | 4096 | measurement_equivalent_control | NO_ANOMALY / false-positive rate | 1/120 (0.83%) [0.02%, 4.56%] | 8/120 (6.67%) [2.92%, 12.71%] | 119/120 (99.17%) [95.44%, 99.98%] |
| calibration | 4096 | subtle | ANOMALY / sensitivity | 113/120 (94.17%) [88.35%, 97.62%] | 119/120 (99.17%) [95.44%, 99.98%] | 7/120 (5.83%) [2.38%, 11.65%] |
| calibration | 4096 | moderate | ANOMALY / sensitivity | 120/120 (100.00%) [96.97%, 100.00%] | 120/120 (100.00%) [96.97%, 100.00%] | 0/120 (0.00%) [0.00%, 3.03%] |
| calibration | 4096 | large | ANOMALY / sensitivity | 120/120 (100.00%) [96.97%, 100.00%] | 120/120 (100.00%) [96.97%, 100.00%] | 0/120 (0.00%) [0.00%, 3.03%] |
| untouched_validation | 256 | identical_control | NO_ANOMALY / false-positive rate | 120/120 (100.00%) [96.97%, 100.00%] | 3/120 (2.50%) [0.52%, 7.13%] | 120/120 (100.00%) [96.97%, 100.00%] |
| untouched_validation | 256 | measurement_equivalent_control | NO_ANOMALY / false-positive rate | 120/120 (100.00%) [96.97%, 100.00%] | 7/120 (5.83%) [2.38%, 11.65%] | 0/120 (0.00%) [0.00%, 3.03%] |
| untouched_validation | 256 | subtle | ANOMALY / sensitivity | 120/120 (100.00%) [96.97%, 100.00%] | 15/120 (12.50%) [7.17%, 19.78%] | 0/120 (0.00%) [0.00%, 3.03%] |
| untouched_validation | 256 | moderate | ANOMALY / sensitivity | 120/120 (100.00%) [96.97%, 100.00%] | 93/120 (77.50%) [68.98%, 84.62%] | 0/120 (0.00%) [0.00%, 3.03%] |
| untouched_validation | 256 | large | ANOMALY / sensitivity | 120/120 (100.00%) [96.97%, 100.00%] | 120/120 (100.00%) [96.97%, 100.00%] | 0/120 (0.00%) [0.00%, 3.03%] |
| untouched_validation | 1024 | identical_control | NO_ANOMALY / false-positive rate | 108/120 (90.00%) [83.18%, 94.73%] | 4/120 (3.33%) [0.92%, 8.31%] | 108/120 (90.00%) [83.18%, 94.73%] |
| untouched_validation | 1024 | measurement_equivalent_control | NO_ANOMALY / false-positive rate | 110/120 (91.67%) [85.21%, 95.93%] | 8/120 (6.67%) [2.92%, 12.71%] | 10/120 (8.33%) [4.07%, 14.79%] |
| untouched_validation | 1024 | subtle | ANOMALY / sensitivity | 119/120 (99.17%) [95.44%, 99.98%] | 45/120 (37.50%) [28.83%, 46.80%] | 1/120 (0.83%) [0.02%, 4.56%] |
| untouched_validation | 1024 | moderate | ANOMALY / sensitivity | 120/120 (100.00%) [96.97%, 100.00%] | 120/120 (100.00%) [96.97%, 100.00%] | 0/120 (0.00%) [0.00%, 3.03%] |
| untouched_validation | 1024 | large | ANOMALY / sensitivity | 120/120 (100.00%) [96.97%, 100.00%] | 120/120 (100.00%) [96.97%, 100.00%] | 0/120 (0.00%) [0.00%, 3.03%] |
| untouched_validation | 4096 | identical_control | NO_ANOMALY / false-positive rate | 4/120 (3.33%) [0.92%, 8.31%] | 7/120 (5.83%) [2.38%, 11.65%] | 4/120 (3.33%) [0.92%, 8.31%] |
| untouched_validation | 4096 | measurement_equivalent_control | NO_ANOMALY / false-positive rate | 1/120 (0.83%) [0.02%, 4.56%] | 4/120 (3.33%) [0.92%, 8.31%] | 119/120 (99.17%) [95.44%, 99.98%] |
| untouched_validation | 4096 | subtle | ANOMALY / sensitivity | 103/120 (85.83%) [78.29%, 91.53%] | 111/120 (92.50%) [86.24%, 96.51%] | 17/120 (14.17%) [8.47%, 21.71%] |
| untouched_validation | 4096 | moderate | ANOMALY / sensitivity | 120/120 (100.00%) [96.97%, 100.00%] | 120/120 (100.00%) [96.97%, 100.00%] | 0/120 (0.00%) [0.00%, 3.03%] |
| untouched_validation | 4096 | large | ANOMALY / sensitivity | 120/120 (100.00%) [96.97%, 100.00%] | 120/120 (100.00%) [96.97%, 100.00%] | 0/120 (0.00%) [0.00%, 3.03%] |

## Pooled metrics by split and shot count

These point estimates pool the frozen design mix (360 positive configurations across three fault strengths; 240 negative configurations across two controls; 600 total per split and shot count). Sensitivity equals recall. Precision is tied to the experiment's 60% anomaly prevalence and should not be transported to a deployment population without its prevalence. No pooled confidence interval is given because fault and control strata have materially different rates; the scenario table above provides stratum-specific exact intervals.

| Split | Shots | Detector | Sensitivity / recall (TP/360) | FPR (FP/240) | Precision (TP/(TP+FP)) | Abstention (n/600) |
|---|---:|---|---|---|---|---|
| calibration | 256 | current_detected | 360/360 (100.00%) | 240/240 (100.00%) | 60.00% | 120/600 (20.00%) |
| calibration | 256 | permutation_detected | 217/360 (60.28%) | 7/240 (2.92%) | 96.88% | 120/600 (20.00%) |
| calibration | 1024 | current_detected | 359/360 (99.72%) | 211/240 (87.92%) | 62.98% | 118/600 (19.67%) |
| calibration | 1024 | permutation_detected | 276/360 (76.67%) | 11/240 (4.58%) | 96.17% | 118/600 (19.67%) |
| calibration | 4096 | current_detected | 353/360 (98.06%) | 2/240 (0.83%) | 99.44% | 127/600 (21.17%) |
| calibration | 4096 | permutation_detected | 359/360 (99.72%) | 13/240 (5.42%) | 96.51% | 127/600 (21.17%) |
| untouched_validation | 256 | current_detected | 360/360 (100.00%) | 240/240 (100.00%) | 60.00% | 120/600 (20.00%) |
| untouched_validation | 256 | permutation_detected | 228/360 (63.33%) | 10/240 (4.17%) | 95.80% | 120/600 (20.00%) |
| untouched_validation | 1024 | current_detected | 359/360 (99.72%) | 218/240 (90.83%) | 62.22% | 119/600 (19.83%) |
| untouched_validation | 1024 | permutation_detected | 285/360 (79.17%) | 12/240 (5.00%) | 95.96% | 119/600 (19.83%) |
| untouched_validation | 4096 | current_detected | 343/360 (95.28%) | 5/240 (2.08%) | 98.56% | 140/600 (23.33%) |
| untouched_validation | 4096 | permutation_detected | 351/360 (97.50%) | 11/240 (4.58%) | 96.96% | 140/600 (23.33%) |

## Diagnosis of the failures

### Current detector: finite-shot null noise plus an inadequate guard

The current detector is `statistics_result.anomaly_detected`, which fires when TVD exceeds `max(threshold, 1/sqrt(shots))`; the call threshold is 0.05. For identical circuits, the diagnosis engine draws the expected and observed samples with separate seeds (`seed` and `seed + 1`), so two empirical distributions from the same underlying law are compared. The finite-shot guard is only 0.0625 at 256 shots and the 0.05 floor applies at 1,024 and 4,096 shots. The saved data show 120/120 false positives in both control strata at 256 shots and 211/240 at 1,024 shots, falling to 2/240 at 4,096 shots across calibration. The untouched validation pattern is similar: 240/240 at 256, 218/240 at 1,024, and 5/240 at 4,096. This is direct evidence that finite-shot variability combined with the current TVD cutoff drives the failures; it is not evidence of actual circuit faults in the controls.

At 256 shots, all 120 identical-control cases in each split are also marked abstained while `current_detected` is true. This means the raw statistics flag and the final diagnosis category are not equivalent outcomes; acceptance metrics intentionally score the raw flag. The distinction should be kept visible when interpreting these results.

### Permutation detector: low-shot power and a strict control certification cutoff

Permutation sensitivity on subtle faults is 11.7-12.5% at 256 shots, 30.0-37.5% at 1,024 shots, and 92.5-99.2% at 4,096 shots. Moderate faults move from 69.2-77.5% at 256 to 100% at 1,024 and 4,096; large faults are 100% in every cell. Similar shot trends in both splits support finite-shot power as the main driver of low-shot sensitivity failures. At 4,096 shots, the calibration subtle cell passes at 119/120, while untouched validation is 111/120 and fails by three. This split difference could reflect template composition or finite-sample variability; the saved one-split-per-role design does not isolate those causes.

Permutation control false positives range from 2/120 to 8/120 (1.7-6.7%). These counts are broadly compatible with a nominal 5% test at this sample size; their intervals are wide. The registered acceptance rule is stricter: it requires no more than one false positive so the exact one-sided 95% upper FPR bound is below 5%. That is a valid certification rule, but it will fail many samples from a test operating near 5%; for example, if true FPR were exactly 5%, the probability of seeing at most one false positive in 120 is only 1.55%. Thus the permutation-control failures alone do not establish that the test is miscalibrated.

### Separation and abstention

The current detector's high sensitivity at 256/1,024 shots is not useful separation: it labels nearly every negative control positive as well. At 4,096 shots, control false positives fall sharply, but subtle-fault sensitivity is 94.2% in calibration and 85.8% in validation; corresponding abstention rates are 5.8% and 14.2%. The permutation detector separates controls much better, while its subtle sensitivity needs the higher shot count. These patterns support both a current-detector calibration/guard problem and a low-shot power problem for the permutation method.

## Metric and evaluation audit

1. **Target counts and statuses are consistent.** All 60 reported target observations and pass booleans match direct counts from valid output records and the frozen registry. No thresholds, labels, scenarios, seeds, or criteria were changed.
2. **Saved count metrics are consistent.** `summary.json` TP, FP, TN, abstention, and configuration counts match the raw records in all 60 split x shot x scenario x detector cells. The saved one-sided false-positive upper bounds also match exact recomputation.
3. **Saved sensitivity lower bounds are not consistent with the claimed confidence rule.** The summary helper sets the binomial survival probability to 0.95 instead of 0.05 when solving a one-sided 95% lower Clopper-Pearson bound. Example: 120/120 should have L95=97.53%, while the summary reports 99.96%; 111/120 should have L95=87.28%, while the summary reports 92.50%. All 36 positive-stratum lower-bound entries are affected. This changes no target status because pass/fail uses the fixed count minima; the confidence values in the saved summary should not be used.
4. **Monte Carlo p-value resolution is finite.** The permutation test uses 999 resamples and add-one correction, giving 0.001 p-value increments. That is consistent with the registry; no stored p-value or detection rule mismatch was found.
5. **Precision is design-mix-specific.** The saved cases contain 60% positive scenarios by design. No deployment prevalence is available, so operational precision cannot be inferred from this dataset alone.

## Confidence and limitations

Per-cell intervals use the 120 distinct templates within a split, shot count, and stratum as the binomial denominator. They quantify uncertainty for that registered stratum under independent-template sampling. The same templates are reused across shot counts, so shot-level outcomes are paired across shot counts and should not be treated as independent when comparing shot levels. The 60 target checks are individual 95% rules; the registry does not apply a family-wise multiplicity adjustment. All intervals are descriptive frequentist estimates from this completed simulation, not hardware measurements.

## Provenance

- Run manifest: `reports/experiments/finite_shot_distribution_calibration_run/run_manifest.json`
- Final summary and registered target results: `reports/experiments/finite_shot_distribution_calibration_run/summary.json`
- Frozen plan and target registry: `tests/fixtures/finite_shot_distribution_calibration.json`
- Output reconciliation: deterministic 3,600-row ledger and result schema in `experiments/finite_shot_calibration_plan.py`
- Confidence interval calculations in this report: exact Clopper-Pearson beta quantiles from raw successes and denominators; two-sided 95% for breakdown rates and one-sided 95% for registered acceptance bounds.
