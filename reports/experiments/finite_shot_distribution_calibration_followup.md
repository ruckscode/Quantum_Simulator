# Finite-shot calibration follow-up: confidence bounds and detector comparison

**Date:** 2026-10-13  
**Source:** `reports/experiments/finite_shot_distribution_calibration_run/`  
**Registry:** `tests/fixtures/finite_shot_distribution_calibration.json`

## Scope and integrity

Read-only reconciliation found **3,600 valid outputs, 0 missing, 0 invalid, and 0 unexpected JSON outputs**. Analysis used the saved per-configuration records; no simulations, training, hardware calls, or design changes were made. The registered target decisions remain **30/60 passed**. Per-configuration outputs, labels, seeds, thresholds, scenarios, and target criteria were preserved.

Changed files in this follow-up:

- `experiments/finite_shot_calibration_plan.py` - corrected exact one-sided lower confidence bound inversion.
- `tests/test_finite_shot_confidence_bounds.py` - focused edge and reference-value tests.
- `reports/experiments/finite_shot_distribution_calibration_run/summary.json` - corrected only the 36 derived `sensitivity_lower_95_one_sided` values; counts, upper bounds, and acceptance decisions were retained.
- This report.

## Confidence-bound defect and repair

The lower one-sided Clopper-Pearson bound for `k` successes in `n` trials solves `P_p(X >= k) = alpha` at `alpha=0.05`. The previous inversion targeted `1-alpha`, producing overly large bounds. The corrected helper returns 0 for zero successes, and the all-success bound is `alpha**(1/n)` (for 120/120: 0.975344598944). Intermediate references: 70/120 = 0.504201076877; 104/120 = 0.804556153802; 114/120 = 0.903703682623; 119/120 = 0.961079153140.

The saved summary was recomputed from its unchanged `tp` and `n_positive` counts. All **36/36 corrected lower bounds** now match a direct recomputation from the 3,600 saved records. The 24 negative-row one-sided upper bounds were independently recomputed and still match. Counts and registered pass/fail results match the underlying saved records; the bounds are descriptive and do not alter the integer count criteria.

Focused test command: `pytest -q --basetemp=.tcal-cp-final-20261013 tests/test_finite_shot_confidence_bounds.py` - **4 passed**. Coverage includes zero successes, all successes, intermediate count reference values, n=0 / n=1 boundaries, and monotonicity.

## Measured detector comparison

Each cell contains 120 saved configurations. Sensitivity is reported for anomaly strata; FPR for controls. Brackets are two-sided exact 95% Clopper-Pearson intervals. Abstention is the saved shared `abstained` flag. These intervals describe each cell; repeated templates and multiple shot levels should not be treated as independent pooled experimental units.

| Split | Shots | Scenario / strength | Metric | TVD detector | Permutation detector | Abstentions |
|---|---:|---|---|---:|---:|---:|
| calibration | 256 | identical_control | FPR | 120/120 (100.0%; 97.0-100.0%) | 5/120 (4.2%; 1.4-9.5%) | 120/120 (100.0%; 97.0-100.0%) |
| calibration | 256 | measurement_equivalent_control | FPR | 120/120 (100.0%; 97.0-100.0%) | 2/120 (1.7%; 0.2-5.9%) | 0/120 (0.0%; 0.0-3.0%) |
| calibration | 256 | subtle | sens | 120/120 (100.0%; 97.0-100.0%) | 14/120 (11.7%; 6.5-18.8%) | 0/120 (0.0%; 0.0-3.0%) |
| calibration | 256 | moderate | sens | 120/120 (100.0%; 97.0-100.0%) | 83/120 (69.2%; 60.1-77.3%) | 0/120 (0.0%; 0.0-3.0%) |
| calibration | 256 | large | sens | 120/120 (100.0%; 97.0-100.0%) | 120/120 (100.0%; 97.0-100.0%) | 0/120 (0.0%; 0.0-3.0%) |
| calibration | 1024 | identical_control | FPR | 104/120 (86.7%; 79.3-92.2%) | 6/120 (5.0%; 1.9-10.6%) | 104/120 (86.7%; 79.3-92.2%) |
| calibration | 1024 | measurement_equivalent_control | FPR | 107/120 (89.2%; 82.2-94.1%) | 5/120 (4.2%; 1.4-9.5%) | 13/120 (10.8%; 5.9-17.8%) |
| calibration | 1024 | subtle | sens | 119/120 (99.2%; 95.4-100.0%) | 36/120 (30.0%; 22.0-39.0%) | 1/120 (0.8%; 0.0-4.6%) |
| calibration | 1024 | moderate | sens | 120/120 (100.0%; 97.0-100.0%) | 120/120 (100.0%; 97.0-100.0%) | 0/120 (0.0%; 0.0-3.0%) |
| calibration | 1024 | large | sens | 120/120 (100.0%; 97.0-100.0%) | 120/120 (100.0%; 97.0-100.0%) | 0/120 (0.0%; 0.0-3.0%) |
| calibration | 4096 | identical_control | FPR | 1/120 (0.8%; 0.0-4.6%) | 5/120 (4.2%; 1.4-9.5%) | 1/120 (0.8%; 0.0-4.6%) |
| calibration | 4096 | measurement_equivalent_control | FPR | 1/120 (0.8%; 0.0-4.6%) | 8/120 (6.7%; 2.9-12.7%) | 119/120 (99.2%; 95.4-100.0%) |
| calibration | 4096 | subtle | sens | 113/120 (94.2%; 88.4-97.6%) | 119/120 (99.2%; 95.4-100.0%) | 7/120 (5.8%; 2.4-11.6%) |
| calibration | 4096 | moderate | sens | 120/120 (100.0%; 97.0-100.0%) | 120/120 (100.0%; 97.0-100.0%) | 0/120 (0.0%; 0.0-3.0%) |
| calibration | 4096 | large | sens | 120/120 (100.0%; 97.0-100.0%) | 120/120 (100.0%; 97.0-100.0%) | 0/120 (0.0%; 0.0-3.0%) |
| untouched_validation | 256 | identical_control | FPR | 120/120 (100.0%; 97.0-100.0%) | 3/120 (2.5%; 0.5-7.1%) | 120/120 (100.0%; 97.0-100.0%) |
| untouched_validation | 256 | measurement_equivalent_control | FPR | 120/120 (100.0%; 97.0-100.0%) | 7/120 (5.8%; 2.4-11.6%) | 0/120 (0.0%; 0.0-3.0%) |
| untouched_validation | 256 | subtle | sens | 120/120 (100.0%; 97.0-100.0%) | 15/120 (12.5%; 7.2-19.8%) | 0/120 (0.0%; 0.0-3.0%) |
| untouched_validation | 256 | moderate | sens | 120/120 (100.0%; 97.0-100.0%) | 93/120 (77.5%; 69.0-84.6%) | 0/120 (0.0%; 0.0-3.0%) |
| untouched_validation | 256 | large | sens | 120/120 (100.0%; 97.0-100.0%) | 120/120 (100.0%; 97.0-100.0%) | 0/120 (0.0%; 0.0-3.0%) |
| untouched_validation | 1024 | identical_control | FPR | 108/120 (90.0%; 83.2-94.7%) | 4/120 (3.3%; 0.9-8.3%) | 108/120 (90.0%; 83.2-94.7%) |
| untouched_validation | 1024 | measurement_equivalent_control | FPR | 110/120 (91.7%; 85.2-95.9%) | 8/120 (6.7%; 2.9-12.7%) | 10/120 (8.3%; 4.1-14.8%) |
| untouched_validation | 1024 | subtle | sens | 119/120 (99.2%; 95.4-100.0%) | 45/120 (37.5%; 28.8-46.8%) | 1/120 (0.8%; 0.0-4.6%) |
| untouched_validation | 1024 | moderate | sens | 120/120 (100.0%; 97.0-100.0%) | 120/120 (100.0%; 97.0-100.0%) | 0/120 (0.0%; 0.0-3.0%) |
| untouched_validation | 1024 | large | sens | 120/120 (100.0%; 97.0-100.0%) | 120/120 (100.0%; 97.0-100.0%) | 0/120 (0.0%; 0.0-3.0%) |
| untouched_validation | 4096 | identical_control | FPR | 4/120 (3.3%; 0.9-8.3%) | 7/120 (5.8%; 2.4-11.6%) | 4/120 (3.3%; 0.9-8.3%) |
| untouched_validation | 4096 | measurement_equivalent_control | FPR | 1/120 (0.8%; 0.0-4.6%) | 4/120 (3.3%; 0.9-8.3%) | 119/120 (99.2%; 95.4-100.0%) |
| untouched_validation | 4096 | subtle | sens | 103/120 (85.8%; 78.3-91.5%) | 111/120 (92.5%; 86.2-96.5%) | 17/120 (14.2%; 8.5-21.7%) |
| untouched_validation | 4096 | moderate | sens | 120/120 (100.0%; 97.0-100.0%) | 120/120 (100.0%; 97.0-100.0%) | 0/120 (0.0%; 0.0-3.0%) |
| untouched_validation | 4096 | large | sens | 120/120 (100.0%; 97.0-100.0%) | 120/120 (100.0%; 97.0-100.0%) | 0/120 (0.0%; 0.0-3.0%) |

The cell table stratifies by split, shot count, control scenario, and fault strength. In negative controls, TVD falsely flags all 120/120 records at 256 shots in each split and 104 to 110/120 at 1,024 shots; at 4,096 shots its control FPR ranges from 1/120 to 4/120. The permutation detector's control false positives are much lower in these cells (2 to 8/120), though their exact intervals remain broad at n=120.

For subtle faults, TVD sensitivity is high at 256 and 1,024 shots but drops at 4,096 shots to 113/120 in calibration and 103/120 in validation. Permutation sensitivity rises with shots: 14 to 15/120 at 256, 36 to 45/120 at 1,024, and 119/120 calibration / 111/120 validation at 4,096. Moderate and large faults are detected at or near ceiling by both methods at 1,024 and 4,096. Saved abstentions are caused by the diagnosis pipeline and reported as measured; the permutation decision is separately counted.

The summary has an overall 60%-positive designed mix, but precision from that artificial prevalence is not a deployment estimate. Per-cell precision is not a useful comparison here because each scenario cell contains only one label class; recall equals sensitivity. No pooled CI is given across heterogeneous scenario and shot strata.

## Why low-shot TVD controls are flagged

The saved implementation uses the rule `TVD > max(configured_threshold, 1/sqrt(shots))`, with configured threshold 0.05. Thus the effective cutoff is 0.0625 at 256 shots and 0.05 at 1,024/4,096. TVD is computed between independently sampled finite-shot distributions over 16 outcomes. Under a matching null, empirical multinomial frequencies still fluctuate; the scalar `1/sqrt(shots)` guard does not use the two distributions' probability vectors, outcome-wise variance, or a calibrated null tail. At low shots, the observed control exceedance rates (100% then 87-92%) are direct evidence that this guard is not calibrated for these controls. This supports finite-sample null variability combined with an inadequately calibrated fixed cutoff; the saved evidence does not isolate contribution from every implementation detail or establish a universal cause.

A statistically grounded shot-aware rule is feasible in principle. The existing pooled-label permutation detector already conditions on the observed pooled category counts and sample sizes and yields much lower control flags here, consistent with a null-calibrated comparison when samples are exchangeable. Its observed weakness is low sensitivity to subtle faults at 256/1,024 shots; the stored results alone do not show that a modified cutoff would recover sensitivity while retaining FPR control.

As a diagnostic only, a control-derived cutoff was estimated from calibration controls: for each shot level, take the higher empirical 95th percentile separately for each control scenario, then use the larger of those two cutoffs. It is an exploratory proposal, not a production rule; n=120 per calibration control makes the tail estimate coarse and uncertain.

| Shots | Calibration identical-control q95 | Calibration measurement-equivalent q95 | Common exploratory cutoff |
|---:|---:|---:|---:|
| 256 | 0.183594 | 0.171875 | 0.183594 |
| 1024 | 0.093750 | 0.090820 | 0.093750 |
| 4096 | 0.044678 | 0.047852 | 0.047852 |

Applying each frozen exploratory cutoff to validation outputs gives:

| Shots | Validation scenario | Metric | Exploratory TVD result (exact two-sided 95% CI) |
|---:|---|---|---:|
| 256 | identical_control | FPR | 3/120 (2.5%; 0.5-7.1%) |
| 256 | measurement_equivalent_control | FPR | 7/120 (5.8%; 2.4-11.6%) |
| 256 | subtle | sens | 13/120 (10.8%; 5.9-17.8%) |
| 256 | moderate | sens | 91/120 (75.8%; 67.2-83.2%) |
| 256 | large | sens | 120/120 (100.0%; 97.0-100.0%) |
| 1024 | identical_control | FPR | 1/120 (0.8%; 0.0-4.6%) |
| 1024 | measurement_equivalent_control | FPR | 5/120 (4.2%; 1.4-9.5%) |
| 1024 | subtle | sens | 41/120 (34.2%; 25.8-43.4%) |
| 1024 | moderate | sens | 120/120 (100.0%; 97.0-100.0%) |
| 1024 | large | sens | 120/120 (100.0%; 97.0-100.0%) |
| 4096 | identical_control | FPR | 6/120 (5.0%; 1.9-10.6%) |
| 4096 | measurement_equivalent_control | FPR | 3/120 (2.5%; 0.5-7.1%) |
| 4096 | subtle | sens | 109/120 (90.8%; 84.2-95.3%) |
| 4096 | moderate | sens | 120/120 (100.0%; 97.0-100.0%) |
| 4096 | large | sens | 120/120 (100.0%; 97.0-100.0%) |

Because this analysis has now inspected the existing validation records, these validation figures are exploratory and cannot serve as a fresh confirmatory test. They do show whether a calibration-only quantile rule is plausible for a new evaluation; the exact output above is computed from saved TVDs without modifying any diagnosis or target.

## Recommendation and limitations

Do not change the production threshold or acceptance criteria from this same calibration. For a separate held-out evaluation, preregister a shot-conditioned permutation p-value procedure (or a calibration-only empirical null cutoff) using only a dedicated calibration partition, preserve both negative-control scenarios, freeze the method before evaluation, and measure FPR and sensitivity by fault strength at each shot count. Include enough independent templates to narrow the tail uncertainty and keep a final untouched test split. Compare against the current fixed TVD rule and existing permutation detector using the registered criteria. This proposal changes no current detector behavior.

The finite-shot calibration supplies 120 records per split/scenario/shot cell. Exact binomial intervals assume independent Bernoulli outcomes within each reported cell; shared templates across shot counts and the deterministic scenario construction limit generalization. The report evaluates saved records only and does not establish performance outside the registered scenarios or seed/template distributions.


## Complete corrected one-sided sensitivity bounds

All 36 positive summary rows are listed below. Bounds are one-sided exact 95% Clopper-Pearson lower limits recomputed from saved true-positive counts.

| Split | Shots | Fault strength | Detector | TP / n | Corrected L95 |
|---|---:|---|---|---:|---:|
| calibration | 1024 | large | current_detected | 120/120 | 0.975344598944 |
| calibration | 1024 | large | permutation_detected | 120/120 | 0.975344598944 |
| calibration | 1024 | moderate | current_detected | 120/120 | 0.975344598944 |
| calibration | 1024 | moderate | permutation_detected | 120/120 | 0.975344598944 |
| calibration | 1024 | subtle | current_detected | 119/120 | 0.961079153140 |
| calibration | 1024 | subtle | permutation_detected | 36/120 | 0.231363963294 |
| calibration | 256 | large | current_detected | 120/120 | 0.975344598944 |
| calibration | 256 | large | permutation_detected | 120/120 | 0.975344598944 |
| calibration | 256 | moderate | current_detected | 120/120 | 0.975344598944 |
| calibration | 256 | moderate | permutation_detected | 83/120 | 0.615081450311 |
| calibration | 256 | subtle | current_detected | 120/120 | 0.975344598944 |
| calibration | 256 | subtle | permutation_detected | 14/120 | 0.071927893330 |
| calibration | 4096 | large | current_detected | 120/120 | 0.975344598944 |
| calibration | 4096 | large | permutation_detected | 120/120 | 0.975344598944 |
| calibration | 4096 | moderate | current_detected | 120/120 | 0.975344598944 |
| calibration | 4096 | moderate | permutation_detected | 120/120 | 0.975344598944 |
| calibration | 4096 | subtle | current_detected | 113/120 | 0.893228274585 |
| calibration | 4096 | subtle | permutation_detected | 119/120 | 0.961079153140 |
| untouched_validation | 1024 | large | current_detected | 120/120 | 0.975344598944 |
| untouched_validation | 1024 | large | permutation_detected | 120/120 | 0.975344598944 |
| untouched_validation | 1024 | moderate | current_detected | 120/120 | 0.975344598944 |
| untouched_validation | 1024 | moderate | permutation_detected | 120/120 | 0.975344598944 |
| untouched_validation | 1024 | subtle | current_detected | 119/120 | 0.961079153140 |
| untouched_validation | 1024 | subtle | permutation_detected | 45/120 | 0.301114347032 |
| untouched_validation | 256 | large | current_detected | 120/120 | 0.975344598944 |
| untouched_validation | 256 | large | permutation_detected | 120/120 | 0.975344598944 |
| untouched_validation | 256 | moderate | current_detected | 120/120 | 0.975344598944 |
| untouched_validation | 256 | moderate | permutation_detected | 93/120 | 0.703322509235 |
| untouched_validation | 256 | subtle | current_detected | 120/120 | 0.975344598944 |
| untouched_validation | 256 | subtle | permutation_detected | 15/120 | 0.078654810263 |
| untouched_validation | 4096 | large | current_detected | 120/120 | 0.975344598944 |
| untouched_validation | 4096 | large | permutation_detected | 120/120 | 0.975344598944 |
| untouched_validation | 4096 | moderate | current_detected | 120/120 | 0.975344598944 |
| untouched_validation | 4096 | moderate | permutation_detected | 120/120 | 0.975344598944 |
| untouched_validation | 4096 | subtle | current_detected | 103/120 | 0.795109865505 |
| untouched_validation | 4096 | subtle | permutation_detected | 111/120 | 0.872774463673 |
