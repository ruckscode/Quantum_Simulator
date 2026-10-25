# Evidence-calibrated diagnosis: exploratory candidate comparison

**Date:** 2026-10-14  
**Status:** Read-only analysis of existing artifacts. No simulations, model training, threshold edits, hardware calls, or changes to registered results.

## Scope and evidence sources

Read the novelty assessment, q95 held-out plan and its JSON preregistration, finite-shot follow-up, `src/quantum_simulator/statistics.py`, `experiments/finite_shot_calibration_plan.py`, `src/quantum_simulator/ml_diagnosis.py`, `src/quantum_simulator/fault_localizer.py`, and focused tests. The complete calibration summary contains 3,600 records and the follow-up independently checked its reported counts and corrected one-sided bounds against the saved outputs. This report uses its calibration and validation tables without changing them. Validation is already inspected and is strictly exploratory here.

No new scoring program or test was run. The existing follow-up provides exact counts and Clopper–Pearson intervals for the baseline and q95 candidate; this report reuses those results and does not recalculate or replace them.

## What the current implementation does

### TVD baseline

`compare_distributions` computes empirical TVD between two normalized distributions and flags if `TVD > max(threshold, 1/sqrt(shots))`; the default threshold is .05. Thus the effective cutoff is .0625 at 256 shots and .05 at 1,024 and 4,096 shots. The finite-shot guard is not a confidence bound or a calibrated null test. It does not use the two count vectors’ category probabilities, their individual sample sizes, or a selected type-I error rate.

### Permutation baseline

The calibration implementation forms pooled category labels from the expected and observed counts, repeatedly repartitions them into the original sample sizes, and compares two-sample TVD. It uses 999 permutations, a deterministic seed, add-one p-value correction, and alpha=.05. This is a conditional two-sample test under exchangeability of the two count vectors. It adapts to pooled category counts and both sample sizes, unlike the scalar TVD guard. It is not automatically valid if the two samples are dependent, generated under different sampling mechanisms, or if labels/categories were selected after looking at outcomes. Its 999 resamples give p-value granularity .001; this is not the observed main low-shot sensitivity limitation, though Monte Carlo error near alpha should be reported or reduced in a future locked method.

### Abstention and confidence

`ml_diagnosis.py` applies a fixed `MIN_CONFIDENCE=.60` to an optional classifier’s largest class probability. Source comments and the model artifact state that those probabilities are uncalibrated, so .60 is operational behavior, not a 60% correctness guarantee. Below the cutoff it returns `INSUFFICIENT_EVIDENCE` without localization. For anomaly/fault predictions, it then depends on the rule-based evidence localizer; unsupported cause evidence also leads to insufficient evidence.

In finite-shot calibration output, `abstained` is computed from the shared diagnosis category being ambiguous/insufficient-evidence. It is not a detector-specific confidence interval or abstention score. In particular, TVD can count a control as detected while the diagnosis pipeline also records abstention; do not interpret the shared column as selective-detector coverage. Existing unit tests exercise mocked classifier probabilities and the low-confidence branch, not empirical confidence calibration, risk-coverage, or abstention performance on independent predictions.

### Localization

`fault_localizer.py` compares operation signatures and ranks insertions, deletions, substitutions, and trace deviations. `hardware_fault_localizer.py` ranks targets using calibration metadata and circuit relevance. These are useful evidence rules, but scores are not calibrated probabilities. Trace-only output differences are deliberately insufficient proof of program fault. Existing controlled Phase 2 includes seven abstentions/not-justified cases and all three localization/detection misses in one family, but has only 36 constructed cases across six families and three seeds. Finite-shot calibration records do not contain source-level faulty statements or independent qubit/gate/source target truth, so localization cannot be scored from those 3,600 rows.

## Why the observed failures occur

### Low-shot TVD control false positives

In validation, fixed-guard TVD flags 120/120 controls at 256 shots and 108/120 identical plus 110/120 measurement-equivalent controls at 1,024 shots. The rate falls to 4/120 and 1/120, respectively, for those controls at 4,096 shots. These counts are direct evidence that the current guard is miscalibrated for the registered null scenarios.

The most supported explanation is finite-sample fluctuation in a 16-outcome, two-sample comparison combined with an inadequately calibrated cutoff: the implementation compares two sampled empirical distributions, while `1/sqrt(n)` supplies neither the correct two-sample null quantile nor category-specific variance. The available evidence does not identify a single universal mathematical cause, nor does it test all edge cases of the implementation. This is a statistical design defect in the guard, not evidence that the TVD arithmetic itself is incorrectly coded.

The permutation detector’s validation FPR is much lower: 3/120 (2.5%) and 7/120 (5.8%) at 256 shots, 4/120 (3.3%) and 8/120 (6.7%) at 1,024, and 7/120 (5.8%) and 4/120 (3.3%) at 4,096 across the two controls. With only 120 observations per cell, intervals are broad and some upper bounds exceed 5%; do not claim a proven ≤5% population FPR.

### Permutation misses on subtle faults

Validation subtle sensitivity is 15/120 (12.5%) at 256 shots, 45/120 (37.5%) at 1,024, and 111/120 (92.5%) at 4,096. Its monotonic increase with shots and high-shot recovery are consistent with limited statistical power when the induced distribution shift is small relative to two-sample sampling variation. Moderate and large cases reach 93/120 and 120/120 at 256, then 120/120 at higher shot counts. This supports an effect-size/shot-budget limitation in the tested regime. No defect has been demonstrated in the permutation implementation; tests verify deterministic results, direction under a shifted distribution, and add-one behavior. However, no independent test validates exchangeability under every generator/scenario, and the pilot does not prove nominal type-I error over a broader population.

### Exploratory q95 TVD candidate

The calibration-only cutoffs frozen in the prior plan were .18359375, .09375, and .0478515625 at 256, 1,024, and 4,096 shots. Applying them to previously inspected validation produces:

| Shots | Control FPR, identical | Control FPR, measurement-equivalent | Subtle sensitivity | Moderate sensitivity | Large sensitivity |
|---:|---:|---:|---:|---:|---:|
| 256 | 3/120 (2.5%) | 7/120 (5.8%) | 13/120 (10.8%) | 91/120 (75.8%) | 120/120 (100%) |
| 1,024 | 1/120 (0.8%) | 5/120 (4.2%) | 41/120 (34.2%) | 120/120 (100%) | 120/120 (100%) |
| 4,096 | 6/120 (5.0%) | 3/120 (2.5%) | 109/120 (90.8%) | 120/120 (100%) | 120/120 (100%) |

These exploratory results show the expected threshold tradeoff, not a novel solution: lower control flags at low shots come with severe subtle-fault sensitivity loss. The q95 estimates use just 120 calibration observations per control and shot level; a nominal empirical 95th percentile is particularly coarse in that tail. Validation reuse forbids treating these rates as confirmatory evidence or selecting this rule by its observed validation score.

## Candidate methods and selection rationale

| Candidate | Statistical/diagnostic principle | What saved data supports | Decision |
|---|---|---|---|
| A. Unchanged pooled-label permutation test | Conditional two-sample null test using pooled counts, fixed sample sizes, and a prespecified alpha. | Already measured on both splits; much lower observed low-shot FPR and low subtle sensitivity. | Keep as a required unchanged baseline. It is a sound comparison method under exchangeability, not the proposed novelty. Future work should assess permutation Monte Carlo error and assumptions. |
| B. Calibration-frozen empirical TVD null cutoff | Estimate control TVD quantile by shot level from calibration only. | Existing q95 candidate and exploratory validation metrics above. | Reject as primary candidate: low-shot sensitivity loss, coarse tail estimates, and no independent validation remaining. Keep as a negative comparator / useful ablation only. |
| C. Effect-size-aware uncertainty interval with an indifference region | Estimate the population two-sample TVD and its uncertainty; flag only when a one-sided lower bound exceeds a prespecified meaningful-effect margin; call normal only when an upper bound is below that margin; otherwise abstain. Calibrate by parametric multinomial bootstrap or a justified simultaneous multinomial confidence region, respecting both finite samples. | Existing count pairs allow a future *exploratory reanalysis*, but the saved aggregate summary does not report this interval, and no such results are claimed here. It cannot be evaluated from reported summary counts alone. | Preferred method-development candidate on principled grounds, not because it won the inspected validation. The indifference margin must have scientific meaning and be fixed from calibration/training data. Coverage, FPR, sensitivity and selective risk remain untested. |
| D. Selective cause/localization layer | After calibrated anomaly evidence, assign cause/target only with direct independent evidence; otherwise return explicit ambiguous/insufficient evidence. Evaluate as a staged decision, with class probabilities calibrated on grouped calibration families if ML remains involved. | Code path and constructed Phase 2 labels show the mechanics and some abstentions; finite-shot records lack cause and target labels. | Required architecture for a diagnosis contribution, but not quantitatively testable from finite-shot outputs alone. ML 0.60 confidence cutoff is not suitable as a calibrated selective rule. |

Candidate C must be specified carefully. A confidence interval for TVD answers a magnitude-uncertainty question; it does not, by itself, test the null of equal distributions at a nominal level. Use a finite-sample calibrated interval/test construction, specify its simultaneous coverage, and keep the anomaly threshold/effect-size margin separate from the type-I error level. Reusing the current permutation p-value and relabeling its p-value as an uncertainty interval would not satisfy this distinction.

## Primary novelty hypothesis

**Hypothesis to test:** On unseen quantum circuit families, a two-stage finite-shot diagnosis rule that (1) separates evidence for a distribution shift from whether its magnitude is practically meaningful, and abstains in the prespecified uncertainty region, then (2) assigns cause/location only when independent program/hardware evidence supports it, will achieve lower *wrong confident diagnosis risk* than unchanged TVD, permutation-only detection, and the existing fixed-confidence pipeline at matched control FPR and minimum coverage, without reducing subtle-fault sensitivity below a preregistered floor.

This is a research hypothesis, not an established novelty claim. “Selective prediction,” confidence intervals, multinomial tests, quantum fault diagnosis, and mutation-based localization all have prior art. The distinction to verify is the *operational integration and evidence*: a finite-shot, effect-size-aware decision contract with calibrated abstention and evidence-grounded cause/localization, evaluated jointly on unseen families at matched FPR/coverage/cost. A literature review must search selective classification, equivalence testing, multinomial two-sample inference, quantum debugging and probabilistic diagnosis before claiming that exact distinction is novel.

## Limits of what was tested

**Tested on existing outputs:** fixed-guard TVD, pooled-label permutation, one calibration-selected q95 TVD cutoff, and the saved diagnosis system's shared abstention summary. Counts and intervals for those results are in the follow-up report. Test code covers core distance/permutation behavior and mocked ML abstention.

**Not tested:** effect-size confidence intervals, an indifference-region rule, calibration of anomaly/cause/localization confidence, selective risk-coverage curves, detector-specific abstention, paired family-cluster inference for a new candidate, the full 3-way cause classification with finite-shot uncertainty, or an independent held-out performance comparison of any newly designed method. The existing validation set is now exploratory for all method selection discussed here.

## Literature correction and gap requiring verification

The prior novelty assessment described Ishimoto et al. (2025) as a preprint. It was presented at EASE 2025 Short Papers/Emerging Results, and bibliographic records give DOI [10.1145/3756681.3757022](https://doi.org/10.1145/3756681.3757022). Its Qiskit study evaluates mutation-based source fault localization on 23 real and 305 artificial defects. This strengthens the existing closest-work warning: source localization is already an evaluated quantum-software method, and our operation mismatch/trace heuristic is not novel as localization.

Other closest competitors already recorded in the research-gap assessment include Huang & Martonosi’s statistical assertions ([ISCA 2019](https://doi.org/10.1145/3307650.3322213)), QDiff ([ASE 2021](https://doi.org/10.1109/ASE51524.2021.9678792)), quantum-circuit diagnosis using fault dictionaries and dynamic binary search ([VTS 2023](https://doi.org/10.1109/VTS56346.2023.10140030)), and projective-measurement cost reduction for quantum testing ([TOSEM 2025](https://doi.org/10.1145/3714468)). They rule out novelty based on statistical tests, distribution comparison, fault diagnosis/localization, or measurement savings separately. No retrieved evidence establishes that these works implement the precise staged selective contract above, but absence from this targeted comparison is not proof of novelty. Verify the full papers, citations, artifacts, and newer related work before manuscript claims.

