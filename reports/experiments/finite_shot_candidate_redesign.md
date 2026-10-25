# Finite-shot candidate redesign: calibration-only feasibility

Date: 2026-10-13

## Decision

**No-go for a new confirmatory run with any candidate evaluated here.** The strongest simple candidate available from saved calibration records—a conditional permutation equality test followed by a minimum observed TVD gate—does not meet the registered subtle-fault sensitivity floor even at 4096 shots. At 256 and 1024 shots it has poor subtle sensitivity. The multinomial likelihood-ratio test (G test) does not meet registered control false-positive requirements at most shot counts, and its asymptotic calibration is not a finite-sample guarantee. A Bonferroni simultaneous confidence-region decision rule is operationally unusable here: it abstains on every control and every subtle and moderate fault, and detects only large faults at 1024 and 4096 shots.

This report uses calibration records only for candidate calculations. Previously inspected validation records are not used for candidate selection or claims. The reports' registered criteria are left unchanged. No simulation or model training was run, and no existing result, threshold, acceptance criterion, or preregistration record was edited.

## Scope and provenance

Read: `finite_shot_design_feasibility_review.md`, `finite_shot_joint_power_analysis.md`, `finite_shot_candidate_selection_protocol.md`, `finite_shot_distribution_calibration_followup.md`, the final calibration summary/registry, detector and calibration-runner implementations, and saved calibration JSON records under `finite_shot_distribution_calibration_run/calibration/`.

The run has 1,800 calibration rows: five strata (two controls and subtle/moderate/large faults), 120 rows per stratum and shot count, at 256, 1024, and 4096 shots. Each row contains the expected and observed 16-outcome distributions, TVD, permutation p-value, and detection output. Reconstructing counts as probability × shot count yielded integer counts summing to the stated shot count for every inspected row. No output files were changed.

The 120 template IDs per stratum are not 120 demonstrated independent circuit families. They share one four-qubit paired-CNOT/phase-tag recipe and vary seeded parameters/interventions. Shots, outcomes, fault variants, and repeated conditions within a template are not additional independent units. Confidence bounds below are exact *binomial arithmetic conditional on treating the 120 row-level indicators as independent Bernoulli trials*. If template-level outcomes are dependent, the bounds are too narrow; the source artifacts do not establish an independent-family sampling design. They are descriptive uncertainty summaries, not population guarantees.

## Why the Bonferroni rule abstains

The candidate constructs marginal two-sided exact Clopper–Pearson intervals for each of 16 multinomial coordinates in each of two samples, allocating total noncoverage `alpha/(2K)` with `alpha=.05`, `K=16`; each tail is `.05/(4×16)=0.00078125`. It intersects the coordinate intervals with the probability simplex and computes a lower and upper bound for TV distance. It asserts anomaly only when `D_L > .05`, asserts no shift only when `D_U <= .05`, and abstains otherwise.

This protects simultaneous coordinate coverage under the binomial marginal model, but the simultaneous box is deliberately wide. At 16 cells and modest counts, even small marginal uncertainty in many coordinates accumulates in the worst-case TV upper bound. The lower bound is also conservative because it must hold for every distribution in the simultaneous region. Consequently, the uncertainty interval often straddles the .05 boundary. The observed pattern supports a **combination** of (i) severe Bonferroni conservatism, (ii) a coordinate-wise box/simplex region that discards multinomial dependence and is wider than a purpose-built joint region, and (iii) weak effect separation for subtle/moderate faults at low shots. The data do not isolate a unique causal share for each component; no alternate exact joint region was implemented for this review.

A scale check illustrates the multiplicity cost, without pretending every cell has probability 1/16: for a representative cell with p=1/16 and the nearest integer count, the Bonferroni marginal CP interval width is 0.1001, 0.0489, and 0.0242 at 256, 1024, and 4096 shots. Summing those 16 representative widths gives 1.60, 0.78, and 0.39 probability mass, respectively (these sums are not themselves TV confidence bounds). This wide coordinate box allows substantial rearrangement among cells before its worst-case distance becomes decisive. The distinction is important: it is evidence of conservative geometry and simultaneous multiplicity, not proof that a different joint confidence region will pass the registered sensitivity floors.

Calibration-only outcomes for the Bonferroni rule (anomaly / no-shift / abstain, per 120 rows):

| Shots | Identical control | Measurement-equivalent control | Subtle | Moderate | Large | Total non-abstaining coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0 / 0 / 120 | 0 / 0 / 120 | 0 / 0 / 120 | 0 / 0 / 120 | 0 / 0 / 120 | 0/600 = 0% |
| 1024 | 0 / 0 / 120 | 0 / 0 / 120 | 0 / 0 / 120 | 0 / 0 / 120 | 120 / 0 / 0 | 120/600 = 20% |
| 4096 | 0 / 0 / 120 | 0 / 0 / 120 | 0 / 0 / 120 | 0 / 0 / 120 | 120 / 0 / 0 | 120/600 = 20% |

Across all shots coverage is 240/1800 = 13.3%. The zero control anomaly count is not evidence of useful FPR control because the method never makes a control decision. It has no accepted no-shift diagnoses either. Sensitivity floors are missed in 7/9 positive shot-by-strength cells. This establishes failure of this *construction and operating point*, not failure of all multinomial confidence regions.

## Candidate definitions and statistical interpretation

1. **Unchanged TVD guard (baseline).** Assert anomaly when empirical `TVD > max(.05, 1/sqrt(n))`. This has no stated sampling-level type-I error guarantee. It is a fixed engineering guard, not a calibrated test.
2. **Unchanged pooled permutation TVD (baseline).** Pool the two samples, permute labels, and compare the observed two-sample TVD with the permutation distribution; saved implementation uses 999 resamples and the add-one p-value. Under exchangeability of the two samples under the null and correctly executed randomization, this is a conditional randomization test; Monte Carlo resolution and randomization error remain. It tests equality, not practical equivalence, and failure to reject does not certify “no fault.”
3. **Conditional permutation + effect gate (exploratory candidate).** Assert anomaly only if the unchanged permutation p-value is `<= .05` and empirical TVD is `> .05`; otherwise abstain. The .05 gate is an exploratory minimum observed effect size, not a confidence bound on population TVD. Conditional on a valid permutation test, intersecting rejection with an additional event cannot increase the null rejection probability, but power falls; this does not provide a calibrated no-shift decision or control the error of a multi-class diagnosis. The .05 practical margin requires domain justification and prospective registration before future data.
4. **Multinomial likelihood-ratio G test (exploratory comparator).** For expected counts `E_i=n p_i` and observed counts `O_i`, use `G²=2 Σ_i O_i log(O_i/E_i)` (zero terms contribute zero), with asymptotic `χ²_15` reference and flag at `p<.05`. The reference assumes the usual large-sample multinomial approximation. Here the minimum expected cell count was 3; 23/1800 (1.28%) of rows had at least one expected cell under 5. Asymptotic p-values are therefore not an exact finite-sample guarantee. A conditional permutation calibration of G² or a validated sparse-cell method would be needed before claiming error control.
5. **G test + effect gate (exploratory).** Assert anomaly if G-test `p<.05` and empirical TVD `>.05`; otherwise abstain. This inherits the G approximation issue and the gate is not a confidence interval. It is evaluated only to show the operating trade-off, not selected for future confirmation.

None of the tests provides calibrated confidence scores. The existing ML max-probability cutoff (`.60`) is not calibrated, and detector abstention fields in saved outputs do not establish probability calibration. No reliability diagram, calibration slope, or Brier score is justified for these test statistics as “confidence.”

## Calibration-only operating results

Cells report `count/120 (rate; one-sided 95% exact CP lower/upper bound)` for positive fault strata as sensitivity, and for controls as false-positive rate (FPR). For each cell, both exact bounds are shown as `L–U`; for zero/all counts the usual exact endpoint convention is used. These bounds assume independent row indicators as described above. Registered sensitivity floors are unchanged: subtle .50/.80/.90, moderate .80/.90/.95, large .95/.95/.95 at 256/1024/4096 shots. Registered control FPR upper-bound ceiling is .05.

### Permutation + observed TVD > .05

| Shots | Scenario | Anomaly assertions | Rate (one-sided 95% CP L–U) |
|---:|---|---:|---:|
| 256 | Identical control (FPR) | 5/120 | .0417 (.0166–.0856) |
| 256 | Measurement-equivalent control (FPR) | 2/120 | .0167 (.0030–.0515) |
| 256 | Subtle sensitivity | 14/120 | .1167 (.0719–.1764) |
| 256 | Moderate sensitivity | 83/120 | .6917 (.6151–.7610) |
| 256 | Large sensitivity | 120/120 | 1.000 (.9753–1.000) |
| 1024 | Identical control (FPR) | 6/120 | .0500 (.0220–.0963) |
| 1024 | Measurement-equivalent control (FPR) | 5/120 | .0417 (.0166–.0856) |
| 1024 | Subtle sensitivity | 36/120 | .3000 (.2314–.3762) |
| 1024 | Moderate sensitivity | 120/120 | 1.000 (.9753–1.000) |
| 1024 | Large sensitivity | 120/120 | 1.000 (.9753–1.000) |
| 4096 | Identical control (FPR) | 1/120 | .0083 (.0004–.0389) |
| 4096 | Measurement-equivalent control (FPR) | 1/120 | .0083 (.0004–.0389) |
| 4096 | Subtle sensitivity | 113/120 | .9417 (.8932–.9723) |
| 4096 | Moderate sensitivity | 120/120 | 1.000 (.9753–1.000) |
| 4096 | Large sensitivity | 120/120 | 1.000 (.9753–1.000) |

The 4096-shot subtle point estimate exceeds .90 but its one-sided lower bound does not meet the .90 floor. At 256 and 1024 shots, subtle and moderate sensitivity are below their point floors. One of two 4096 control upper bounds meets .05 exactly under the independent-row calculation; the other also meets it. This narrow apparent pass does not rescue the positive-cell failures or dependence concern.

For this positive-assertion/otherwise-abstain rule, **positive-decision coverage** is the fraction of all 600 rows at a shot count that receive an anomaly assertion; it is not full diagnostic coverage. The other rows are abstentions, including controls and missed faults. Positive-decision coverage is 184/600=30.7% at 256 (7/240 controls and 117/360 faults asserted), 287/600=47.8% at 1024 (11/240 controls and 276/360 faults), and 355/600=59.2% at 4096 (2/240 controls and 353/360 faults). Corresponding abstention rates are 69.3%, 52.2%, and 40.8%. Among anomaly assertions, false discovery fraction / selective risk is 7/184=3.8%, 11/287=3.8%, and 2/355=0.56%; precision at this dataset's designed 60% fault prevalence is 96.2%, 96.2%, and 99.4%. These are dataset-mixture quantities, not deployment precision. Since there is no accepted “normal” output, these must not be presented as conventional all-class coverage or selective risk for a complete classifier.

### G-test and G-test + effect gate

| Shots | Scenario | G test count / rate (one-sided 95% CP L–U) | G + TVD>.05 count / rate (one-sided 95% CP L–U) |
|---:|---|---|---|
| 256 | Identical control | 6/120 .0500 (.0220–.0963) | 6/120 .0500 (.0220–.0963) |
| 256 | Measurement-equivalent control | 1/120 .0083 (.0004–.0389) | 1/120 .0083 (.0004–.0389) |
| 256 | Subtle | 16/120 .1333 (.0855–.1954) | 16/120 .1333 (.0855–.1954) |
| 256 | Moderate | 86/120 .7167 (.6413–.7839) | 86/120 .7167 (.6413–.7839) |
| 256 | Large | 120/120 1.000 (.9753–1.000) | 120/120 1.000 (.9753–1.000) |
| 1024 | Identical control | 4/120 .0333 (.0115–.0747) | 4/120 .0333 (.0115–.0747) |
| 1024 | Measurement-equivalent control | 5/120 .0417 (.0166–.0856) | 5/120 .0417 (.0166–.0856) |
| 1024 | Subtle | 40/120 .3333 (.2621–.4109) | 40/120 .3333 (.2621–.4109) |
| 1024 | Moderate | 120/120 1.000 (.9753–1.000) | 120/120 1.000 (.9753–1.000) |
| 1024 | Large | 120/120 1.000 (.9753–1.000) | 120/120 1.000 (.9753–1.000) |
| 4096 | Identical control | 7/120 .0583 (.0277–.1068) | 1/120 .0083 (.0004–.0389) |
| 4096 | Measurement-equivalent control | 9/120 .0750 (.0397–.1272) | 1/120 .0083 (.0004–.0389) |
| 4096 | Subtle | 119/120 .9917 (.9611–.9996) | 113/120 .9417 (.8932–.9723) |
| 4096 | Moderate | 120/120 1.000 (.9753–1.000) | 120/120 1.000 (.9753–1.000) |
| 4096 | Large | 120/120 1.000 (.9753–1.000) | 120/120 1.000 (.9753–1.000) |

G-test positive-decision coverage is 229/600=38.2%, 289/600=48.2%, and 375/600=62.5%; abstention is 61.8%, 51.8%, and 37.5%. Selective risk among positive assertions is 7/229=3.1%, 9/289=3.1%, and 16/375=4.3%, respectively. With the effect gate it is 7/229=3.1%, 9/289=3.1%, and 2/355=0.56%; positive-decision coverage/abstention is 38.2%/61.8%, 48.2%/51.8%, and 59.2%/40.8%. These are not full-classifier coverage/risks. G-test alone fails the .05 control upper-bound criterion in several cells, most notably both 4096-shot controls. G+gate decreases those high-shot control flags but misses the registered 4096 subtle sensitivity lower-bound requirement (.893 < .90). At lower shot counts it does not resolve subtle-fault separation.

### Fixed-guard TVD baseline context

Saved calibration detector counts from the follow-up report: at 256 shots it flags 120/120 in both control strata and all positive strata except this statement is based on the shared baseline result; at 1024 it flags 104/120 identical controls, 107/120 measurement-equivalent controls, 119/120 subtle, and 120/120 moderate/large; at 4096 it flags 1/120 in each control, 113/120 subtle, and 120/120 moderate/large. The current guard trades catastrophic low-shot control FPR for high sensitivity; it is not an error-calibrated rule. Permutation suppresses low-shot control flags substantially but reduces subtle sensitivity (14/120 at 256; 36/120 at 1024; 119/120 at 4096). The conditional permutation + gate has the same counts as permutation through 1024 and improves 4096 controls, but loses six of 119 subtle flags at that shot count.

## Methodological conclusion and novelty

The empirical failure is not only implementation. The fixed TVD guard's low-shot false positives arise because sampling fluctuations are compared against a deterministic distance threshold without a null sampling distribution. The permutation baseline addresses that equality-test problem and has far fewer low-shot control flags, but the saved calibration shows the expected power trade-off for subtle faults. The Bonferroni rule adds a valid-under-model simultaneous coverage target but pays for it with a wide 16-coordinate region and an unhelpful three-way decision. G-test gives a useful benchmark but its nominal asymptotic reference is not enough to claim the desired finite-sample FPR.

Neither a likelihood-ratio test nor “test plus effect threshold” is itself a novel contribution. Closest relevant methods and the project novelty assessment already identify statistical goodness-of-fit testing and finite-shot uncertainty handling as established ideas. A defensible contribution would need a specific, validated selective-diagnosis procedure that (a) controls false alarms under an explicit sampling/exchangeability model, (b) gives a meaningful no-fault or abstain result through a separately justified equivalence/decision boundary, and (c) improves a predeclared coverage-versus-risk or fault-sensitivity objective across independently sampled circuit families. This saved one-topology calibration cannot establish that distinction. No clear methodological novelty claim survives this feasibility pass by itself.

## Recommendation and next step

Do not launch the proposed held-out run with the Bonferroni candidate or promote either exploratory candidate. Preserve the registered acceptance criteria. If the project continues, first specify a prospective design on calibration data that separates: a level-controlled anomaly test; a practical-effect/equivalence margin justified independently of these labels; and an abstention/coverage objective. Use a joint multinomial or conditional randomization procedure with a finite-sample error interpretation, and select it only on calibration families. Do not call the current 120 seeded templates independent families; define genuinely distinct family units first. Freeze the full method, thresholds, seeds, split, and analysis before a new, family-disjoint evaluation. If no candidate can satisfy the existing sensitivity/FPR floors at feasible shot cost, narrow the paper to a documented negative/feasibility result or revise the scientific question prospectively—not the criteria after seeing outcomes.

**Conclusion: no feasible candidate is established with the available calibration evidence.** The conditional permutation + TVD gate is the most promising low-cost exploratory comparator, but it fails the registered subtle sensitivity bound at the highest shot count and is not yet an adequately novel or independently validated method.


