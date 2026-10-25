# Finite-shot design feasibility review

**Date:** 2026-10-14  
**Status:** Statistical audit and read-only calibration analysis. No simulation, training, production change, or edit to an existing result/preregistration was made.

## Executive decision

**No-go for the proposed effect-size-aware selective method as currently specified, and no-go for the 3,600-, 9,000-, or 13,500-configuration confirmatory run now.** The exact Bonferroni multinomial confidence-region construction in the candidate-selection protocol was evaluated on calibration records only. It abstained on all controls, subtle faults, and moderate faults at every shot count; it detected only large faults, 120/120 at 1,024 and 4,096 shots. Its total coverage was 0% at 256 and 20% at 1,024/4,096, below the revised proposal's 70% overall minimum. The FPR of zero is vacuous with zero control coverage.

The power calculations in the previous report are mostly correct, but the proposed 900-family design does not automatically establish the joint all-target claim. At n=900 per cell, under an illustrative candidate sensitivity vector and true control FPR .025, the probability all 15 CP gates pass is about .952 if endpoint pass events are independent; Frechet bounds allow a much broader range when dependence is unknown. Adding a minimum-coverage gate lowers the joint pass probability. Family-level dependence is not estimable from the existing one-recipe calibration.

The smallest design capable of answering the statistical **anomaly detection** question is therefore not currently established. First replace or substantially improve the candidate interval construction using calibration data only, with a prespecified useful-coverage target. Then simulate the joint acceptance rule under family-level dependence estimated from a genuinely multi-family development set. If a novel staged anomaly/cause/localization method cannot be distinguished from established statistical testing and quantum fault-diagnosis methods, narrow the paper to a benchmark/negative-result contribution instead of financing a large run.

## Artifacts and code inspected

Read the four requested reports, `tests/fixtures/finite_shot_distribution_calibration.json`, `reports/experiments/finite_shot_distribution_calibration_run/summary.json`, the calibration output records, `src/quantum_simulator/statistics.py`, `experiments/finite_shot_calibration_plan.py`, `src/quantum_simulator/ml_diagnosis.py`, `src/quantum_simulator/fault_localizer.py`, and `src/quantum_simulator/hardware_fault_localizer.py`.

The calibration output folder has 1,800 calibration records and 1,800 validation records (3,600 total), with 120 templates per split, 5 scenario strata, and 3 shot levels. The registered design changes interventions while reusing the reference template across all conditions. It uses one fixed four-qubit paired-CNOT/phase-tag recipe. The records are distinct generated templates conditional on this recipe, not independent samples from a broad population of circuit families. They contain outcome distributions and intervention labels, but not independently adjudicated source-statement/qubit/gate localization truth.

## Independent audit of Clopper–Pearson calculations

For x successes among n independent Bernoulli units, the one-sided 95% exact lower confidence bound is

\[
L(x,n)=\begin{cases}0,&x=0,\\ Q_{.05}\{\mathrm{Beta}(x,n-x+1)\},&x>0.\end{cases}
\]

For y false positives among n independent controls, the one-sided 95% exact upper bound is

\[
U(y,n)=\begin{cases}1,&y=n,\\ Q_{.95}\{\mathrm{Beta}(y+1,n-y)\},&y<n.\end{cases}
\]

These match the project's `_cp_lower` / `_cp_upper` definitions. An acceptance cell passes iff its sensitivity lower bound is at least the registered floor or its FPR upper bound is at most .05. They are exact binomial limits only for independent Bernoulli units; row-level repetition within a circuit family violates that assumption for family-population inference.

### Why 59 zero-error controls only makes the FPR criterion attainable

With zero false positives,

\[
U(0,n)=1-.05^{1/n}.
\]

At n=58 this is approximately .05034, above .05. At n=59 it is approximately .04951, just below .05. Thus 59 independent controls with **zero** false positives is the smallest possible sample that could pass; one error fails. It is an attainability limit, not adequate power or a robust estimate. For sensitivity, all successes give (L(n,n)=.05^{1/n}), so floors .50/.80/.90/.95 first become attainable at n=5/14/29/59, respectively. This likewise assumes perfect outcomes and says nothing about power.

### Independent reproduction of cited power figures

For FPR ceiling .05, define (k_f(n)=\max\{y:U(y,n)\le.05\}). Under true FPR (p), exact power is

\[
\Pr\{X\le k_f(n)\}=\sum_{y=0}^{k_f(n)}{n\choose y}p^y(1-p)^{n-y}.
\]

SciPy `beta.ppf` and `binom.cdf` give:

| n | largest passing false-positive count (k_f) | power at true FPR .025 |
|---:|---:|---:|
| 120 | 1 | .1954 (19.5%) |
| 240 | 6 | .6063 (60.6%) |
| 400 | 12 | .7940 |
| 420 | 13 | .8280 |
| 600 | 20 | .9197 |
| 800 | 29 | .9796 |
| 900 | 34 | .9920 |

Therefore the earlier 19.5% and 60.6% figures are correct under independent controls and the true-FPR=.025 planning alternative. The observed calibration registry also confirms n=120: one false positive has upper bound .0389208 and two have .051534, so its max of one is correct. These power values are not candidate estimates; .025 is an illustrative alternative, not a registered fact.

### Sensitivity power examples

At n=120, exact minimum detections for registered floors .50/.80/.90/.95 are 70/104/114/119. Under illustrative true sensitivities .65/.90/.95/.98, the powers are .947/.910/.606/.305. At n=240, the same floor/alternative pairs yield approximately .999/.997/.904/.793. At n=400: approximately 1.000/1.000/.981/.938. At n=900: all exceed .999. These are selected alternatives; small margins above high sensitivity floors materially increase required n. They must not be mistaken for expected candidate performance.

## Calibration-only evaluation of the stated confidence-region rule

### Exact rule evaluated

The selection protocol proposed (K=16), overall confidence .95, and δmin=.05. For each of two multinomial samples, I formed two-sided exact marginal Clopper–Pearson intervals with per-tail alpha/(4K)=.00078125. Bonferroni across 2K coordinates yields at least .95 simultaneous coverage; intersecting each interval box with the simplex gives a valid (conservative) multinomial confidence region. I evaluated:

* (D_L=\min_{p\in C_p,q\in C_q}\tfrac12\|p-q\|_1), using a linear program with auxiliary absolute-difference variables.
* (D_U=\max_{p\in C_p,q\in C_q}\tfrac12\|p-q\|_1), using exact exhaustive maximization over the (2^{16}=65,536) TVD sign/subset forms. For subset A, the region extrema are (p_{max}(A)=\min(\sum_{A}u_p,1-\sum_{A^c}l_p)) and (q_{min}(A)=\max(\sum_A l_q,1-\sum_{A^c}u_q)); (D_U=\max_A[p_{max}(A)-q_{min}(A)]).
* State rule: anomaly if (D_L>.05); no meaningful shift if (D_U\le.05); abstain otherwise.

Counts were reconstructed from the saved normalized expected/observed distributions multiplied by shots; all 1,800 calibration records reconstructed to exactly the registered shot total. No validation records were used. The calculation used the exact LP lower extremum and exhaustive upper extremum, not loose coordinate-wise outer bounds. No result was written back into project output records or the summary.

### Calibration results (exploratory candidate feasibility only)

| Shots | Scenario | Candidate anomaly / n | No-shift / n | Abstain / n | Sensitivity or FPR |
|---:|---|---:|---:|---:|---:|
| 256 | identical control | 0/120 | 0/120 | 120/120 | FPR 0/120; coverage 0 |
| 256 | measurement-equivalent control | 0/120 | 0/120 | 120/120 | FPR 0/120; coverage 0 |
| 256 | subtle | 0/120 | 0/120 | 120/120 | sensitivity 0%; below .50 floor |
| 256 | moderate | 0/120 | 0/120 | 120/120 | sensitivity 0%; below .80 floor |
| 256 | large | 0/120 | 0/120 | 120/120 | sensitivity 0%; below .95 floor |
| 1,024 | identical control | 0/120 | 0/120 | 120/120 | FPR 0/120; coverage 0 |
| 1,024 | measurement-equivalent control | 0/120 | 0/120 | 120/120 | FPR 0/120; coverage 0 |
| 1,024 | subtle | 0/120 | 0/120 | 120/120 | sensitivity 0%; below .80 floor |
| 1,024 | moderate | 0/120 | 0/120 | 120/120 | sensitivity 0%; below .90 floor |
| 1,024 | large | 120/120 | 0/120 | 0/120 | sensitivity 100%; passes .95 point floor |
| 4,096 | identical control | 0/120 | 0/120 | 120/120 | FPR 0/120; coverage 0 |
| 4,096 | measurement-equivalent control | 0/120 | 0/120 | 120/120 | FPR 0/120; coverage 0 |
| 4,096 | subtle | 0/120 | 0/120 | 120/120 | sensitivity 0%; below .90 floor |
| 4,096 | moderate | 0/120 | 0/120 | 120/120 | sensitivity 0%; below .95 floor |
| 4,096 | large | 120/120 | 0/120 | 0/120 | sensitivity 100%; passes .95 point floor |

Overall candidate coverage is 0/600 at 256, 120/600=20% at 1,024, and 120/600=20% at 4,096. Across shots, it covers 240/1,800=13.3%, well below the revised proposal's 70% overall target. The 0/120 FPR is uninformative because it abstains on all negative controls. Selective risk is 0 among covered cases at 1,024/4,096 only because the only covered cases are large positive faults; it cannot be reported as evidence of useful selective performance without the very low coverage disclosed.

This directly evaluates the specific Bonferroni coordinate-region construction in the proposed protocol. It does not prove every valid confidence-region method will fail. It does show that this implementation cannot support the registered sensitivity floors or a useful coverage target on the existing calibration set and is not ready for a held-out run. Replacing it with a narrower likelihood-ratio region or a parametric bootstrap is a method change; such a candidate would need calibration-only development and a newly frozen protocol, never a validation-selected claim.

The previous validation q95 numbers remain exploratory: low-shot subtle sensitivity 13/120 and 41/120. They were not used in the candidate computation above, parameter selection, or confirmatory inference.

## Clarifying independent units

Configuration count is not sample size. One configuration is one scenario × shot observation, but the registered experiment reuses each template across five scenarios and three shot counts. Those 15 rows are paired and correlated. Multiple shots are samples within a configuration, not independent diagnosis units. Fault variants applied to the same template are within-template paired outcomes. Repeating seeds or templates under one fixed topology broadens parameter draws conditional on that topology; it does not create new circuit-family clusters.

The proposed designs mean:

| Design | Configurations | Proposed template/family structure | Independent family units per cell | What it can support |
|---|---:|---|---:|---|
| 3,600 revised proposal | 3,600 | 240 templates in 20 families × 12 templates; each template has 15 conditions | 20 family clusters, 240 template observations (not independent across family) | Paired pilot/exploration. At ICC .05/.10/.20, a crude design-effect effective n is 155/109/71. Cannot give precise family-generalization intervals with only 20 clusters. |
| 9,000 old q95 plan | 9,000 | 600 templates from the same registered fixed topology/phase-tag recipe × 15 conditions | At most 1 topology family as currently specified; 600 conditional templates per cell | Checks candidate behavior on new seeds/templates conditional on the same recipe. Not a broad unseen-family test. |
| 13,500 revised power proposal | 13,500 | 900 genuinely new circuit families × 1 template/family × 15 conditions | 900 family units per cell if families are independently sampled and truly distinct | Supports family-level CP per cell and paired cross-condition analysis, subject to construction feasibility and explicit target-population definition. |

For the 3,600 proposal, design effect (DE=1+(m-1)\rho) with m=12 gives nominal n=240 divided by DE: 155 at ρ=.05, 109 at .10, and 71 at .20. This is a sensitivity analysis, not measured ICC. The old calibration uses one topology family and cannot estimate ρ. A 900-unit design only avoids this clustering concern if each unit is an independently drawn family; renaming 900 templates from one generator as 900 families would not satisfy it.

## Joint power and criterion dependence

There are nine registered sensitivity cells and six registered FPR cells. The global pass is an intersection-union rule: all 15 cells must meet their one-sided .05 CP limits. This conjunction controls false acceptance at .05 without Bonferroni across the component tests. It does **not** imply high joint power. A separate sensitivity or FPR rate does not stand in for probability that every gate passes.

Using illustrative true sensitivities [.65,.90,.95] for subtle across shots, [.90,.95,.98] for moderate, and [.98,.98,.98] for large, plus true control FPR .025, the exact marginal powers and dependence bounds are in the companion `finite_shot_joint_power_analysis.md`. For n=900, marginal powers are near .992 or higher and the independent-endpoint joint pass probability is about .952. Frechet bounds based only on those marginals are about [.951,.992]. At n=400, the independent-endpoint all-15 probability is only about .187 and the Frechet lower bound is zero. These are hypothetical power calculations, not measured candidate rates.

Minimum useful coverage must be part of the joint decision. The prior revised proposal says overall coverage ≥.70 and ≥.50 in each ambiguity stratum, but does not define the ambiguity stratum operationally or state whether the coverage gate needs a confidence lower bound. Candidate calibration coverage is 13.3% overall, so it fails even the point threshold. For n=900, if true coverage were .80, a point threshold of .70 would have very high power under independent families; a lower confidence-bound coverage gate would require a separately frozen interval and a joint-power calculation. We cannot estimate future candidate coverage from the old data after the observed candidate's low calibration coverage, nor can we infer criterion dependence from one topology.

Paired superiority claims add at least two baselines and multiple endpoints. With paired discordance q=.15 and a 10-point difference, a single two-sided .05 comparison needs roughly n=110 independent pairs for 80% power by normal approximation. The new candidate's discordance is not available until it is viable; do not borrow the old q95 validation performance. Use paired family resampling and hierarchical/Holm testing for candidate-versus-TVD and candidate-versus-permutation claims.

## Design comparison: runtime and memory

The run completed 3,600 configurations in 3,192.3 seconds (53.2 min). Linear scaling is shown only as a baseline; actual runtime requires safe preflight because the old Windows lock error has no captured root cause and OneDrive caused manifest contention.

| Design | Sampled shot executions | Linear runtime from completed run | Planning runtime | Storage estimate / memory |
|---|---:|---:|---:|---|
| 3,600 configurations | 12.9024 million | 53 min | 1–3 h | ~8.2 MB outputs; reserve 25 MB; reserve ≥1 GiB RAM |
| 9,000 configurations | 32.256 million | 133 min | 2.25–6 h in old preregistration | ~20.4 MB outputs; reserve 50 MB; ≥1 GiB RAM |
| 13,500 configurations | 48.384 million | 199.5 min | 4–8 h proposed | ~30.6 MB outputs; reserve 75 MB; ≥1 GiB RAM |

Arithmetic assumes two sampled circuits per configuration and 256+1,024+4,096 shots per each of five strata per template. The one-process active permutation working set is below 1 MiB; Python/runtime/output retention dominates memory, so stream results. Runtime/storage are local simulation estimates, not hardware costs.

## Novelty and publication scope

The project cannot claim novelty for applying statistical tests, comparing output distributions, quantum fault diagnosis/localization, or abstention alone. Huang & Martonosi (ISCA 2019) establish quantum statistical assertions; Li et al. (VTS 2023) use fault dictionaries plus dynamic binary-search localization; Oldfield et al. (TOSEM 2025) study distribution oracles and measurement cost; Ishimoto et al. (EASE 2025) evaluate mutation-based quantum source localization. The 2025 EASE paper makes source localization an especially close competitor.

The only candidate distinction is a *tested decision contract* that separates evidence for a practically meaningful output shift from evidence for a cause/target, abstains when uncertainty crosses a preregistered effect margin, and evaluates risk/coverage/cost on unseen circuit families. The evaluated confidence-region construction does not make this contribution viable: it is too conservative for the registered sensitivity and coverage gates. A less-conservative finite-sample confidence method could be a method contribution only if it has a precise coverage justification, beats unchanged permutation and established diagnosis methods at matched FPR/coverage, and generalizes beyond this project's single current circuit recipe. Otherwise narrow to a data/benchmark or negative-result paper.

## Recommended next action

Do not run any of the three proposed held-out configurations now. Keep the completed outputs and registered criteria untouched. First decide whether the paper is limited to anomaly detection or intends joint source/hardware localization. If localization remains central, build an independent labeled dataset and implement faithful literature baselines before sizing the experiment. If anomaly detection is the target, develop a less conservative candidate using calibration data only, explicitly model reference-sample uncertainty, and set a minimum useful coverage requirement. Then run a *joint* power analysis using independent family-level data and a frozen operating alternative. Proceed only if calibration feasibility passes all sensitivity/FPR/coverage floors and the required number of genuinely independent families is available. The current evidence supports a no-go on the large run.

