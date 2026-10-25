# Calibration-only candidate-selection protocol

**Date:** 2026-10-14  
**Status:** design only; no new simulations, threshold changes, or edits to the existing registry/results. This protocol is a proposed successor for a future, separately preregistered study. The prior 9,000-configuration q95 preregistration remains unchanged and unexecuted.

## Evidence and inspected implementations

This protocol builds on the research-gap assessment, candidate comparison, revised selective-diagnosis proposal, finite-shot follow-up, `src/quantum_simulator/statistics.py`, `experiments/finite_shot_calibration_plan.py`, `src/quantum_simulator/ml_diagnosis.py`, `src/quantum_simulator/fault_localizer.py`, and the actual registry `tests/fixtures/finite_shot_distribution_calibration.json`.

The saved calibration directory has 3,600 completed configuration outputs (plus a manifest and summary); the fixture has 120 templates in one fixed paired-CNOT/phase-tag circuit recipe per split. The calibration and validation templates have differing seeds and fingerprints, but both use the same narrow recipe. Therefore they are independent generated draws conditional on that recipe, not a sample of independent circuit families supporting broad family-level inference. The saved output rows contain two sampled distributions and scenario labels; they do not contain independently adjudicated source-line/qubit/gate localization truth.

The TVD baseline uses `TVD > max(.05, 1/sqrt(shots))`, without a null-calibrated error guarantee. The permutation baseline pools the two samples and uses 999 random label permutations, add-one p-values and alpha=.05. `ml_diagnosis.py` abstains below a fixed top-class probability of .60, but the model probabilities are explicitly uncalibrated; this is not an empirical confidence guarantee. Existing calibration `abstained` values are a shared downstream diagnosis category, not detector-specific coverage. Program/hardware localization is rule-based and its scores are not calibrated probabilities.

Existing outputs support descriptive comparisons of TVD, permutation, and the calibration-only q95 cutoff. Reused validation figures are exploratory and already exposed; they are never candidate-selection or confirmatory evidence here. The q95 candidate's validation sensitivity of 13/120 at 256 shots and 41/120 at 1,024 shots, together with its control reductions, remains the historical exploratory result and is not changed.

## Explicit candidate: simultaneous multinomial TVD interval with an indifference margin

For a fixed, preregistered outcome alphabet of $K=16$ bitstrings, let independent count vectors be $X\sim\mathrm{Multinomial}(n_r,p)$ and $Y\sim\mathrm{Multinomial}(n_o,q)$. The target effect size is the population total variation distance

\[
D(p,q)=\tfrac12\sum_{j=1}^{K}|p_j-q_j|.
\]

Both finite samples contribute uncertainty. The candidate does not use the observed TVD as if one distribution were known exactly.

### Finite-sample uncertainty construction

Construct simultaneous confidence regions $C_p$ and $C_q$ for the two multinomial probability vectors, separately at overall confidence $1-\alpha=.95$. A conservative, auditable construction is:

1. For each of the $K$ fixed categories in each sample, form an exact two-sided Clopper–Pearson interval for its marginal probability with noncoverage $\alpha/(2K)$.
2. Intersect the coordinate intervals with the probability simplex (nonnegative components summing to one) to obtain $C_p,C_q$. Bonferroni across the $2K$ marginal intervals gives simultaneous coverage at least .95, without requiring independence among category counts.
3. Compute
\[
D_L=\min_{p'\in C_p,q'\in C_q}D(p',q'),\qquad
D_U=\max_{p'\in C_p,q'\in C_q}D(p',q').
\]

If the two multinomial samples are independent and their categories and sample sizes were fixed before looking at the data, then $D(p,q)\in[D_L,D_U]$ whenever both regions cover. Under those assumptions the interval has coverage at least .95 by construction; the Bonferroni set may be conservative. This is a finite-sample coverage argument conditional on valid marginal intervals, not a claim that the rule has been implemented, optimized, or empirically validated. Implement the extrema solver with numerical verification before a future run; optimization error must be bounded conservatively so it cannot spuriously raise $D_L$ or lower $D_U$.

The project already defines “subtle” effects with ideal TVD beginning at .05. Freeze $\delta_{min}=.05$ as the minimum practically meaningful output shift, provided scientific owners agree that this pre-existing boundary is a meaningful effect size. It is not selected to maximize results on old validation. If that interpretation is rejected, do not tune a replacement on validation; pause for a separately justified calibration-only domain choice.

### Three-way selective decision

Use the same family confidence level $1-\alpha=.95$:

* **ANOMALY:** $D_L>\delta_{min}$.
* **NO MEANINGFUL SHIFT:** $D_U\le\delta_{min}$.
* **ABSTAIN:** otherwise, including intervals that cross the margin.

The decision distinguishes statistical uncertainty from practical effect size. A zero-distance null lies within the no-meaningful-shift region. Since $D_L>\delta_{min}$ implies the confidence region excludes every distance at or below the margin, its false-positive probability is at most the interval's noncoverage probability under the stated sampling assumptions. This is a conservative bound, not an unconditional quantum-hardware guarantee. The candidate must abstain when a confidently meaningful shift cannot be separated from a shift at/below the margin.

This differs from:

* **Fixed-guard TVD:** one empirical distance compared with `.05` or `1/sqrt(shots)`; it supplies neither two-sample uncertainty limits nor a meaningful-effect confidence statement.
* **Pooled-label permutation:** tests equality of two distributions at a null-significance level. It does not by itself say whether a statistically detectable difference exceeds a domain-meaningful TVD margin, and it has no indifference-zone decision. It remains a required unchanged baseline.
* **Current ML abstention:** uses an uncalibrated model maximum probability and .60 operational cutoff; this candidate abstains from a two-sample uncertainty interval, not classifier confidence.

### Cause and localization gate

The candidate's first output is only a distribution-shift state. It must not infer “program fault,” “hardware anomaly,” qubit, gate, or source location from TVD alone. Following an ANOMALY decision, a separate evidence stage may assert a cause/target only if independent circuit-difference or hardware-calibration evidence supports it and no incompatible cause remains; otherwise it returns `ABSTAIN_CAUSE`. The saved finite-shot calibration lacks localization ground truth, so it cannot tune or score this stage. Any future cause/localization claim requires newly labeled held-out source/circuit records and a compatible baseline; otherwise the paper's claim stays at selective anomaly detection.

## Calibration-only selection and freezing

1. **No validation selection.** The untouched-validation split has already been read and is permanently exploratory. Use only original `calibration` rows for historical data analysis. Keep a template and all its scenario/shot records together in every fold.
2. **Declare the rule before parameter fitting.** Fix (K=16), the TVD definition, exact marginal interval method, optimization solver/tolerance, confidence level .95, (\delta_{min}=.05) subject to domain approval, abstention semantics, exclusions, and missing-output handling. The registered sensitivity floors and ≤.05 FPR bounds remain as they are; no criterion is relaxed.
3. **Calibration-only checks.** Using grouped calibration folds, assess numerical solver coverage against known multinomial probabilities only in analytical checks or a separately labeled development audit; assess interval width, abstention, sensitivity, FPR, and stability by shot/scenario. Because all existing templates use one recipe, these checks cannot estimate between-family variance and cannot establish external validity. Do not report fold-level estimates as independent confirmation.
4. **Candidate selection rule.** The interval construction is chosen for its explicit two-sample estimand and conservative finite-sample coverage logic, not because it wins a validation metric. If alternative interval constructions are tested, compare only in grouped calibration folds and freeze the candidate once. No old validation outcomes can break ties or select (\delta_{min}).
5. **Calibrate no model confidence.** Do not use `.60` or stored classifier probabilities as uncertainty calibration. If the future study includes classifier probabilities, fit/calibrate only on grouped development families and freeze the mapping before test access; this is optional and outside the primary interval candidate.
6. **Freeze record.** Hash source code, protocol, interval solver, configuration generator, seeds, family/template IDs, margins, alpha, thresholds, criteria, exclusions, and a complete dependency lock before generating the new test outputs. Changes after test access require a new held-out split.

## Leakage-resistant new split

Every held-out circuit family must be structurally distinct from all calibration/validation recipe IDs, Phase 1/2 families, candidate development data, and ML training families. Each family contributes one independent template/seed; that template is paired over the five registered strata and three shot levels. Thus each held-out family contributes 15 configuration records, but only one Bernoulli unit per family in any specific stratum × shot cell. No family, source skeleton, template hash, reference fingerprint, template seed, sampling seed, or fault intervention seed may overlap prior artifacts. Check uniqueness and disjointness by canonical structure hash as well as IDs and seeds; seed disjointness alone is not sufficient.

Freeze scenario and shot definitions, configuration IDs, all seeds and reference generation logic before execution. Store each sampled reference/observed count pair immutably, record whether the reference probabilities are themselves finite-shot estimates, and never use held-out references to tune the interval, margin, or source gate. Reference distribution uncertainty is included in (C_p) whenever reference counts are sampled; if ideal exact probabilities are used instead, define the one-sample regime in the preregistration and do not apply the two-sample formula unchanged.

## Endpoints and reporting

At each of 15 registered cells (two controls × three shots; three fault strengths × three shots), retain every configuration. Abstention is a separate outcome; it is a missed anomaly for sensitivity and is never silently removed from denominators. Report sensitivity/recall, FPR, specificity, abstention fraction, coverage, selective risk among non-abstained outputs, interval width, and exact one-sided Clopper–Pearson bounds for binary rates. Report precision only with the designed prevalence stated and do not present it as deployment precision.

For diagnosis, report cause accuracy and top-1/top-k localization only if independent ground truth exists; report confident wrong-cause/target risk, `ABSTAIN_CAUSE` coverage, and exact/joint diagnosis success. The calibration records do not support localization metrics.

Use paired comparisons on identical count pairs. Report paired discordant counts and paired family-level bootstrap intervals for candidate-minus-baseline differences in sensitivity, FPR, coverage and selective risk. For two baseline/candidate primary comparisons, use a fixed hierarchical testing plan or Holm adjustment for comparative claims. The absolute “all registered cells pass” acceptance claim is an intersection-union test: each cell must pass its registered one-sided .05 CP criterion, so no alpha correction is required for the conjunction's type-I error. That does **not** provide simultaneous 95% coverage for every separately worded cell claim, nor does it provide high probability that all gates pass. Apply Holm/Bonferroni for families of secondary per-cell claims and show unadjusted cell intervals as descriptive where clearly labeled.

Use CP intervals only for independent family-level Bernoulli units. Do not treat repeated configurations of the same family as independent observations. Paired analyses and all uncertainty about across-family generalization resample families, not individual rows. Predeclare how a failed, missing, unsupported, or inconclusive cell is reported: it remains in its denominator or is reported as a protocol failure; never replace it post hoc.

## Literature and novelty boundary

The closest verified prior methods include statistical assertions for quantum programs (Huang & Martonosi, ISCA 2019, [DOI](https://doi.org/10.1145/3307650.3322213)); differential output testing across quantum software stacks (Wang et al., ASE 2021, [DOI](https://doi.org/10.1109/ASE51524.2021.9678792)); fault-dictionary and dynamic localization of quantum circuits (Li et al., VTS 2023, [DOI](https://doi.org/10.1109/VTS56346.2023.10140030)); and mutation-based quantum source localization (Ishimoto et al., EASE 2025, [DOI](https://doi.org/10.1145/3756681.3757022)). These works rule out novelty from statistical testing, output distribution comparisons, or fault localization as categories. Systematic literature work remains necessary on finite-sample multinomial confidence regions, equivalence/indifference testing, selective prediction, and diagnosis under finite-shot sampling.

The narrow possible contribution is an empirically tested contract that (i) separates “statistically supported meaningful output shift” from “cause known,” (ii) abstains when uncertainty crosses a meaningful effect boundary, and (iii) measures family-held-out risk/coverage/cost. The confidence-region mathematics is established statistical machinery, not itself a novel contribution. Novelty rests on a precise quantum diagnosis formulation, faithful competitive baselines and a reproducible family-held-out demonstration. This remains an unverified hypothesis.

## Recommendation

Do calibration-only analytical and focused implementation work first. Do not launch the 9,000-case q95 run. The finite-shot artifacts show why the scalar guard is inadequate but do not evaluate this candidate or its interval coverage. Do not proceed to held-out simulation unless the interval implementation is mathematically checked, a meaningful .05 TVD boundary is approved, the sample-size/power plan below is satisfied, independent family units can be generated, and source/location labels are available for any corresponding claim. Otherwise narrow the paper to anomaly detection or stop the novelty claim.

