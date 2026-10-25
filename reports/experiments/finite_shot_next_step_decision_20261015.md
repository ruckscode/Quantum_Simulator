# Finite-shot next-step decision (2026-10-15)

## Evidence base

This planning review uses the characterization audit, its preregistration, and its result report. The prior experiment is complete and immutable. The saved all-shot null decision is FAIL because the 256-shot cell has 680/2,000 false positives (34.0%; one-sided exact 95% upper bound 35.78%). At 1,024 shots the observed rate is 114/2,000 (5.7%), with upper bound 6.63%; the result fails the registered upper-bound gate but does not determine whether the underlying rate is below or above 6%. At 4,096 shots it is 0/2,000, with upper bound 0.15%, satisfying that cell's gate.

The detector was fixed at `TVD > max(0.05, 1/sqrt(shots))`; there were no development draws or threshold selection. The smallest alternative (true TVD 0.025) had decreasing observed power with higher shots (44.0%, 21.4%, 2.0%), consistent with its being below the fixed 0.05 cutoff at the two larger shot counts. No alternative power pass criterion exists in the preregistration.

## Decisions

### 1. Remaining scientific question

For the specified four-outcome null and the unchanged detector at 1,024 shots, is the long-run false-positive probability at most 0.06? The existing point estimate is 0.057, but the current 2,000 replicates are too few for the registered one-sided upper confidence bound to certify 0.06. The 256-shot issue is not similarly unresolved: its observed rate and interval are far above the registered ceiling.

The experiment does not determine whether another detector or threshold can meet a chosen operating target while preserving useful alternative power. It also says nothing about circuit-derived distributions, QMutBench, independent families, hardware, or diagnosis.

### 2. Change detector or acceptance design?

**Neither should be changed in the existing experiment.** Its preregistration and result remain fixed. Evidence does justify describing the existing heuristic as not calibrated to the 6% target at 256 shots under this synthetic null; it does not justify changing production behavior or choosing a replacement cutoff from these evaluation outcomes. Changing the acceptance design after observing results would compromise the registered decision. The 1,024-shot failure is failure to certify under the stated criterion, not evidence that the point rate is above 6%.

### 3. Smallest useful next action

Prepare a **separate preregistration for an independent 1,024-shot, null-only confirmation**. Do not append samples to or rewrite the already registered result. Use fresh deterministic stream identifiers and analyze this new sample independently. Keep the null hypothesis and one-sided 95% exact upper-bound criterion at 0.06; specify the sample size, seed/version recording, and stop rules before generating any counts.

A planning calculation used the prior 5.7% point estimate as a design alternative and the 6% upper-bound target. Under the binomial model, approximately **38,228 new null pairs** gives about **80.2%** probability that the one-sided exact 95% Clopper-Pearson upper bound is at most 0.06 if the true null flag probability is 0.057. This is a proposed design target, not a preregistered requirement or a guarantee. It will have less power if the true rate is closer to 0.06; at exactly 0.06, a finite sample cannot be expected to certify an upper bound at or below the boundary with high probability. If a scientifically meaningful margin from 6% is desired, it must be specified before the follow-up instead of treating 5.7% as known truth.

**Proposed new criterion:** for that independent follow-up only, call the 1,024-shot null operating point certified against the target if its one-sided exact 95% upper FPR bound is <=0.06. Plan for at least 80% probability of certification at FPR=0.057. This reuses the numerical bound from the old preregistration but creates a new sample-size/power plan; it is not an existing criterion for a new experiment. The original experiment remains failed overall regardless of the follow-up.

**Estimated cost:** the completed run used 6.0 CPU seconds for 12,000 count-pairs. Linear extrapolation gives about 19 CPU seconds for 38,228 pairs at 1,024 shots (roughly 3.19 times the prior total pair count). This is a coarse estimate from the prior run, not a measured benchmark for a new process; environment and overhead can change it. No sampling was run for this planning review.

### 4. Effect on the research sequence

- **Statistical characterization:** retain the failed result. The proposed small follow-up would clarify the 1,024-shot cell only. It cannot cure the 256-shot failure. Any new detector/cutoff work would need a separate development/confirmation protocol and must not use the old evaluation outcomes for tuning.
- **Independent data provenance:** proceed as a separate evidence-acquisition stream if authorized. This synthetic null question neither satisfies nor replaces the QMutBench lineage, completeness, independent-label, or review gates; the current eligible-root count remains zero according to the audited project record.
- **Controlled mutation detection:** remains blocked until independently verified recipe roots, behavioral labels, and compatible immutable execution/measurement configuration are available. No conclusion here supports mutation-detection performance.
- **Family-separated classical ML:** remains downstream of provenance, label, and family-split gates. This statistical characterization supplies no training data, labels, independent families, or ML evidence and does not authorize model fitting.

## Explicit unknowns

- The true 1,024-shot null flag probability is not established to be either <=0.06 or >0.06 by the existing interval.
- The 5.7% estimate is a planning input from one finite synthetic run, not a known operating rate.
- The available artifacts do not establish whether a different detector/cutoff can jointly achieve a specified false-positive target and useful alternative power; no alternative power floor was registered.
- The follow-up cost estimate is extrapolated. No new-run environment, NumPy version, or runtime has been measured here.

## Recommended next action

Draft and independently review the separate 1,024-shot null-only follow-up preregistration, including a defensible margin/power target and the proposed sample size. Do not execute it until that protocol is frozen. In parallel, keep QMutBench evidence acquisition on its own provenance-gated path; do not allow either activity to advance mutation evaluation or ML before the required independent roots and labels exist.
