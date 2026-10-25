# Finite-shot candidate redesign comparison

Date: 2026-10-13

## Executive comparison

Calibration-only comparison of the existing fixed-guard TVD and pooled permutation tests, a permutation-plus-effect gate, asymptotic multinomial G test, G test plus effect gate, and the previously proposed Bonferroni confidence-region rule. **No candidate is feasible for confirmatory evaluation under the unchanged registered criteria.** Existing validation outputs were not used here.

| Rule | Statistical meaning | Calibration behavior | Registered operating-point assessment |
|---|---|---|---|
| Fixed-guard TVD | Deterministic empirical-distance guard; no calibrated type-I error | Very high low-shot sensitivity and severe low-shot control flags; 256-shot controls 120/120, 1024-shot controls 104/120 and 107/120 | Fails FPR at low shots; not a calibrated detector |
| Pooled permutation TVD | Conditional equality test under exchangeability; 999 Monte Carlo shuffles, add-one p | Control flags at 256: 5/120 and 2/120; subtle: 14/120, 36/120, 119/120 as shots rise | Better low-shot control behavior than fixed guard, but subtle sensitivity far below .50/.80 floors at 256/1024; 4096 one-sided lower bound .893 < .90 |
| Permutation + TVD>.05 | Permutation rejection AND observed practical-effect gate; otherwise abstain | Same as permutation at 256/1024; at 4096 controls 1/120 each and subtle 113/120 | Appears to meet control FPR upper bounds at 4096 under independent-row assumption, but subtle lower bound .893 < .90; no no-fault acceptance rule |
| G test, asymptotic | Likelihood-ratio homogeneity test with χ²(15) reference | Strong sensitivity at 4096; flags 7/120 and 9/120 controls | Control upper bounds exceed .05 for both controls at 4096; asymptotic calibration is not exact with some sparse expected cells |
| G + TVD>.05 | Approximate G test AND observed effect gate | At 4096 controls 1/120 each; subtle 113/120 | Same subtle lower-bound failure as gated permutation; at low shots poor subtle sensitivity; no calibrated equivalence decision |
| Bonferroni simultaneous regions | Simultaneous marginal exact intervals plus simplex and TV distance bounds | Coverage 0/600 at 256, 120/600 at 1024 and 4096; abstains on all controls/subtle/moderate | Fails 7/9 registered positive cells; operational no-go |

## Metrics and calculations

For each scenario and shot-count cell there are 120 calibration rows. A one-sided 95% exact Clopper–Pearson bound is calculated from `Beta^{-1}(.05;x,n−x+1)` for the lower bound when `x>0`, and `Beta^{-1}(.95;x+1,n−x)` for the upper bound when `x<n`; boundary cases are 0 and 1. These bounds assume independent Bernoulli row outcomes. Because each condition shares a single circuit-family recipe and the artifacts do not demonstrate family-level independence, treat bounds as row-level descriptive calculations and potentially anti-conservative for broader populations.

The candidate effect gate was exactly `TVD > .05`; it is an exploratory decision margin and the observed TVD is not a confidence bound for true population TVD. The G statistic is `2Σ O_i log(O_i/E_i)` with zero terms set to zero, `df=15`, asymptotic p-value threshold .05. Expected cell counts below five occur in 23/1800 calibration rows (1.28%), minimum expected count 3. No exact small-cell guarantee is claimed.

### Permutation + effect gate results

| Shots | Scenario | x/120 | Rate | One-sided 95% exact interval |
|---:|---|---:|---:|---:|
| 256 | Identical control FPR | 5 | .0417 | .0166–.0856 |
| 256 | Measurement-equivalent control FPR | 2 | .0167 | .0030–.0515 |
| 256 | Subtle sensitivity | 14 | .1167 | .0719–.1764 |
| 256 | Moderate sensitivity | 83 | .6917 | .6151–.7610 |
| 256 | Large sensitivity | 120 | 1.000 | .9753–1.000 |
| 1024 | Identical control FPR | 6 | .0500 | .0220–.0963 |
| 1024 | Measurement-equivalent control FPR | 5 | .0417 | .0166–.0856 |
| 1024 | Subtle sensitivity | 36 | .3000 | .2314–.3762 |
| 1024 | Moderate sensitivity | 120 | 1.000 | .9753–1.000 |
| 1024 | Large sensitivity | 120 | 1.000 | .9753–1.000 |
| 4096 | Identical control FPR | 1 | .0083 | .0004–.0389 |
| 4096 | Measurement-equivalent control FPR | 1 | .0083 | .0004–.0389 |
| 4096 | Subtle sensitivity | 113 | .9417 | .8932–.9723 |
| 4096 | Moderate sensitivity | 120 | 1.000 | .9753–1.000 |
| 4096 | Large sensitivity | 120 | 1.000 | .9753–1.000 |

The existing permutation results and gated counts coincide through 1024 shots. At 4096, the observed-effect gate removes six control assertions and six subtle assertions. Positive-assertion coverage is 30.7%, 47.8%, 59.2% across increasing shot count; positive-assertion abstention is 69.3%, 52.2%, 40.8%. Positive assertion selective risk is 3.8%, 3.8%, and 0.56%, respectively, under the experiment's 60% fault prevalence. These figures condition on an assertion-only rule: no non-rejection is accepted as “normal.” They are not full diagnostic coverage or deployment risk.

### G and G + gate results

| Shots | Scenario | G x/120 (rate; lower–upper) | G + gate x/120 (rate; lower–upper) |
|---:|---|---|---|
| 256 | Identical control | 6 (.0500; .0220–.0963) | 6 (.0500; .0220–.0963) |
| 256 | Measurement-equivalent control | 1 (.0083; .0004–.0389) | 1 (.0083; .0004–.0389) |
| 256 | Subtle | 16 (.1333; .0855–.1954) | 16 (.1333; .0855–.1954) |
| 256 | Moderate | 86 (.7167; .6413–.7839) | 86 (.7167; .6413–.7839) |
| 256 | Large | 120 (1.000; .9753–1.000) | 120 (1.000; .9753–1.000) |
| 1024 | Identical control | 4 (.0333; .0115–.0747) | 4 (.0333; .0115–.0747) |
| 1024 | Measurement-equivalent control | 5 (.0417; .0166–.0856) | 5 (.0417; .0166–.0856) |
| 1024 | Subtle | 40 (.3333; .2621–.4109) | 40 (.3333; .2621–.4109) |
| 1024 | Moderate | 120 (1.000; .9753–1.000) | 120 (1.000; .9753–1.000) |
| 1024 | Large | 120 (1.000; .9753–1.000) | 120 (1.000; .9753–1.000) |
| 4096 | Identical control | 7 (.0583; .0277–.1068) | 1 (.0083; .0004–.0389) |
| 4096 | Measurement-equivalent control | 9 (.0750; .0397–.1272) | 1 (.0083; .0004–.0389) |
| 4096 | Subtle | 119 (.9917; .9611–.9996) | 113 (.9417; .8932–.9723) |
| 4096 | Moderate | 120 (1.000; .9753–1.000) | 120 (1.000; .9753–1.000) |
| 4096 | Large | 120 (1.000; .9753–1.000) | 120 (1.000; .9753–1.000) |

The gating rule improves the observed 4096 control rates but its one-sided confidence bound for subtle sensitivity misses .90. G-only also fails control upper bounds in several cells. Its asymptotic p-values do not establish finite-sample control. Neither result qualifies as an independently validated improvement.

## Interpretation of coverage, abstention, risk, and confidence

All candidate tests here output “anomaly” or “abstain.” A p-value above .05 is not a probability that the null is true and does not license a “no fault” call. For each positive-only rule, report (i) anomaly assertion rate as positive-decision coverage, (ii) abstention as one minus that rate, (iii) FPR among controls, (iv) sensitivity among faults, and (v) false discovery fraction among asserted positives. Because the data mixture has 360 faults and 240 controls per shot count, the positive predictive value depends on the designed 60% prevalence. No deployment prevalence or calibrated score is available. A selective risk/coverage curve for a complete classifier is not supported by these artifacts.

The current ML score threshold is not calibrated, and source-localization heuristics have no ground-truth localization labels in these calibration outputs. Localization accuracy and confidence calibration cannot be measured from this dataset.

## What is and is not established

**Established descriptively on these saved calibration records:** fixed-guard TVD flags controls frequently at low shots; permutation testing suppresses many of these flags but lowers subtle sensitivity; the Bonferroni rule abstains broadly; the G and gated-G calculations produce the rates above; and the simple observed-effect gate removes both control and subtle assertions at 4096.

**Not established:** independent generalization to new circuit families; valid family-level confidence intervals; finite-sample type-I error for asymptotic G; a statistically calibrated effect margin; calibrated anomaly confidence; a no-fault/equivalence diagnosis; localization performance; causal decomposition of Bonferroni conservatism versus the chosen region; or a novel method outperforming prior selective testing methods.

## Recommendation

**No-go. Do not launch a new simulation using these candidates as confirmatory methods.** The most plausible exploratory path is a conditional randomization test combined with a prospectively justified equivalence margin and a separately specified abstention boundary. But calibration data do not provide adequate subtle sensitivity at the current shot counts under the effect-gated rule, and the existing seeded templates do not establish the independent-family count needed for generalization. Before any new holdout, define independent family units, resolve whether a missed fault is an abstention or a negative diagnosis, justify the practical margin without reusing evaluation outcomes, and preregister a family-level power design. If that cannot satisfy the registered floors at feasible cost, revise the paper question toward a bounded feasibility/negative result rather than run the 9,000/13,500 configuration plan.

