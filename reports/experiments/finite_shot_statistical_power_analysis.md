# Finite-shot selective-diagnosis power and sample-size analysis

**Date:** 2026-10-14  
**Purpose:** decide whether the proposed 3,600-configuration study can support its registered claims. Calculations use exact one-sided 95% Clopper–Pearson limits and binomial power under stated independent-unit assumptions. No simulation, training, new candidate scoring, or hardware run was performed.

## Data provenance and independent-unit problem

The actual registry is `tests/fixtures/finite_shot_distribution_calibration.json`; the saved run is `reports/experiments/finite_shot_distribution_calibration_run/`. It contains 120 templates per split, five scenario strata, three shot counts, and 3,600 configuration outputs over calibration and untouched validation. The template recipe is one fixed four-qubit paired-CNOT topology with phase-tag variation. The 120 seeds create distinct parameterized templates within that recipe, but not 120 independent circuit families. Each template repeats across conditions and shot counts. The existing reports' per-cell CP intervals assume independent Bernoulli outcomes; they are descriptive for this fixed recipe and do not quantify across-family generalization.

The earlier 3,600-configuration proposal uses 240 calibration-family templates? Its 600-template proposed q95 design repeats each template over 5 scenarios × 3 shots; the more recent revised 3,600 plan proposed 240 templates grouped into 20 families. Both designs have repeated conditions within template and family. Neither design can treat all configuration rows as independent units for population-wide claims. The protocol below avoids that by one held-out template per independently sampled circuit family. Repeated scenarios and shots remain paired within family and are handled as a cluster in cross-condition comparisons.

## Exact registered acceptance rule

For sensitivity with (x) detections among (n_+) positives, the one-sided exact lower bound is
\[
L(x,n_+)=\mathrm{Beta}^{-1}(0.05;x,n_+-x+1),
\]
with (L(0,n)=0). A sensitivity cell passes its registered floor (s_0) iff (L(x,n_+)\ge s_0).

For (y) false positives among (n_-) controls, the one-sided exact upper bound is
\[
U(y,n_-)=\mathrm{Beta}^{-1}(0.95;y+1,n_--y),
\]
with (U(n,n)=1). The FPR cell passes iff (U(y,n_-)\le0.05).

The project registry's sensitivity floors remain: subtle [.50, .80, .90], moderate [.80, .90, .95], large [.95, .95, .95] at [256, 1024, 4096] shots. Each of two control scenarios at each of three shot counts has the same maximum FPR .05. These 15 cell criteria are not changed here. For any proposed sample size, the accepted integer count must be recalculated from the equations; do not reuse the preregistration's n=600 integer limits at another n.

### Bare attainability is not useful power

With all positive cases detected, $L(n,n)=0.05^{1/n}$. Therefore the smallest $n$ at which each sensitivity floor can be met even with perfect observations is:

| Sensitivity floor | Minimum independent positive units, all successes |
|---:|---:|
| .50 | 5 |
| .80 | 14 |
| .90 | 29 |
| .95 | 59 |

With zero false positives, $U(0,n)=1-0.05^{1/n}$; the smallest $n$ making a ≤.05 upper bound attainable is **59 independent negative units with zero false positives**. At n=59, a single false positive fails.

These are attainability limits only. They do not mean the criteria have acceptable power. At n=120, exact minimum detections are 70/120, 104/120, 114/120, and 119/120 for floors .50/.80/.90/.95; at most 1/120 false positives passes the .05 upper-bound rule, matching the original registry.

## Exact-binomial power calculations

For a registered floor $s_0$, let $k_s(n)$ be the smallest detection count with $L(k_s,n)\ge s_0$. Under true sensitivity $s_1$, per-cell power is $P[\mathrm{Binom}(n,s_1)\ge k_s(n)]$. For the FPR test let $k_f(n)$ be the largest false-positive count with $U(k_f,n)\le.05$; at true FPR $f_1$, power is $P[\mathrm{Binom}(n,f_1)\le k_f(n)]$. Numerical values below were computed with SciPy's beta quantiles and binomial CDF/survival function, equivalent to the exact equations above.

### Registered sensitivity floors: n and example power

| n per cell | Minimum detections for floors [.50, .80, .90, .95] | Power at true rates [.65, .90, .95, .98] against those floors |
|---:|---|---|
| 120 | [70, 104, 114, 119] | [.947, .910, .606, .305] |
| 240 | [134, 203, 224, 234] | approximately [>.99, .997, .904, .793] |
| 400 | [217, 334, 371, 388] | [>.99, >.99, .981, .938] |
| 900 | [476, 741, 826, 866] | [>.99, >.99, >.99, >.99] |

The alternative rates are examples, not predictions of the new candidate. In particular, true sensitivity .98 is only a three-point margin above the .95 floor; n=240 has about .793 power, n=400 about .938, and n=900 exceeds .99. If true sensitivity is only .55 against the .50 floor, n=400 has about .638 power and n=900 about .904. Data in the old calibration cannot establish new-family operating rates.

### FPR power, the limiting registered gate

| Independent controls n | Maximum accepted false positives under exact upper bound | Power if true FPR=.01 | if .025 | if .03 |
|---:|---:|---:|---:|---:|
| 120 | 1 | .662 | .195 | .086 (approx.) |
| 240 | 6 | .989 | .606 | .391 (approx.) |
| 400 | 12 | >.99 | .794 | .576 |
| 420 | 13 | >.99 | .828 | about .63 |
| 600 | 20 | >.99 | .920 | .83 (approx.) |
| 800 | 29 | >.99 | .980 | .93 (approx.) |
| 900 | 34 | >.99 | .992 | .96 (approx.) |

The project requires upper FPR ≤.05, not a true FPR of .025; .025 is an illustrative operating alternative to show power. At the current n=120 per control×shot cell, an actually .025-FPR method passes the exact bound only about 19.5% of the time. At n=240 it is about 60.6%; at n=400 about 79.4%. Thus the proposed 240-template/3,600-configuration study is underpowered for an 80% per-cell demonstration of .025 FPR, even before accounting for family dependence. It would still be useful for estimation, but not a strong confirmatory all-cell pass claim.

## Paired method comparisons

All detectors use the same count pairs, so comparisons should be paired. For binary decisions define $p_{10}=P(A=1,B=0)$, $p_{01}=P(A=0,B=1)$, paired difference $\Delta=p_{10}-p_{01}$, and discordance $q=p_{10}+p_{01}$. A rough large-sample standard error is
\[
SE(\hat\Delta)\approx\sqrt{(q-\Delta^2)/n}.
\]
For one two-sided .05 comparison and 80% power, the normal approximation is
\[
n\approx\frac{(1.96+0.84)^2(q-\Delta^2)}{\Delta^2}.
\]
For a 10-point paired improvement and q=.15, this is about 110 independent paired units. This is only an illustration: candidate-vs-baseline discordance is unknown because the interval candidate has not been scored. The q95 candidate's previous validation outcomes are exploratory and may not be used to claim independent paired power. For the actual study, estimate q only from calibration records, use exact/clustered paired procedures, and apply the preregistered multiplicity correction to comparisons.

Repeated shot counts and scenarios from one template are paired and correlated. Resample whole independent families for paired bootstrap intervals. Do not apply unclustered McNemar or CP to repeated records as though each row were independent.

## Clustering, multiplicity, abstention, and coverage

### Circuit-family dependence

The prior proposed 20-family × 12-template design has only 20 top-level clusters. Its nominal 240 cases per cell would have effective sample size approximately (n_{eff}=n/[1+(m-1)\rho]) under a simple equal-cluster design effect, where m=12 and intraclass correlation (ICC) (\rho) is unknown: 155 at ICC=.05, 109 at .10, and 71 at .20. This approximation is not a replacement for a cluster-binomial model, but shows why 240 rows can carry much less information. Worse, the existing calibration contains one topology recipe, so it cannot estimate between-family ICC. Do not insert an assumed .05 ICC and describe it as measured.

For the primary exact-binomial acceptance analysis, use **one independently generated template from each new circuit family**, so every cell has one binary observation per independent family. Pair its conditions across scenarios and shot levels, and use family-cluster paired inference across endpoints. If multiple templates per family are necessary, use a preregistered beta-binomial/GEE or family-cluster bootstrap analysis as primary and do not claim ordinary CP intervals are valid for the population rate.

### 15 simultaneous acceptance cells

The global acceptance claim requires all nine sensitivity cells and all six control FPR cells to pass. This is an intersection-union test: the null is that at least one required cell fails, and the global claim is accepted only if every cell passes its one-sided .05 criterion. No Bonferroni correction is required to control the type-I error of that **conjunction**. However:

* Individual per-cell claims advertised as a family of discoveries need Holm/Bonferroni or clear descriptive labeling.
* Paired superiority claims against two baselines and multiple strata require hierarchical testing or multiplicity adjustment.
* An intersection-union gate can have low probability that all targets pass even when each target separately has .80 power. Under independent endpoints, .80^15 is only 3.5%. If one conservatively seeks ≥.80 joint power by Bonferroni allocation of failure probability, each of 15 targets needs power ≥ (1-.20/15=.9867). Correlations may change this; calculate joint power using a preregistered family-level simulation based on calibration-only paired dependence, with sensitivity analyses over unmeasured family ICC.

At n=900 independent units per cell and true FPR=.025, the exact FPR-cell power is about .992; if the six control cells were independent, all six would pass with probability about .953. At n=400, per-cell power is about .794 and six-cell joint power under independence is only about .25. Actual cell dependence is expected because conditions share templates/families, so these products are illustrations; the joint distribution must be estimated or simulated under a frozen design.

### Abstention and selective risk

Every configuration stays in the denominator. For anomaly sensitivity, abstention is a non-detection. It is not an FPR event, but report the abstention numerator/rate and exact two-sided interval. Define coverage as fraction receiving a non-abstaining state. Define selective risk as errors among non-abstained divided by covered cases and report its interval; do not let low coverage make conditional accuracy look good. Also report unconditional joint correct diagnosis per all eligible cases, which counts abstentions as not successful. The old shared `abstained` output is not an interval-based detector-specific abstention and cannot estimate this candidate's coverage or selective risk.

## Minimum defensible evaluation size and resource estimate

The exact CP FPR gate with 80% power at true FPR=.025 needs approximately 420 independent negatives for an individual cell (13 false positives are allowed; exact binomial power is about .828). Sensitivity can require comparable or larger n when the true rate is only slightly above its floor; at true .98 against floor .95, approximately 400 independent positives gives .938 power. To make the **all-15-target** claim plausible, allow up to roughly 900 independent families per cell: FPR=.025 then has .992 power per control cell; sensitivity alternatives .65/.90/.95/.98 against floors .50/.80/.90/.95 each have power above .99. This is conditional on these alternatives being scientifically plausible and must be checked with a joint power analysis. It is not a guarantee.

Recommended confirmatory design if candidate selection/preflight succeeds:

* **900 new independent circuit families**, one held-out template and one seed per family; no family/template/reference/seed/source skeleton overlap with any prior phase, calibration, validation, or ML training data.
* Pair each family across 5 registered strata (two controls + subtle/moderate/large) and 3 registered shots (256/1,024/4,096): 15 conditions per family.
* Total: 13,500 configurations; 27,000 sampled circuit executions; 48,384,000 shot-circuit executions; 13,486,500 permutation resamples for one 999-resample baseline.
* At n=900, exact CP pass counts are at least [476, 741, 826, 866] detections for sensitivity floors [.50,.80,.90,.95], and at most 34 false positives for an upper FPR limit .05. Recompute via the locked exact implementation and report these counts before run.

This is larger than the revised 3,600-case idea and the old 9,000-case q95 plan. It is the cost of independent family units and useful power for all registered cells. Do not replace families with repeated templates from a few families to keep nominal n high. If 900 genuinely distinct families are not feasible, narrow the confirmatory claim to fewer predeclared shot/strength endpoints without changing the historical registry, or report a pilot/estimation study that cannot make the all-target claim.

### Compute, memory, runtime

Each configuration compares two sampled circuits at one shot count. Total shot executions are
\[
2\times900\times5\times(256+1024+4096)=48{,}384{,}000.
\]

The completed run took 3,192.3 seconds for 3,600 configurations. Linear extrapolation to 13,500 is 11,971 seconds (about 199.5 minutes / 3.3 hours). Due to the unresolved Windows `RuntimeError: can't allocate lock`, OneDrive manifest contention, and host variance, budget **4–8 hours** after a separately authorized safe preflight. This is local simulation runtime, not hardware runtime. At approximately 2.27 kB per saved record, raw JSON is around 30.6 MB; reserve at least 75 MB for metadata/logs. Stream one record at a time; permutation active vectors are below 1 MiB, but reserve 1 GiB available RAM for Python/runtime and safety. Use local non-OneDrive staging, one process/manifest writer, create-only per-output commits, tested resume, and checksums.

## Unresolved limitations and recommendation

1. The existing calibration recipe cannot estimate family-level ICC or generalization; any ICC scenario in power planning is assumption, not measurement.
2. The new candidate's interval width, coverage, abstention, and sensitivity are untested; existing calibration summaries do not report these outcomes. Build and mathematically verify the confidence-region optimizer before asking for any held-out run.
3. A cause/localization claim needs independent target truth and a faithful comparison to quantum fault-dictionary/dynamic diagnosis and EASE 2025 mutation-based localization. Current finite-shot records do not support that claim.
4. The 9,000-configuration q95 study is not justified. A 3,600-configuration study with 240 templates/20 families is not enough for strong all-cell family-generalization claims: cellwise FPR power is only ~.61 at true .025 even before clustering, and only 20 clusters cannot stabilize cluster uncertainty.
5. A 13,500-configuration/900-independent-family study is statistically more defensible under the explicit alternative-rate assumptions, but high cost and family construction burden remain. Before execution, require calibration-only proof that the candidate is viable, a cluster-aware joint-power analysis, sufficient labeled families, and a preregistration frozen before any test results are accessed.

**Recommendation: NO-GO for the proposed 3,600-configuration confirmatory study and the old 9,000-case q95 study. Conditional GO only for the 900-independent-family design after the method and power assumptions survive calibration-only development and focused preflight. If 900 independent families are unavailable, narrow the estimand/claims rather than pretending repeated templates are independent.**

