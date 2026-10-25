# Joint power analysis for finite-shot acceptance criteria

**Date:** 2026-10-14  
**Status:** Reproducible analytical calculations; no quantum simulations or training.

## Questions and assumptions

The proposed acceptance claim requires all 9 sensitivity cells (three strengths × three shot levels) and all 6 negative-control FPR cells (two controls × three shot levels) to pass one-sided exact 95% Clopper–Pearson bounds. The revised design additionally requires minimum useful coverage. The unit is a **new independent circuit family**, with one held-out template per family; each template is repeated across strata and shot levels, so outcomes across the 15 cells within a family are paired and dependent.

The 900-family design's per-cell binomial calculations are valid only if the families are independently sampled from the declared target population and each family contributes one observation to that cell. If each family instead contributes multiple templates, ordinary CP limits on template rows are invalid for across-family inference. The current saved calibration has one topology recipe and cannot estimate family-level ICC or the joint distribution of cell outcomes.

All power inputs below are hypothetical operating alternatives, not candidate estimates:

| Cell | Registered floor | Illustrative true sensitivity |
|---|---:|---:|
| subtle, 256 / 1,024 / 4,096 | .50 / .80 / .90 | .65 / .90 / .95 |
| moderate, 256 / 1,024 / 4,096 | .80 / .90 / .95 | .90 / .95 / .98 |
| large, all shots | .95 | .98 / .98 / .98 |
| each negative control, each shot | FPR ≤.05 | true FPR=.025 |

These alternatives require owner agreement. In particular, the .025 true FPR is chosen to quantify power below the .05 limit; the project does not establish that the candidate will achieve .025. Any actual study must present sensitivity analyses across more pessimistic alternatives, including sensitivity close to the registered floor and FPR near .04.

## Exact marginal power equations

For alpha=.05, define the exact bounds

\[
L(x,n)=Q_{.05}(\mathrm{Beta}(x,n-x+1)),\quad U(y,n)=Q_{.95}(\mathrm{Beta}(y+1,n-y)).
\]

The success threshold for sensitivity floor s is (k_s(n)=\min\{x:L(x,n)\ge s\}); its power at true sensitivity p is (\pi_s(n,p)=\Pr[\mathrm{Binom}(n,p)\ge k_s(n)]). The maximum acceptable false-positive count is (k_f(n)=\max\{y:U(y,n)\le.05\}); FPR-gate power at true rate r is (\pi_f(n,r)=\Pr[\mathrm{Binom}(n,r)\le k_f(n)]).

The following compact SciPy pseudocode reproduces the table (SciPy `beta.ppf`, `binom.sf`, and `binom.cdf`):

```python
alpha = 0.05
L = lambda x, n: 0.0 if x == 0 else beta.ppf(alpha, x, n-x+1)
U = lambda y, n: 1.0 if y == n else beta.ppf(1-alpha, y+1, n-y)
k_s = min(x for x in range(n+1) if L(x, n) >= floor)
k_f = max(y for y in range(n+1) if U(y, n) <= 0.05)
sensitivity_power = binom.sf(k_s-1, n, true_sensitivity)
fpr_power = binom.cdf(k_f, n, true_fpr)
```

At zero false positives, (U(0,n)=1-.05^{1/n}). This first falls to .05 or below at n=59; that requires zero errors and is only attainability. With n=120, kf=1; with n=240, kf=6; with n=400, kf=12; with n=900, kf=34.

## Marginal powers for all 15 registered gates

The exact pass counts and power under the illustrative alternatives are:

| n independent units per cell | Sensitivity minimum counts for [.50,.80,.90,.95] | Power for respective alternatives [.65,.90,.95,.98] | FPR max count | FPR power at .025 |
|---:|---|---|---:|---:|
| 120 | [70,104,114,119] | [.9467,.9101,.6063,.3054] | 1 | .1954 |
| 240 | [134,203,224,234] | [.9986,.9969,.9043,.7925] | 6 | .6063 |
| 400 | [217,334,371,388] | [>.999,.999,.9810,.9381] | 12 | .7940 |
| 900 | [476,741,826,866] | [>.999,>.999,>.999,>.9998] | 34 | .9920 |

Scanning integer n from 1 upward, the first n at which the exact FPR gate reaches at least 80% power at true FPR .025 is **361** (k_f=11; power .8030). Because k_f changes in integer steps, power is not monotone for every increment of n: n=400 permits 12 errors and has .794 power, whereas n=420 permits 13 and has .828. Use a stable planned n and assess nearby discrete boundaries rather than reporting “about 420” as a continuous exact minimum.

The mapping from floors to true rates follows the table above; .65 is used for the .50 floor, .90 for .80, .95 for .90, and .98 for .95. These are per-cell powers. They do not establish probability of passing all cells.

## Joint pass probability and dependence bounds

Let (A_i) be the event that primary cell i passes, i=1,…,15, with marginal pass probability (p_i). Without a joint dependence model, the sharp general Frechet bounds are

\[
\max\left(0,1-\sum_{i=1}^{15}(1-p_i)\right)\le \Pr(\cap_i A_i)\le\min_i p_i.
\]

If one assumes all cell pass events independent, the joint probability is (\prod_i p_i). Independence is not supported: all conditions share circuit families, references, and often fault-template structure. These estimates are scenarios, not identified joint probabilities.

| n per cell | Independent-endpoint probability all 15 CP gates pass | General Frechet lower | General Frechet upper | Probability all 6 FPR gates pass if independent |
|---:|---:|---:|---:|---:|
| 120 | effectively 0 | 0 | .1954 | .000055 |
| 240 | .0159 | 0 | .6063 | .0496 |
| 400 | .1867 | 0 | .7940 | .251 |
| 900 | .9523 | .9513 | .9920 | .953 |

The n=120 all-six-FPR product is (.1954^6\approx5.5\times10^{-5}), already showing why “80% power per cell” is not an 80% all-cell design. At n=400, independent all-FPR pass probability is about .251 despite .794 per-cell power. At n=900 it is about .953 for the six FPR gates. For all 15 gates, n=900 gives independent joint power .952 and Frechet bounds [.951,.992] under these particular alternatives. This is a favorable-looking but conditional result; it depends on rates materially above every floor.

If true subtle/256 sensitivity were only .55 (five points above the .50 floor), its n=900 marginal power is about .904, and joint all-target power could be much lower. At n=400 that marginal power is only about .638. No existing result shows that the proposed confidence-region method will achieve the assumed .65/.90/.95/.98 vector; on the saved calibration data, the specific Bonferroni interval candidate abstains on all subtle cases.

The all-15 conjunction is an intersection-union acceptance rule. Component alpha=.05 does not need multiplicity correction to protect type-I error of the global conjunction, because every criterion must pass. But if the paper separately claims cell-wise discoveries, use a multiplicity-adjusted family or label intervals descriptive. Power is different: to guarantee ≥.80 joint power by a simple union-bound allocation across 15 criteria, target per-criterion power ≥ (1-.20/15=.9867). This is conservative and often impractical. At n=900 the FPR power at true .025 is .992, but floor-adjacent sensitivity may fail the .9867 requirement. A preregistered family-level joint-power simulation should replace independence only after a defensible joint model and alternatives are frozen.

### Coverage gate

The revised proposal asks for overall coverage ≥.70 and coverage ≥.50 in each ambiguity stratum. It did not define “ambiguity stratum” precisely or establish whether the threshold is a point estimate or a lower confidence bound. The calibration evaluation of the candidate produced 0/600 covered cases at 256 shots and 120/600 at each of 1,024 and 4,096; pooled across shots, coverage is 240/1,800=.133. Therefore the candidate fails even a .70 point threshold on calibration.

For a future candidate with true overall coverage .80, if coverage is tested by an independent family-level mean and the criterion is merely the observed point estimate ≥.70, Hoeffding's inequality gives a failure probability no greater than (\exp[-2N(.10)^2]): at N=900 this is (e^{-18}\approx1.5\times10^{-8}). This bound treats each family’s coverage proportion (between 0 and 1) as one independent observation and is conservative. A 95% lower-bound gate is a different criterion and requires its exact family-level method and power to be frozen. Ambiguity-specific coverage requires a prespecified, label-based ambiguity definition and enough families in that stratum; it cannot be inferred from the current shared `abstained` field.

Coverage does not multiply as an independent 16th event unless formally defined as a separate acceptance gate. If one assumes a coverage-gate pass probability .80 at n=900 and independence from the 15 CP gates, joint pass probability becomes .9523×.80=.762. Frechet bounds using the CP-gate bounds and .80 coverage probability are [.751,.80]. The correlation is unknown, so this is only a sensitivity illustration. Given candidate calibration coverage .133, the current candidate cannot use such an assumed .80.

## Paired comparisons and family dependence

The candidate, TVD and permutation detectors must be scored on identical count pairs. For paired binary detections let discordance q be the chance that methods disagree and difference Δ be the candidate-minus-baseline success probability. A normal approximation gives

\[
n\approx\frac{(1.96+.84)^2(q-\Delta^2)}{\Delta^2}
\]

for 80% power in a single two-sided .05 test. If q=.15 and Δ=.10, n≈110 independent pairs. This excludes clustering and multiplicity. The candidate was not scored before this analysis, so its q is unknown; use calibration-only paired discordances for planning, never the inspected validation to assert future power. Resample circuit families for paired bootstrap and use hierarchical/Holm control for two baseline comparisons and repeated endpoint claims.

Under a simple equal cluster-size approximation, the 20-family ×12-template proposal has design effect (1+(12-1)\rho). For 240 nominal templates per cell, effective n is about 155, 109, or 71 at ICC .05/.10/.20. The ICC is not estimable from the one-topology calibration. Its 20 families also give few independent clusters for bootstrap tail estimation. The old 9,000 plan has 600 new seeds/templates under a fixed recipe, so it has little or no independent family diversity; n=600 conditional draws do not fix family-level generalization.

## Design cost comparison

Every template contributes 5 strata × 3 shots. Each configuration has two sampled count vectors, so shot executions equal (2N_{templates}\times5\times(256+1024+4096)).

| Plan | Templates / independent families | Configurations | Shot executions | Linear runtime from 3,600 in 53.2 min | Generalization interpretation |
|---|---|---:|---:|---:|---|
| Revised small plan | 240 templates / 20 families ×12 | 3,600 | 12.9024 M | 53 min | Pilot; clustering and only 20 families prevent strong generalization claim. |
| Prior q95 plan | 600 templates / one fixed topology recipe | 9,000 | 32.256 M | 133 min | New seeds but same recipe; conditional replication, not unseen-family generalization. |
| Family-level plan | 900 independent families ×1 template | 13,500 | 48.384 M | 199.5 min | Statistically interpretable per-cell CP and family-level comparison if family sampling is real and target population is defined. |

Practical planning ranges remain 1–3 h, 2.25–6 h, and 4–8 h, respectively, because the previous Windows lock-allocation error is unresolved and OneDrive manifest contention can distort throughput. Reserve ≥1 GiB RAM and use streaming, one writer, one process, local non-OneDrive staging and safe-resume preflight. The sequential permutation vector itself is <1 MiB; runtime/output retention dominate.

The 13,500 count is the smallest of these proposed designs with 900 independent family units per cell, but this analysis does **not** establish it as sufficient for the actual candidate: it fails calibration feasibility and the true operating rates and joint dependence are unknown. A 9,000 configuration design with 600 per cell could have .920 FPR-cell power at true .025, but only if those are 600 independent controls; current specification does not create 600 independent circuit families. Even under ideal independence, six FPR gates all pass with probability about .92^6=.606.

## Conclusion

1. The cited CP calculations are correct: n=59 zero-error controls is the first possible pass; n=120 and n=240 powers at true FPR .025 are 19.5% and 60.6%.
2. The proposed Bonferroni confidence-region candidate is not feasible as written on calibration: it abstains on all controls, subtle and moderate cases; overall coverage is 13.3%, and sensitivity floors fail in 7 of the 9 positive cells. Only large faults at 1,024 and 4,096 shots reach 120/120; all six FPR point-rate bounds pass, but only because controls are always abstained and the coverage gate fails.
3. The joint pass estimate for 900 families is conditional on strong alternative rates and independent family units. Unknown dependence, unknown candidate feasibility, and undefined ambiguity coverage prevent a firm joint-power assurance.
4. Do not run 3,600, 9,000, or 13,500 configurations now. Rework the candidate on calibration data only, define the coverage gate, identify truly independent families, and recalculate joint power from a defensible family-level model. If the method cannot meet registered sensitivity/FPR/coverage criteria on calibration without using validation, stop or narrow the claim.

