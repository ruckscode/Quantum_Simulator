# Revised proposal: selective diagnosis under finite-shot uncertainty

**Date:** 2026-10-14  
**Status:** Proposed only; no candidate has been fitted to a new split and no simulation is authorized or performed. This replaces the 9,000-configuration q95 confirmatory proposal for future planning; it does not replace or amend the earlier preregistration or its results.

## Decision and objective

Do not execute the existing 9,000-configuration q95 TVD run. The rule has exploratory validation subtle sensitivity of 10.8% at 256 shots and 34.2% at 1,024 shots, despite lower control flags. The calibration q95 tail is based on 120 controls per cell. Those facts do not justify its confirmatory compute cost.

Develop one statistically defined candidate plus two diagnostic ablations:

1. **Primary candidate — effect-size-aware selective test:** estimate finite-shot uncertainty for the population distance between reference and observed outcome distributions, accounting for both samples. Freeze a meaningful TVD margin `delta_min` using calibration data and domain rationale. Return `ANOMALY` only if a one-sided lower confidence bound for TVD exceeds `delta_min`; return `NO_MEANINGFUL_SHIFT` only if an upper bound is below it; otherwise `ABSTAIN`. The interval procedure must be calibrated under multinomial sampling, including sparse cells, and validate coverage by simulation/preflight separate from the scientific held-out evaluation.
2. **Ablation 1 — null-significance only:** unchanged pooled-label permutation at alpha=.05, preserving 999 resamples initially for exact baseline reproducibility. If a new candidate also uses permutation, prespecify a larger repetition count or Monte Carlo error tolerance before scoring.
3. **Ablation 2 — magnitude threshold without uncertainty:** calibration-frozen TVD cutoff, with the already evaluated q95 rule retained as an explicitly exploratory comparator. Keep production fixed-guard TVD unchanged as the historical baseline.

Then apply an evidence gate: cause and target claims require independent operation/circuit or hardware calibration evidence. If evidence is absent, contradictory, or multiple causes remain compatible, abstain at cause/localization level even if an output anomaly is detected. This gate is evaluated separately from the statistical anomaly decision.

## Calibration protocol (no use of old validation for selection)

1. Treat all 3,600 existing calibration and validation records as previously observed; prior validation must never be called a fresh test. It may be reported only as exploratory.
2. For candidate selection, use only the original `calibration` records. Keep each template and all of its scenarios/shot levels together. Use grouped cross-fitting by template/family within calibration to estimate interval coverage, operating characteristics, and parameter stability; do not split individual scenario rows across folds.
3. Freeze the interval algorithm, confidence level, `delta_min`, anomaly alpha, abstention rule, shot policy, cause-evidence hierarchy, and any calibration map. Select `delta_min` by a declared scientific effect-size interpretation and calibration-only tradeoff; do not tune to best validation accuracy. If no defensible meaningful-effect margin can be justified, stop the candidate rather than selecting a purely empirical optimum.
4. Reference-distribution leakage: all observations sharing a reference template remain in one grouped fold. Any reference distribution used to tune a threshold must be generated from calibration families only; held-out reference fingerprints are generated independently and never used for tuning. Record the reference generator, finite reference shots, and sampling seed. If reference probabilities are exact simulator probabilities rather than estimates, state this different uncertainty regime explicitly.
5. Freeze the complete protocol and code/config hashes in a new preregistration before creating/opening any new held-out results. A disjointness script must check template family IDs, template hashes, reference fingerprints, source/circuit skeleton IDs, all seeds, and intervention recipes against calibration, old validation, Phases 1/2, and any ML training set. Any overlap invalidates the affected family before outcome inspection.
6. Existing calibration data can identify failure modes but is small (120 records per scenario/shot cell); uncertainty estimates selected there may be unstable. Report grouped cross-fit variation and do not treat calibration cross-fit as confirmation.

## Smallest proposed held-out evaluation

Use **20 genuinely new circuit families × 12 independent templates/seeds per family = 240 templates**. For every template, make paired records for both controls plus subtle, moderate, and large fault strata at all three shot counts (256, 1,024, 4,096). This yields:

| Quantity | Planned value |
|---|---:|
| Templates | 240 |
| Circuit families | 20 |
| Conditions per template | 5 strata × 3 shot counts = 15 |
| Configurations | 3,600 |
| Two sampled count vectors per configuration | 7,200 circuit-sample executions |
| Shot-circuit executions | 2 × 240 × 5 × (256 + 1,024 + 4,096) = 12,902,400 |
| Permutation work for baseline | 3,596,400 resamples at 999/configuration |

The 20-family design keeps template pairing for method comparisons and supplies more independent family clusters than the 9,000-case plan’s 600 templates without increasing the number of configurations beyond the already completed calibration. A rough paired-power rationale is to target a minimum meaningful 10 percentage-point change in paired detection/correct-diagnosis rate with discordant method outcomes around 15%; under independent pairs this is on the order of 110 pairs for 80% power at two-sided 5%. With 12 templates per family and an assumed within-family ICC of .05, design effect is approximately 1.55 and effective n is about 155 per stratum/shot cell. These are planning assumptions, not a guarantee: estimate discordance/ICC from grouped calibration-only data, conduct a family-cluster-aware power simulation, and increase family count or declare the study underpowered if the assumptions fail. With only 20 family clusters, cluster-level intervals will still be coarse.

All detectors score identical sampled count pairs. The candidate, baselines, and all criteria are frozen before held-out outputs are read. A smaller design may be adopted only if calibration-only power analysis shows it can resolve the preregistered minimum effect with adequate family-cluster robustness; otherwise 3,600 configurations is a minimum practical proposal, not a target to cut automatically.

## Endpoints, uncertainty, and paired comparisons

### Primary endpoint

**Wrong confident joint diagnosis risk at matched coverage:** among all positive and negative cases, count an error when the system confidently asserts an incorrect anomaly state, cause, or target. Report abstentions separately; compare methods at common coverage levels using frozen confidence/abstention rules. Primary superiority requires lower wrong-confident risk than both unchanged TVD and unchanged permutation at the same coverage within ±2 percentage points, while satisfying absolute control and sensitivity gates.

This endpoint is only valid if each positive has independently adjudicated cause and target truth. If the planned dataset contains only injected scenario labels and no independent target truth, limit the primary claim to anomaly detection/selective anomaly detection and remove cause/localization superiority from primary claims.

### Required metrics

* By shot count, control scenario, fault strength, and family: sensitivity/recall, FPR, specificity, precision at the designed prevalence (clearly descriptive), abstention rate, coverage, selective risk, cause accuracy, top-1/top-k localization, and joint exact diagnosis.
* Exact one-sided 95% Clopper–Pearson lower limits for sensitivity and upper limits for FPR, with two-sided intervals for descriptive cell rates. Since template/family outcomes are clustered, also give paired family-cluster bootstrap intervals (resample families, then templates within family) and per-family distributions. Do not treat the three shot levels from one template as independent.
* For paired detector deltas, use identical configurations and a preregistered family-cluster paired bootstrap interval for sensitivity, FPR, coverage and selective-risk differences. Include discordant counts; McNemar tests can be supplementary but do not account for within-family clusters on their own.
* Confidence-bound coverage for the proposed TVD magnitude interval should be assessed in an independent null/known-shift coverage audit. It is not enough that the method reduces FPR on the evaluation set.
* Runtime/cost: sampling executions and shots, detector CPU time, wall-clock by stage, peak memory, serialized output size, and cost per supported correct diagnosis. Do not combine local simulation wall-clock with projected quantum-hardware cost.

### Proposed preregistered gates (owner approval required)

These are new proposed criteria, not changes to the completed calibration registry:

* Control FPR one-sided 95% upper bound ≤ .05 for each control × shot cell.
* Sensitivity one-sided 95% lower bound ≥ .50 subtle/256; ≥ .80 subtle/1,024; ≥ .90 subtle/4,096; ≥ .80 moderate/256; ≥ .90 moderate/1,024; ≥ .95 moderate/4,096; and ≥ .95 large at all shot levels. These mirror the previously registered floors for comparability, without rewriting the historical results.
* Coverage ≥ .70 overall and no less than .50 in any prespecified ambiguity stratum. Wrong confident cause/location rate ≤ .10 among non-abstained cases if valid target labels exist.
* Primary candidate must improve wrong-confident risk by at least 10 percentage points over each baseline at matched coverage, with paired family-cluster interval excluding zero; otherwise no primary novelty claim. It must not lose more than 5 percentage points in subtle sensitivity versus permutation at each shot level. Adjust for the declared family of primary tests or use a single hierarchical primary test.

## Resource estimate and execution preconditions

3,600 configurations require 12.90 million shot-circuit executions, the same sampled execution count as the completed 3,600-case calibration (the 9,000 plan required 32.256 million). At observed calibration runtime 3,192.3 seconds for 3,600 configurations, a direct linear estimate is about 53 minutes; use a broad **1–3 hour** planning window until a separately authorized pilot establishes throughput. This is a local simulator estimate, not hardware execution time.

If saved output size scales from the prior 9,000-plan estimate of 20.4 MB, 3,600 records are roughly 8.2 MB; reserve at least 25 MB for metadata, logs, and safety margin. Two 16-bin vectors of 64-bit counts need only about 256 bytes per record (about 0.9 MB total), excluding JSON overhead. Sequential permutation working memory is small; reserve at least 1 GiB available RAM because the previous Windows lock-allocation failure has no captured traceback/system telemetry and cannot be attributed confidently to RAM alone.

Before any later run: test safe resume and create-only per-configuration outputs; stage outside OneDrive; use one manifest writer and one worker initially; confirm process/lock behavior and disk space; verify a small preflight with focused tests; checksum every output; never replace valid records. The old run’s sequential runner and manifest retry logic reduce some risks but do not explain the earlier `can't allocate lock` error. Runtime cannot be a firm estimate before a safe, authorized preflight.

## Baselines and literature checks before preregistration

Required unchanged baselines: current fixed-guard TVD and current pooled-label permutation. Also compare the q95 rule only as the already explored, calibration-selected ablation. Implement a Monte Carlo multinomial goodness-of-fit or confidence-set baseline only after specifying whether the expected distribution is exact or sampled; it must model both sample uncertainties. For diagnosis/localization, compare the closest feasible fault-dictionary/dynamic-localization method from Li et al. (VTS 2023) and mutation-based source localization from Ishimoto et al. (EASE 2025, DOI 10.1145/3756681.3757022) only on compatible truth and source mappings. If an implementation is not faithful/comparable, document that instead of claiming direct superiority.

The literature distinction still requires systematic search and full-paper review: selective prediction/abstention, equivalence testing, multinomial two-sample confidence regions, quantum software diagnosis, and probabilistic/mutation localization. Current work establishes that statistical tests and quantum localization are not novel categories; it does not prove that the proposed staged, effect-size-aware selective contract is new.

## Go / no-go

**No-go** for the existing q95 9,000-configuration study and for claiming the current fixed-confidence abstention as evidence-calibrated. **Conditional go** for calibration-only development of the effect-size-aware interval plus independent evidence gate. Proceed to a new 3,600-configuration held-out proposal only if grouped calibration analysis supports a defensible effect margin and cluster-aware power, the target labels support the intended claim, candidate interval coverage passes its own preflight, the source/seed/reference split is machine-verified disjoint, the literature review finds a meaningful distinction, and every criterion is frozen before held-out output is inspected. If any gate fails, do not run; narrow to an anomaly-detection study or stop.

