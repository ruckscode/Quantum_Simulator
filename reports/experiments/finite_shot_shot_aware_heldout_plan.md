# Preregistered held-out shot-aware detector evaluation

**Status:** designed, hashed, and not executed  
**Date:** 2026-10-14  
**Preregistration record:** `finite_shot_shot_aware_heldout_preregistration.json`

## Research question and hypothesis

The question is whether a shot-conditioned empirical-null cutoff for total variation distance (TVD) can address the current detector's low-shot false positives on negative controls while preserving useful sensitivity across fault strengths. The confirmatory hypothesis is that the candidate will meet all frozen, stratum-specific sensitivity and false-positive criteria on new templates and seeds, and will reduce low-shot control flags versus the unchanged fixed-guard TVD baseline. The existing pooled-label permutation detector is a second unchanged comparator.

The targeted failures are: (1) current TVD flags 100% of both controls at 256 shots and roughly 87-92% at 1,024 shots; (2) its false-positive behavior depends strongly on shot count; and (3) the permutation detector has low measured sensitivity for subtle faults at 256/1,024 shots. These findings come from saved simulation outputs, not hardware measurements.

## Candidate selected from calibration data only

For each shot count, calculate the higher empirical 95th percentile of saved TVD values separately for `identical_control` and `measurement_equivalent_control` in the **calibration split**. Freeze the larger of those two values. Flag an anomaly iff TVD is strictly greater than that cutoff. This yields:

| Shots | Identical-control q95 | Measurement-equivalent q95 | Frozen candidate cutoff |
|---:|---:|---:|---:|
| 256 | 0.18359375 | 0.17187500 | 0.18359375 |
| 1,024 | 0.09375000 | 0.09082031 | 0.09375000 |
| 4,096 | 0.04467773 | 0.04785156 | 0.04785156 |

The candidate is frozen before the new held-out results are opened. Do not re-estimate thresholds from the new split. The baselines remain unchanged: current TVD uses configured threshold 0.05 and the existing `max(0.05, 1/sqrt(shots))` guard; permutation uses the current pooled-label test with 999 resamples, add-one p-value correction, and alpha 0.05. All methods use the same expected/observed sampled count pairs for each configuration.

The q95 cutoffs are a plausible shot-aware null adjustment, not a guaranteed FPR control. Their tail estimates use only 120 calibration examples per control and are coarse. The follow-up's exploratory validation outcomes also show that the candidate is not expected to satisfy the subtle sensitivity floors at 256 and 1,024 shots. This evaluation therefore has a substantial chance of confirming failure. No production threshold or registered criterion is changed.

## New held-out split and leakage check

Create a single new split called `new_heldout_20261014`, using the existing frozen `fixed_pilot_topology_phase_tags` template recipe and all five registered scenarios. Generate 600 new reference templates; reuse each reference across the five interventions and all three shot counts. The proposed seed namespaces are:

- Template seeds: 60,271,014 through 60,271,613.
- Sampling seeds: derive from root 70,361,015 plus `10,000 * stratum_index + 10 * template_index + shot_index`, with zero-based indices, 5 strata, 600 templates, and shot indices 0-2.
- Template identities use `new_heldout_20261014:template:NNNN`.

Before writing this plan, the proposed 600 template identities, template seeds, all 9,000 sampling seeds, and generated reference fingerprints were compared with all **3,600 valid saved output records**. Those records contain 240 prior templates and 3,600 sampling seeds. The proposed split has **zero overlap** in identities, seeds, or reference fingerprints. The preregistration JSON records the registry and corrected-summary SHA-256 digests and a frozen-configuration digest.

## Metrics, uncertainty, and acceptance criteria

Primary metrics are sensitivity in each fault-strength x shot cell and false-positive rate (FPR) in each negative-control scenario x shot cell. Compute separate one-sided exact 95% Clopper-Pearson lower bounds for sensitivity and upper bounds for FPR. Report rates, numerators, denominators, bounds, and raw confusion counts. Do not pool across shot counts or scenarios.

At 600 configurations per cell, the one-sided exact criteria correspond to these minimum detection counts for 600 positives: 321/600 for 50%, 497/600 for 80%, 553/600 for 90%, and 580/600 for 95%. For each negative-control cell, at most 20/600 false positives yields a one-sided exact 95% upper bound no greater than 5%. The candidate passes only if every positive cell meets its registered sensitivity lower-bound minimum and every control cell has FPR upper bound at most 5%. The sensitivity floors are unchanged from the existing calibration registry:

| Fault strength | 256 shots | 1,024 shots | 4,096 shots |
|---|---:|---:|---:|
| Subtle | 50% | 80% | 90% |
| Moderate | 80% | 90% | 95% |
| Large | 95% | 95% | 95% |

The global candidate claim requires all component gates to pass; per-cell one-sided alpha is 0.05 under an intersection-union decision. Report both control scenarios separately. Do not adjust a cutoff or criterion based on held-out results. Score the same held-out configurations for all three detectors.

No configuration is removed because of abstention. An abstention counts as a non-detection (false negative) for sensitivity, is not counted as an anomaly flag for FPR, and is shown separately with its two-sided exact 95% Clopper-Pearson interval. Also report precision for the designed mix as descriptive only; the 60% positive prevalence does not represent deployment prevalence. Compare candidate-minus-baseline sensitivity and FPR within cells using paired template bootstrap percentile intervals (10,000 resamples, deterministic bootstrap seed 90,261,014), resampling the same template indices jointly across methods. These comparisons are secondary and do not replace the absolute acceptance gates.

## Workload and resource estimate

| Quantity | Estimate |
|---|---:|
| Configurations | 9,000 |
| Sampled shot-circuit executions | 32,256,000 |
| Permutation resamples | 8,991,000 |
| Saved result JSON | about 20.4 MB; reserve 50 MB with logs and metadata |
| Active permutation working set | under 1 MB per sequential comparison |
| Python memory | about 100-300 MB if all result objects are retained; reserve at least 1 GiB available RAM |
| Runtime | empirical linear projection about 133 minutes; plan for 130-360 minutes |

Arithmetic: `2 circuits * 600 templates * 5 strata * (256 + 1024 + 4096) shots`. The completed 3,600-configuration run took 3,192.3 seconds (53.2 minutes), giving about 133 minutes by linear scaling. The earlier pilot-based estimate for that completed workload was higher than actual; the wider 130-360 minute planning window accommodates host load, circuit depth, permutation processing, and Windows filesystem variability.

The prior `RuntimeError: can't allocate lock` has no captured traceback or system-resource telemetry in the saved artifacts. The inspected calibration runner is sequential and does not create worker locks, so the root cause remains unresolved. To reduce process-lock and file-contention risks, execute only as one process with one worker, from a local non-OneDrive NTFS staging directory such as `C:\Quantum_Simulator_runs\finite_shot_shot_aware_heldout_run`. Do not update the manifest from a OneDrive-synced folder during execution, do not run concurrent resumptions, and copy the finalized checksummed directory to a new `reports/experiments/` run folder only after completion. This mitigation does not establish the cause of the earlier error.

## Tests and execution boundary

Focused metadata tests verify workload arithmetic, deterministic ledger generation, unique configuration and sampling identifiers, disjointness from prior template identities/seeds/fingerprints, and the frozen candidate cutoffs and criteria. They do not launch a simulator or read/write calibration outputs. No held-out configuration has been simulated.

## Decision needed before execution

The design freezes the candidate and acceptance rules; the statistical definitions are not left open. The owner should decide whether the **9,000-configuration, roughly 2.25-6 hour evaluation is worthwhile as a confirmatory falsification study**, given the candidate's already observed validation sensitivity of only 13/120 subtle detections at 256 shots and 41/120 at 1,024 shots. If the goal is to improve performance rather than independently confirm the candidate's failure, pause and select a different candidate using calibration data only, then create a new preregistration and a new held-out split before any run. Also confirm that local non-OneDrive staging and 1 GiB available memory can be provided. Execution is not authorized by this design document.
