# Scientific audit: finite-shot synthetic characterization (2026-10-15)

## Scope and audited artifacts

This is a read-only audit of the runner, its focused tests, preregistration, saved results, and experiment report listed in the request. No detector, protocol, fixture, result, or acceptance criterion was changed. The experiment and any training job were not rerun.

## Verified design and implementation

The endpoint is a two-sample anomaly flag: draw a reference count vector and a submitted count vector, normalize each to an empirical distribution, compute their total variation distance (TVD), and flag iff

`empirical_TVD > max(0.05, 1 / sqrt(shots))`.

The comparison call passes `threshold=0.05` and the common shot count. Thus the effective cutoffs are 0.0625 at 256 shots and 0.05 at 1,024 and 4,096 shots. Equality to the cutoff is not flagged. Hellinger and Jensen-Shannon values are computed by the shared statistics routine but do not determine this detector decision.

The reference distribution is p=(0.50, 0.25, 0.15, 0.10). For the null, submitted q=p. For alternatives, mass delta is transferred from the first outcome to the second: delta is 0.025, 0.05, 0.10, or 0.20, giving q=(0.50-delta, 0.25+delta, 0.15, 0.10), whose TVD from p is delta. At each of three shot counts, the code independently samples reference and submitted multinomial counts. It uses 2,000 null replicates and 500 replicates for every alternative. The detector threshold is fixed; there are zero development draws. All registered draws form the evaluation set. There is no separate train/development/evaluation split in this synthetic experiment, and no threshold selection step.

Each count-pair is one reference multinomial vector plus one submitted vector for one condition and shot count. Planned work is 3 x 2,000 = 6,000 null pairs plus 3 x 4 x 500 = 6,000 alternative pairs: 12,000 total pairs / 24,000 count vectors.

## Null counts and independent confidence-bound check

The table separates observed false positives and their empirical rate from the preregistered one-sided 95% exact Clopper-Pearson upper confidence bound. I independently recomputed the bounds from (x,n) using the beta-quantile identity `BetaPPF(0.95, x+1, n-x)` (SciPy `beta.ppf`); values agree with the stored bounds within 5.2e-15.

| Shots | False positives / null pairs | Observed FPR | Stored upper 95% | Independent upper 95% | Gate: upper <= 0.06 |
|---:|---:|---:|---:|---:|---|
| 256 | 680 / 2,000 | 0.3400 (34.00%) | 0.35783917 | 0.35783917 | FAIL |
| 1,024 | 114 / 2,000 | 0.0570 (5.70%) | 0.06627158 | 0.06627158 | FAIL |
| 4,096 | 0 / 2,000 | 0.0000 (0.00%) | 0.00149674 | 0.00149674 | PASS |

The all-shot null gate therefore fails. Diagnostic one-sided 95% lower bounds (not registered acceptance quantities) are about 0.32251, 0.04871, and 0, respectively. In particular, the 256-shot result is not a borderline point estimate: the lower bound is far above 0.06. At 1,024 shots, the observed rate is below 0.06, but the interval extends above it; this run does not establish that the rate is at most 0.06 there.

## Alternative rates, intervals, and criteria

The exact two-sided 95% Clopper-Pearson intervals in the saved results were independently recalculated from the detection counts using beta quantiles; all agree to floating-point precision. No minimum alternative power or per-cell alternative acceptance criterion was preregistered. Consequently, none of these alternatives can be labeled pass or fail against a power criterion; they are descriptive estimates only.

| True TVD | Shots | Detections / 500 | Power | Exact two-sided 95% interval | Preregistered alternative verdict |
|---:|---:|---:|---:|---:|---|
| 0.025 | 256 | 220 | 0.440 | [0.3960, 0.4848] | No criterion; descriptive |
| 0.025 | 1,024 | 107 | 0.214 | [0.1788, 0.2526] | No criterion; descriptive |
| 0.025 | 4,096 | 10 | 0.020 | [0.0096, 0.0365] | No criterion; descriptive |
| 0.050 | 256 | 304 | 0.608 | [0.5637, 0.6510] | No criterion; descriptive |
| 0.050 | 1,024 | 349 | 0.698 | [0.6557, 0.7380] | No criterion; descriptive |
| 0.050 | 4,096 | 360 | 0.720 | [0.6784, 0.7590] | No criterion; descriptive |
| 0.100 | 256 | 470 | 0.940 | [0.9155, 0.9592] | No criterion; descriptive |
| 0.100 | 1,024 | 499 | 0.998 | [0.9889, 0.9999] | No criterion; descriptive |
| 0.100 | 4,096 | 500 | 1.000 | [0.9926, 1.0000] | No criterion; descriptive |
| 0.200 | 256 | 500 | 1.000 | [0.9926, 1.0000] | No criterion; descriptive |
| 0.200 | 1,024 | 500 | 1.000 | [0.9926, 1.0000] | No criterion; descriptive |
| 0.200 | 4,096 | 500 | 1.000 | [0.9926, 1.0000] | No criterion; descriptive |

The preregistered directional expectation that power would generally increase with shots is not true for the smallest alternative. Its true TVD (0.025) is below the fixed 0.05 cutoff at 1,024 and 4,096 shots; as sampling noise contracts, fewer empirical distances cross that fixed cutoff. This is consistent with the stated rule and observed counts, but does not independently identify a causal mechanism beyond this synthetic setup.

## Why the null criterion failed

The evidence supports a detector-operating-characteristic explanation at 256 shots: the existing heuristic cutoff yields a 34% empirical null flag rate, with an exact lower confidence bound around 32.3%, versus the registered 6% ceiling. The detector compares two noisy empirical distributions and its `max(0.05, 1/sqrt(shots))` guard is not a calibrated level-0.06 two-sample test. Finite-shot sampling variation is the source of empirical TVD under q=p; the chosen rule fails to suppress that variation enough at 256 shots. The large count and interval make ordinary replicate sampling uncertainty an inadequate explanation for the gap from 6% at this shot count.

At 1,024 shots, the empirical rate is 5.7%, but 2,000 null replicates leave an upper exact bound of 6.63%. The preregistered conservative confidence-bound gate is not met, while the data do not show that the underlying rate exceeds 6%. That cell is unresolved relative to the 6% boundary because of finite Monte Carlo precision near the boundary. At 4,096 shots, zero flags gives an upper bound of 0.15%, satisfying the registered cell criterion.

There is no evidence that threshold selection caused the failure: the preregistration records zero development draws and no threshold selection, and the runner always supplies the existing 0.05 threshold. Nor is there a discrepancy in the registered distributions or replicate counts. The all-cells acceptance rule is conservative by design (failure to certify is a failure under its stated definition); it correctly produces overall FAIL because at least one cell fails, but its 1,024-shot outcome should not be interpreted as proof that the true FPR exceeds 6%.

## Leakage and implementation consistency

The implementation matches the preregistered p, shot counts, null and four q alternatives, replicate counts, two-stream seed formula, roles, threshold, and resource pair accounting. The result's implementation and statistics-module hashes match the preregistered hashes and current files; the preregistration hash in results also matches. There are 24,000 planned stream seeds (12,000 pairs x two roles); recomputing them from the recorded policy found 24,000 unique 32-bit seeds and no collisions. This check does not prove statistical independence of a pseudorandom generator; it verifies distinct seed values for this ledger.

No development sample, adaptive threshold choice, or held-out result feedback appears in the runner. Therefore there is no development/evaluation leakage within this protocol. Calling the replicate set ?evaluation? is more accurate than calling it a separate ?held-out? test: no development split was used because no tuning was performed.

## Resource and reproducibility audit

The saved manifest reports planned=actual=maximum=12,000 pairs and one worker. The fixed loop structure independently gives the same total. The implementation checks the pair count before execution and the processed count after each sampled pair; for this fixed plan it ends at 12,000. (If the plan were later expanded, the post-sample `processed > MAX_PAIRS` check would detect an excess only after generating the first over-cap pair; the current fixed workload does not hit that edge.) It records 6.0 CPU seconds and 6.047 wall seconds, below the 600-second CPU cap. The runner enforces a 256 MiB `tracemalloc` peak and a 256 MiB serialized JSON limit; recorded traced peak is 1,070,257 bytes and the saved results file is 5,882 bytes. The memory measurement is traced Python allocation, not total process RSS; the preregistration explicitly names the cap as traced memory, so total resident memory is unresolved by these artifacts.

The hash-derived seeds, fixed loop order, and seeded NumPy generators support deterministic sampled counts within a compatible NumPy/RNG implementation. A focused test confirms that one pair repeats for the same input and seed. It does not rerun and compare the full 12,000-pair result. The artifacts omit Python and NumPy versions, and `default_rng` does not promise identical streams across all future NumPy versions. Also, recorded CPU/wall timing fields are inherently variable, so the entire result JSON is not expected to be byte-identical across runs even if all statistical counts match. The saved artifacts support reproducibility of the procedure, but not unconditional cross-version bitwise reproduction.

## Criterion-by-criterion verdict

| Criterion | Verdict | Basis |
|---|---|---|
| Exact planned/max pair count | PASS for this run | 12,000 planned and actual pairs; fixed loops sum to 12,000. |
| CPU-time cap <= 600 seconds | PASS | 6.0 seconds recorded. |
| Traced-memory cap <= 256 MiB | PASS as preregistered | 1,070,257 bytes recorded; not a total-RSS measurement. |
| Serialized-output cap <= 256 MiB | PASS | 5,882-byte result file; code checks before writing. |
| Null upper FPR bound <= 0.06 at 256 shots | FAIL | 0.35783917. |
| Null upper FPR bound <= 0.06 at 1,024 shots | FAIL / boundary not certified | 0.06627158, while point estimate is 0.057. |
| Null upper FPR bound <= 0.06 at 4,096 shots | PASS | 0.00149674. |
| Overall preregistered null gate | FAIL | Requires all three shot cells to pass. |
| Alternative power acceptance | Not specified | No alternative power floor or pass/fail rule exists in preregistration. |
| No threshold tuning on evaluation draws | PASS | No development draws or selection step; fixed pre-existing threshold. |
| Hash/configuration consistency | PASS | Code, stats module, and preregistration hashes agree; registered settings align. |
| Cross-version bitwise reproducibility | UNRESOLVED | Runtime versions not recorded; full run was not repeated. |
| Total process-RSS limit | UNRESOLVED | Only traced allocation was measured. |

## Recommendation (separate from audit findings)

Treat this as a negative characterization of the current heuristic under the specified synthetic null, especially at 256 shots. Do not use its threshold as evidence of a calibrated false-positive level or as support for program-mutation detection or three-way diagnosis. If future work evaluates a different detector or cutoff, first preregister its statistical target and any development-only calibration procedure, then evaluate it on new independent draws without modifying these saved artifacts. Record Python/NumPy versions and, if a total-memory budget is required, measure process RSS explicitly.

## Lightweight audit checks

- Read the five requested artifacts and the shared `compare_distributions` implementation to verify the decision rule.
- Parsed preregistration and results JSON; verified implementation, statistics-module, and preregistration SHA-256 consistency.
- Recomputed all three null upper bounds and all 12 alternative two-sided intervals independently with beta quantiles; stored values agree to floating-point precision.
- Recomputed all 24,000 stream seeds; no duplicate seed values were found.
- No experiment, sampling run, training, ingestion, hardware job, or QML job was rerun.
