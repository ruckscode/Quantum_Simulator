# Synthetic finite-shot TVD characterization (2026-10-15)

**Scope:** bounded simulator-independent multinomial sampling experiment preregistered in `finite_shot_synthetic_characterization_preregistration_20261015.json`. Machine-readable results: `finite_shot_synthetic_characterization_results_20261015.json`. Existing detector and frozen 45-case diagnosis evaluation were not changed.

## Preregistered design

Reference distribution `p=(0.50, 0.25, 0.15, 0.10)`; independent reference and submitted multinomial count vectors at 256, 1,024, and 4,096 shots. Null `q=p`; alternatives transfer mass from outcome 1 to outcome 2 by TVD 0.025, 0.05, 0.10, and 0.20. There were 2,000 null pairs per shot count and 500 alternative pairs per effect and shot cell.

Each pair means one independent reference count vector plus one submitted count vector for one condition and shot count. Planned and actual total: **12,000 count-pairs** (6,000 null and 6,000 alternative), or 24,000 multinomial count vectors. Seeds were derived for each stream from the first 32 bits, big-endian, of SHA-256 over `protocol_id|shot_count|condition_id|replicate_index|stream_role`.

The fixed detector was `TVD > max(0.05, 1/sqrt(shots))`; no threshold was selected or tuned. Null false-positive rates use one-sided exact 95% Clopper-Pearson upper bounds. Alternative detection rates use two-sided exact 95% Clopper-Pearson intervals. The only acceptance gate was upper null FPR bound <= 0.06 at every shot count; no alternative minimum-power criterion was specified.

## Results

| Condition | True TVD | Shots | Detections / trials | Rate | 95% interval / upper bound |
|---|---:|---:|---:|---:|---:|
| null_equal | 0.000 | 256 | 680 / 2000 | 0.3400 | upper 0.3578 (one-sided) |
| tvd_025 | 0.025 | 256 | 220 / 500 | 0.4400 | [0.3960, 0.4848] (two-sided) |
| tvd_050 | 0.050 | 256 | 304 / 500 | 0.6080 | [0.5637, 0.6510] (two-sided) |
| tvd_100 | 0.100 | 256 | 470 / 500 | 0.9400 | [0.9155, 0.9592] (two-sided) |
| tvd_200 | 0.200 | 256 | 500 / 500 | 1.0000 | [0.9926, 1.0000] (two-sided) |
| null_equal | 0.000 | 1024 | 114 / 2000 | 0.0570 | upper 0.0663 (one-sided) |
| tvd_025 | 0.025 | 1024 | 107 / 500 | 0.2140 | [0.1788, 0.2526] (two-sided) |
| tvd_050 | 0.050 | 1024 | 349 / 500 | 0.6980 | [0.6557, 0.7380] (two-sided) |
| tvd_100 | 0.100 | 1024 | 499 / 500 | 0.9980 | [0.9889, 0.9999] (two-sided) |
| tvd_200 | 0.200 | 1024 | 500 / 500 | 1.0000 | [0.9926, 1.0000] (two-sided) |
| null_equal | 0.000 | 4096 | 0 / 2000 | 0.0000 | upper 0.0015 (one-sided) |
| tvd_025 | 0.025 | 4096 | 10 / 500 | 0.0200 | [0.0096, 0.0365] (two-sided) |
| tvd_050 | 0.050 | 4096 | 360 / 500 | 0.7200 | [0.6784, 0.7590] (two-sided) |
| tvd_100 | 0.100 | 4096 | 500 / 500 | 1.0000 | [0.9926, 1.0000] (two-sided) |
| tvd_200 | 0.200 | 4096 | 500 / 500 | 1.0000 | [0.9926, 1.0000] (two-sided) |

**Preregistered decision: FAIL.** Null upper bounds are 0.3578 at 256 shots, 0.0663 at 1,024, and 0.0015 at 4,096. The first two exceed 0.06; the 4,096-shot cell meets the gate. At true TVD 0.025, power was 0.44, 0.214, and 0.02 as shots increased. The fixed 0.05 minimum cutoff makes this small effect harder to flag as sampling noise contracts.

## Reproducibility, validation, and limits

Protocol ID: `synthetic-two-sample-tvd-v1-20261015`. Implementation SHA-256: `a40a1d81a0fd1a7aaadcb187535484849a80c14f6b6165ed32501974a9e63a9f`; statistics module SHA-256: `2d90d01a024aa527a4dd7805a09ba0216e35afceb3e718d979e9ed5399212d82`. The preregistration SHA-256 is recorded in the result JSON. One worker used 6.0 CPU seconds (6.05 wall seconds); peak traced Python allocation was 1,070,257 bytes. Pair cap was met exactly; runtime and serialized-output caps were not reached.

Exact binomial interval inversion uses log-PMF summation in the experiment runner to avoid overflow at 2,000 trials; no dependencies were added. The output is deterministic for the recorded code, configuration, and seed policy.

This experiment samples synthetic distributions only. It does not simulate a quantum circuit, validate the detector on QMutBench or independent circuit families, establish program-fault ground truth, support the separate three-way diagnosis task, or provide physical-hardware evidence. QMutBench eligibility remains zero verified recipe roots.

Focused tests: 4 passed. The established bounded suite completed with 816 passed, 1 failed, and 2 deselected. Its sole failure was `tests/test_reference_diagnosis_generalization.py::test_heldout_generalization_outcomes_and_targets`, reporting the same seven frozen mismatches; its fixture, labels, thresholds, and outcomes are unchanged.
