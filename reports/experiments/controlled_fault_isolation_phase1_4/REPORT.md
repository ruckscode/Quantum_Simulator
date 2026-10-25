# Controlled Fault-Isolation — Phase 1.4 Seed Sensitivity

Scenario protocol: `controlled_fault_isolation_phase1_v1` (the same nine predeclared scenarios and ground-truth rules used in Phase 1).
Output/evaluation version: `controlled_fault_isolation_phase1_4_seed_sensitivity_v1`. The output version identifies this repeated-seed study; it does not rename the unchanged scenario protocol.

## Design

- Cases: 9 scenarios × 5 seeds = 45 runs.
- Seeds: 20261014, 20261015, 20261016, 20261017, 20261018. These are five fixed seeds following the Phase 1 seed 20261013; the baseline run itself is not included again.
- Shots: 512 per case, unchanged from Phase 1. Holding shots fixed isolates sensitivity to seed-dependent sampling; this does not estimate shot-count sensitivity.
- Scenario definitions, hardware profiles, ground-truth labels, and diagnosis threshold are reused unchanged. Ground truth comes from the declared scenario table, not model outputs.
- Repetitions share the same small circuit family. Treat rates as descriptive repeated-sampling results, not independent observations or evidence of generalization.
- Execution uses the local simulator only; this is not physical-hardware validation.

## Aggregate results across scenario-seed runs

- Statuses: {'COMPLETED': 45, 'FAILED': 0, 'UNSUPPORTED': 0}
- Accuracy: 1.000; balanced accuracy: 1.000; macro-F1: 1.000.
- Detection: 35/35 (1.000); false alarms 0/10.
- Localization: 30/30 supported target-cases (1.000).
- Abstentions: 10; failures: 0; unsupported: 0.

## Per-seed metrics

| Seed | Accuracy | Balanced accuracy | Macro-F1 | Detection | False alarms | Localization | Abstentions |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20261014 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 2 |
| 20261015 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 2 |
| 20261016 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 2 |
| 20261017 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 2 |
| 20261018 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 2 |

## Per-scenario variability

| Scenario | Truth | Detection rate | Classification accuracy | Localization rate (n) | Abstentions | TVD mean [min, max] |
|---|---|---:|---:|---:|---:|---:|
| healthy_ideal | NORMAL | 0.000 | 1.000 | n/a | 0/5 | 0.0000 [0.0000, 0.0000] |
| readout_noise_q0 | HARDWARE_ANOMALY | 1.000 | 1.000 | 1.000 (5) | 0/5 | 0.3891 [0.3398, 0.4434] |
| readout_noise_q1 | HARDWARE_ANOMALY | 1.000 | 1.000 | 1.000 (5) | 0/5 | 0.4051 [0.3906, 0.4219] |
| gate_noise_x_q0 | HARDWARE_ANOMALY | 1.000 | 1.000 | 1.000 (5) | 0/5 | 0.4078 [0.3789, 0.4336] |
| coherence_healthy_control | NORMAL | 0.000 | 1.000 | n/a | 0/5 | 0.0000 [0.0000, 0.0000] |
| coherence_degradation_q0 | HARDWARE_ANOMALY | 1.000 | 1.000 | 1.000 (5) | 0/5 | 0.6250 [0.5938, 0.6504] |
| program_fault_ideal | PROGRAM_FAULT | 1.000 | 1.000 | 1.000 (5) | 0/5 | 1.0000 [1.0000, 1.0000] |
| program_fault_plus_readout_noise | AMBIGUOUS / INSUFFICIENT_EVIDENCE | 1.000 | 1.000 | 1.000 (5) | 5/5 | 0.5879 [0.5605, 0.6035] |
| deviation_without_cause_evidence | AMBIGUOUS / INSUFFICIENT_EVIDENCE | 1.000 | 1.000 | n/a | 5/5 | 0.5000 [0.5000, 0.5000] |

Full per-scenario/per-seed outcomes, evidence, settings, and candidates are in `scenario_seed_results.csv` and the case JSON files. `scenario_variability.csv` and `robustness_results.json` preserve the same summaries and provenance.

## Limits

Five fixed seeds at one shot count quantify only seed sensitivity for these constructed scenarios. The nine scenarios share one circuit family; no confidence interval or generalization claim is warranted. Ground truth is scenario-defined, and all results are simulator-derived.
