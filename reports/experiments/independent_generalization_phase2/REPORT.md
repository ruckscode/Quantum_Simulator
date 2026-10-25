# Independent Generalization and Fault Localization — Phase 2

Protocol: `independent_generalization_phase2_v1`. This report uses only new Phase 2 scenarios; Phases 1, 1.3, and 1.4 are inventory references, not observations included in these metrics.

## Prior-phase coverage and Phase 2 novelty

Phases 1, 1.3, and 1.4 used the same two-qubit X/measurement circuit family: readout noise on q0/q1, X-gate error on q0, an X-based program mutation at index 0, and one identity-duration/T1/T2 pair. Phase 1.3 fixed duration-only evidence semantics; Phase 1.4 repeated those same nine scenarios across seeds.

Phase 2 introduces four-qubit register usage; RX/RY/RZ/RZZ sequences; q2/q3 readout and gate locations; q0-q2 RZZ and q1-q3 CX couplings; program mutations at operation index 2; a program/hardware pair with matched observed distributions; readout strengths 0.04, 0.055, 0.20, and 1.0; a subtle RZZ gate error; and overlapping/unknown-cause cases. These are new scenario structures relative to the prior nine, although several Phase 2 strength variants intentionally share a circuit family.

The q1-q3 coupling case supplies both a CX gate error (applied by the simulator) and matching coupling-edge calibration metadata (used for edge localization). Edge error metadata alone is not applied as a shot-noise process by the current simulator.

## Protocol and provenance

- Scenarios: 12; seeds: [20262001, 20262002, 20262003]; shots per run: 512; total runs: 36.
- Scenario ground truth and affected components are declared in `scenario_manifest.json` before execution; they do not depend on diagnosis output.
- Each scenario-seed result records circuit family, circuit and observed circuit, hardware/noise settings, observable statistics, evidence, missing evidence, candidates, and localization assessment.
- Source revision: `4ac81ad0e278c6045592a4e1c4df518512b04841`; working tree dirty: `True`. Source hashes are recorded in the manifest.
- Aggregate summaries were reconstructed from the saved case JSON files after correcting the per-seed settings field; no circuit was rerun. Rebuild code provenance is recorded in `aggregate_metrics.json`.
- All executions use the local simulator. No physical hardware was used.

## Results

- Run status: {'COMPLETED': 36, 'FAILED': 0, 'UNSUPPORTED': 0}; unsupported and failed runs remain separate from diagnosis errors.
- Ground-truth counts: {'NORMAL': 6, 'PROGRAM_FAULT': 6, 'HARDWARE_ANOMALY': 18, 'AMBIGUOUS / INSUFFICIENT_EVIDENCE': 6}.
- Classification: accuracy 0.917, balanced accuracy 0.958, macro-F1 0.922; abstentions 7.
- Detection: 22/30 positives (0.733); false positives 0/6 (0.000); false negatives 8.
- Localization: {'correct': 24, 'incorrect': 0, 'not_justified': 3, 'failed': 0, 'unsupported': 0}; correct rate across target cases 0.889.

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| NORMAL | 6 | 0.750 | 1.000 | 0.857 |
| PROGRAM_FAULT | 6 | 1.000 | 1.000 | 1.000 |
| HARDWARE_ANOMALY | 18 | 1.000 | 0.833 | 0.909 |
| AMBIGUOUS / INSUFFICIENT_EVIDENCE | 6 | 0.857 | 1.000 | 0.923 |

### Confusion matrix (truth rows, prediction columns)

| Truth \ Predicted | NORMAL | PROGRAM_FAULT | HARDWARE_ANOMALY | AMBIGUOUS / INSUFFICIENT_EVIDENCE | FAILED | UNSUPPORTED | OTHER |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| PROGRAM_FAULT | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| HARDWARE_ANOMALY | 2 | 0 | 15 | 1 | 0 | 0 | 0 |
| AMBIGUOUS / INSUFFICIENT_EVIDENCE | 0 | 0 | 0 | 6 | 0 | 0 | 0 |

### Per-scenario and per-seed results

| Scenario | Family | Truth | Prediction | Detection | Localization | Abstain | Seed | TVD |
|---|---|---|---|---|---|---:|---:|---:|
| healthy_bell_q2_q3 | phase2_bell_q2_q3 | NORMAL | NORMAL | False | NOT_APPLICABLE | False | 20262001 | 0.017578125 |
| healthy_rzz_phase_depth | phase2_rzz_phase_q0_q2 | NORMAL | NORMAL | False | NOT_APPLICABLE | False | 20262001 | 0.027010798517136073 |
| program_fault_ry_to_rz_q2 | phase2_rotation_readout_q2 | PROGRAM_FAULT | PROGRAM_FAULT | True | CORRECT | False | 20262001 | 1.0 |
| hardware_readout_inversion_q2 | phase2_rotation_readout_q2 | HARDWARE_ANOMALY | HARDWARE_ANOMALY | True | CORRECT | False | 20262001 | 1.0 |
| program_fault_remove_rzz_q0_q2 | phase2_rzz_phase_q0_q2 | PROGRAM_FAULT | PROGRAM_FAULT | True | CORRECT | False | 20262001 | 0.0711781828074088 |
| hardware_readout_q3_moderate | phase2_cx_q1_q3_readout | HARDWARE_ANOMALY | HARDWARE_ANOMALY | True | CORRECT | False | 20262001 | 0.162109375 |
| hardware_readout_q3_near_threshold | phase2_cx_q1_q3_readout | HARDWARE_ANOMALY | HARDWARE_ANOMALY | False | CORRECT | False | 20262001 | 0.044921875 |
| hardware_readout_q3_subthreshold | phase2_cx_q1_q3_readout | HARDWARE_ANOMALY | NORMAL | False | NOT_JUSTIFIED | False | 20262001 | 0.037109375 |
| hardware_rzz_error_near_threshold | phase2_rzz_phase_q0_q2 | HARDWARE_ANOMALY | HARDWARE_ANOMALY | False | CORRECT | False | 20262001 | 0.025057673517136073 |
| hardware_cx_coupler_q1_q3 | phase2_cx_edge_q1_q3 | HARDWARE_ANOMALY | HARDWARE_ANOMALY | True | CORRECT | False | 20262001 | 0.080078125 |
| overlap_ry_to_rz_plus_readout_q2 | phase2_rotation_readout_q2 | AMBIGUOUS / INSUFFICIENT_EVIDENCE | AMBIGUOUS / INSUFFICIENT_EVIDENCE | True | CORRECT | True | 20262001 | 0.78515625 |
| cause_unknown_phase_observation | phase2_unknown_q3_rotation | AMBIGUOUS / INSUFFICIENT_EVIDENCE | AMBIGUOUS / INSUFFICIENT_EVIDENCE | True | NOT_APPLICABLE | True | 20262001 | 0.5 |
| healthy_bell_q2_q3 | phase2_bell_q2_q3 | NORMAL | NORMAL | False | NOT_APPLICABLE | False | 20262002 | 0.03515625 |
| healthy_rzz_phase_depth | phase2_rzz_phase_q0_q2 | NORMAL | NORMAL | False | NOT_APPLICABLE | False | 20262002 | 0.04159637185414237 |
| program_fault_ry_to_rz_q2 | phase2_rotation_readout_q2 | PROGRAM_FAULT | PROGRAM_FAULT | True | CORRECT | False | 20262002 | 1.0 |
| hardware_readout_inversion_q2 | phase2_rotation_readout_q2 | HARDWARE_ANOMALY | HARDWARE_ANOMALY | True | CORRECT | False | 20262002 | 1.0 |
| program_fault_remove_rzz_q0_q2 | phase2_rzz_phase_q0_q2 | PROGRAM_FAULT | PROGRAM_FAULT | False | CORRECT | False | 20262002 | 0.048654381144415094 |
| hardware_readout_q3_moderate | phase2_cx_q1_q3_readout | HARDWARE_ANOMALY | HARDWARE_ANOMALY | True | CORRECT | False | 20262002 | 0.1875 |
| hardware_readout_q3_near_threshold | phase2_cx_q1_q3_readout | HARDWARE_ANOMALY | HARDWARE_ANOMALY | True | CORRECT | False | 20262002 | 0.064453125 |
| hardware_readout_q3_subthreshold | phase2_cx_q1_q3_readout | HARDWARE_ANOMALY | AMBIGUOUS / INSUFFICIENT_EVIDENCE | True | NOT_JUSTIFIED | True | 20262002 | 0.0546875 |
| hardware_rzz_error_near_threshold | phase2_rzz_phase_q0_q2 | HARDWARE_ANOMALY | HARDWARE_ANOMALY | False | CORRECT | False | 20262002 | 0.03769012185414237 |
| hardware_cx_coupler_q1_q3 | phase2_cx_edge_q1_q3 | HARDWARE_ANOMALY | HARDWARE_ANOMALY | True | CORRECT | False | 20262002 | 0.123046875 |
| overlap_ry_to_rz_plus_readout_q2 | phase2_rotation_readout_q2 | AMBIGUOUS / INSUFFICIENT_EVIDENCE | AMBIGUOUS / INSUFFICIENT_EVIDENCE | True | CORRECT | True | 20262002 | 0.798828125 |
| cause_unknown_phase_observation | phase2_unknown_q3_rotation | AMBIGUOUS / INSUFFICIENT_EVIDENCE | AMBIGUOUS / INSUFFICIENT_EVIDENCE | True | NOT_APPLICABLE | True | 20262002 | 0.5 |
| healthy_bell_q2_q3 | phase2_bell_q2_q3 | NORMAL | NORMAL | False | NOT_APPLICABLE | False | 20262003 | 0.005859375 |
| healthy_rzz_phase_depth | phase2_rzz_phase_q0_q2 | NORMAL | NORMAL | False | NOT_APPLICABLE | False | 20262003 | 0.027924496854142368 |
| program_fault_ry_to_rz_q2 | phase2_rotation_readout_q2 | PROGRAM_FAULT | PROGRAM_FAULT | True | CORRECT | False | 20262003 | 1.0 |
| hardware_readout_inversion_q2 | phase2_rotation_readout_q2 | HARDWARE_ANOMALY | HARDWARE_ANOMALY | True | CORRECT | False | 20262003 | 1.0 |
| program_fault_remove_rzz_q0_q2 | phase2_rzz_phase_q0_q2 | PROGRAM_FAULT | PROGRAM_FAULT | True | CORRECT | False | 20262003 | 0.06191925314585763 |
| hardware_readout_q3_moderate | phase2_cx_q1_q3_readout | HARDWARE_ANOMALY | HARDWARE_ANOMALY | True | CORRECT | False | 20262003 | 0.21875 |
| hardware_readout_q3_near_threshold | phase2_cx_q1_q3_readout | HARDWARE_ANOMALY | HARDWARE_ANOMALY | False | CORRECT | False | 20262003 | 0.048828125 |
| hardware_readout_q3_subthreshold | phase2_cx_q1_q3_readout | HARDWARE_ANOMALY | NORMAL | False | NOT_JUSTIFIED | False | 20262003 | 0.025390625 |
| hardware_rzz_error_near_threshold | phase2_rzz_phase_q0_q2 | HARDWARE_ANOMALY | HARDWARE_ANOMALY | False | CORRECT | False | 20262003 | 0.025971371854142368 |
| hardware_cx_coupler_q1_q3 | phase2_cx_edge_q1_q3 | HARDWARE_ANOMALY | HARDWARE_ANOMALY | True | CORRECT | False | 20262003 | 0.12109375 |
| overlap_ry_to_rz_plus_readout_q2 | phase2_rotation_readout_q2 | AMBIGUOUS / INSUFFICIENT_EVIDENCE | AMBIGUOUS / INSUFFICIENT_EVIDENCE | True | CORRECT | True | 20262003 | 0.7890625 |
| cause_unknown_phase_observation | phase2_unknown_q3_rotation | AMBIGUOUS / INSUFFICIENT_EVIDENCE | AMBIGUOUS / INSUFFICIENT_EVIDENCE | True | NOT_APPLICABLE | True | 20262003 | 0.5 |

Full source configurations, case outputs, and aggregate metrics are saved in the sibling JSON/CSV files and `cases/` directory.

## Limitations

This is a small constructed simulator study over six new circuit families, with three fixed seeds and 512 shots. Strength variants and paired cases share circuits; runs are not independent samples of generalization. Near-threshold outcomes are sensitive to this finite-shot design. The ground truth is scenario-defined, not independently adjudicated. Simulator calibration profiles do not establish physical-hardware calibration fidelity or generalization.
