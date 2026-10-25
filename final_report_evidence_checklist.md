# Final Report Evidence Checklist

This checklist maps claims in `final_research_report.md` to checked-in source, tests, and saved artifacts. Metric values below are transcribed from saved results unless explicitly labeled as arithmetic derived from a saved confusion matrix. No experiment, training run, data generation, or test run was performed for this report.

## 1. Project scope and system claims

| Report claim | Supporting evidence | Qualification to retain |
|---|---|---|
| Project objective: local quantum execution, distribution comparison, program localization, hardware calibration evidence, and evidence-preserving diagnosis. | `README.md`, §§1–5, 9–13; `project_finalization_audit.md`; `project_finalization_checklist.md`. | This is the project's stated objective and implemented local scope. It is not a general physical quantum platform. |
| Statevector simulator applies supported gates and samples measurement; circuit API includes measurement, reset, and barrier operations. | `src/quantum_simulator/quantum_simulator.py`; `src/quantum_simulator/gates.py`; `src/quantum_simulator/quantum_circuit.py`; `tests/test_simulator.py`; `tests/test_gates.py`; `tests/test_gate_correctness.py`. | Say supported gates/operations as implemented; do not imply all quantum instructions or physical execution. |
| HardwareModel stores backend-independent qubit, gate, connectivity, and calibration properties and validates probabilities/times. | `src/quantum_simulator/hardware_model.py`; `README.md`, §12. | Model values are caller-supplied or generic, not necessarily measured hardware calibration. |
| Gate calibration models probabilistic operation skips; readout error flips measurement bits; relaxation code includes duration-dependent T1/T2 effects. | `src/quantum_simulator/quantum_simulator.py`, `run_circuit`, `_apply_hardware_relaxation`; `tests/test_noise_model_validation.py`; `tests/test_simulator.py`. | T1/T2 code is simulator behavior, not evidence of validated physical units or device dynamics. First-wave coverage was excluded. |
| Distribution metrics are TVD, Hellinger, and JS; finite-shot effective threshold uses `max(configured threshold, 1/sqrt(shots))`. | `src/quantum_simulator/statistics.py`; `tests/test_statistics.py`. | This is a project threshold rule, not a calibrated significance test. |
| Program localizer compares operation sequences; hardware localizer scores relevant qubit/gate/edge evidence; diagnosis composes results. | `src/quantum_simulator/fault_localizer.py`; `hardware_fault_localizer.py`; `diagnosis_engine.py`; corresponding tests. | Operation localization assumes a supplied expected/reference sequence; hardware localization depends on supplied HardwareModel evidence. |
| `diagnose_circuit` runs a supplied circuit ideally and optionally with hardware model; `diagnose_execution` can compare separate expected/observed circuits/distributions. | `src/quantum_simulator/diagnosis_engine.py`; `README.md`, §§9–10; `tests/test_end_to_end_diagnosis.py`; `tests/test_diagnosis_engine.py`. | The high-level function does not take a distinct reference circuit. |
| Trace-only deviations are not treated as direct program-fault evidence; cancellation/mixed evidence is retained. | `src/quantum_simulator/diagnosis_engine.py`; `tests/test_stage3_controlled_diagnosis_evaluation.py`; `tests/test_diagnosis_engine.py`; `data/qmutbench_ml/final/controlled_diagnosis_evaluation.md/.json`. | Controlled cases test designed paths, not population-level accuracy. `AMBIGUOUS` and `INSUFFICIENT_EVIDENCE` enum members share one serialized value; summary/evidence fields carry the distinction. |
| Observable ML interface extracts five features, validates classifier outputs, defaults to confidence 0.60, and requires localization evidence for fault predictions. | `src/quantum_simulator/ml_diagnosis.py`; `src/quantum_simulator/observable_model_adapter.py`; `tests/test_ml_diagnosis_integration.py`. | Tests use mock classifiers; this does not establish real trained-model inference. |

## 2. Quantitative claims and verification

### QMutBench-derived binary study

| Claim in report | Verified saved source | Check / caveat |
|---|---|---|
| 10,003 records; 9,792 mutant/PROGRAM_FAULT and 211 original/NORMAL. | `data/qmutbench_ml/final/evaluation_report.json`; `dataset_audit.json`. | Counts sum to 10,003; labels align exactly with record type in the audit. |
| Five trace features: max/mean/median probability difference, deviation concentration, high-deviation fraction. | `evaluation_report.json`; `training_data_label_audit.json`. | These differ from observable adapter features. |
| 211 original records are all-zero; 1,258 mutant records are all-zero. | `dataset_audit.json`; `zero_feature_diagnostic.json`. | Audit reports 1,469 all-zero rows total. |
| Earlier split: 8,051 train / 1,952 test; 169 / 42 circuit groups, zero overlap. | `evaluation_report.json`. | Separate from the threshold-tuned partition below. |
| Earlier untuned logistic metrics: accuracy 0.8294057; balanced accuracy 0.9128272; macro-F1 0.5529706; NORMAL P/R 0.112/1.0; PROGRAM_FAULT P/R 1.0/0.8256545. | `evaluation_report.json`. | Report rounds to 0.829 / 0.913 / 0.553; recall 0.826. |
| Later split: train/validation/test = 5,920 / 2,408 / 1,675; groups = 126 / 42 / 43; pairwise group overlap zero. | `threshold_tuning_report.json`. | Split counts and group counts are explicitly recorded in that report. |
| Threshold 0.8100000000000002 selected on validation groups; validation metric rows are separate from held-out test. | `threshold_tuning_report.json`. | Threshold selected on validation; test groups held out. |
| Later test: 1,675 records (43 NORMAL, 1,632 PROGRAM_FAULT); accuracy 0.8770149, balanced accuracy 0.9368873, macro-F1 0.6135781; NORMAL P/R 0.1726908/1.0; PROGRAM_FAULT P/R 1.0/0.8737745. | `threshold_tuning_report.json`; `final_project_summary.json`. | Rounded values in report are 0.877, 0.937, 0.614, and 0.173/1.000, 1.000/0.874. |
| Later confusion matrix `[[1426,206],[0,43]]`, actual rows/predicted columns in PROGRAM_FAULT, NORMAL order. | `threshold_tuning_report.json`; `final_project_summary.json`. | 206 false NORMAL predictions are explicitly also recorded in final summary. |

### Controlled three-class simulator ML

| Claim in report | Verified saved source | Check / caveat |
|---|---|---|
| 270 reports; 30 families; 9/family; 90/class. | `data/qmutbench_ml/final/grouped_model_comparison.json`, `source`; `grouped_model_comparison.md`; `reports/experiments/full_dataset/dataset_manifest.json`. | Balanced controlled simulator scenario labels. |
| Five observable features and excluded leakage fields. | `grouped_model_comparison.json`, `source`; `src/quantum_simulator/ml_diagnosis.py`. | Labels target only; IDs, family identity, intervention metadata, calibration, diagnosis outputs and trace features excluded. |
| Outer protocol: 30-fold leave-one-family-out GroupKFold; 29 families/261 rows train, one family/9 rows test, 3/class in test; inner five-fold family GroupKFold, macro-F1 selection. | `grouped_model_comparison.json`, `evaluation`; `grouped_model_comparison.md`. | Preprocessing fitted within training folds; fixed LR is unsearched. |
| Fixed logistic OOF: accuracy/balanced accuracy 0.562963; macro-F1 0.543431; recalls NORMAL 0.900, PROGRAM_FAULT 0.422222, HARDWARE_ANOMALY 0.366667. | `grouped_model_comparison.json`, `out_of_fold_metrics.logistic_fixed_baseline`; `grouped_model_comparison.md`. | Report rounds to three decimals. |
| Fixed LR confusion `[[81,0,9],[32,38,20],[45,12,33]]`, labels NORMAL/PROGRAM_FAULT/HARDWARE_ANOMALY. | `grouped_model_comparison.json`; `grouped_model_comparison.md`. | Used for per-class recall and error description. |
| Tuned logistic OOF: accuracy 0.585185; balanced accuracy 0.585185; macro-F1 0.575644; hardware recall 0.422222. | `grouped_model_comparison.json`, `out_of_fold_metrics.logistic_tuned_C`; `grouped_model_comparison.md`. | Highest pooled candidate score; not a selected deployment model. |
| Tuned-vs-fixed family: better/tied/worse 7/18/5; mean family macro-F1 delta 0.037. | `grouped_model_comparison.md`; `grouped_model_comparison.json`, `fold_consistency_vs_existing_logistic`. | Descriptive paired family comparison; no significance test. |
| Majority dummy macro-F1 0.166667. | `grouped_model_comparison.json`, `out_of_fold_metrics.majority_dummy`; `grouped_model_comparison.md`. | Rounded to 0.167. |
| Model comparison table metrics and hardware P/R/F1 for tuned LR, RBF-SVM, fixed LR, LDA, Extra Trees, Random Forest, Dummy. | `grouped_model_comparison.md`, “Out-of-fold results”; `grouped_model_comparison.json`, `out_of_fold_metrics`. | Display values rounded to three decimals directly from pooled metrics. |
| Fixed LR misses 57/90 hardware cases; tuned LR misses 52/90. | `grouped_model_comparison.json`, both confusion matrices. | Arithmetic derived from the actual hardware rows: 90−33 and 90−38; no new evaluation was performed. |
| Exact cross-class feature collisions exist. | `grouped_model_comparison.md`, “Error analysis”; `grouped_model_comparison.json`, collision analysis. | Supports feature identifiability limitation, not proof that all classifier errors are caused by collisions. |

### Controlled diagnosis and integrations

| Claim in report | Verified saved source | Check / caveat |
|---|---|---|
| Stage-three fixture has one normal, five program, and four hardware cases, including T1/T2. | `tests/test_stage3_controlled_diagnosis_evaluation.py`, `CASES`; `controlled_diagnosis_evaluation.md`. | Case counts describe a constructed test fixture only. |
| QMutBench adapter: 11 executed/diagnosed of 12; one unsupported; no reliable category-compatible GT; metrics null. | `reports/benchmarks/qmutbench_validation_report.json`. | Do not describe as accuracy. |
| Bugs4Q focused integration: 8 selected cases, 3 NO_ANOMALY and 5 PROGRAM_FAULT. | `reports/bugs4q_focused_integration_report.json`; `README.md`, §15. | Integration counts, not benchmark performance metrics. |
| Proposed first wave: 240 planned reports, T1/T2 excluded, preflight BLOCKED/STOP, no reports generated. | `data/qmutbench_ml/final/first_wave_spec_v3_review.md`; `first_wave_execution_preflight.md/.json`. | Planned counts are not observations. |
| No live IBM result. | `reports/ibm_validation_report.json`: `real_hardware_executed=false`, `real_ibm_status=NOT_EXECUTED`, offline fixture. | No physical hardware validation claim. |
| Historical full-suite reports show 280, 89, and 96 passed in different files. | `README.md`; `reports/project_status.json`; `reports/reproducibility_report.json`. | Distinct historical snapshots; not rerun or combined for this report. |

## 3. ML interpretation to retain

- [ ] Keep the binary QMutBench-derived and three-class controlled simulator evaluations in separate sections and tables.
- [ ] Describe QMutBench labels as record-type-derived, not independent verified ground truth.
- [ ] State that the tuned binary threshold was selected on validation groups and report held-out results separately.
- [ ] Describe family-held-out evaluation as protection against within-family leakage, not proof of arbitrary-program generalization.
- [ ] Report the weak hardware recall, family-level inconsistency, exact collisions, small family count, and absence of uncertainty/significance analysis.
- [ ] Do not call the tuned LR a validated winner or any current three-class classifier reliable.
- [ ] State that no compatible three-class observable artifact is present; NPZ models belong to the binary trace-feature schema and mock tests do not establish real inference.

## 4. Hardware scope and first-wave disposition

- [ ] State plainly that no real-device/IBM job or real-hardware performance evaluation is evidenced. The IBM report is an offline fixture.
- [ ] Explain the nuance around T1/T2: source and some prior controlled tests implement/use modeled relaxation, but T1/T2 is explicitly excluded from the proposed first wave; no physical time-unit contract is established.
- [ ] Describe the 240-report wave as planned only. The v3 design review was not execution authorization; preflight says BLOCKED/STOP and no reports were generated.
- [ ] Do not convert planned report counts, seeds, or static observability calculations into experimental observations.

## 5. Background and references

- [ ] State that no external related-work bibliography was supplied or verified; make no unsupported novelty/comparison claims.
- [ ] Use only repository source, tests, docs, and saved artifacts listed in the report's “References and evidence sources” section.
- [ ] Do not fabricate external citations or imply project reports are peer-reviewed publications.

## 6. Final document quality checks

- [ ] Report organization includes abstract, introduction/motivation, problem/objectives, background/related work, architecture, simulator/noise model, datasets, ML methodology, results, error/localization analysis, discussion, limitations/threats, conclusion/future work, and references/evidence sources.
- [ ] Quantitative tables state their protocol and denominators and separate the 1,952-record and 1,675-record binary tests.
- [ ] All percentages/counts have a nearby artifact citation or appear in this checklist's quantitative map.
- [ ] Ambiguous and insufficient evidence are described using the implementation's actual enum alias and result evidence fields.
- [ ] Existing historical files are preserved; the prior `final_research_report_draft.md` remains unchanged.
- [ ] No experiment, training, data generation, package installation, hardware access, or production-code modification was performed during polishing.
