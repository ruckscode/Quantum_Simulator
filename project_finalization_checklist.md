# Project Finalization Checklist

**Project:** Hardware-Aware Data-Driven Testing and Fault Localization for Quantum Programs  
**Decision:** FINALIZE WITH EXISTING EVIDENCE  
**Purpose:** Prepare the existing project for an accurate final report and presentation. This is a checklist, not the final report. Do not reconstruct missing evidence or alter historical artifacts.

## 1. Evidence map for major claims

Check each item only after the final report/presentation cites the named evidence and includes the qualification alongside the claim.

| Status | Claim that can be made | Existing evidence to cite | Required qualification |
|---|---|---|---|
| [ ] | The project implements a local statevector simulator and circuit API with supported gates, measurement, and finite-shot execution. | `src/quantum_simulator/quantum_simulator.py`; `src/quantum_simulator/quantum_circuit.py`; `tests/test_simulator.py`; `tests/test_gates.py`; `tests/test_gate_correctness.py`; `README.md` §§5–8. | Describe implemented supported operations; do not imply a general-purpose quantum runtime or physical device simulator. |
| [ ] | Distribution differences are measured with statistical metrics and a finite-shot tolerance. | `src/quantum_simulator/statistics.py`; `tests/test_statistics.py`; `tests/test_noise_model_validation.py`; `README.md` §§5, 9. | A statistical anomaly says outcomes differ under the configured criterion; it does not identify a cause by itself. |
| [ ] | The rule-based diagnosis integrates operation comparisons, relevant hardware calibration evidence, and distribution evidence; it can retain ambiguity or insufficient evidence. | `src/quantum_simulator/diagnosis_engine.py`; `src/quantum_simulator/fault_localizer.py`; `src/quantum_simulator/hardware_fault_localizer.py`; `tests/test_diagnosis_engine.py`; `tests/test_fault_localizer.py`; `tests/test_hardware_fault_localizer.py`; `tests/test_stage3_controlled_diagnosis_evaluation.py`. | Distinguish direct operation mismatch from a trace-only deviation. A trace deviation alone is not evidence that the program is faulty. Controlled cases demonstrate paths, not expected field accuracy. |
| [ ] | High-level diagnosis runs a supplied circuit ideally and optionally with a `HardwareModel`; lower-level diagnosis compares supplied expected/observed circuits and distributions. | `src/quantum_simulator/diagnosis_engine.py` (`diagnose_circuit`, `diagnose_execution`); `tests/test_end_to_end_diagnosis.py`; `tests/test_diagnosis_engine.py`; `README.md` §§9–11. | `diagnose_circuit` does not accept a distinct reference circuit. Use `diagnose_execution` for explicit reference/actual comparison. |
| [ ] | Controlled local tests exercise normal, program-edit, and synthetic hardware conditions and selected component/operation localization. | `tests/test_stage3_controlled_diagnosis_evaluation.py`; `data/qmutbench_ml/final/controlled_diagnosis_evaluation.md/.json`; `reports/experiments/readout_anomaly_controlled_local.json`; `reports/experiments/program_fault_seeded_local.json`. | Synthetic models and seeded scenarios are not physical device measurements or population-level validation. |
| [ ] | The project has Bugs4Q/QMutBench adapters and an optional IBM integration. | `bugs4q_validation.py`; `bugs4q_integration.py`; `qmutbench_validation.py`; `ibm_validation.py`; `tests/test_bugs4q_validation.py`; `tests/test_bugs4q_integration.py`; `tests/test_ibm_validation.py`; `reports/benchmarks/bugs4q_validation_report.json`; `reports/benchmarks/qmutbench_validation_report.json`; `reports/ibm_validation_report.json`. | Report selected integration counts as adapter/fixture evidence, not benchmark accuracy. The IBM report is offline fixture validation; no live hardware claim. Availability of external/official datasets is not established by adapters alone. |
| [ ] | The QMutBench-derived binary study has saved data, predictions, threshold results, and model parameter files. | `data/qmutbench_ml/qmutbench_results.csv`; `qmutbench_manifest.json`; `qmutbench_failures.csv`; `final/evaluation_report.json`; `final/threshold_tuning_report.json`; `final/final_project_summary.json`; `final/test_predictions.csv`; `final/tuned_test_predictions.csv`; `final/threshold_validation_curve.csv`; `final/numpy_logistic_model.npz`; `final/numpy_logistic_model_tuned.npz`. | State the labels are record-type-derived (original vs mutant), class-imbalanced, and not independent proof of behavioral fault. Do not treat these binary models as hardware classifiers or the production observable adapter model. |
| [ ] | The controlled three-class ML evidence uses the production five observable features and avoids feeding labels/family IDs as features. | `data/qmutbench_ml/final/grouped_model_comparison.md/.json`; `grouped_model_comparison_predictions.csv`; source feature extractor `src/quantum_simulator/ml_diagnosis.py`; source reports under `reports/experiments/full_dataset/` and `dataset_manifest.json`. | This is controlled simulator-only evidence with synthetic intervention labels, 30 families, exact feature collisions, small family folds, and no significance test. |
| [ ] | The saved audit found no compatible model artifact for real inference through the observable-only ML adapter. | `data/qmutbench_ml/final/ml_artifact_integration_audit.md/.json`; `src/quantum_simulator/observable_model_adapter.py`; `tests/test_ml_diagnosis_integration.py`. | Adapter code and mock routing tests do not mean there is a trained/deployable model. The portable expected JSON artifact is absent. The two saved NPZ files have incompatible binary trace features/classes. |
| [ ] | Project verification has been reported as historical test evidence, not freshly rerun for finalization. | `README.md` §15 reports 280 passed; `reports/project_status.json` records an older 89; `reports/reproducibility_report.json` records 96. | Explain these are different historical snapshots. Do not combine them or say this checklist reran the test suite. |

## 2. Best-supported reproducible ML evaluation

Use this as the primary grouped ML evaluation when discussing generalization across controlled circuit families. Point to the saved Markdown, JSON, and OOF predictions; do not claim this checklist reran the pipeline.

| Item | Supported statement |
|---|---|
| Data | 270 saved controlled simulator reports, 30 circuit families, 9 reports/family, 90 each of NORMAL, PROGRAM_FAULT, HARDWARE_ANOMALY. |
| Inputs | Five production observables: TVD, JS divergence, Hellinger distance, observed entropy, and observed maximum probability. IDs, family names, intervention metadata, labels, calibration, diagnosis output, and trace localization features are excluded from inputs. |
| Primary baseline | Fixed StandardScaler + LogisticRegression(C=1, max_iter=2000), unsearched. Outer leave-one-family-out GroupKFold: 30 folds, each held-out family contributes 9 rows (3/class). The nested comparison uses 5-fold grouped inner CV on the 29 outer-training families to tune candidate models; preprocessing is fitted within training folds. |
| Baseline pooled OOF | Accuracy 0.563; balanced accuracy 0.563; macro-F1 0.543. Recall: NORMAL 0.900, PROGRAM_FAULT 0.422, HARDWARE_ANOMALY 0.367. Majority-dummy macro-F1 0.167. |
| Tuned candidate | `logistic_tuned_C` reaches pooled OOF macro-F1 0.576, balanced accuracy 0.585, and hardware recall 0.422. Its family macro-F1 is better/tied/worse than fixed LR on 7/18/5 families; treat as a candidate, not validated winner. |
| Limitations | Only 30 controlled simulator families; 3 examples/class/family; each family fold is coarse; outer training sets overlap; exact different-label feature collisions exist; no significance test; labels are controlled intervention labels; no physical device data. |

**Artifacts:** `data/qmutbench_ml/final/grouped_model_comparison.md`, `.json`, and `_predictions.csv`; report source: `reports/experiments/full_dataset/`; feature implementation: `src/quantum_simulator/ml_diagnosis.py::extract_observable_features`.

### Keep the binary QMutBench protocol separate

The QMutBench derived binary result is a separate result, not a competing score for the three-class grouped protocol.

- Dataset: 10,003 records (9,792 mutants / PROGRAM_FAULT; 211 originals / NORMAL), with five trace deviation features. All original records have zero features; 1,258 mutant records also have zero features.
- `evaluation_report.json` reports 8,051 training and 1,952 test rows grouped by original-circuit identity (169 train groups, 42 test groups, zero group overlap). This is the earlier untuned evaluation.
- The later threshold-tuned artifacts select threshold 0.81 on validation and report 1,675 held-out rows (43 NORMAL / 1,632 PROGRAM_FAULT): accuracy 0.877, balanced accuracy 0.937, macro-F1 0.614; NORMAL precision 0.173 and recall 1.0; PROGRAM_FAULT precision 1.0 and recall 0.874.
- Keep the 1,952-row and 1,675-row results distinct. The labels follow record type, and the strong association of zero features with originals exposes construction signal. This is preliminary dataset-specific binary classification, not independent behavioral validation or hardware diagnosis.

**Artifacts:** `data/qmutbench_ml/final/evaluation_report.json`, `threshold_tuning_report.json`, `final_project_summary.json`, `test_predictions.csv`, `tuned_test_predictions.csv`, `threshold_validation_curve.csv`, `dataset_audit.json`, `training_data_label_audit.md/.json`, `zero_feature_diagnostic.json`.

## 3. Demonstrable capabilities

- [ ] Demonstrate a supported circuit through the local simulator and show its measurement counts. Cite `README.md` §§7–8 and `tests/test_simulator.py`.
- [ ] Show finite-shot distribution comparison and explain the configured tolerance. Cite `src/quantum_simulator/statistics.py` and `tests/test_statistics.py`.
- [ ] Show a direct operation mismatch (for example, expected X vs actual H) producing a program candidate. Cite `tests/test_diagnosis_engine.py` and `tests/test_fault_localizer.py`.
- [ ] Show calibration relevant to the circuit contributing hardware candidates (readout, gate, or connectivity case). Cite `tests/test_stage3_controlled_diagnosis_evaluation.py` and `tests/test_hardware_fault_localizer.py`.
- [ ] Show ambiguity or insufficient evidence when evidence conflicts or is incomplete. Cite `tests/test_diagnosis_engine.py`.
- [ ] If discussing ML integration, show feature extraction and mock classifier routing/abstention only. Say real artifact inference is unavailable and unverified. Cite `src/quantum_simulator/ml_diagnosis.py`, `tests/test_ml_diagnosis_integration.py`, and `ml_artifact_integration_audit.md`.
- [ ] Treat controlled-diagnosis case totals as coverage of deliberately constructed cases, not estimates of accuracy on unseen programs/hardware.

## 4. Claims to qualify or omit

| Topic | Finalization rule |
|---|---|
| Real hardware / IBM | **Omit any claim of real-hardware validation, hardware performance, backend superiority, or live IBM execution.** Say IBM integration is optional and offline fixture-tested; the checked-in IBM report is not a live job. Cite `reports/ibm_validation_report.json`, `reports/project_status.json`, `README.md` §13. |
| T1/T2 relaxation | **Do not claim that the proposed first wave evaluated T1/T2.** The v3 spec explicitly excludes it; the preflight marks it blocked/excluded. Generic `HardwareModel` fields or prior controlled reports that contain T1/T2 values do not establish a validated physical relaxation model or unit convention. Cite `first_wave_experiment_spec_v3.json`, `first_wave_spec_v3_review.md`, `first_wave_execution_preflight.md`. |
| 240-report first wave | Describe it only as a reviewed proposal with `READY FOR OWNER APPROVAL` review status and `BLOCKED / STOP` execution preflight; no reports were generated. The blockers are same-shot realized RX skip capture and safe generation/resume controls. Do not report planned reports as observations. Cite `first_wave_spec_v3_review.md`, `first_wave_execution_preflight.md/.json`. |
| Reliable three-class ML diagnosis | **Do not claim reliable/general three-class diagnosis.** The strongest grouped fixed baseline has macro-F1 0.543 and hardware recall 0.367; tuned candidate hardware recall is 0.422, with inconsistent family gains and feature collisions. Synthetic interventions are not field observations. |
| Deployable ML diagnosis | **Do not claim a trained model is integrated in production.** Default inference has no compatible backend artifact. NPZ QMutBench models are binary and use trace features, unlike the five observable features/three output classes required by the adapter. |
| QMutBench “fault accuracy” | Qualify as classification of supplied QMutBench-derived record labels. The original/mutant record type maps directly to NORMAL/PROGRAM_FAULT; labels are not independent behavior outcomes. Include class imbalance, zero-feature audit, and low NORMAL precision for the tuned held-out result. |
| Simulator as a physical noise model | Describe only the mechanisms actually implemented and controlled. Do not generalize to arbitrary hardware noise, calibrated device behavior, or hardware causality. |
| Benchmark quality/accuracy | Adapter integration counts are not benchmark accuracy, recall, F1, mutation score, or verified localization without independent ground truth. |

## 5. Recommended final research report structure

This is an outline for a later writing step; **do not start drafting the report as part of checklist completion.**

1. **Title and abstract:** identify the simulator and evidence-based diagnosis contribution; summarize the bounded ML findings and explicitly state simulator-only/no real-hardware validation.
2. **Problem and objectives:** distinguish output anomaly detection, program localization, hardware evidence, and uncertainty/abstention.
3. **System design:** simulator, circuit representation, hardware model, statistics, program localizer, hardware localizer, diagnosis engine, optional ML adapter boundary.
4. **Implementation and verification:** describe implemented behavior, relevant tests, controlled demos, and historical test-result provenance. Separate tested code paths from unavailable external integrations.
5. **Data and labels:** describe the QMutBench-derived sample and failure records, record-type-derived labels, class imbalance and zero-feature audit; then describe the 270-row controlled simulator dataset and its intervention labels/family structure.
6. **Evaluation protocols:** give separate, exact sections for (a) QMutBench binary evaluation, distinguishing 1,952-row untuned from 1,675-row threshold-tuned results, and (b) grouped controlled three-class evaluation, including outer/inner family grouping, training-only preprocessing and saved predictions.
7. **Results:** report per-class metrics and confusion matrices, prioritize macro-F1/balanced accuracy over accuracy alone, and compare fixed LR with the tuned candidate and dummy baseline. Summarize rule-based controlled cases separately from ML.
8. **Threats to validity and limitations:** label validity, class imbalance, zero vectors, feature collisions, small number of families, synthetic channels, mechanism scope, missing compatible production ML artifact, missing external data/live backend evidence, and lack of significance analysis.
9. **Conclusion and future work:** conclude only supported claims; list independent labels, broader held-out families, safe first-wave design resolution, and real hardware as future work if desired. State no new experiment is part of this finalization.
10. **Reproducibility appendix:** artifact paths, manifests, predictions, model schema status, software/test snapshots, and explicit unavailable evidence.

## 6. Recommended presentation structure

1. **Motivation and scope** — the distinction between an output deviation and its cause.
2. **System architecture** — circuit/simulator → statistics → program and hardware evidence → diagnosis/abstention.
3. **Implemented demonstration** — one program mismatch, one controlled hardware calibration case, and one ambiguous/insufficient case; label all evidence as local/synthetic.
4. **Evaluation datasets** — clearly separate derived QMutBench binary labels and controlled simulator three-class data.
5. **Protocol and leakage controls** — family-level holdout, nested tuning, training-only scaler; show why random row splits are not the reported protocol.
6. **Results** — fixed grouped baseline first, tuned candidate second, class recalls/confusion matrix and dummy baseline; binary QMutBench result in a separate panel with label caveat.
7. **Limitations and missing evidence** — no real device, no T1/T2 first-wave evidence, weak hardware recall, no compatible deployed ML model; identify the 240-report wave as blocked and deferred.
8. **Close** — state what is implemented, what evidence supports it, and what future work would require. Do not imply deployment readiness.

## 7. Missing evidence to disclose, not reconstruct

- [ ] No live IBM/backend execution or real-device calibration/result evidence is verified. Offline fixture artifacts are not substitutes.
- [ ] No completed 240-report first-wave dataset, realized same-shot RX skip counts, or first-wave validation results exist. Planned counts/seeds/circuits are design metadata only.
- [ ] No T1/T2 experiment from the proposed wave exists; that arm is explicitly blocked/excluded. No physical time-unit calibration is established by generic model values.
- [ ] No compatible three-class observable ML model artifact exists at `src/quantum_simulator/models/observable_logistic_regression.json`; do not fabricate or convert the binary NPZ model into a different schema/class set.
- [ ] No independent behavioral ground-truth audit for every original/mutant label is present. Mutation status and record type do not prove observable fault behavior.
- [ ] No confidence interval/significance test or broad family-generalization guarantee is reported for the 270-row ML study.
- [ ] External official Bugs4Q or full QMutBench data availability is not established by adapter code or a report that names a dataset; cite only data actually present.
- [ ] No fresh test run was performed during this finalization audit/checklist work; historical totals of 280, 89, and 96 come from distinct saved documentation snapshots and must not be combined.
- [ ] Do not infer exact CSV-wide reproducibility or bitwise cross-environment reproduction beyond saved report provenance; prior audit inspected selected files and did not rerun all evaluations.

## 8. Minimum completion sequence and stop conditions

- [ ] **Freeze the claim boundary.** Stop when all final materials call QMutBench results record-label classification, three-class ML controlled simulator evidence, and hardware/live IBM validation absent.
- [ ] **Attach exact artifacts to each headline number.** Stop when each result has its report/prediction source, label definition, split unit and metric denominator next to it; keep the two binary test sizes separate.
- [ ] **Prepare the report and slide outline from sections 5–6.** Stop when every claimed capability has code/test/report evidence and every unsupported claim has been omitted or expressly qualified.
- [ ] **Preserve the evidence tree.** Do not overwrite/delete/rename historical reports or predictions; do not run the first wave or retrain for this finalization. Stop with the current inventory and disclosure of gaps.

No new experiment, model training, data generation, package installation, or production-code change is needed to complete this checklist.
