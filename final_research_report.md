# Hardware-Aware Data-Driven Testing and Fault Localization for Quantum Programs

**Final research report**  
**Evidence reviewed through:** 12 October 2026

## Abstract

Quantum program testing must distinguish an output deviation from its possible causes: a program change, modeled hardware behavior, finite sampling, or incomplete evidence. This project combines a local statevector simulator, finite-shot distribution comparison, operation-level program fault localization, and calibration-informed hardware candidate reporting. The diagnosis workflow preserves evidence and can withhold a cause when inputs are incomplete or conflicting.

The saved machine-learning (ML) evidence comprises two separate studies. A QMutBench-derived binary study classifies record-type-derived labels. Under its later group-separated, validation-threshold protocol, the held-out test accuracy is 0.877 and macro-F1 is 0.614; NORMAL precision is 0.173. These labels are not independent behavioral ground truth. A controlled three-class simulator study uses five observable distribution features and leave-one-circuit-family-out evaluation. Its fixed logistic-regression baseline has pooled out-of-fold macro-F1 0.543 and HARDWARE_ANOMALY recall 0.367. A tuned logistic candidate reaches macro-F1 0.576 and hardware recall 0.422, but its family-level gains are inconsistent. The evidence does not establish reliable general three-class diagnosis.

The project demonstrates implemented simulator and rule-based diagnosis/localization paths in controlled tests. It does not establish real-device validation or a deployable trained model for the observable three-class adapter. T1/T2 relaxation code and prior controlled cases exist, but T1/T2 coverage was excluded from the proposed 240-report first wave; that wave was blocked before execution. Findings and limitations below are confined to the checked-in source and saved artifacts.

## 1. Introduction and motivation

Quantum program outcomes can vary because of program defects, execution noise, sampling variation, or missing context. A difference between expected and observed measurement distributions indicates an output deviation under a selected threshold; by itself it does not say whether the program, hardware, or finite sampling caused the difference. Diagnosis therefore requires evidence beyond an output distance.

The project, **Hardware-Aware Data-Driven Testing and Fault Localization for Quantum Programs**, addresses this problem through a local simulator and an evidence-based workflow. It compares distributions, compares expected and observed operations where available, and evaluates supplied calibration information relevant to the circuit. Its stated design permits ambiguous or insufficient-evidence conclusions instead of forcing a single cause (`README.md`, §§1–5, 9–13).

This report uses three evidence classes. Source and tests support statements about implemented behavior; saved evaluation files support reported metrics and protocols; unavailable evidence is called out explicitly. No saved result is treated as evidence merely because another document refers to it.

## 2. Problem statement and research objectives

The problem is to identify and localize potential quantum-program faults while considering hardware behavior and finite-shot observations, without equating a statistical deviation with a cause. The project README states these objectives:

- Execute supported circuits in a local simulator.
- Compare expected and observed measurement distributions using statistical distances and finite-shot tolerance.
- Localize differences between expected and observed circuit operations.
- Evaluate calibration evidence relevant to circuit components.
- Preserve evidence and return a diagnosis consistent with the available inputs.
- Keep the core simulator usable without a specific IBM backend, account, or live connection.

The ML work asks whether summary features from execution outcomes can classify selected labels. The existing artifacts support a preliminary QMutBench-derived binary experiment and a controlled simulator-only three-class comparison. Neither result establishes an end-to-end, deployable ML system for real hardware.

## 3. Background and related work

The repository frames the technical problem around three interacting observations: a circuit's expected behavior, its observed measurement distribution, and hardware calibration context. Its implementation separates statistical anomaly detection from operation comparison and calibration-based hardware candidate scoring. This separation matters because the same output deviation can be consistent with more than one cause, while matching aggregate distributions do not necessarily erase direct operation or calibration evidence (`README.md`, §§2–5; `src/quantum_simulator/diagnosis_engine.py`).

The project materials reviewed for this report do not contain a scholarly related-work review or a verified external bibliography. Accordingly, this section describes the project's own problem framing and architecture only. No novelty claim or comparison with prior research is made. A literature review remains an evidence gap if the report is intended for academic publication.

## 4. System architecture

The implementation follows this data flow:

```text
Circuit or instruction sequence
               |
               v
       Circuit representation
               |
               v
 Ideal or HardwareModel execution
               |
               v
 Expected/observed outcomes and circuits
       |                       |
       v                       v
 Distribution comparison   Operation comparison
       |                       |
       +-----------+-----------+
                   v
      Relevant calibration evidence
                   |
                   v
    Evidence-preserving diagnosis
```

`QuantumSimulator` performs local execution. `statistics.py` compares distributions. `fault_localizer.py` compares operation sequences and may use a trace to rank suspicious steps. `hardware_fault_localizer.py` evaluates supplied qubit, gate, and edge calibration context. `diagnosis_engine.py` composes the results. The observable-only ML adapter is a separate optional inference boundary; an ML class prediction is not sufficient on its own to establish a fault cause (`src/quantum_simulator/quantum_simulator.py`, `statistics.py`, `fault_localizer.py`, `hardware_fault_localizer.py`, `diagnosis_engine.py`, `ml_diagnosis.py`).

The API levels differ. `diagnose_circuit` runs the supplied circuit ideally and optionally with a `HardwareModel`; it does not take a distinct reference circuit. `diagnose_execution` accepts expected and observed distributions and optional expected and observed circuits, so explicit reference/actual comparison is available at that level (`README.md`, §§9–10; `src/quantum_simulator/diagnosis_engine.py`).

## 5. Simulator and noise-model implementation

### 5.1 Statevector execution

The simulator represents an (n)-qubit pure state as a complex statevector, applies supported gate matrices, and samples measurement outcomes. The package circuit API covers the operations implemented in the gate and simulator layers, including one- and multi-qubit gates, rotations, measurement, reset, and barriers (`src/quantum_simulator/quantum_simulator.py`, `gates.py`, `quantum_circuit.py`; `tests/test_simulator.py`, `tests/test_gates.py`, `tests/test_gate_correctness.py`).

### 5.2 HardwareModel and modeled effects

`HardwareModel` stores backend-independent qubit properties, gate calibrations, connectivity, and calibration metadata. Probability fields are validated in [0, 1], and durations and coherence values are validated as nonnegative (`src/quantum_simulator/hardware_model.py`). A configured or default value is model input; it is not automatically an empirical backend measurement.

The simulator's calibrated gate-error behavior samples whether a matching operation is skipped. Readout error is applied as measurement bit flips. The source also implements a duration-dependent relaxation update: an amplitude-relaxation step based on T1 and additional dephasing based on T2 when positive values and a positive gate duration are supplied (`src/quantum_simulator/quantum_simulator.py::_apply_hardware_relaxation`). These are simulator mechanisms. The generic model does not establish a physical time unit or validate a device calibration pipeline, so the report does not interpret supplied values as measured physical times.

### 5.3 Distribution comparison

`compare_distributions` normalizes the input distributions and computes TVD, Hellinger distance, and Jensen–Shannon divergence. TVD is the primary anomaly measure. When a shot count (N) is supplied, the implementation compares TVD with `max(configured_threshold, 1/sqrt(N))` (`src/quantum_simulator/statistics.py`). This is the project's finite-shot tolerance rule; it is not presented as a calibrated hypothesis test or universal significance criterion.

## 6. Diagnosis and fault-localization workflow

`diagnose_execution` composes the available evidence:

1. If both expected and observed distributions are supplied, compare them and determine whether TVD exceeds the effective threshold.
2. If expected and observed circuits are supplied, compare operation signatures and retain candidates for direct changes such as replacement, insertion, or removal.
3. If a hardware model is supplied, identify calibration candidates and retain those relevant to the circuit.
4. Separate direct program mismatch evidence from trace-only deviations. A trace deviation without an operation mismatch is not treated as proof of a program fault.
5. Return the evidence, missing-evidence details, candidates, and diagnosis summary. When evidence is absent or competing, do not infer a unique cause.

The operation localizer aligns instruction sequences and scores differences (`src/quantum_simulator/fault_localizer.py`). The hardware localizer reports relevant qubit, gate, or edge candidates from the supplied model (`src/quantum_simulator/hardware_fault_localizer.py`). Controlled tests cover program edits on otherwise healthy hardware, hardware-only settings with an unchanged circuit, missing calibration/distributions, and cases with simultaneous program and hardware evidence (`tests/test_diagnosis_engine.py`, `tests/test_fault_localizer.py`, `tests/test_hardware_fault_localizer.py`, `tests/test_stage3_controlled_diagnosis_evaluation.py`).

One API detail affects interpretation: `DiagnosisCategory.AMBIGUOUS` and `DiagnosisCategory.INSUFFICIENT_EVIDENCE` currently have the same enum value, `"AMBIGUOUS / INSUFFICIENT_EVIDENCE"` (`src/quantum_simulator/diagnosis_engine.py`). They are therefore not distinct serialized category labels. The result summary, evidence list, and missing-evidence list describe whether the issue is conflicting evidence or incomplete evidence. The report uses “ambiguous” and “insufficient evidence” as semantic descriptions, not as distinct machine-readable category values.

The saved controlled diagnosis evaluation describes cancellation cases where output distributions match while direct operation and calibration evidence remain. In those cases the engine preserves the remaining evidence and reports the shared ambiguous/insufficient category rather than treating distribution equality alone as a normal diagnosis (`data/qmutbench_ml/final/controlled_diagnosis_evaluation.md/.json`; `tests/test_stage3_controlled_diagnosis_evaluation.py`). These controlled cases demonstrate specific routing behavior; they do not estimate general diagnostic accuracy.

## 7. Datasets and preprocessing

### 7.1 QMutBench-derived binary dataset

The processed table has **10,003 records**: **9,792 mutants** labeled `PROGRAM_FAULT` and **211 originals** labeled `NORMAL`. The saved classifier is a NumPy-only logistic-regression model. Its five inputs are trace deviation summaries: maximum, mean, and median probability differences, deviation concentration, and fraction of high-deviation observations (`data/qmutbench_ml/final/evaluation_report.json`, `dataset_audit.json`). The audit shows record type and label are perfectly aligned: all originals are NORMAL and all mutants are PROGRAM_FAULT. Every original has a zero feature vector; **1,258 mutant records** also have zero features (`dataset_audit.json`, `zero_feature_diagnostic.json`). These are QMutBench-derived record labels, not independent verification that each original is behaviorally correct or each mutant causes an observable fault.

### 7.2 Controlled three-class simulator dataset

The grouped ML comparison uses **270 saved reports** from **30 circuit families**, with **9 reports per family** and **90 examples per class**: `NORMAL`, `PROGRAM_FAULT`, and `HARDWARE_ANOMALY` (`data/qmutbench_ml/final/grouped_model_comparison.md/.json`; source reports in `reports/experiments/full_dataset/`). It uses the five production observable features: TVD, JS divergence, Hellinger distance, observed entropy, and observed maximum probability. Identifiers, family identity, intervention metadata, calibration, diagnosis outputs, trace/localization features, and other post-outcome fields are excluded from model inputs; labels are targets only (`grouped_model_comparison.json`).

These labels describe controlled simulator scenarios. They are not labeled physical hardware events. The grouped comparison excludes cancellation cases from forced three-class classification; the separate evidence-gated diagnosis behavior remains relevant for ambiguous/mixed cases (`grouped_model_comparison.json`, `controlled_diagnosis_evaluation.md`).

## 8. ML methodology and evaluation protocol

### 8.1 QMutBench binary protocols

Two saved binary evaluations use different partitions and must be reported separately.

| Saved evaluation | Partition and selection protocol | Held-out result |
|---|---|---|
| Earlier, untuned (`evaluation_report.json`) | 8,051 train and 1,952 test records; 169 train groups and 42 test groups by original-circuit identity; zero group overlap. The report gives a logistic-regression baseline. | Accuracy 0.829; balanced accuracy 0.913; macro-F1 0.553; NORMAL precision 0.112, recall 1.000; PROGRAM_FAULT precision 1.000, recall 0.826. |
| Later, threshold-tuned (`threshold_tuning_report.json`, `final_project_summary.json`) | Grouped train/validation/test partition: 5,920 / 2,408 / 1,675 records and 126 / 42 / 43 groups, respectively. There is zero pairwise group overlap. Threshold 0.8100000000000002 was selected on validation groups. | Test accuracy 0.877; balanced accuracy 0.937; macro-F1 0.614; NORMAL precision 0.173, recall 1.000; PROGRAM_FAULT precision 1.000, recall 0.874. |

The tuned test confusion matrix, actual rows and predicted columns ordered `PROGRAM_FAULT`, `NORMAL`, is `[[1426, 206], [0, 43]]`. This means 206 of 1,632 held-out mutant-labeled records were predicted NORMAL, while all 43 NORMAL-labeled records were predicted NORMAL (`threshold_tuning_report.json`, `final_project_summary.json`). High accuracy should not obscure the low NORMAL precision or the record-type/feature construction relationship.

The partition difference matters: the later threshold-tuned test set contains 1,675 records, not the earlier 1,952. Scores from these two reports do not share the same held-out rows and are not directly substituted for one another. Neither protocol removes the label-validity limitation.

### 8.2 Family-held-out three-class protocol

The primary controlled three-class protocol is 30-fold leave-one-circuit-family-out GroupKFold. In each outer fold, **29 families / 261 reports** are used for training and **1 family / 9 reports** for testing, with **3 examples per class** in that test family. Each family is held out once, so the outer test folds are mutually disjoint by family. For tuned candidates, 5-fold GroupKFold on the outer-training families selects hyperparameters by mean inner-fold macro-F1. Scaling is fit inside training folds for scaled models; no outer test information is used for preprocessing or tuning. The fixed logistic baseline is unsearched. Candidate model grids are predeclared and fold-level selections are saved (`grouped_model_comparison.json`, “evaluation”; `grouped_model_comparison.md`).

Family-held-out evaluation measures transfer to a circuit family absent from training. That is a stronger leakage control than randomly dividing rows when related scenarios share a family. It does not show transfer to arbitrary programs. Each outer test fold is small, there are only **30** controlled families, outer training sets overlap between folds, and no confidence interval or significance test is provided. The pooled out-of-fold metrics are descriptive for this saved scenario set.

### 8.3 Observable ML adapter

The observable adapter expects the ordered inputs `total_variation_distance`, `js_divergence`, `hellinger_distance`, `observed_entropy_bits`, and `observed_max_probability`, and three classes (`NORMAL`, `PROGRAM_FAULT`, `HARDWARE_ANOMALY`). It validates output probabilities and abstains below its default confidence threshold of **0.60**; predicted fault classes also require independent rule-based localization evidence (`src/quantum_simulator/ml_diagnosis.py`).

The saved binary NPZ models use different trace features and only two classes. The expected portable observable-model JSON file is absent, so the default inference backend is unavailable. Existing adapter tests use mock classifiers; they do not validate real artifact loading or inference (`ml_artifact_integration_audit.md/.json`, `tests/test_ml_diagnosis_integration.py`). The adapter design is implemented, but no compatible trained model is demonstrated.

## 9. Results and quantitative comparisons

### 9.1 Family-held-out three-class results

The table reports saved pooled out-of-fold metrics for all algorithms in the grouped comparison. The class metrics shown for HARDWARE_ANOMALY are precision/recall/F1. The fixed logistic row is the predeclared baseline; the tuned logistic row is a candidate selected by nested grouped CV.

| Model | Accuracy | Balanced accuracy | Macro-F1 | Hardware P/R/F1 |
|---|---:|---:|---:|---:|
| Logistic regression, tuned C | 0.585 | 0.585 | 0.576 | 0.551 / 0.422 / 0.478 |
| RBF-SVM | 0.570 | 0.570 | 0.547 | 0.636 / 0.389 / 0.483 |
| Logistic regression, fixed C=1 | 0.563 | 0.563 | 0.543 | 0.532 / 0.367 / 0.434 |
| Regularized LDA | 0.511 | 0.511 | 0.513 | 0.407 / 0.489 / 0.444 |
| Extra Trees | 0.522 | 0.522 | 0.484 | 0.500 / 0.278 / 0.357 |
| Random Forest | 0.496 | 0.496 | 0.462 | 0.382 / 0.233 / 0.290 |
| Majority dummy | 0.333 | 0.333 | 0.167 | 0.333 / 1.000 / 0.500 |

Source: `data/qmutbench_ml/final/grouped_model_comparison.md`, with detailed folds and metrics in `.json` and saved OOF predictions in `grouped_model_comparison_predictions.csv`. Values are rounded to three decimals from the saved pooled metrics.

For the fixed baseline, per-class recall is **0.900 NORMAL**, **0.422 PROGRAM_FAULT**, and **0.367 HARDWARE_ANOMALY**. Its confusion matrix (actual rows; predicted columns in `NORMAL`, `PROGRAM_FAULT`, `HARDWARE_ANOMALY` order) is `[[81, 0, 9], [32, 38, 20], [45, 12, 33]]`. The tuned logistic candidate's macro-F1 is 0.576; its mean family macro-F1 difference from fixed logistic is 0.037, and it is better/tied/worse on 7/18/5 held-out families, respectively (`grouped_model_comparison.md`). This variation does not support claiming a reliable winner or statistically significant improvement.

The fixed baseline misses 57 of 90 hardware-anomaly examples in pooled OOF predictions; the tuned candidate misses 52 of 90 (`grouped_model_comparison.json`, confusion matrices). This is a substantial negative result for a hardware-diagnosis claim. The saved feature-collision analysis also reports exact five-feature vectors shared across different labels, which makes those cases indistinguishable to classifiers using only this representation (`grouped_model_comparison.md/.json`).

### 9.2 QMutBench-derived binary results

| Metric, later threshold-tuned held-out set | Result |
|---|---:|
| Test records | 1,675 (43 NORMAL; 1,632 PROGRAM_FAULT) |
| Selected threshold | 0.8100000000000002 |
| Accuracy | 0.877 |
| Balanced accuracy | 0.937 |
| Macro-F1 | 0.614 |
| NORMAL precision / recall | 0.173 / 1.000 |
| PROGRAM_FAULT precision / recall | 1.000 / 0.874 |

Source: `data/qmutbench_ml/final/threshold_tuning_report.json` and `final_project_summary.json`. These metrics measure agreement with original-versus-mutant-derived labels in this held-out partition. They do not independently establish behavioral fault detection, and the low NORMAL precision means many predicted NORMAL records are mutant-labeled examples.

### 9.3 Controlled diagnosis and adapter results

The controlled diagnosis artifacts document selected normal, program-fault, readout, gate, coherence, two-qubit, mixed-evidence, cancellation, and robustness scenarios. For example, the stage-three test fixture covers one normal case, five program-fault cases, and four hardware cases, including readout, gate, T1/T2, and two-qubit conditions (`tests/test_stage3_controlled_diagnosis_evaluation.py`). This is designed-case coverage, not a statistical sample of fault prevalence or general accuracy.

The QMutBench validation report records **11 executed and diagnosed cases out of 12**, with one unsupported. It reports zero reliable diagnosis-category-compatible ground-truth cases, so accuracy, precision, recall, F1, and a confusion matrix are unavailable (`reports/benchmarks/qmutbench_validation_report.json`). The focused Bugs4Q report covers **8 selected cases** (3 diagnosed `NO_ANOMALY`, 5 `PROGRAM_FAULT`), but the README explicitly characterizes these as integration results rather than benchmark accuracy (`reports/bugs4q_focused_integration_report.json`; `README.md`, §15).

## 10. Error analysis and fault-localization findings

### 10.1 ML errors and representational limits

On the QMutBench-derived tuned test set, 206 mutant-labeled records were classified as NORMAL and no NORMAL-labeled records were classified as PROGRAM_FAULT. This asymmetric error pattern explains the low NORMAL precision despite perfect NORMAL recall. The label audit and zero-feature analysis show a systematic relationship between record type and inputs: all originals have zero features, but a substantial number of mutants do too (`final_project_summary.json`, `dataset_audit.json`, `zero_feature_diagnostic.json`). These results are compatible with learning properties of the dataset construction and do not isolate a general mutation signature.

For the three-class experiment, the fixed baseline's pooled confusion matrix shows confusion between program and hardware classes as well as missed hardware cases. The tuned model raises pooled macro-F1 modestly, but its family-level results are better than fixed logistic on only 7 families, tied on 18, and worse on 5. Exact cross-class feature collisions further limit identifiability from the five outcome summaries alone. The results support additional independent evidence as a need; they do not support a claim that a more complex model resolves the causal distinction.

### 10.2 Operation and hardware localization

Operation-level localization is supported when expected and observed instruction sequences are supplied. The tests verify examples such as gate replacement, insertion, deletion, and parameter changes, and confirm that direct mismatches yield program candidates (`tests/test_diagnosis_engine.py`, `tests/test_fault_localizer.py`, `tests/test_stage3_controlled_diagnosis_evaluation.py`). This is candidate localization against a supplied reference, not automatic discovery of an unknown correct program.

Hardware localization uses the supplied model to identify relevant qubit, gate, or connectivity evidence. Controlled hardware-only tests verify that calibration candidates can be reported while direct program evidence remains empty; trace-deviation candidates may be retained as diagnostics but are not elevated to direct program-fault evidence (`tests/test_stage3_controlled_diagnosis_evaluation.py`, `tests/test_hardware_fault_localizer.py`). The system cannot infer that a calibration parameter reflects a real device unless the caller supplies verified backend data.

The diagnosis engine preserves conflicting evidence in cancellation and mixed-cause cases. Missing hardware calibration is reported as missing evidence, and an output anomaly without program or hardware evidence does not justify assigning either fault cause (`tests/test_diagnosis_engine.py`, `controlled_diagnosis_evaluation.md`). Because AMBIGUOUS and INSUFFICIENT_EVIDENCE share the same enum string, downstream users must inspect summaries and evidence fields to understand why certainty was withheld.

## 11. Discussion

The project's strongest contribution in the inspected evidence is the separation of anomaly detection from causal attribution. A distribution difference is evaluated statistically, while program localization depends on operation comparison and hardware localization depends on relevant calibration input. This design allows the diagnosis to retain uncertainty rather than treat output disagreement alone as proof of a program fault or hardware fault.

The family-held-out ML protocol reduces a common evaluation risk: related scenarios from one circuit family are not split between training and test within a fold. Its measured performance remains limited. The fixed baseline has macro-F1 0.543 and hardware recall 0.367; the tuned candidate has macro-F1 0.576 and hardware recall 0.422 but inconsistent family-level gains. These results do not support reliable general three-class diagnosis. No significance test was reported, and the size and construction of the scenario set constrain inference.

The QMutBench-derived binary result addresses a different target with a different feature schema. Its threshold-tuned score should not be pooled with or compared as if it were the three-class experiment. The labels are record-type-derived, the data are imbalanced, all original records have zero features, and NORMAL precision remains low. Accordingly, the binary result is best presented as a dataset-specific preliminary classification result.

The observable ML adapter is not a deployed inference system in the current artifact state. Its schema checks, confidence abstention, and localization gate are implemented, but the available NPZ models belong to an incompatible binary trace-feature task. No compatible three-class model was loaded or evaluated. Any claim of model-assisted production diagnosis must be qualified accordingly.

## 12. Limitations and threats to validity

### 12.1 Construct and label validity

The QMutBench-derived binary labels encode original-versus-mutant record type and are not independent behavioral ground truth. Some mutants may survive, be equivalent, or not change measured behavior; the saved report does not establish a verified outcome for every record. The five trace features are zero for every original and for many mutants, creating a strong relationship between features and record type (`dataset_audit.json`, `training_data_label_audit.md/.json`, `zero_feature_diagnostic.json`).

The controlled three-class labels describe assigned simulator interventions. They do not establish hardware fault categories observed in real devices. Exact feature collisions across classes show that some different labels share the same input vector.

### 12.2 Evaluation validity

The three-class comparison has only 30 circuit families and 3 records per class in each held-out family. Outer training partitions overlap, family scores are coarse, and no confidence intervals or significance tests are reported. The repeated-feature and ablation analyses also use small, overlapping family-held-out splits; they are exploratory and do not establish a consistent feature improvement (`repeated_feature_evaluation.md`, `controlled_feature_ablation.md`).

The QMutBench binary reports use two different saved test partitions. These results must remain separate. Their strong class imbalance and record-type construction labels constrain the interpretation of aggregate metrics.

### 12.3 Simulator, hardware, and deployment scope

The simulator implements selected modeled effects and the `HardwareModel` can hold T1/T2 values, gate errors, readout errors, and connectivity. This is not a general physical noise model. Although T1/T2 behavior is represented in code and included in some prior controlled test cases, the proposed first-wave design explicitly marked T1/T2 `BLOCKED_AND_EXCLUDED`, and its time-unit convention was not verified (`first_wave_spec_v3_review.md`, `first_wave_execution_preflight.md`). No T1/T2 result from that proposed wave exists.

There is no verified real-device execution or calibration evidence. The IBM report is an offline fixture; it records `real_hardware_executed: false` and `real_ibm_status: NOT_EXECUTED` (`reports/ibm_validation_report.json`). No physical-hardware validation claim is supported.

The saved binary NPZ models are not compatible with the observable adapter's five-feature, three-class schema. The expected portable JSON model is absent; actual model inference is therefore unavailable through the default backend (`ml_artifact_integration_audit.md/.json`). Mock-classifier integration tests do not close this gap.

The proposed **240-report first-wave experiment was blocked and not executed**. The v3 specification review marked the design ready for owner approval, not execution. The saved preflight decision is `BLOCKED / STOP`, with no reports generated. It identifies two unresolved execution/data-integrity requirements: recording realized RX gate skips in the same sampled execution stream, and specifying safe isolated generation, checkpoint/resume, duplicate handling, failure/resource stops, and reconciliation (`first_wave_spec_v3_review.md`, `first_wave_execution_preflight.md/.json`). Planned counts or seeds are not observed results.

### 12.4 Reproducibility and evidence completeness

The quantitative results in this report are transcribed from saved reports and prediction artifacts; experiments and tests were not rerun for this final report. Repository documents contain historical suite totals of 280, 89, and 96 passed tests from different snapshots (`README.md`, `reports/project_status.json`, `reports/reproducibility_report.json`). They are not combined here and do not constitute a fresh test run. Full row-by-row CSV replay, saved-model prediction replay, and cross-environment bitwise reproduction were not performed.

## 13. Conclusion and future work

The project implements a local statevector simulator, configurable simulator-side hardware effects, statistical distribution comparison, operation-level program candidates, and calibration-informed hardware candidates. The rule-based diagnosis workflow can preserve direct evidence and withhold a unique cause when evidence is missing or conflicting. Controlled tests demonstrate selected local scenarios, including program edits, synthetic hardware conditions, cancellation, and ambiguity.

The ML findings are preliminary and protocol-specific. In the QMutBench-derived binary evaluation, threshold 0.81 yields held-out macro-F1 0.614, but the labels are record-type-derived and NORMAL precision is 0.173. In the controlled three-class family-held-out evaluation, the fixed logistic baseline has macro-F1 0.543 and hardware recall 0.367. The tuned logistic candidate reaches macro-F1 0.576 and hardware recall 0.422, but its gains are inconsistent across families. These results do not support reliable general three-class diagnosis.

There is no compatible trained observable three-class artifact, no real-device validation, and no executed 240-report first wave. T1/T2 relaxation appears in simulator implementation and prior controlled cases, but is excluded from the first-wave design and is not supported as a physically calibrated effect by the evidence reviewed. The defensible project conclusion is a tested local simulator and evidence-based diagnosis workflow accompanied by exploratory ML baselines with substantial label, data, and generalization limitations.

Future work, if stronger research claims are desired, includes independent behavioral labels for original and mutant programs; more held-out circuit families with predeclared analysis and uncertainty estimates; training and provenance-checking a model for the adapter's exact observable schema; validating physical units before relaxation claims; and separately authorized live-device studies. If the first-wave simulator experiment is revived, its saved preflight blockers must be resolved before collection. None of these future activities is represented as completed here.

## 14. References and evidence sources

No external scholarly sources or formal related-work bibliography were present in the project materials reviewed. To avoid fabricating references, this report cites repository sources and saved evidence only. The following are the principal sources for claims in the report:

1. Project objectives, architecture, terminology, and stated limitations: `README.md`.
2. Statevector execution, gate handling, and modeled hardware effects: `src/quantum_simulator/quantum_simulator.py`, `src/quantum_simulator/gates.py`, `src/quantum_simulator/quantum_circuit.py`, `src/quantum_simulator/hardware_model.py`.
3. Statistics, program localization, hardware localization, diagnosis, and observable adapter: `src/quantum_simulator/statistics.py`, `fault_localizer.py`, `hardware_fault_localizer.py`, `diagnosis_engine.py`, `ml_diagnosis.py`, `observable_model_adapter.py` (all under `src/quantum_simulator/`).
4. Relevant software tests: `tests/test_simulator.py`, `test_statistics.py`, `test_diagnosis_engine.py`, `test_fault_localizer.py`, `test_hardware_fault_localizer.py`, `test_stage3_controlled_diagnosis_evaluation.py`, `test_ml_diagnosis_integration.py`.
5. QMutBench-derived ML data and evaluation: `data/qmutbench_ml/final/evaluation_report.json`, `threshold_tuning_report.json`, `final_project_summary.json`, `dataset_audit.json`, `zero_feature_diagnostic.json`, `test_predictions.csv`, `tuned_test_predictions.csv`, `numpy_logistic_model.npz`, `numpy_logistic_model_tuned.npz`.
6. Controlled three-class ML evaluation and saved predictions: `data/qmutbench_ml/final/grouped_model_comparison.md`, `grouped_model_comparison.json`, `grouped_model_comparison_predictions.csv`; source reports and manifest in `reports/experiments/full_dataset/`.
7. Controlled diagnosis, label, and ML artifact audits: `data/qmutbench_ml/final/controlled_diagnosis_evaluation.md/.json`, `training_data_label_audit.md/.json`, `ml_artifact_integration_audit.md/.json`.
8. First-wave proposal and disposition: `data/qmutbench_ml/final/first_wave_experiment_spec_v3.md/.json`, `first_wave_spec_v3_review.md/.json`, and root `first_wave_execution_preflight.md/.json`.
9. Benchmark and IBM adapter reports: `reports/benchmarks/qmutbench_validation_report.json`, `reports/bugs4q_focused_integration_report.json`, and `reports/ibm_validation_report.json`.
10. Historical project/test status: `reports/project_status.json` and `reports/reproducibility_report.json`.

### Remaining evidence gaps

- No live backend execution, physical-device calibration capture, or real-hardware performance evaluation.
- No independent behavioral ground truth for every QMutBench-derived original/mutant label; the benchmark adapter report also lacks diagnosis-category-compatible ground truth for its evaluated cases.
- No compatible trained three-class observable ML artifact or real artifact inference result.
- No executed 240-report first wave or realized per-shot RX skip counts.
- No T1/T2 coverage in that first-wave design and no validated physical time-unit contract for interpreting relaxation parameters.
- No confidence interval/significance analysis or broad generalization guarantee for the controlled three-class evaluation.
- No fresh test run or full independent replay of all saved predictions during this review.
- No external scholarly related-work sources were supplied or verified; the bibliography is therefore limited to repository evidence.

## 15. Finalization milestone and current-tree validation (2026-10-15)

This section supersedes older statements elsewhere in this report that say no fresh validation was performed. The repository contained substantial pre-existing user changes when this milestone began; these were preserved. No source implementation changes were made for this milestone because the frozen failure below is explained by the fixed method and finite-shot sampling, and the request prohibits tuning that acceptance set.

### 15.1 Bounded validation

| Validation | Exact command / scope | Result |
|---|---|---|
| Focused frozen evaluator | `.\.venv\Scripts\python.exe -m pytest tests/test_reference_diagnosis_generalization.py -q --tb=short --basetemp .t-finalization-focused` | **1 failed, 1 passed** in 66.59s. The unchanged `test_heldout_generalization_outcomes_and_targets` fails on the seven frozen outcomes; the forensic explanation test passes. |
| Bounded repository suite | `.\.venv\Scripts\python.exe -m pytest tests -q --tb=short -k "not bounded_recipe_comparison_120_templates_and_finite_shot_subset and not independent_calibration_fixture_and_permutation_comparison" --basetemp .t-finalization-suite` | **794 passed, 1 failed, 0 skipped, 2 deselected** (797 collected) in 259.71s. The sole failure is the same frozen 7/45 test. The two deselected tests remain excluded under the registered finite-shot NO-GO decision. |
| Whitespace / JSON | `git diff --check`; Python `json.loads` on modified tracked JSON files | `git diff --check` exited successfully (Git emitted only existing LF-to-CRLF normalization warnings); `reports/experiment_schema.json` and `reports/project_status.json` parsed successfully before the final status update. |

No claim is made that the full suite passes. Historical test totals elsewhere in the repository are snapshots from earlier trees and should not be substituted for this run.

### 15.2 Frozen 45-case diagnosis evaluation

The test evaluates 45 constructed held-out circuit cases through `diagnose_circuit`, comparing detection, category, abstention, and source/hardware localization against frozen expected values. Ground-truth scenario names and mutation metadata are withheld from that API call. The acceptance expectations, cases, threshold, and selection rules were not changed. Seven of 45 detection/classification/abstention outcomes still disagree; source localization targets pass and none of these seven cases expects hardware localization.

| Frozen case | Exact ideal TVD | Registered sampled TVD | Observed explanation |
|---|---:|---:|---|
| `qft_like_3q_program_fault_11` | 0.0078559 | 0.01953125 | Expected positive, but signal is below the fixed 0.05 threshold and the 0.0625 256-shot guard; source target is localized. |
| `qft_like_3q_program_fault_29` | 0.0012597 | 0.00000000 | Same sub-threshold expected-positive limitation; source target is localized. |
| `qft_like_3q_program_fault_47` | 0.0112994 | 0.01953125 | Same sub-threshold expected-positive limitation; source target is localized. |
| `ring_cluster_4q_correct_11` | 0 | 0.0859375 | Finite-shot false positive for identical ideal output distributions. |
| `ring_cluster_4q_correct_29` | 0 | 0.1250000 | Finite-shot false positive for identical ideal output distributions. |
| `ring_cluster_4q_equivalent_11` | 0 | 0.0859375 | Commuting CZ reorder preserves the exact output distribution; sampling false positive plus syntactic operation candidates produces the observed ambiguous diagnosis. |
| `ring_cluster_4q_equivalent_29` | 0 | 0.1250000 | Same observational-equivalence and finite-shot false-positive mechanism. |

These explanations are reproducible simulator/method diagnostics, not independent evidence of physical causation. They show both sensitivity limits for weak expected positives and specificity limits for identical-distribution controls, consistent with the separate registered calibration NO-GO finding. The forensic regression protects this characterization; it does not change the failing acceptance test. Evidence: `reports/experiments/heldout_diagnosis_root_cause_20261014.json` and `tests/test_reference_diagnosis_generalization.py`.

### 15.3 Dataset provenance and evidence boundaries

The saved QMutBench-derived original/mutant labels are not independent behavioral ground truth, and no independently adjudicated lineage is available for the historical 270-row simulator ML dataset. Synthetic fixtures establish deterministic software behavior only. They do not establish externally representative distributions or real-device causation.

Simulator output, controlled interventions, and calibration parameters supplied to `HardwareModel` are not physical-hardware observations. IBM integration is optional and offline-tested; no live backend execution, retrieved device calibration, or physical fault attribution is demonstrated. Generic calibration freshness is policy-driven: a caller must supply a positive maximum age, acquisition timestamps remain caller-reported, snapshot `created_at` is distinct, and reported provenance is not independently authenticated. IBM preflight uses a separate backend-specific policy.

### 15.4 Supported reproduction entry points

From the repository root, with the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe demo.py
.\.venv\Scripts\python.exe -m pytest tests -q --tb=short -k "not bounded_recipe_comparison_120_templates_and_finite_shot_subset and not independent_calibration_fixture_and_permutation_comparison" --basetemp .t-finalization-suite
```

The suite command is expected to report the current frozen failure documented above; use a new workspace-local basetemp if `.t-finalization-suite` already exists. Core package use and circuit entry points are described in the README. Optional IBM and benchmark integrations require their respective external data, credentials, or dependencies and are not part of the validated offline core path.

### 15.5 Claims matrix

| Proposed claim | Classification | Basis / limit |
|---|---|---|
| Supported circuit building, statevector execution, gate/measurement operations, and finite-shot counts work for covered contracts. | **Demonstrated** | Current bounded software tests and simulator source; coverage is finite and does not imply all possible circuits. |
| Operation candidates can be localized when a reference operation sequence is supplied. | **Demonstrated** | Focused localization and diagnosis tests; this is not automatic recovery of unknown intended source. |
| Calibration-linked candidates and evidence-based abstention work on controlled local scenarios. | **Demonstrated only on synthetic fixtures** | Tests use supplied models/interventions, not independently measured hardware faults. |
| Saved QMutBench-derived binary metrics reflect original/mutant record labels. | **Preliminary** | Grouped saved partitions exist, but labels are record-type-derived and not independent behavioral truth. |
| Historical 270-row three-class ML scores establish independent generalization. | **Unsupported** | Historical rows lack complete independent lineage and ground truth; later provenance checks reject the data for current fitting. |
| A compatible trained observable three-class classifier is integrated and validated. | **Unsupported** | Adapter contract and mock routing are tested; default compatible model artifact and real artifact inference are absent. |
| Current finite-shot detector meets the frozen 45-case acceptance contract. | **Blocked** | Current bounded run retains the same 7/45 outcome mismatch; two registered calibration comparison tests remain deselected under NO-GO. |
| The workflow identifies physical hardware faults or has been validated on an IBM device. | **Unsupported** | No live job, physical calibration capture, or independent device ground truth. |
| A caller-reported calibration timestamp proves calibration freshness/authenticity. | **Unsupported** | Freshness needs an explicit policy; source provenance is not independently authenticated. |
| The proposed finite-shot campaign is ready to execute. | **Blocked** | Existing design disposition is NO-GO and execution is not authorized. |

### 15.6 Final verdict and stop disposition

**Software readiness:** core simulator, circuit API, diagnosis/localization interfaces, and uncertainty routing are implemented and exercised by a broad bounded suite, with **794 passing tests**. The tree is not test-clean: one frozen diagnosis acceptance test fails. This is a release blocker for any release criterion requiring all frozen acceptance tests to pass or claiming that the 45-case detector acceptance is met. The smallest necessary next action is a separately reviewed, independent development/validation protocol that can decide whether a new detector criterion is warranted without fitting or tuning on these frozen cases; until then preserve the failure and do not claim general diagnosis acceptance.

**Research validity:** no independent ML generalization or reliable physical-hardware fault-diagnosis claim is established. The historic ML numbers are descriptive records of their saved datasets/protocols; present them as historical and preliminary, with their lineage and label limits. The simulator and controlled fixture results support software behavior only.

**Release blockers:** the frozen 7/45 acceptance failure blocks a clean release against that test contract. The proposed finite-shot study remains NO-GO and unauthorized.

**Future research:** independent behavioral labels and lineage; an independently sampled family-held-out evaluation with predeclared criteria and uncertainty analysis; a compatible observable model evaluated only after data eligibility is established; authenticated calibration provenance and physical-device validation; and a reviewed redesign of finite-shot calibration. No new backlog item was selected or launched during this finalization pass.

The finalization work is documented and stopped here. The milestone is **not complete as a clean release** while the frozen acceptance blocker remains; the research conclusions are explicitly bounded by the evidence above.

## 16. Next independent research phase

The bounded follow-up design, candidate hypotheses, ML evidence protocol, provenance gates, and stop conditions are in the [independent research handoff](reports/independent_research_handoff_20261015.md). That document is a planning artifact only; it does not authorize new tests, model fitting, simulation campaigns, or hardware execution.

## 17. QMutBench bounded ingestion update (2026-10-15)

This dated update records a later, separately authorized data-integration milestone; it does not revise the historical ML results or claims above. The official QMutBench metadata CSV was acquired from the [published endpoint](https://enautmendi.github.io/QMutBench/data/all_data.csv), pinned to repository commit [01f5e7700c051fe715fc368b307deaa7e6869ba8](https://github.com/EnautMendi/QMutBench/commit/01f5e7700c051fe715fc368b307deaa7e6869ba8), with SHA-256 3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778. The official source has 723,079 metadata records and 375 Origin_program IDs. The previous local CSV hash/size discrepancy is explained by LF versus CRLF line endings: all rows and columns match when parsed.

A separate streaming, resumable pipeline and provenance manifest now preserve raw source files, normalized parent/mutant records, simulated observations, failures, and checkpoints independently. Its 12-record smoke test used one GHZ original group: 12 records parsed and simulated, four had changed exact computational-basis distributions, eight were indistinguishable in that observable, and none failed. These are simulator-only pipeline observations, not independent fault labels or an ML evaluation. The QMutBench Killed field remains raw metadata; it is not mapped to the project's NORMAL, PROGRAM_FAULT, or HARDWARE_FAULT labels.

The official original-to-mutant grouping provides a source-parent relation, but does not attest to 375 independent recipe roots; QMutBench also documents MQT Bench as an upstream source. The ML preflight is blocked with zero eligible independent roots, no independently adjudicated behavioral labels, no preregistered minimum family count, and no locked train/calibration/test partitions. No model was fit and no generalization or hardware claim is added.

The pipeline details and artifacts are in [the QMutBench data-pipeline report](reports/qmutbench_data_pipeline_20261015.md). Focused tests pass (9 passed). The requested bounded suite, with both finite-shot NO-GO tests deselected, reports 803 passed, 1 failed, and 2 deselected. The sole failure is the same frozen 45-case acceptance test with the same seven mismatches; the frozen fixture, thresholds, outcomes, and test remain unchanged.

### 17.1 QMutBench lineage audit and separate detection eligibility (2026-10-15)

A deterministic bounded audit now recomputes the pinned-manifest hashes for all 24 locally staged QASM artifacts and the official metadata CSV; its results are in `reports/qmutbench_lineage_audit_20261015.json`. This verifies current local bytes against the local manifest. It does not authenticate the upstream commit (reported unsigned), establish complete corpus custody, or prove independent recipe roots. In the bounded smoke bundle, the original ID and QASM hash establish circuit identity, and `Origin_program` plus the documented directory relationship establishes a parent association. The QMutBench documentation's MQT Bench link is upstream-documented but no per-origin MQT recipe/ancestor map is locally available. Generator/run ancestry is not attested. The smoke observations have a recorded ideal/no-noise simulator configuration, but are simulator outputs rather than labels. `Killed` remains raw metadata.

For the distinct output-only program-mutation detection question, the result is **zero independently eligible recipe roots** and no evaluation manifest. One locally represented GHZ group cannot form family-separated partitions; full root reconciliation, complete prior-inventory reconciliation, independent behavioral labels with blinded adjudication, and compatible independently sourced observations are absent. This does not alter or combine with the three-way normal/program-fault/hardware-fault diagnosis task. The frozen 45-case seven-mismatch result remains unchanged.

The focused lineage tests pass (3 passed). The requested bounded suite reports 805 passed, 2 failed, and 2 deselected. The frozen 45-case acceptance test still has the same seven mismatches. The additional failure, `test_incomplete_file_is_invalid_and_preserved_for_recovery`, raises `FileNotFoundError` when its recovery step expects a nested parent directory under the new pytest basetemp. No frozen artifact, threshold, or criterion changed.

Recovery-test follow-up: the failure came from the test combining the long production output path with a fresh nested pytest temporary path beyond Windows' effective path limit. The test now uses a short test-owned output root while retaining all invalid-file and byte-preservation checks, and the atomic writer uses a short temporary staging prefix. Recovery plus lineage focused tests pass (11 passed). The bounded suite now reports 807 passed, 1 failed, and 2 deselected; its sole failure is the frozen 45-case acceptance test with the same seven mismatches. Zero eligible QMutBench recipe roots remains the lineage-audit decision.
