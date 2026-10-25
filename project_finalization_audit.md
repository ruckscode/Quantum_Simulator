# Project Finalization Audit

**Project:** Hardware-Aware Data-Driven Testing and Fault Localization for Quantum Programs  
**Audit date:** 2026-10-12  
**Scope:** Read-only review of source, tests, reports, datasets, manifests, predictions, and saved model artifacts. No simulations, training, tests, package installation, or production changes were performed for this audit.

## Decision

**FINALIZE WITH EXISTING EVIDENCE**

The project can be finalized honestly if its claims stay within the implemented simulator, evidence-based diagnosis, mock-tested ML adapter contract, binary QMutBench-derived classification experiment, and controlled simulator-only ML comparisons. The saved evidence does not support a real-hardware claim, a production ML inference claim, or general hardware-fault detection performance. The proposed 240-report first wave is not required to close the original project at that scope. It would add a new, narrow simulator study; its latest preflight is blocked and it should remain unexecuted unless the project is deliberately expanded to make that study a claim.

## A. Implemented and tested functionality

- The packaged simulator and circuit API implement statevector execution, supported gates, measurement and finite-shot sampling. The `HardwareModel` provides configurable readout, gate, coherence and connectivity properties. Statistical comparison includes total variation, Jensen-Shannon and Hellinger measures and a finite-shot guard.
- The rule-based diagnosis path (`diagnose_execution`) combines distribution anomaly status with direct operation mismatch evidence and relevant calibration-backed hardware candidates. It can return `NO_ANOMALY`, `PROGRAM_FAULT`, `HARDWARE_ANOMALY`, `AMBIGUOUS` or `INSUFFICIENT_EVIDENCE`. Trace-only deviations are deliberately not treated as proof of a program fault. Controlled tests cover normal, program edits, readout/gate/coherence hardware interventions, localization, missing evidence and mixed evidence.
- `diagnose_circuit` runs a supplied circuit ideally and optionally with a hardware model. It does not accept a distinct reference circuit. Direct expected/observed circuit comparison is supported by the lower-level `diagnose_execution` API.
- The observable-only ML integration extracts five distribution features, validates classifier schema/probabilities, abstains below confidence 0.60, and requires independent rule-based localization support before a fault label becomes a definitive diagnosis. The default backend intentionally raises `ModelBackendUnavailable` because no compatible trained artifact is present. Integration tests use mock classifiers; no real model inference is verified.
- Bugs4Q/QMutBench adapters and offline IBM integration are implemented. Checked-in evidence documents selected adapter cases and fixtures; these are not benchmark-ground-truth accuracy evaluations. README says the latest full test suite had 280 passing tests with a writable temporary directory; older status/reproducibility reports give 89 and 96 tests, respectively. Those are historical snapshots, not a single current audit rerun.

## B. ML evidence and exact protocols

### Best-supported current baseline

For the project's five production observables and three controlled simulator labels, the strongest defensible baseline is the **fixed StandardScaler + logistic regression (C=1, max_iter=2000)** in `grouped_model_comparison`. It is the predeclared fixed baseline, not the highest-scoring tuned candidate. Across 30 mutually exclusive leave-one-circuit-family-out outer folds, its pooled out-of-fold accuracy/balanced accuracy are 0.563/0.563 and macro-F1 is 0.543. Class recall is NORMAL 0.900, PROGRAM_FAULT 0.422, HARDWARE_ANOMALY 0.367 (90 examples per class). The majority dummy macro-F1 is 0.167. This is descriptive evidence on 270 controlled local simulator reports from 30 families, not a validated general detector.

The nested comparison's tuned logistic model ranks highest descriptively: macro-F1 0.576, balanced accuracy 0.585 and hardware recall 0.422. Its mean family macro-F1 gain over fixed logistic is 0.037, but it beats fixed logistic on only 7 of 30 families, ties on 18, and loses on 5. Treat this as a candidate comparison, not a selected or validated winner. Each outer test fold has only 9 rows (3/class); there are exact conflicting feature vectors, only 30 families, overlapping training sets among outer folds, and no significance test.

### Other saved evaluations

| Artifact / question | Dataset and exact protocol | Supported result and limit |
|---|---|---|
| `evaluation_report.json`, `threshold_tuning_report.json`, `final_project_summary.json` | 10,003 QMutBench-derived records; five trace-difference features; 8,051 train and 1,952 test records, grouped by original-circuit identity (169 train groups, 42 test groups, zero overlap). The later threshold-tuned report uses a validation-selected threshold 0.81 and reports a held-out set of 1,675 records (43 NORMAL, 1,632 PROGRAM_FAULT); prediction CSVs and saved NumPy model exist. | Tuned held-out accuracy 0.877, balanced accuracy 0.937, macro-F1 0.614; NORMAL precision 0.173/recall 1.0, PROGRAM_FAULT precision 1.0/recall 0.874. The earlier untuned report has 1,952 test records and different metrics. Preserve this distinction; do not mix protocols. Labels are record-type construction labels: all 211 originals are NORMAL and all 9,792 mutants PROGRAM_FAULT. All 211 originals have zero features, and 1,258 mutant rows also have zero features. This supports only a preliminary, imbalanced dataset-specific binary evaluation, not independent behavioral truth. |
| `controlled_diagnosis_evaluation.*` | Deterministic hand-designed local cases evaluated by rule-based diagnosis, not ML. | Exercises decision paths and selected localization scenarios; case coverage is not a population estimate. |
| `grouped_model_comparison.*` | 270 saved simulator reports; five observable-only features; labels used only as targets; 30 families, balanced 90/class. Leave-one-family-out outer GroupKFold (30 folds), 5-fold GroupKFold inner CV on outer training families; inner macro-F1 selects predeclared model grids; preprocessing fit within training folds. Fixed logistic is unsearched. | Best reproducible broad ML baseline and protocol described above. Saved OOF predictions, metrics, fold selections, collision audit and reports exist. Reproducible from saved report artifacts and code/environment in principle; this audit did not rerun it. |
| `controlled_three_class_feasibility.*`, `expected_mode_shift_evaluation.*`, `repeated_feature_evaluation.*`, `controlled_feature_ablation.*` | Small family-grouped, controlled simulator evaluations with explicit prediction artifacts. The feasibility split uses 18 train / 6 validation / 6 test families (54 test rows, 18/class); repeated comparison uses 50 overlapping six-family test folds; feature ablations compare fixed configurations. | These reveal weak/unstable class separation, low hardware recall, overlap/collisions, and no reliable feature gain. Repeated folds are dependent; results are exploratory. They do not replace the single nested grouped comparison or show hardware performance. |

The 270-row report set and ML predictions are present. The two `.npz` models are also present, but both implement the incompatible five-feature binary trace schema (`PROGRAM_FAULT`, `NORMAL`). They cannot serve the observable-only three-class adapter. The expected `src/quantum_simulator/models/observable_logistic_regression.json` is absent. Do not infer trained model availability from reports that describe the adapter or from the mere presence of an NPZ file.

## C. Scientific limitations versus correctness defects

### Scientific or scope limitations

- QMutBench labels encode original-versus-mutant record type, are highly imbalanced, and do not independently establish whether behavior is faulty. Zero features occur for every original and many mutants. Reported classification can exploit dataset construction and does not establish general program fault detection.
- Three-class ML data are controlled simulator interventions, with 30 fixed families and three repetitions per class/family. Features can collide across labels and do not uniquely identify the cause. Hardware-anomaly recall for the best nested tuned candidate is 0.422. Small per-family test folds, limited mechanisms and no uncertainty/significance analysis constrain inference.
- Synthetic `HardwareModel` channels and calibration values are not physical device observations or calibrated backend evidence. There is no real-hardware validation. The simulator is not a general physical noise model.
- The proposed first wave excludes T1/T2, covers only specified readout flips and RX operation skips and combinations, and treats its four-family test as exploratory. Its results could not support broad hardware claims.

### Verified implementation/data-integrity limitations (not demonstrated diagnosis misclassification defects)

- Production inference is unavailable without an explicitly supplied compatible classifier; the saved binary NPZ artifacts are intentionally incompatible with the three-class observable adapter. Existing mock tests do not establish real-artifact loading or inference.
- The sampler's hardware gate-error path samples an operation skip but the ordinary `run_shots` report does not expose per-shot `gate_failed` events. The trace path captures gate failure separately, not for the sampled report stream. This is a real observability/data-integrity blocker for the first-wave requirement to report realized skips, not by itself a defect in diagnosis results.
- The v3 wave preflight also found no approved resumable/atomic, duplicate-safe, bounded execution protocol. Existing bulk paths can rerun/overwrite or delete reports; they must not be used for that wave as-is.
- No reproducible, concrete production misclassification or simulator correctness bug was established by the reviewed saved audits. The audit is not a fresh full test run, so it cannot certify the current checkout beyond the checked-in test/report evidence.

## D/E. Essential claims, deferrals, and the 240-report proposal

Essential to finishing the original project honestly: describe implemented simulator and circuit behavior; explain the evidence-gated rule diagnosis and its ambiguity/insufficient-evidence behavior; report controlled offline validation without implying physical hardware; report the QMutBench-derived binary result with its exact label and split caveats; and state that the production ML adapter has no compatible model artifact.

Can be deferred as future research: the 240-report first wave, real IBM execution, independent behavioral relabeling, broader circuit-family evidence, additional feature/model search, T1/T2/unit semantics, and improving/exporting a production three-class ML model. These are needed only if making expanded performance or deployment claims. A direct reference-circuit parameter in the high-level wrapper is also future work; the lower-level comparison exists.

**The 240-report first wave is not necessary to finish this project honestly at the scope above.** Its latest v3 review says READY FOR OWNER APPROVAL, not authorization. The saved execution preflight says BLOCKED / STOP, with no reports generated. B1 requires same-shot RX skip event capture or an owner-approved change to count requirements. B2 requires a safe isolated runner protocol with atomic writes, status/checkpoint/resume/duplicate handling, bounded retries/resource stop, and final reconciliation. Those blockers are relevant only if choosing to collect that new wave. Existing grouped simulator evidence and controlled rule-diagnosis tests already allow a bounded, transparent project conclusion; the wave would not supply real-hardware evidence or cure QMutBench label validity.

## F. Minimum remaining tasks and stop conditions

1. **Adopt the claim boundary in the final project write-up.** Say that the ML study is preliminary and simulator/dataset-specific, that the three-class ML comparison is weak, and that no real-hardware validation or compatible production ML artifact exists. **Stop when** every summary, abstract and conclusion uses those boundaries and no result is described as physical hardware or deployed ML inference.
2. **Use the audit's evidence inventory as the reproducibility index.** Link each headline result to the exact report, predictions, manifest/model file and split protocol; distinguish the earlier 1,952-row binary evaluation from the later validation-threshold / 1,675-row test evaluation. **Stop when** each cited result points to an existing artifact and its protocol/label caveat is adjacent.
3. **Preserve the 240-report wave as deferred/blocked.** Do not generate reports. If later promoted to a project requirement, first resolve B1 and B2 in a separately reviewed plan and define new claims; do not infer realized skips from bitstrings or trace samples. **Stop now** with no experiment, runner, or code change.
4. **Close the project at the stated evidence boundary.** Keep source and historical artifacts untouched; do not connect the binary NPZ artifact to the observable adapter. **Stop when** project status calls QMutBench accuracy dataset-specific, controlled ML simulator-only, and hardware/live IBM phases partial/unavailable as appropriate.

No task in this audit requires another experiment, training run, package change, or source edit.

## Artifact inventory (verified present unless marked otherwise)

| Area | Evidence reviewed |
|---|---|
| Simulator / diagnosis | `src/quantum_simulator/quantum_simulator.py`, `hardware_model.py`, `diagnosis_engine.py`, `fault_localizer.py`, `hardware_fault_localizer.py`, `ml_diagnosis.py`, `observable_model_adapter.py`; relevant tests include `tests/test_simulator.py`, `test_diagnosis_engine.py`, `test_ml_diagnosis_integration.py`, `test_stage3_controlled_diagnosis_evaluation.py`, hardware/localizer and gate tests. |
| Project goals/status | `README.md`, `reports/project_status.json`, `reports/reproducibility_report.json`, `data/qmutbench_ml/final/FINAL_PROJECT_REPORT.md`. README states the original goals, future work and latest 280-pass suite snapshot; status records Bugs4Q and live IBM phases as partial. |
| QMutBench derived data | `data/qmutbench_ml/qmutbench_results.csv`, `qmutbench_manifest.json`, `qmutbench_failures.csv`; `final/dataset_audit.json`, `training_data_label_audit.*`, `zero_feature_diagnostic.json`, `evaluation_report.json`, `final_project_summary.json`, `test_predictions.csv`, `tuned_test_predictions.csv`, `threshold_tuning_report.json`, `threshold_validation_curve.csv`, `model_comparison.csv`. Manifests/failure records exist; the external full official dataset should not be assumed available. |
| Saved models | `final/numpy_logistic_model.npz`, `final/numpy_logistic_model_tuned.npz` exist and were schema-audited. Neither matches the observable 3-class adapter. Expected portable JSON model under `src/quantum_simulator/models/` is absent. |
| Controlled simulator ML | 270 reports under `reports/experiments/full_dataset/`, `dataset_manifest.json`, and final grouped comparison JSON/Markdown/OOF predictions; controlled diagnosis, three-class feasibility/error analysis, expected-mode shift, repeated feature and ablation reports/predictions; hardware coverage and feature audits. |
| Proposed new wave | `first_wave_experiment_spec_v3.{md,json}`, `first_wave_spec_v3_review.{md,json}`, `first_wave_execution_preflight.{md,json}` at root; latest v3 preflight is BLOCKED and states no reports were generated. The output directory was absent at preflight. |
| Historical/local validation | `reports/` demo, experiment, Bugs4Q, QMutBench and IBM fixture reports. The project documents offline fixture/simulator provenance; no live hardware evidence is claimed. |

## Audit method and limits

Files were inspected without running project commands. Presence and selected report content were verified; referenced paths were not all hashed or independently recomputed. Saved metrics are reported as artifacts state them. The review did not validate every row of every CSV or reload the NPZ weights for prediction. No claim is made that saved evidence is bitwise reproducible on every environment.

**Recommendation: FINALIZE WITH EXISTING EVIDENCE**
