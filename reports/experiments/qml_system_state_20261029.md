# QML system state audit — 2026-10-29

Repository `qmutbench-master-optimization` at `a2d51be22fd985bbdd10655f7e31dc77b9d19ed6`. This read-only audit preserves the existing working tree.

## Current status

- 19 acquisition-inventory entries and 19 archive entries are recorded; 19 listed source paths exist locally.
- 12 of 18 task-matrix rows are marked complete or existing champion; this count includes distinct source/task rows, not unique datasets.
- Saved QNN artifact replay audit status: **verified**. It records checkpoint archive validation, metric recalculation, and gradient checks for the newer completed tasks.
- Saved multi-source training artifact audit status: **completed_artifact_audit**; its scope is development artifacts and it records that the frozen test was not opened.
- New four-epoch QNN candidate sweeps: **9 tasks, 90 checksummed fold checkpoints, and 90 passing fold gradient checks**.
- Matched classical controls are complete for all nine tasks and both feature widths. The QNN has higher macro F1 than the descriptively highest fixed control in 1 of 18 task/width cells, ties in 1, and trails in 16. Each paired interval resamples whole source groups 1,000 times.
- Available RAM at audit: **1115.0 MiB**; full-data QMutBench training is below its 1,024 MiB floor and sampled binary training is below its 2,048 MiB startup requirement.
- Fresh binary QNN preflight: **blocked_memory_guard** — Only 1113 MiB available; estimated peak requires 2048 MiB to preserve the 1024 MiB floor
- QMutBench binary training is not recorded as complete. Its historical WOO/OPO outcome target is distinct from hardware anomaly attribution.
- IBM hardware fault attribution remains unavailable because verified device observations and independent labels are absent.
- Latest saved broad simulator suite: 794 passed, 1 failed, 0 skipped, 2 registered finite-shot NO-GO tests deselected (797 collected). Its sole failure is the unchanged frozen 45-case diagnosis acceptance with 7 mismatches; this audit did not rerun it.

## Source and task findings

The detailed 19-entry acquisition inventory and the per-task completion matrix are embedded in the JSON report. QMutBench, Bugs4Q, QSimBench, Bugs4Q-Robust, QBugs, Qolumbina ideal-backend, LintQ warning triage, and quantum-platform comment tasks have source-specific target definitions. Their scores remain separate and development-only.

Implemented QML families include angle and re-uploading VQCs, amplitude-encoded and hybrid-readout QNNs, QSVC, and a quantum reservoir with classical readout. Existing evidence includes 4,575 saved QNN OOF predictions replayed across nine source runs with maximum probability error below 1.2e-16, plus separate integrity/gradient audits for the newer QNN tasks.

**Extraction reconciliation:** the Bugs4Q Zenodo archive contains 337 files and passes ZIP CRC validation. Its 41 nested `.git` metadata files account for 29,110,733 bytes; excluding `.git`, both archive and extracted tree contain 296 data files / 177,433 bytes. The prior apparent count mismatch is resolved.

## Verified evaluation summary

- Existing QMutBench binary classical champion: 578,117 development records, 305 origin groups, five-fold grouped OOF macro F1 **0.7933**. This is historical and below the 0.85 objective; it is not a QML result or hardware diagnosis.
- QMutBench three-class QNN work: 205 records / 12 OPO-bearing origins / three grouped folds. The best saved QNN ensemble macro F1 is about **0.434**, versus matched Extra Trees about **0.467**. It is exploratory and class-balanced.
- QSimBench backend fingerprint QNN: 300 observations / 150 circuit-setting groups / five folds; the angle re-uploading model has about **0.660** accuracy and macro F1. The label is backend identity, not a fault label.
- Bugs4Q-Robust pass/fail QNN: 2,220 repeated observations / 37 issue groups / five folds; the resource-light amplitude hybrid has about **0.514** accuracy and macro F1. The saved matrix has no matched classical score for this exact feature run.
- New four-epoch hashed-text QNN results include Bugs4Q pooled macro F1 **0.494**, Bugs4Q-NA **0.470**, QBugs **0.718**, Qolumbina ideal **0.452**, LintQ warning triage **0.426**, and quantum-platform comments **0.894**. Same-input classical controls were completed; only the Bugs4Q core-only transfer 8-feature QNN exceeded the descriptively top fixed control (+3.38 macro-F1 points). The full table and paired group-bootstrap intervals are in [the matched comparison report](../qml_standalone_accuracy_optimization_20261028/remaining_task_sweeps/matched_qnn_classical_comparison.md).
- The OOF-highest classical control is reported descriptively after scoring all controls; it was not selected on an independent validation/confirmation set. These are development results and do not establish generalization to unseen source projects beyond the registered grouped folds.

## Validation performed

Focused command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv\Scripts\python.exe -m pytest --basetemp=.pytest-qml-audit-base tests\test_qmutbench_binary_qnn.py -q` — **27 passed, 1 skipped**. The skipped case is the real PennyLane gradient/update/resume smoke, gated on 2,048 MiB available RAM. Binary configuration validation passed; the fresh training preflight was safely blocked below its 2,048 MiB requirement. These checks do not constitute a new real-data fit.
The simulator's frozen acceptance failure remains preserved at its existing 7/45 mismatch. This QML-only suite is separate verification and was not counted as simulator validation.

The QML foundation and multiclass forward/backward suite also passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv\Scripts\python.exe -m pytest --basetemp=.pytest-qml-foundation-base qmutbench_qml\tests\test_foundations_and_multiclass.py -q` — **5 passed**.

## Reproduction commands

Resume/reconcile the fixed four-epoch development QNN candidates: `LIGHT_QNN_RESUME=1`, `LIGHT_QNN_EPOCHS=4`, `LIGHT_QNN_BATCH=64`, `LIGHT_QNN_SEED=28102026`, then run `experiments/train_remaining_standalone_qml_tasks.py`. Run the matched controls with `experiments/evaluate_qnn_candidate_matched_classical.py`; regenerate integrity reports with `experiments/audit_qml_candidate_sweeps.py` and `experiments/audit_matched_qnn_classical_comparison.py`.

## Remaining work and blockers

- Run the QMutBench binary and full-data grouped comparisons only after measured RAM meets the explicit startup/in-run gates.
- Execute the separate prepared 14-model/imbalance-method external suite on a supported CPU runtime; local bundle validation is not execution evidence.
- Normalize and deduplicate QML-platform issue workbooks and build validated task manifests for Bug-Pattern/QChecker before modeling.
- Resolve Mute-Alg legacy Qiskit pickle compatibility without relabeling or altering originals.
- Obtain authorized IBM observations with calibration, mappings, timestamps, and independently adjudicated labels before hardware-attribution modeling.

No test labels from a frozen QMutBench confirmation set were read in generating this audit. Machine-readable source/task rows, branch refs, resource state, provenance hashes, and blockers are in `qml_system_state_20261029.json`.
