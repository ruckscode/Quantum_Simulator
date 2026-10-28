# Multi-source quantum ML campaign status

**Status: nine supervised tasks have complete grouped-fold training artifacts; QMutBench has development OOF results but no recoverable model checkpoint.** Bugs4Q-Robust and QSimBench have defensible task manifests, but local QNN attempts stopped at the 500 MiB guard before producing checkpoints. The combined CPU training notebook is prepared but not run. The campaign remains incomplete. No model is promoted to production. The QMutBench frozen test was not opened.

## 2026-10-28 execution audit and bundle readiness

The refreshed standard-library audit recomputes 85 saved model/ensemble OOF rows and 327 fold rows from development predictions. It separately records the 578,117-record QMutBench binary champion and the 205-record three-class research task; neither is pooled with Bugs4Q or the other source tasks. The cross-source file contains 24 source-specific transfer results for the Bugs4Q core/NA pooled and source-restricted runs. These are small, within-task grouped OOF comparisons, not a universal cross-dataset score.

The external notebook now embeds SHA-256 hashes for every bundled task manifest, the QNN runner, and the classical runner, and verifies them before installing dependencies or training. It pins the versions recorded by the existing QNN environment (`numpy 2.5.3`, `pandas 3.0.6`, `PennyLane 0.45.1`, `scikit-learn 1.9.1`, `psutil 7.2.2`) and explicitly uses CPU `default.qubit`; a GPU is unnecessary. Notebook syntax and ZIP integrity were checked locally, but **the notebook has not been executed**.

The bundled broad classical suite has 14 baseline models plus training-fold-only random over/undersampling and continuous-space SMOTE variants, using the same five grouped outer folds for each task. Those results do not exist yet. The suite reports fold/aggregate classification metrics and sampled process RSS, but Kaggle must run it before any baseline or imbalance method can be called evaluated. Borderline-SMOTE, XGBoost, LightGBM, CatBoost, stacking, and QMutBench's separate binary champion are not part of that unrun suite. See `campaign_completion_matrix.csv` for the task-level split between completed evidence and external work still pending.

## Completed trained tasks

| Task | Grouped data | QNN result | Best matched classical result | Outcome |
|---|---:|---:|---:|---|
| Bugs4Q core source-state classification | 31 issues, 62 paired snapshots | Best macro F1 0.499 (amplitude); accuracy 0.500 | Logistic: macro F1 0.526, accuracy 0.532 | Classical control leads |
| Bugs4Q-NA additional source-state classification | 31 non-overlapping issues, 62 snapshots | Angle hybrid: macro F1 0.508, accuracy 0.516 | Extra Trees: macro F1 0.517, accuracy 0.532 | Classical control leads; QNN equal-probability ensemble macro F1 0.548 in OOF diagnostic |
| QBugs HornBro circuit state | 750 correct/buggy pairs, 15 scenario groups, 1,500 snapshots | Angle hybrid: macro F1 0.774, accuracy 0.775 | Random forest: macro F1 0.843, accuracy 0.843 | Classical control leads |
| Quantum platform bug annotation comments | 319 comments, grouped by 314 commits (223 bug / 96 FP after filtering) | Angle hybrid: macro F1 0.819, accuracy 0.853, balanced accuracy 0.809 | Random forest: macro F1 0.897, accuracy 0.915, balanced accuracy 0.889 | Metadata comment task; classical control leads |
| Qolumbina ideal-backend seeded-mutant detection | 182 variants, 26 program groups, 5 folds | Angle VQC: macro F1 0.502, accuracy 0.621, balanced accuracy 0.524 | Extra Trees: macro F1 0.505, accuracy 0.615, balanced accuracy 0.533 | Synthetic mutant detection; essentially tied, not real bug classification |
| LintQ warning triage | 290 TP/FP warnings, 228 issue/file groups, 5 folds | Angle hybrid: macro F1 0.526, accuracy 0.566, balanced accuracy 0.557 | Random forest: macro F1 0.640, accuracy 0.672, balanced accuracy 0.691 | Classical control leads; warning-level task, not program state |
| QSimBench backend fingerprint | 300 Aer/fake_kyiv observations, 150 algorithm-size groups | Not trained; import-time RAM floor stopped local execution | Not trained | Derived backend identification task; no bug/fix semantics |
| Bugs4Q-Robust execution outcome | 2,220 repeated-run observations, 37 issue groups | Not trained; first optimizer step hit RAM guard | Not trained | Predict pass/fail from program and test harness; execution outcome, not source-state target |
| QMutBench three-class mutation task | 205 records, 12 origins, 3 grouped folds | QNN: macro F1 0.300, accuracy 0.322, balanced accuracy 0.318 | Extra Trees: macro F1 0.467, accuracy 0.493, balanced accuracy 0.485 | Development-only research result; predictions and curves exist, but no parameter checkpoint |

The Bugs4Q core and Bugs4Q-NA standalone runs used four QNN encodings/architectures with matched classical controls, gradient checks, and grouped folds. The source-only transfer runs and pooled run used identical five-fold group partitions across the same 124-record, 62-issue combined manifest. In the pooled run, the hybrid QNN macro F1 was 0.498 and Extra Trees was 0.524. Training only on core yielded hybrid macro F1 0.477 overall (0.464 on held-out NA and 0.490 on held-out core); training only on NA yielded 0.516 overall (0.508 on held-out NA and 0.512 on held-out core). These small samples do not establish robust transfer gains. The source-specific OOF diagnostics are reported in each `source_performance.json`.

The binary QNN models optimize supervised class-weighted binary cross-entropy with PennyLane `default.qubit` analytic backpropagation. All nine complete source tasks have finite-difference gradient checks and saved parameters; replay reproduced 4,575 QNN OOF probabilities with maximum absolute error below 1.2e-16. Training histories, fold checkpoints, OOF probabilities, fold metrics, per-class metrics, and grouped uncertainty outputs are stored beside each `summary.json`. The transfer outputs are in `bugs4q_joint/qnn_train_core_complete/` and `bugs4q_joint/qnn_train_na_complete/`.

## Candidate-invalid and incomplete branches

- `quantum_platform_223/qnn_hybrid/` used annotation fields whose presence directly encoded bug versus false-positive status. This is label-construction leakage; it is **invalid for evaluation and excluded**. The corrected comment-only model uses only comment text and masks explicit label words, though semantic clues may remain.
- `bugs4q_joint/qnn_train_core/` is an interrupted early attempt after RAM fell below 500 MiB. Its partial artifacts are not scored. The complete rerun is separately stored and finished above the 500 MiB floor.
- `hornbro_mute_alg/` contains 169 extracted pickles, but restricted loading fails (`state is not a dictionary`) under Qiskit 1.4.5; Qiskit 2.5 also rejects the legacy circuit register state. No examples were counted or trained.
- Qolumbina's ideal-backend DOSS task is now trained from mutant source code and pre-run backend features, with held-out labels derived only from the study's repeated test outcomes. It is a synthetic seeded-mutant detection task, not a real-bug task. The broader 546-row multi-backend attempt remains partial and is not scored.
- QSimBench supplies circuit execution traces rather than bug/fix labels. We prepared an Aer-versus-fake_kyiv fingerprint task from output histograms and group by algorithm-size setting; its QNN did not train locally because post-import available RAM fell below the runner's safe floor.
- LintQ's manually annotated TP/FP warning triage task is trained with issue/file-grouped folds. The 55 ambiguous NW records are excluded; results do not establish program-level bug/fix classification.
- Bugs4Q-Robust now has a separate program-plus-test pass/fail task. Its local QNN attempt stopped below the 500 MiB floor before saving any checkpoint; the repeated runs for each issue are grouped together.
- QML platform issue workbooks were downloaded but not deduplicated/schema-normalized; license and label validation remain open. They were not trained.

Other acquired sources remain incomplete or unsuitable for a current supervised task: Bugs4Q-Robust and QSimBench have valid tasks but local fits hit the memory floor before checkpoints; Mute-Alg legacy pickle deserialization fails on the available Qiskit versions; QML platform workbooks need label/license normalization; and Bug-Pattern/QChecker have no validated supervised manifest yet. The QMutBench three-class artifacts lack a recoverable parameter checkpoint. These branches remain explicitly uncompleted in `campaign_completion_matrix.csv`; inspection is not counted as training.

## Reproducibility and key artifacts

- Integrity-checked dataset inventory: `verified_dataset_inventory.json` (download, extraction, and data status fields; archive hashes/CRC and completeness checks).
- Dataset/task registry: `dataset_task_inventory.json` (source status joined to task definitions, model status, and caveats).
- External bundle validation: `external_bundle_validation.json` (archive CRC, embedded input hashes, manifest schemas/counts, and notebook syntax; training remains explicitly unexecuted).
- Source/task status matrix: `campaign_completion_matrix.csv`.
- Recomputed model comparison: `comprehensive_results.csv`; task winners: `best_model_by_task.csv`; fold scores: `fold_level_results.csv`.
- Source-specific Bugs4Q transfer results: `cross_source_generalization.csv`.
- Bugs4Q core QNNs: `bugs4q_standalone/qnn_gradient_fix/`.
- Bugs4Q-NA QNNs: `bugs4q_na/qnn/`.
- Bugs4Q pooled and source-only comparisons: `bugs4q_joint/`.
- QBugs QNN: `qbugs/qnn_hybrid/`.
- Quantum platform comment-only model: `quantum_platform_223_comment_only/qnn_hybrid/`.
- Qolumbina ideal-backend QNN: `qolumbina_detection_ideal/qnn/`.
- LintQ warning triage QNN: `lintq_warning_triage/qnn/`.
- Bugs4Q-Robust task manifest: `bugs4q_robust_execution/normalized_manifest.csv`.
- QSimBench task manifest: `qsimbench_backend_fingerprint/normalized_manifest.csv`.
- Self-contained external runner: `Remaining_Source_QNN_Training.ipynb` (bundle: `remaining_source_qnn_training_bundle.zip`). It has not been executed. Required external step: import/open the notebook and run all cells in a compatible Kaggle CPU session; then retrieve `remaining_source_qnn_results.zip` from Kaggle output to verify the two QNNs and nine-task classical/imbalance suite.
- QMutBench research-only QNN campaign: `reports/experiments/qmutbench_three_class_qnn_campaign_20261027/`.

All completed evaluations use development folds only. No frozen-test data was used for feature selection, training, or ensemble selection. Production model files were not changed. The local runner enforces a 500 MiB available-RAM floor; incomplete attempts are retained with their partial logs for audit.
