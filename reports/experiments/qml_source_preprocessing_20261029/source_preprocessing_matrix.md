# Source preprocessing and execution status — 2026-10-29

Inventory sources: 19; current task matrix rows: 18.
The current CSV contains 18 task rows; all are represented below. The acquisition inventory has 19 source entries.

## Task status

| Dataset/task | Existing training status | Records | Groups | Validation/readiness | Evidence / next action |
|---|---|---:|---:|---|---|
| Bugs4Q official framework / source-state | COMPLETE | 62 | 31 | READY_FOR_TRAINING | No exact duplicate task-feature fingerprints cross the registered validation folds. |
| Bugs4Q-NA / source-state | COMPLETE | 62 | 31 | READY_FOR_TRAINING | No exact duplicate task-feature fingerprints cross the registered validation folds. |
| Bugs4Q pooled / source-state integration | COMPLETE | 124 | 62 | READY_FOR_TRAINING | No exact duplicate task-feature fingerprints cross the registered validation folds. |
| Bugs4Q core-only transfer / source-state transfer | COMPLETE_SOURCE_RESTRICTED_RESUMABLE_QNN_PILOT | 124 | 62 | READY_FOR_TRAINING | No exact duplicate task-feature fingerprints cross the registered validation folds. |
| Bugs4Q-NA-only transfer / source-state transfer | COMPLETE_SOURCE_RESTRICTED_RESUMABLE_QNN_PILOT | 124 | 62 | READY_FOR_TRAINING | No exact duplicate task-feature fingerprints cross the registered validation folds. |
| QBugs HornBro subset / correct/buggy circuit state | COMPLETE | 1500 | 750 | BLOCKED_FEATURE_LABEL_AMBIGUITY | Exact model-input text/metadata fingerprints recur across folds with conflicting correct/buggy labels; reconcile source pairs or evaluate a duplicate-component split before new training. |
| Quantum platform annotations / issue-comment classification | COMPLETE | 319 | 314 | BLOCKED_CROSS_FOLD_DUPLICATE_COMMENTS | Exact comment text is repeated across issue groups and validation folds; regroup duplicate comment components before new training. |
| Qolumbina empirical study / ideal-backend seeded-mutant detection | COMPLETE | 182 | 26 | READY_FOR_TRAINING | No exact duplicate task-feature fingerprints cross the registered validation folds. |
| LintQ / manual warning triage TP vs FP | COMPLETE | 290 | 228 | BLOCKED_FEATURE_LABEL_AMBIGUITY | The current warning_message-only feature repeats across folds with conflicting TP/FP labels; include justified context or redesign the split/feature policy before new training. |
| Bugs4Q-Robust / test-harness pass/fail prediction | COMPLETE_GROUPED_QNN | 2220 | 37 | READY_FOR_TRAINING | No exact duplicate task-feature fingerprints cross the registered validation folds. |
| QSimBench / Aer vs fake_kyiv backend fingerprinting | COMPLETE_GROUPED_QNN | 300 | 150 | READY_EXPLORATORY_WITH_FEATURE_COLLISIONS | Quantized output descriptors recur across held-out circuit settings with both backend labels; preserve this limitation in any generalization claim. |
| QMutBench / three-class mutation diagnosis | COMPLETE_DEVELOPMENT_OOF_WITH_VERSIONED_RESUMABLE_CHECKPOINT_RECOVERY | 205 | 12 | READY_FOR_TRAINING_AND_CHECKPOINT_REPLAY_VERIFIED | No exact duplicate task-feature fingerprints cross the registered validation folds. Original campaign checkpoint status remains no_real_data_qnn_parameter_checkpoint; a separate versioned recovery verified 9 optimizer/RNG checkpoints and replayed 615 OOF probability rows. This historical mutation-outcome task is separate from hardware fault diagnosis. |
| Mute-Alg / original vs synthetically buggy QASM v2 | COMPLETE_RESUMABLE_QNN_PILOT | 9140 | 812 | READY_FOR_TRAINING | No exact duplicate task-feature fingerprints cross the registered validation folds. Upstream HornBro repository lists MIT; the local dataset-only snapshot lacks its repository license file and commit ID. The 169 source-file hashes are pinned in `source_provenance.json`. |
| Bug-Pattern / No validated supervised manifest | NOT_TRAINED | 20 git files |  | INELIGIBLE_UNVALIDATED | Source audit found only 9 local notebooks across four categories; README-linked external issue examples are not local rows. Notebook titles/markdown expose category labels, and there are no clean negatives or enough independent groups for a leakage-safe split. See bug_pattern_qchecker_source_audit.json. |
| QChecker / No validated supervised manifest | NOT_TRAINED | 39 git files |  | INELIGIBLE_UNVALIDATED | Source audit found 31 instance files across nine directory labels, one syntax error, severe class imbalance, no clean negatives, and no verified per-file detector outcomes; README uses MI while the local directory is IM. No defensible leakage-safe supervised split is established. See bug_pattern_qchecker_source_audit.json. |
| QML platform workbooks / Schema/license normalization incomplete | NOT_TRAINED | 43 source archive members |  | BLOCKED | The 39 workbooks parse as OOXML, but their exports have heterogeneous and sometimes shifted columns. The integrated Conclusion workbook has 481 issue rows: 339 unique issues have clean 0/1 quantum-related labels (220/119), 140 rows are unlabeled or malformed, and two duplicate issue rows were removed. Ten repositories collapse to nine repository/title components, one containing 264 rows. The source inventory found no reuse license, so model training remains blocked pending rights clarification and a leakage-safe supervised split; see the workbook label audit. |
| QMutBench / binary Survivor vs WOO/OPO merged detection (existing development champion) | PARTIAL_FULL_DEVELOPMENT_QNN_RESUMABLE | 578117 | 305 | PREPROCESSING_READY_TRAINING_IN_PROGRESS_RESUMABLE | Full-development streaming run is safely checkpointed in fold 1 after 51 chunks / 26112 training-record exposures and 816 optimizer updates (837.3 cumulative training seconds; minimum recorded available RAM 1748.0 MiB; peak RSS 468.0 MiB); observed-rate estimate for all configured folds/epochs is 164.8 hours before evaluation; chunk audit verified all hashes, gradients, and parameter changes. This is partial training with no OOF metrics yet. Resume the versioned full-development run; frozen test untouched. |
| Nine binary source tasks / Broad grouped classical models and training-fold-only imbalance methods | COMPLETE_LOCAL_9_TASK_CLASSICAL_SCREEN | 5,055 across nine validated task adapters | 1,606 across nine task-local groups | COMPLETE_LOCAL_CLASSICAL_SCREEN | The refreshed source-isolated v2 suite completed on nine validated task adapters (5,055 rows): 180 model/task summaries and 900 fold/model metrics across 20 models and five grouped folds per task. OOF IDs, labels, groups, folds, and aggregate/per-fold metrics replayed with zero audit issues. It is an independent classical CV screen, not matched to QNN feature/split choices. MLP convergence warnings at the configured 200-iteration limit were observed and retained. The separate external Kaggle notebook remains unexecuted. |

## Resumable pilot runs

| Task | Rows/groups/folds | QNN macro F1 | Classical macro F1 | Resume state |
|---|---:|---:|---:|---|
| bugs4q_core_source_state | 62/31/5 | 0.4493 | 0.4609 | verified (1 resumes) |
| bugs4q_core_only_transfer_source_restricted_v2 | 124/62/5 | 0.5403 | 0.5065 | verified (1 resumes) |
| bugs4q_joint_source_state | 124/62/5 | 0.4935 | 0.4967 | verified (1 resumes) |
| bugs4q_na_source_state | 62/31/5 | 0.4609 | 0.4999 | verified (2 resumes) |
| bugs4q_na_only_transfer_source_restricted_v2 | 124/62/5 | 0.4919 | 0.5065 | verified (1 resumes) |
| bugs4q_robust_execution | 2220/37/5 | 0.5269 | 0.6075 | verified (1 resumes) |
| lintq_warning_triage_context_v2 | 290/217/5 | 0.4269 | 0.5433 | verified (1 resumes) |
| lintq_warning_triage_context_v2_balanced | 290/217/5 | 0.4185 | 0.5433 | verified (1 resumes) |
| mute_alg_hornbro_qasm_v2 | 9140/812/5 | 0.4894 | 0.4997 | verified (1 resumes) |
| qbugs_qasm_v2_source_state | 1496/748/5 | 0.6845 | 0.7464 | verified (1 resumes) |
| qolumbina_ideal_seeded_mutant_detection | 182/26/5 | 0.4518 | 0.4303 | verified (1 resumes) |
| qolumbina_multibackend_seeded_mutant_detection | 546/26/5 | 0.4810 | 0.3445 | verified (1 resumes) |
| qolumbina_multibackend_seeded_mutant_detection_balanced | 546/26/5 | 0.4161 | 0.3445 | verified (1 resumes) |
| quantum_platform_comment_v2_balanced | 319/304/5 | 0.5894 | 0.8393 | verified (1 resumes) |

## Invalidated pilot artifacts

- `bugs4q_core_only_transfer`: INVALID_FOR_TRANSFER_TASK; The resumable pilot runner performed mixed-source grouped cross-validation. The registered contract requires each fold to train only on Bugs4Q_core and evaluate the held-out fold with the Bugs4Q_NA_additional transfer subset reported separately. This run therefore cannot support the transfer claim. See `reports\experiments\resumable_qnn_pilot_bugs4q_core_transfer_20261029\task_validity_correction.json`. Artifacts are preserved and excluded from valid pilot summaries.

## Separate grouped classical screen

Status: VERIFIED; 5 eligible tasks, 100 aggregate rows and 500 fold/model rows. This is not QNN-matched because it uses TF-IDF/SVD features and its own deterministic GroupKFold splits.
Four duplicate-risk tasks were excluded from this local screen; the external nine-task Kaggle bundle remains unexecuted.
See `reports/experiments/local_grouped_classical_suite_eligible_20261029/report.md`.

## Refreshed nine-source classical suite

Status: VERIFIED; 9 tasks, 180 model/task results, 900 fold/model metrics. This is an independent classical screen, not a matched QNN evaluation.
The OOF ID/label/group/fold checks and aggregate/per-fold metric replays passed. MLP convergence warnings at the configured 200 iterations were observed and retained.
See `reports/experiments/local_grouped_classical_suite_nine_source_v2_20261029/report.md`.

## Source-restricted Bugs4Q transfer pilots

Status: VERIFIED; 2 directions verified with training-source-only folds, separate transfer-source metrics, matched classical OOF, and resume/checkpoint receipts.
See `bugs4q_transfer_pilot_audit.json`. A prior mixed-source run is preserved but invalidated for a transfer claim.

## Additional task variants

These variants extend the campaign task matrix while preserving its original 18 rows.

| Source/task | Readiness | Rows/groups | Evidence |
|---|---|---:|---|
| lintq / warning_triage_context_v2 | READY_EXPLORATORY_WITH_LABEL_AMBIGUITY; conflict-excluded variant trained | 287/216 retained in variant | The original 290-row task retains 3 contradictory labels in one byte-identical code-context component. A separately versioned variant rejected all three component rows, reused the original grouped folds for the remaining records, and completed 5-fold QNN OOF: accuracy 74.91%, macro-F1 45.40%, balanced accuracy 51.14%. See reports/experiments/lintq_context_v2_conflict_excluded_20261029/run_manifest.json. |
| mute_alg_hornbro / compatible_subset_qasm_v1 | INELIGIBLE_PARTIAL_SOURCE_LOW_GROUP_DIVERSITY | 5440/11 | The compatibility-selected subset has 5440 accepted rows but only 11 exact-QASM/pair components; 77 of 169 files loaded and 92 failed restricted Qiskit deserialization. Another 1720 rows were excluded across conflicting duplicate fingerprints. This partial, compatibility-selected sample is too narrow for a representative training claim; resolve Qiskit serialization and expand independent circuits first. |
| qbugs_hornbro / correct_buggy_circuit_state_qasm_v2 | READY_FOR_TRAINING | 1496/748 | READY: 1496 OpenQASM records in 748 paired groups; 4 records (2 complete source pairs) were explicitly rejected because exact OpenQASM fingerprints had contradictory labels. Source pickle, input, split, and rejection hashes verified; no exact QASM feature crosses folds. A separate QNN-only candidate sweep on the authoritative processed split completed both architectures (five grouped folds, 1,496 OOF rows per model). Amplitude: accuracy/balanced accuracy 68.38%, macro-F1 68.30%; angle re-upload: 73.26% accuracy/balanced accuracy, 73.22% macro-F1. All source, split, label, group, OOF, and ten checkpoint hashes verified. See reports/experiments/qbugs_qasm_v2_candidate_sweep_20261030/run_manifest.json. |
| quantum_platform_bugs / comment_only_classification_v2 | READY_FOR_TRAINING | 319/304 | No exact duplicate task-feature fingerprints cross the registered validation folds. |
| qolumbina / seeded_mutant_detection_multibackend | READY_EXPLORATORY_SEVERE_BACKEND_CLASS_IMBALANCE | 546/26 | All 546 rows across ideal/fake_algiers/fake_brooklyn follow the source majority-detection rule. The established five-fold original-program grouping is reused; no program crosses folds. fake_brooklyn has only one negative row; outcomes are exploratory seeded-mutant detection, not hardware attribution. Two resumable pilots, a QNN-only two-architecture grouped candidate sweep (1,092 OOF rows), and a five-fold inverse-frequency weighted-loss QNN follow-up are audit-verified. Candidate-sweep hashes and row/group/fold/label alignment were verified. Both unweighted candidates predicted only the positive class; the weighted follow-up reached 43.53% balanced accuracy. Exploratory under severe 40/506 imbalance; not hardware attribution. See reports/experiments/qolumbina_multibackend_candidate_sweep_20261029/run_manifest.json and weighted_loss_followup/run_manifest.json. |

## Validated source-isolated manifests

18 manifests/indexes were reconciled, including the six prior manifests (2,595 accepted rows), additional source-specific task variants, QMutBench development data and its binary SQLite index (578,117 rows). The 2,220-row Bugs4Q-Robust v1 output with an incorrect run-index contract is preserved as rejected evidence; corrected v2 has all rows accepted.

## Source task resume commands

- `.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\bugs4q_core_source_state.json --chunk-size 64`
- `.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\bugs4q_na_source_state.json --chunk-size 32`
- `.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\bugs4q_joint_source_state.json --chunk-size 64`
- `.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\bugs4q_core_only_transfer.json --chunk-size 64`
- `.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\bugs4q_na_only_transfer.json --chunk-size 64`
- `.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\qsimbench_backend_fingerprint.json --chunk-size 64`
- `.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\bugs4q_robust_execution.json --chunk-size 128`
- `.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\lintq_warning_triage.json --chunk-size 64`
- `.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\qbugs_correct_buggy_circuit_state.json --chunk-size 64`
- `.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\quantum_platform_comment_classification.json --chunk-size 64`
- `.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\qolumbina_ideal_detection.json --chunk-size 64`
- `.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\qmutbench_three_class_exploratory.json --chunk-size 64`
- `.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\lintq_warning_triage_context_v2.json --chunk-size 64`
- `.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\mute_alg_compatible_subset_qasm_v2_qiskit_compat_v2.json --chunk-size 128`

## Other resume commands

- QMutBench binary development index: `.venv\Scripts\python.exe experiments\prepare_qmutbench_binary_development_index.py`.
- Resume a pilot only if interrupted before completion: `.venv\Scripts\python.exe experiments\run_resumable_qnn_pilot.py --resume --epochs 4` (use the matching `--task`, including `qbugs_qasm_v2_source_state` or `mute_alg_hornbro_qasm_v2`, for a non-default task).
- Rebuild the LintQ context component task: `.venv\Scripts\python.exe experiments\prepare_lintq_context_v2.py` then `.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\lintq_warning_triage_context_v2.json --chunk-size 64`.
- Rebuild the duplicate-aware platform comment task: `.venv\Scripts\python.exe experiments\prepare_quantum_platform_comment_v2.py` then `.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\quantum_platform_comment_v2.json --chunk-size 64`.
- Rebuild the QBugs QASM task with the Qiskit 1.4.5 environment: `.qbugs1_env\Scripts\python.exe experiments\prepare_qbugs_qasm_v2.py` then `.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\qbugs_correct_buggy_circuit_state_qasm_v2.json --chunk-size 128`.
- QNN full-development QMutBench training only after memory preflight passes: inspect `.venv\Scripts\python.exe -m qmutbench_qml.training.run_binary_development --help`; use the established `--mode full-development` and `--resume` options.
- Rebuild integrity/readiness matrix: `.venv\Scripts\python.exe experiments\audit_qml_task_semantics.py` followed by `.venv\Scripts\python.exe experiments\build_qml_preprocessing_status.py`.

No source task is called clean solely because its columns parsed. Semantic feature collisions, group leakage, target ambiguity, and source-specific blockers are recorded above. The preserved frozen simulator diagnosis remains 7 mismatches among 45 outcomes.
