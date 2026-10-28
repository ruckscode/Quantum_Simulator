# Standalone QML Accuracy Optimization Campaign

**Evaluation scope:** grouped development out-of-fold (OOF) predictions only. The frozen test was not read or used for selection; final held-out test accuracy is unavailable and intentionally blank.

The campaign preserves prior grouped QNN runs as incumbents and adds local, CPU-only statevector optimization for Bugs4Q-Robust and QSimBench, plus a QMutBench finalist seed rerun. Scores are task-specific and should not be compared across different targets.

## Per-task results

| Task | Prediction target | Records / groups | Grouped split | Label distribution | Original best QML accuracy | Selected development accuracy | Macro F1 | Balanced accuracy | Training seconds | Change | Selected model | Final test accuracy |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Bugs4Q core | source-state fixed vs buggy | 62 / 31 | issue-group 5-fold (bug_id) | `{"0": 31, "1": 31}` | 51.61% | 51.61% | 49.7297% | 51.6129% | 162.85436049999998 | +0.00 pp | angle_reupload_vqc | unavailable |
| Bugs4Q core-only transfer | source-state fixed vs buggy | 124 / 62 | shared issue-group 5-fold; training restricted to core (bug_id) | `{"0": 62, "1": 62}` | 50.00% | 50.00% | 47.7007% | 50.0% | 116.2697448 | +0.00 pp | angle_hybrid_readout | unavailable |
| Bugs4Q pooled | source-state fixed vs buggy | 124 / 62 | shared issue-group 5-fold (bug_id) | `{"0": 62, "1": 62}` | 50.00% | 50.00% | 49.791% | 50.0% | 243.97170849999998 | +0.00 pp | angle_hybrid_readout | unavailable |
| Bugs4Q-NA | source-state fixed vs buggy; equal probability ensemble diagnostics | 62 / 31 | issue-group 5-fold OOF diagnostics (bug_id) | `{"0": 31, "1": 31}` | 54.84% | 54.84% | 54.7917% | 54.8387% |  | +0.00 pp | quantum_equal_probability_mean | unavailable |
| Bugs4Q-NA-only transfer | source-state fixed vs buggy | 124 / 62 | shared issue-group 5-fold; training restricted to NA (bug_id) | `{"0": 62, "1": 62}` | 51.61% | 51.61% | 51.6003% | 51.6129% |  | +0.00 pp | angle_hybrid_readout | unavailable |
| LintQ | manual TP vs FP warning | 290 / 228 | issue/file-group 5-fold (bug_id) | `{"0": 74, "1": 216}` | 56.55% | 56.55% | 52.5726% | 55.7307% | 14.3235232 | +0.00 pp | angle_hybrid_readout | unavailable |
| QBugs HornBro subset | correct vs buggy circuit | 1500 / 750 | paired circuit 5-fold (bug_id) | `{"0": 750, "1": 750}` | 77.47% | 77.47% | 77.4087% | 77.4667% | 240.76087180000002 | +0.00 pp | angle_hybrid_readout | unavailable |
| QMutBench three-class research | Survivor/WOO/OPO | 205 / 12 | saved origin-group 3-fold OOF; fixed equal probability mean (origin_program) | `{"0": 72, "1": 68, "2": 65}` | 46.83% | 47.32% | 41.8858% | 46.0596% | 1.8660812999951304 | +0.49 pp | multi_seed_probability_ensemble_amplitude_lr008 | unavailable |
| Qolumbina ideal | seeded mutant detected vs not detected | 182 / 26 | program-group 5-fold (bug_id) | `{"0": 32, "1": 150}` | 62.09% | 62.09% | 50.1726% | 52.4167% | 46.11504789999999 | +0.00 pp | angle_vqc | unavailable |
| Quantum platform comment task | bug comment vs false-positive comment | 319 / 314 | commit-group 5-fold (bug_id) | `{"0": 96, "1": 223}` | 85.27% | 85.27% | 81.8743% | 80.86% | 236.1916943 | +0.00 pp | angle_hybrid_readout | unavailable |
| bugs4q_robust_execution | bugs4q_robust_execution | 2220 / 37 | five deterministic group-disjoint folds; all paired/source variants held by bug_id (bug_id; issue groups) | `{"0": 1110, "1": 1110}` | 51.35% | 51.35% | 51.3158% | 51.3514% | 12.0115 | +0.00 pp | amplitude_hybrid_3q_depth2 | unavailable |
| qsimbench_backend_fingerprint | qsimbench_backend_fingerprint | 300 / 150 | five deterministic group-disjoint folds; all paired/source variants held by bug_id (bug_id; algorithm-size groups) | `{"0": 150, "1": 150}` | 66.00% | 66.00% | 65.9985% | 66.0% | 10.8382 | +0.00 pp | angle_reupload_hybrid_4q_depth2 | unavailable |

## Per-class selected-model results

| Task | Per-class precision / recall / F1 / support |
|---|---|
| Bugs4Q core | `{"0": {"f1": 0.39999999999999997, "precision": 0.5263157894736842, "recall": 0.3225806451612903, "support": 31}, "1": {"f1": 0.5945945945945946, "precision": 0.5116279069767442, "recall": 0.7096774193548387, "support": 31}}` |
| Bugs4Q core-only transfer | `{"0": {"f1": 0.3673469387755102, "precision": 0.5, "recall": 0.2903225806451613, "support": 62}, "1": {"f1": 0.5866666666666667, "precision": 0.5, "recall": 0.7096774193548387, "support": 62}}` |
| Bugs4Q pooled | `{"0": {"f1": 0.5303030303030304, "precision": 0.5, "recall": 0.5645161290322581, "support": 62}, "1": {"f1": 0.46551724137931033, "precision": 0.5, "recall": 0.43548387096774194, "support": 62}}` |
| Bugs4Q-NA | `{"0": {"f1": 0.5333333333333333, "precision": 0.5517241379310345, "recall": 0.5161290322580645, "support": 31}, "1": {"f1": 0.5625, "precision": 0.5454545454545454, "recall": 0.5806451612903226, "support": 31}}` |
| Bugs4Q-NA-only transfer | `{"0": {"f1": 0.5081967213114755, "precision": 0.5166666666666667, "recall": 0.5, "support": 62}, "1": {"f1": 0.5238095238095237, "precision": 0.515625, "recall": 0.532258064516129, "support": 62}}` |
| LintQ | `{"0": {"f1": 0.38834951456310685, "precision": 0.30303030303030304, "recall": 0.5405405405405406, "support": 74}, "1": {"f1": 0.6631016042780749, "precision": 0.7848101265822784, "recall": 0.5740740740740741, "support": 216}}` |
| QBugs HornBro subset | `{"0": {"f1": 0.7626404494382022, "precision": 0.8056379821958457, "recall": 0.724, "support": 750}, "1": {"f1": 0.7855329949238579, "precision": 0.7493946731234867, "recall": 0.8253333333333334, "support": 750}}` |
| QMutBench three-class research | `{"0": {"f1": 0.586046511627907, "precision": 0.4405594405594406, "recall": 0.875, "support": 72}, "1": {"f1": 0.43243243243243246, "precision": 0.5581395348837209, "recall": 0.35294117647058826, "support": 68}, "2": {"f1": 0.2380952380952381, "precision": 0.5263157894736842, "recall": 0.15384615384615385, "support": 65}}` |
| Qolumbina ideal | `{"0": {"f1": 0.25806451612903225, "precision": 0.19672131147540983, "recall": 0.375, "support": 32}, "1": {"f1": 0.7453874538745388, "precision": 0.8347107438016529, "recall": 0.6733333333333333, "support": 150}}` |
| Quantum platform comment task | `{"0": {"f1": 0.7403314917127071, "precision": 0.788235294117647, "recall": 0.6979166666666666, "support": 96}, "1": {"f1": 0.8971553610503281, "precision": 0.8760683760683761, "recall": 0.9192825112107623, "support": 223}}` |
| bugs4q_robust_execution | `{"0": {"f1": 0.5263157894736842, "precision": 0.5128205128205128, "recall": 0.5405405405405406, "support": 1110}, "1": {"f1": 0.5, "precision": 0.5142857142857142, "recall": 0.4864864864864865, "support": 1110}}` |
| qsimbench_backend_fingerprint | `{"0": {"f1": 0.6577181208053691, "precision": 0.6621621621621622, "recall": 0.6533333333333333, "support": 150}, "1": {"f1": 0.6622516556291391, "precision": 0.6578947368421053, "recall": 0.6666666666666666, "support": 150}}` |

## Executed optimization

- **Bugs4Q-Robust:** six 4-epoch QNN configurations were trained on five issue-grouped folds; the angle re-upload, 4-qubit, depth-2, learning-rate 0.07 candidate was repeated with two additional seeds. Candidate OOF accuracies were 55.41%, 48.65%, and 54.05% (mean 52.70%, sample SD 3.58 percentage points). The fixed three-seed probability ensemble reached 50.00%. The previous 51.35% QNN remains selected because the apparent gain is seed-unstable and the ensemble is lower.
- **Bugs4Q core:** two 4-epoch QNNs were trained on the saved five-fold issue-group split. Their OOF accuracy was 45.16% (3-qubit amplitude) and 50.00% (4-qubit angle re-upload), both below the 51.61% incumbent; the incumbent remains selected. Predictions and fold checkpoints are in `remaining_task_sweeps/bugs4q_core/`.
- **QSimBench:** six 4-epoch configurations were trained on five algorithm-size grouped folds. The same finalist across three seeds reached 65.00%, 62.67%, and 63.33% (mean 63.67%, sample SD 1.20 pp); the probability ensemble reached 64.33%. The prior 66.00% candidate remains selected.
- **QMutBench three-class:** the prior bounded six-configuration, 24-epoch QNN sweep was followed by a 24-epoch rerun of its strongest single model (2-qubit amplitude hybrid, depth 2, learning rate 0.08), on the saved three-fold origin split. The extra seed scored 46.34%; the two-seed probability ensemble reached 47.32% accuracy, 41.89% macro F1, and 46.06% balanced accuracy, up 0.49 pp from the previous 46.83% QNN ensemble. The additional seed 2027 was attempted twice but failed the runner's 700 MiB preflight; no training began for that seed. The two-seed ensemble is selected provisionally on development OOF accuracy.
- **Other eight standalone tasks:** their existing grouped QNN configuration comparisons and OOF predictions are retained as the incumbents. A further local sweep began for Bugs4Q-NA but stopped when available RAM fell to 494.1 MiB, below the 500 MiB floor. That task has partial checkpoints but no complete summary/OOF score and is marked unscored in `remaining_task_sweeps/remaining_task_sweep_status.json`. The remaining task-specific runners were not completed because the environment has no PennyLane, Torch, or Qiskit, and local RAM subsequently fell to about 12 MiB. No external execution was used.

## Protocol and resources

The new binary runs used identical deterministic five-fold group-disjoint partitions, training-fold min-max scaling, supervised cross-entropy, Adam, and validated parameter-shift gradients. The QMutBench runs reused the saved origin-group folds and fold-specific selected features with training-only scaling. OOF probabilities were recomputed for seed ensembles by aligned sample ID, label, group, and fold. No frozen-test data, classical model tuning, cross-source training, or cloud compute was used.

Training used the local NumPy CPU statevector implementation. No GPU was detected. Available system RAM fluctuated around the safety floor; the QMutBench third-seed attempts stopped at the explicit 700 MiB preflight. The prior broader binary sweep that fell below 500 MiB remains partial and unscored.

## Artifacts

- Candidate and fold-level rows: `qml_standalone_candidate_results.csv`.
- Selected per-task development model: `task_winners.csv`.
- Seed variability and probability-ensemble metrics: `multiseed_finalist_summary.csv`, `multiseed_ensemble_summary.csv`.
- Development OOF predictions: `qml_standalone_predictions/`.
- Fold checkpoints and preprocessing metadata: `qml_standalone_checkpoints/`.
- Configuration, task labels/groups, and blocked-run status: `qml_standalone_run_manifest.json`.
- Independent checks: `qml_standalone_audit.md`.

**Status:** local optimization improved the QMutBench development ensemble by 0.49 percentage points. The new Bugs4Q core sweep did not beat its incumbent; the binary finalists did not produce better seed-ensemble results. Final test results remain unavailable; the campaign does not establish unseen-test accuracy. The remaining tasks are blocked by the local RAM floor and missing task-specific QML libraries.
