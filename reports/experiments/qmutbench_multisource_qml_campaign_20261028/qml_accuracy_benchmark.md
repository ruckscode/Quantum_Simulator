# QML Accuracy Benchmark

**Scope:** development grouped out-of-fold predictions only. Final-test accuracy is blank for every model because optimization remains in progress and held-out/frozen test data remain untouched. Tasks have different labels and cohorts; compare only within a task.

| Dataset/task | Best development QML model | Accuracy | Macro F1 | Balanced accuracy | Matched baseline accuracy |
|---|---|---:|---:|---:|---:|
| Bugs4Q core | angle_reupload_vqc | 51.6129% | 49.7297% | 51.6129% | 53.2258% (logistic_4d) |
| Bugs4Q core-only transfer | angle_hybrid_readout | 50.0% | 47.7007% | 50.0% | 51.6129% (random_forest_4d) |
| Bugs4Q pooled | angle_hybrid_readout | 50.0% | 49.791% | 50.0% | 52.4194% (extra_trees_4d) |
| Bugs4Q-NA | quantum_equal_probability_mean | 54.8387% | 54.7917% | 54.8387% | 53.2258% (extra_trees_4d) |
| Bugs4Q-NA-only transfer | angle_hybrid_readout | 51.6129% | 51.6003% | 51.6129% | 51.6129% (logistic_4d) |
| LintQ | angle_hybrid_readout | 56.5517% | 52.5726% | 55.7307% | 67.2414% (random_forest_4d) |
| QBugs HornBro subset | angle_hybrid_readout | 77.4667% | 77.4087% | 77.4667% | 84.3333% (random_forest_4d) |
| QMutBench three-class research | equal_probability_qnn_ensemble | 46.8293% | 43.4176% | 45.8865% | 49.2683% (extra_trees_4f) |
| Qolumbina ideal | angle_vqc | 62.0879% | 50.1726% | 52.4167% | 61.5385% (extra_trees_4d) |
| Quantum platform comment task | angle_hybrid_readout | 85.2665% | 81.8743% | 80.86% | 91.5361% (random_forest_4d) |
| bugs4q_robust_execution | amplitude_hybrid_3q_depth2 | 51.3514% | 51.3158% | 51.3514% | not evaluated |
| qsimbench_backend_fingerprint | angle_reupload_hybrid_4q_depth2 | 66.0% | 65.9985% | 66.0% | not evaluated |

## Completed QNN outcomes

- Bugs4Q-Robust: 2,220 observations and 37 issue groups, five grouped folds. Best completed QNN: 3-qubit depth-2 amplitude hybrid, 51.35% accuracy, 51.32% macro F1, 51.35% balanced accuracy; ten fold checkpoints saved.
- QSimBench: 300 observations and 150 algorithm-size groups, five grouped folds. Best completed QNN: 4-qubit depth-2 angle re-upload hybrid, 66.00% accuracy, 66.00% macro F1 and balanced accuracy; ten fold checkpoints saved.
- QMutBench three-class: 205 development records and 12 origins, using saved three-fold origin splits and fold-selected features. The 24-epoch sweep ensemble reached 46.83% accuracy, 43.42% macro F1, and 45.89% balanced accuracy; best single QNN reached 46.34% accuracy. The matched Extra Trees baseline reached 49.27% accuracy on the same pilot. QNN member checkpoints are saved.
- Expanded 20-epoch binary QNN sweep stopped after the runtime guard detected 473.4 MiB available, below the 500 MiB floor. Partial weights are retained but unscored. Completed baseline QNN runs remain intact.

## Artifacts

- Candidate-level metrics, per-class results, configurations, runtime, checkpoint and OOF paths: `qml_model_accuracy_benchmark.csv`.
- Per-task development winner: `best_qml_model_by_task.csv`.
- Independent metric and checkpoint archive audit: `completed_qnn_artifact_audit.json`.
- Incomplete sweep evidence: `qnn_hyperparameter_sweep_status.json`.
- QML-only external sweep: `QML_Only_External_Sweeps.ipynb` / `qml_only_external_sweep_bundle.zip`; validated but not executed.
- No held-out test score is reported.
