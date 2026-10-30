# Matched QNN and classical comparison — 2026-10-29

All entries are five-fold grouped development OOF results. Each classical model used the same fixed hashed-text feature representation, fold assignments, and eligible training rows as its QNN counterpart. No frozen test was accessed.

| Task | QNN | Features | QNN macro F1 | Best fixed classical macro F1 | QNN minus control (pp) |
|---|---|---:|---:|---:|---:|
| Bugs4Q core | angle_reupload_4q_depth2_lr0035 | 4 | 0.499 | extra_trees_balanced 0.564 | -6.56 |
| Bugs4Q core | amplitude_3q_depth1_lr004 | 8 | 0.449 | decision_tree_balanced 0.503 | -5.39 |
| Bugs4Q-NA | angle_reupload_4q_depth2_lr0035 | 4 | 0.470 | decision_tree_balanced 0.563 | -9.34 |
| Bugs4Q-NA | amplitude_3q_depth1_lr004 | 8 | 0.461 | extra_trees_balanced 0.577 | -11.58 |
| Bugs4Q pooled | angle_reupload_4q_depth2_lr0035 | 4 | 0.459 | random_forest_balanced 0.516 | -5.74 |
| Bugs4Q pooled | amplitude_3q_depth1_lr004 | 8 | 0.494 | extra_trees_balanced 0.516 | -2.26 |
| Bugs4Q core-only transfer | angle_reupload_4q_depth2_lr0035 | 4 | 0.468 | random_forest_balanced 0.505 | -3.79 |
| Bugs4Q core-only transfer | amplitude_3q_depth1_lr004 | 8 | 0.540 | logistic_balanced 0.506 | +3.38 |
| Bugs4Q-NA-only transfer | angle_reupload_4q_depth2_lr0035 | 4 | 0.510 | random_forest_balanced 0.540 | -3.04 |
| Bugs4Q-NA-only transfer | amplitude_3q_depth1_lr004 | 8 | 0.492 | extra_trees_balanced 0.531 | -3.93 |
| LintQ | angle_reupload_4q_depth2_lr0035 | 4 | 0.426 | random_forest_balanced 0.669 | -24.29 |
| LintQ | amplitude_3q_depth1_lr004 | 8 | 0.426 | random_forest_balanced 0.669 | -24.37 |
| QBugs HornBro subset | angle_reupload_4q_depth2_lr0035 | 4 | 0.718 | random_forest_balanced 0.832 | -11.37 |
| QBugs HornBro subset | amplitude_3q_depth1_lr004 | 8 | 0.706 | random_forest_balanced 0.848 | -14.20 |
| Qolumbina ideal | angle_reupload_4q_depth2_lr0035 | 4 | 0.440 | random_forest_balanced 0.481 | -4.10 |
| Qolumbina ideal | amplitude_3q_depth1_lr004 | 8 | 0.452 | majority_dummy 0.452 | +0.00 |
| Quantum platform comment task | angle_reupload_4q_depth2_lr0035 | 4 | 0.894 | random_forest_balanced 0.903 | -0.81 |
| Quantum platform comment task | amplitude_3q_depth1_lr004 | 8 | 0.483 | extra_trees_balanced 0.912 | -42.86 |

The ‘best fixed classical’ column is the highest OOF score among five predeclared controls and is shown descriptively; it is not an independently selected final model. Per-class metrics, fold-level scores, confusion matrices, and each control result are in `matched_qnn_classical_comparison.json`.

The QNN configurations are 3-qubit depth-1 amplitude encoding of eight hashed features and 4-qubit depth-2 angle re-uploading of four features. Both use deterministic statevector simulation with four training epochs and fixed Adam settings. Classical controls are majority dummy, balanced Logistic Regression, balanced depth-4 Decision Tree, balanced Random Forest, and balanced Extra Trees.

These small source-specific tasks support a development comparison only. Qolumbina labels seeded mutation outcomes; LintQ labels warning triage; QSimBench elsewhere in the project labels backend identity. None implies physical hardware fault attribution.
