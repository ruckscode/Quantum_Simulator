# QML source/task semantic validation audit

Tasks audited: 18; verified: 18; review required: 0. Frozen test accessed: no.

| Source/task | Manifest status | Training eligibility | Rows accepted/rejected | Groups | Folds | Shared feature fingerprints | Cross-fold duplicate rows | Conflicting-label duplicate fingerprints | Issues |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| bugs4q_joint / core_only_transfer | VERIFIED | READY_FOR_TRAINING | 124/0 | 62 | 5 | 0 | 0 | 0 | none |
| bugs4q_core / source_state | VERIFIED | READY_FOR_TRAINING | 62/0 | 31 | 5 | 0 | 0 | 0 | none |
| bugs4q_joint / source_state | VERIFIED | READY_FOR_TRAINING | 124/0 | 62 | 5 | 0 | 0 | 0 | none |
| bugs4q_joint / na_only_transfer | VERIFIED | READY_FOR_TRAINING | 124/0 | 62 | 5 | 0 | 0 | 0 | none |
| bugs4q_na / source_state | VERIFIED | READY_FOR_TRAINING | 62/0 | 31 | 5 | 0 | 0 | 0 | none |
| bugs4q_robust / harness_execution_outcome | VERIFIED | READY_FOR_TRAINING | 2220/0 | 37 | 5 | 0 | 0 | 0 | none |
| lintq / warning_triage | VERIFIED | BLOCKED_FEATURE_LABEL_AMBIGUITY | 290/0 | 228 | 5 | 35 | 115 | 8 | none |
| lintq / warning_triage_context_v2 | VERIFIED | READY_EXPLORATORY_WITH_LABEL_AMBIGUITY | 290/0 | 217 | 5 | 0 | 0 | 0 | none |
| mute_alg_hornbro / compatible_subset_qasm_v1 | VERIFIED | INELIGIBLE_PARTIAL_SOURCE_LOW_GROUP_DIVERSITY | 5440/0 | 11 | 5 | 0 | 0 | 0 | none |
| mute_alg_hornbro / compatible_subset_qasm_v2 | VERIFIED | READY_FOR_TRAINING | 9140/0 | 812 | 5 | 0 | 0 | 0 | none |
| qbugs_hornbro / correct_buggy_circuit_state | VERIFIED | BLOCKED_FEATURE_LABEL_AMBIGUITY | 1500/0 | 750 | 5 | 67 | 701 | 34 | none |
| qbugs_hornbro / correct_buggy_circuit_state_qasm_v2 | VERIFIED | READY_FOR_TRAINING | 1496/0 | 748 | 5 | 0 | 0 | 0 | none |
| qmutbench / three_class_mutation_outcome_exploratory | VERIFIED | READY_FOR_TRAINING | 205/0 | 12 | 3 | 0 | 0 | 0 | none |
| qolumbina_ideal / seeded_mutant_detection_ideal | VERIFIED | READY_FOR_TRAINING | 182/0 | 26 | 5 | 0 | 0 | 0 | none |
| qsimbench / backend_fingerprint | VERIFIED | READY_EXPLORATORY_WITH_FEATURE_COLLISIONS | 300/0 | 150 | 5 | 50 | 171 | 15 | none |
| quantum_platform_bugs / comment_only_classification | VERIFIED | BLOCKED_CROSS_FOLD_DUPLICATE_COMMENTS | 319/0 | 314 | 5 | 7 | 15 | 0 | none |
| quantum_platform_bugs / comment_only_classification_v2 | VERIFIED | READY_FOR_TRAINING | 319/0 | 304 | 5 | 0 | 0 | 0 | none |
| qmutbench / binary_mutation_outcome | VERIFIED | PREPROCESSING_READY_TRAINING_BLOCKED_MEMORY | 578117/0 | 305 | 5 | None | None | None | none |

## Duplicate-feature decisions

- **lintq / warning_triage — BLOCKED_FEATURE_LABEL_AMBIGUITY:** The current warning_message-only feature repeats across folds with conflicting TP/FP labels; include justified context or redesign the split/feature policy before new training.
- **lintq / warning_triage_context_v2 — READY_EXPLORATORY_WITH_LABEL_AMBIGUITY:** Full rule/warning/source-context duplicates are grouped within folds; 3 records in 1 identical-context components retain conflicting manual labels.
- **qbugs_hornbro / correct_buggy_circuit_state — BLOCKED_FEATURE_LABEL_AMBIGUITY:** Exact model-input text/metadata fingerprints recur across folds with conflicting correct/buggy labels; reconcile source pairs or evaluate a duplicate-component split before new training.
- **qsimbench / backend_fingerprint — READY_EXPLORATORY_WITH_FEATURE_COLLISIONS:** Quantized output descriptors recur across held-out circuit settings with both backend labels; preserve this limitation in any generalization claim.
- **quantum_platform_bugs / comment_only_classification — BLOCKED_CROSS_FOLD_DUPLICATE_COMMENTS:** Exact comment text is repeated across issue groups and validation folds; regroup duplicate comment components before new training.
