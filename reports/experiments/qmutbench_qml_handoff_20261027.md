# QMutBench classical handoff for QML - 2026-10-27

## Task and labels

Binary mutation outcome on QMutBench. Map `WOO` and `OPO` to detected (1), and `Survivor` to undetected (0). Preserve `record_id` alignment and group all validation by `Origin_program`. Development data contains 578,117 records across 305 origins. Use the existing saved grouped folds; never create random row-level splits or bring the historically reused frozen test into model selection.

## Frozen classical reference

Champion: mutation-aware Extra Trees, fixed threshold 0.5; 100 trees, max depth 22, minimum leaf size 15, `max_features="sqrt"`, `class_weight="balanced_subsample"`, one job. Model SHA-256: `7532be26b6959e7e62277220f0c0b78e648d723a0be35d2b693d68e1c9ebf085`.

Exact ordered 23-feature schema:

`algorithm`, `operation`, `gate`, `dominant_state`, `qubits`, `gates`, `Position_int`, `position_fraction`, `qasm_diff_orig_span`, `qasm_diff_mut_span`, `qasm_diff_total_span`, `qasm_diff_expected_gate_match`, `qasm_diff_qubit_set_equal`, `qasm_diff_qubits_touched`, `qasm_diff_removed_arity`, `qasm_diff_added_arity`, `qasm_diff_exact_single_edit`, `qasm_diff_is_local`, `qasm_diff_removed_gate`, `qasm_diff_added_gate`, `qasm_diff_alignment_class`, `qasm_diff_local_prev_gate`, `qasm_diff_local_next_gate`.

Five-fold grouped OOF metrics: macro F1 **0.793314**, accuracy **0.793329**, balanced accuracy **0.794087**. The selected model improved over the legacy eight-feature Extra Trees control; the paired origin-bootstrap 95% macro-F1 difference interval was +1.13 to +2.98 pp. This is the full-feature context, not a directly matched comparison with the four-scalar QML representation.

The lightweight Replace-only tree trial was rejected. The production bundle and champion remain unchanged.

## QML matched comparison protocol

Use the isolated `qmutbench_qml/` implementation. Its bounded pilots select four scalar inputs inside each training fold from the same available metadata and mutation-diff feature information, fit scaling within training folds, and use the exact same four transformed columns, record IDs, and folds for QML and classical controls. Compare each QML family to matched Extra Trees, Logistic Regression, and RBF SVC using macro F1 primary plus accuracy, balanced accuracy, per-class results, and paired origin-level uncertainty. Report runtime, memory, circuit depth, qubits, and circuit evaluations where relevant. The 23-feature Extra Trees result is context only because it uses a different representation.

## Artifacts

- Five-fold champion OOF and saved fold assignment: `data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/mutation_diff_oof.npz`.
- Champion feature/cache manifests and mutation-difference cache: `data/qmutbench_ml/mutation_aware_v1/best_model_metadata.json` and `data/qmutbench_ml/qasm_mutation_diff_v1/`.
- Matched QML records/features/splits/results: `qmutbench_qml/results/` and its `README.md`.
- Classical full-feature report: `reports/experiments/qmutbench_final_classical_campaign_20261027.md`.
- Final Replace-specialist screen and rejected decision: `reports/experiments/qmutbench_replace_specialist_20261027/research_report.md`.

## Test caveat

The historical 144,962-record frozen set has prior reuse. Its 81.84% macro F1 is retrospective only and must not be used to select or tune QML candidates. No pristine untouched-origin test is available in this handoff. Do not claim quantum advantage from execution success or a mismatched baseline.
