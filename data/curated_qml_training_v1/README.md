# Curated QML training bundle (v1)

This folder contains only a small, auditable set for the bounded work. The three sources remain separate tasks; their labels are not pooled.

- `qmutbench_binary/`: sampled five-fold, origin-grouped QMutBench pilot data. Includes selected metadata, selected QASM-diff features, and fold/role manifest. The parent development split has 578,117 rows and 305 origins; the local bundle contains only the sampled records used for train/validation roles. Frozen-test rows are excluded.
- `qolumbina_ideal/`: prior grouped OOF data/results for the 182-row, 26-program ideal-backend seeded-mutant detection task.
- `qbugs_correct_vs_buggy/`: prior grouped OOF data/results for the 1,500-row paired correct-versus-buggy circuit snapshot task.
- `source_registry.csv`: counts and task definitions.
- `training_results.json`: completed benchmark results and limitations.

QMutBench predicts Survivor vs WOO/OPO mutation outcomes; Qolumbina predicts empirical DOSS detection of seeded mutants; QBugs predicts correct vs buggy circuit snapshots. They are related but not the same target, so each retains its own model and accuracy.

Two QMutBench sampled five-fold OOF comparisons are recorded. The initial 2,560-row pilot scored Binary VQC 57.66% accuracy / 56.92% macro-F1, Extra Trees 57.11% / 56.76%, and Logistic Regression 55.70% / 55.49%. The larger matched 10,240-row sample scored Binary VQC 59.25% / 57.07%, Extra Trees 59.35% / 57.64%, and Logistic Regression 58.90% / 55.51%. The larger run used the same origin-grouped reference folds, balanced validation support, and reduced four-feature schema across all three models. On this matched sample, Extra Trees is narrowly ahead of the VQC, so the QNN has not shown a performance advantage.

Two 15,360-row sampled OOF runs used 1,024 training and 1,536 validation examples per class per fold, eight epochs, and the same reduced four-feature schema. Seed 20261028 scored VQC 58.29% accuracy / 56.47% macro-F1, Extra Trees 58.74% / 56.36%, and Logistic Regression 58.18% / 55.21%. Seed 20261029 scored VQC 58.44% / 55.61%, Extra Trees 58.55% / 56.53%, and Logistic Regression 58.65% / 54.86%. Across the two runs, the descriptive mean accuracy / macro-F1 was VQC 58.37% / 56.04%, Extra Trees 58.64% / 56.45%, and Logistic Regression 58.41% / 55.03%. These two sampled runs show no consistent QNN advantage: VQC macro-F1 was slightly higher in the first seed and lower than both classical controls in the repeat. The means are descriptive across resampled development OOF runs, not an independent test estimate.

These are sampled development OOF scores, not full-data or frozen-test estimates. They are not directly comparable to the full-data mutation-aware classical champion (79.33% accuracy / 79.33% macro-F1 / 79.41% balanced accuracy on 578,117 development rows) because the larger classical result uses all development rows and a richer mutation-aware feature schema. The earlier matched runs used PennyLane `default.qubit` on CPU. A later 2,048-training-records-per-class/fold run used `lightning.gpu` with `adjoint` via a Kaggle-only runtime monkeypatch; its summary is recorded separately and flagged for fold-2 prediction and runtime consistency audits. The repository trainer still does not expose that GPU choice as a supported configuration. The original full QNN campaign remains checkpointed and incomplete.

QMutBench sample selection was reconstructed deterministically from the pinned input archive and logged Kaggle configuration; the source checksum and version caveat are recorded in `qmutbench_binary/dataset_config.json`. The original Kaggle sample manifest was not directly exported, so exact row-for-row equivalence is not claimed. Qolumbina and QBugs OOF files are copied from completed, source-specific campaign outputs. The QBugs summary copy’s stale dataset description was corrected from the normalized manifest and OOF alignment; the original campaign output was left untouched. `curation_audit.json` records the row, label, and group-fold checks.


A matched QML-vs-classical report with per-seed metrics, fold dispersion, runtime, and limitations is available in `matched_qml_classical_comparison.md`. Both models were evaluated on the same sampled folds/features; compute time was not equalized.

## Integration boundary

These QMutBench results are benchmark evidence for a separate mutation-outcome classifier. They are not the simulator diagnosis pipeline accuracy and are not currently an inference-time component of `diagnose_circuit()`: the QMutBench target and reduced feature contract do not, by themselves, identify program or hardware faults from arbitrary submitted circuits. Integration should be claimed only after an adapter supplies the same versioned features at inference and an end-to-end grouped evaluation validates it against labeled diagnosis outcomes. The matched Extra Trees metric is the fair direct classical comparison for the QNN experiments in this folder.
