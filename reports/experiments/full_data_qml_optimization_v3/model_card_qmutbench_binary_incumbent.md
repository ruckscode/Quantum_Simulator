# Model card: QMutBench binary development incumbent

- Model: classical ExtraTrees, mutation-aware feature model; this is not a QNN.
- Target: killed mutant (WOO or OPO)=1; Survivor=0. Mapping verified against 578,117 saved development OOF rows.
- Evaluation: grouped 5-fold OOF over 578,117 records and 305 QMutBench origins; 79.33% accuracy, 79.33% macro F1, 79.41% balanced accuracy. No frozen-test data is reported here.
- Intended use: research baseline for QMutBench mutation outcomes; does not diagnose fault cause, localize bugs, or predict execution reproducibility.
- Limitations: classical model; outcome distribution and mutation-generation process are benchmark-specific. Full QASM-enhanced QNN comparison is outstanding.
- Artifacts: `data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/summary.json`, `data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/mutation_diff_oof.npz`, and `data/qmutbench_ml/master_optimization_v1/best_model_metadata.json`.
