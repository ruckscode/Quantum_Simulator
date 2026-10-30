# QMutBench Logistic Regression development tuning

The target maps WOO/OPO to detected (1) and Survivor to undetected (0). Tuning used only the verified 578,117 development records across the saved three-fold origin-disjoint OOF split. The frozen test was not read.

The existing balanced `liblinear`, C=1 baseline and every candidate use the same seven historical features, records, and folds. Numeric scaling, imputation, and one-hot category vocabularies were fit within each training fold.

Grid: C=0.001, 0.0316228, 1, 31.6228, 1000; solvers=liblinear, lbfgs, saga; class weights=['balanced', None]; max_iter=500; tol=0.0001. Selection used mean fold macro F1.

| Model | Accuracy | Balanced accuracy | Macro F1 (pooled OOF) | Fold macro F1 mean +/- SD |
|---|---:|---:|---:|---:|
| baseline_liblinear_C1_balanced | 69.27% | 69.35% | 69.27% | 69.26% +/- 0.81% |
| tuned_17 | 69.39% | 69.38% | 69.36% | 69.36% +/- 0.62% |

Selected configuration: C=1, solver=lbfgs, class_weight=None. Selected minus baseline pooled OOF macro-F1: +0.097 percentage points. This is a small development difference; the candidate was frozen before final-test evaluation.

`fold_metrics.json` includes per-class metrics and convergence iteration counts for every configuration and fold. `oof_predictions.csv` contains paired row predictions for the baseline and selected candidate. The full grid and fold assignment are in `configurations.csv` and `split_manifest.csv`.

Runtime: 455.78s. Source SHA-256: `3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778`. Split manifest SHA-256: `7d203ac4cd6878a1caaf19eb768bb9a1e4f322b42fc56475cdc603591cc250ce`.
