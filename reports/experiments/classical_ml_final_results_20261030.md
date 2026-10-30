# Classical ML optimization: final source-by-source results

**Completed:** grouped, nested model searches across nine source adapters, sparse KNN comparisons, the same-target pooled source task, and the matched QMutBench classical benchmark. Model/feature selection used inner folds only; reported source scores are pooled outer-fold out-of-fold predictions. Frozen test data was not accessed.

## Source-specific results

Each task searched 104 word/character SVM, logistic, SGD, and ComplementNB configurations plus 32 cosine KNN configurations. We selected the candidate inside each training partition using inner GroupKFold accuracy and scored it once on outer held-out groups. This reduces the winner-selection bias in the original “best of 20” screen.

| Source/task | Rows / groups | Pooled OOF accuracy | Macro-F1 | Balanced accuracy | Inner-selected model family by folds |
|---|---:|---:|---:|---:|---|
| qbugs_qasm_v2 (correct vs. buggy circuit) | 1,496 / 748 | 92.71% | 92.69% | 92.71% | linear text models (5/5 folds) |
| quantum_platform_comment_v2 (comment label classification) | 319 / 304 | 94.04% | 92.67% | 91.29% | linear text models (5/5 folds) |
| bug_correctness_three_source_pooled (merged correct/fixed vs. buggy) | 1,620 / 810 | 89.63% | 89.60% | 89.63% | linear text models (5/5 folds) |
| qsimbench_exploratory (backend identity (exploratory)) | 300 / 150 | 82.33% | 82.33% | 82.33% | linear text models (5/5 folds) |
| lintq_context_v2_conflict_excluded (warning TP vs. FP, conflict-excluded) | 287 / 216 | 79.44% | 67.92% | 65.91% | sparse_cosine_knn (4/5 folds), linear text models (1/5 folds) |
| qolumbina_ideal (ideal-backend detector outcome) | 182 / 26 | 84.07% | 54.16% | 54.69% | linear text models (5/5 folds) |
| bugs4q_robust_execution (harness pass vs. fail) | 2,220 / 37 | 56.76% | 56.73% | 56.76% | sparse_cosine_knn (1/5 folds), linear text models (4/5 folds) |
| bugs4q_core (source state) | 62 / 31 | 51.61% | 51.56% | 51.61% | sparse_cosine_knn (5/5 folds) |
| bugs4q_na (source state) | 62 / 31 | 50.00% | 48.92% | 50.00% | sparse_cosine_knn (2/5 folds), linear text models (3/5 folds) |
| bugs4q_pooled (source state, core + NA) | 124 / 62 | 49.19% | 47.97% | 49.19% | sparse_cosine_knn (5/5 folds) |

Notes: Qolumbina has a 82.42% majority-class baseline; its 84.07% accuracy corresponds to only 54.16% macro-F1 and 54.69% balanced accuracy. LintQ excludes the entire group with three conflicting exact-context labels. Robust execution has repeated identical code within issue groups, so training deduplicated same-label/same-group code with duplicate-count weights while scoring every held-out execution row. QSimBench remains exploratory because descriptor collisions are documented. Bugs4Q core/NA have only 31 independent groups each, so fold estimates are noisy.

## QMutBench (full development dataset)

The strongest completed full-feature classical result remains the mutation-aware ExtraTrees model: **79.33% accuracy, 79.33% macro-F1, 79.41% balanced accuracy** on 578,117 development rows / 305 origins. The matched full-feature ExtraTrees baseline scored 77.34% accuracy. A completed HistGradientBoosting screen scored 77.77% with mutation-difference features. These are grouped development OOF results, not frozen-test results.

For direct comparison to the QNN four-feature representation, eight freshly trained models used the exact saved five-fold split and fold-local encoder rule. Their pooled OOF accuracy:

| Classifier | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| extra_trees | 57.00% | 56.61% | 56.73% |
| hist_gradient_boosting | 56.93% | 56.48% | 56.64% |
| random_forest | 56.79% | 56.46% | 56.54% |
| sgd_logistic | 55.83% | 55.39% | 55.54% |
| logistic | 55.73% | 55.00% | 55.34% |
| linear_svc | 55.69% | 54.95% | 55.30% |
| logistic_balanced | 55.47% | 55.41% | 55.68% |
| dummy_prior | 51.84% | 34.14% | 50.00% |

The best four-feature control was ExtraTrees at **57.00% accuracy**. This narrower control is not the full-feature QMutBench benchmark; its purpose is a feature-matched comparison with the QNN.

## Merged sources

The valid three-source target merge (Bugs4Q core + Bugs4Q-NA + QBugs; correct/fixed=0 vs. buggy=1) scored **89.63% accuracy**, **89.60% macro-F1**, and **89.63% balanced accuracy** under the nested model-selection procedure. QBugs alone scored 92.71%, so the merge did not improve accuracy. The earlier merged KNN screen at 80.37% was replaced by this inner-selected nested result.

Bugs4Q core + NA alone scored **49.19%** in the nested source-state task. Different targets such as QMutBench mutation outcome, comments, detector outcome, and execution pass/fail were not combined into one meaningless aggregate score.

## Why several scores remain low

- Bugs4Q core and NA each have 62 rows across only 31 issue groups. Holding out entire issues leaves very little training data per fold; the observed 50–52% accuracy is close to chance.
- Robust execution has repeated reports but only 37 independent issue groups, and accuracy is 56.76% after group-safe validation. Repeated runs do not provide 2,220 independent examples.
- Qolumbina is highly imbalanced. Accuracy close to its 82.42% majority baseline does not mean strong minority detection.
- Some source targets are genuinely different tasks. A high comment-classification score does not mean strong bug detection or hardware diagnosis.

## Artifacts

- Nested source results and per-fold predictions: `reports/experiments/classical_nested_combined_search_v2_20261030/`
- Source text-model search: `experiments/run_classical_source_nested_search.py`
- Sparse KNN nested search: `experiments/run_classical_sparse_knn_nested.py`
- QMutBench matched four-feature metrics and OOF rows: `reports/experiments/qmutbench_classical_development_matched_20261030/`
- QMutBench full-feature five-fold reference: `data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/summary.json`
- Three-source pooled input integrity audit: `reports/experiments/qml_source_preprocessing_20261029/classical_nine_source_v2_inputs/bug_correctness_three_source_pooled_audit.json`
- Earlier broad 20-model screens: `reports/experiments/local_grouped_classical_suite_nine_source_v2_20261029/classical_suite_results.csv` and `reports/experiments/local_grouped_classical_suite_lintq_conflict_excluded_20261030/classical_suite_results.csv`
