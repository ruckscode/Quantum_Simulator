# QMutBench final classical ML campaign - 2026-10-27

## Verdict

The best-supported development model remains mutation-aware Extra Trees with 23 features, fixed threshold 0.5, 100 trees, depth 22, minimum leaf size 15, `sqrt` feature sampling, and `balanced_subsample` weights. Its five-fold grouped OOF scores are macro F1 **0.793314**, accuracy **0.793329**, and balanced accuracy **0.794087** on 578,117 development records from 305 origins. The 85% target was not reached. No new candidate displaced it in this campaign.

The paired origin bootstrap versus the legacy eight-feature Extra Trees baseline gives a macro-F1 difference of +1.13 to +2.98 pp (2,000 resamples). This is evidence for the mutation-aware feature set within this development corpus; feature discovery and model choice used the same corpus, so it is not external validation.

The highest available frozen-test result remains the historical, retrospective 0.818375 macro F1 (accuracy 0.818504; balanced accuracy 0.818599), versus 0.788428 for the older full-development eight-feature model on the same 144,962 records and 70 origins. Its paired origin bootstrap interval was +1.55 to +4.52 pp. The test has been reused historically and was not accessed for this campaign's feature/model selection. No pristine untouched-origin test was available; future external generalization remains unverified.

## Execution checklist and integrity

- [x] Inspected branch `qmutbench-master-optimization`, HEAD `869aa55034d5016eb4bb4661f5390a7ff38cb8b7`, existing uncommitted research artifacts, model metadata, feature caches, grouped OOF assignments, prior reports, and QML pilot README/results.
- [x] Confirmed source SHA-256 `3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778`; labels WOO/OPO=1 and Survivor=0; dev counts 299,704/278,413 and 305 origins.
- [x] Verified the selected model artifact SHA-256 `7532be26b6959e7e62277220f0c0b78e648d723a0be35d2b693d68e1c9ebf085`, schema, threshold, and grouped OOF record/origin alignment from saved metadata/manifests.
- [x] Reused completed grouped folds, cached features, predictions, and experiment reports. No test data was loaded for this campaign.
- [x] Preserved the incomplete structural confirmation checkpoints and documented why fold 5 was not resumed.
- [x] Ran focused tests and syntax/output checks; see verification below.

No confirmed label, cache-alignment, or preprocessing defect was found in the audited results. The 15 QASM mutation-difference features are derived from original/mutant circuit syntax and operation metadata, not labels or origin names. Feature one-hot vocabularies and the QML pilot's scaling/feature selection are fit within training folds. The documented common-prefix/common-suffix alignment can be ambiguous on repeated/reordered gates; this is a feature limitation, not a confirmed leakage defect. OOF validation was origin-grouped, but related algorithm families can still occur across distinct origin names; the task estimates generalization to new origins, not unseen algorithm families.

## Error analysis

### Five-fold champion OOF

Confusion matrix, true rows `[survivor=0, detected=1]`, predicted columns `[0,1]`:

`[[226818, 51595], [67885, 231819]]`

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Survivor (0) | 0.7696 | 0.8147 | 0.7915 | 278,413 |
| Detected (1) | 0.8180 | 0.7735 | 0.7951 | 299,704 |

There are more false negatives (67,885) than false positives (51,595); detected-class recall is the weaker side. Balanced class support means the result is not explained by a severe global label imbalance. The 5-fold origin comparison improved accuracy on 246 origins, worsened on 57, and tied on 2, so gains are not limited to a few groups.

The saved three-fold subgroup analysis (a separate, earlier protocol; do not mix its values with the five-fold aggregate) identifies `Replace` as a plausible specialist direction: macro F1 0.7485 on 120,676 records, compared with 0.7921 for Add (437,983) and 0.8124 for Remove (19,458). Difficult algorithm slices include `grover-noancilla` (0.4831, n=10,095), `vqe` (0.6519, n=21,728), `ae` (0.6621, n=14,289), and `qwalk-noancilla` (0.6748, n=5,417). Several gate slices also have low scores (e.g. rxx 0.5857, h 0.6425, sx 0.6830). These are descriptive OOF slices with unequal class mixtures, not causal evidence that the metadata itself explains errors. The cached subgroup report does not provide per-class metrics for each slice; no per-slice causal claim is made.

No saved analysis established a prediction-confidence threshold that improves macro F1. Probability estimates are uncalibrated. The previously tested fixed threshold remains 0.5.

## Candidate ledger and prior classical evidence

| Candidate / hypothesis | Development result | Decision |
|---|---|---|
| Legacy 8-feature Extra Trees | 5-fold macro F1 0.773387; accuracy 0.773397; BA 0.774698 | Matched control; retain for context. |
| Mutation-aware 23-feature Extra Trees | 5-fold macro F1 0.793314; accuracy 0.793329; BA 0.794087; paired origin CI vs legacy +1.13 to +2.98 pp | **Champion retained.** |
| Nested model/threshold/calibration selection | Macro F1 0.792801 vs fixed-threshold nested reference 0.793450 | Rejected; retain fixed 0.5. |
| Logistic Regression tuned configuration | Macro F1 0.693626 on prior matched three-fold screen vs 0.692658 balanced baseline | Negligible gain; no repeated grid. |
| Random Forest | 0.75891 vs matched Extra Trees 0.76628 on prior grouped screen | Rejected; redundant to repeat. |
| Histogram Gradient Boosting | 0.77717; improved its own matched control (0.76847) by 0.87 pp, but below champion | Not promoted. CatBoost/XGBoost/LightGBM unavailable; no install justified. |
| Mutation feature ablation | Removing five gate/alignment context fields scored 0.76475 vs matched 0.76628; fold changes  +0.38, -0.46, -0.38 pp | Retain useful block; not a five-fold ablation. |
| Legacy/champion 50:50 probability blend | Macro F1 0.78528 | Rejected; no complementary improvement. |
| Threshold/calibration and class-weight screens | Prior nested threshold did not improve fixed threshold; current class-weight recipe is balanced-sub-sample | No repeated search; no evidence supports it. |
| Add eight native structure features to champion | Fold-1 screen +1.85 pp; clean folds 2-4 pooled -0.13 pp, paired origin 95% CI -0.55 to +0.27 pp; fold 5 interrupted | Rejected; not promoted. Fold 5 remains incomplete. |
| Operation-specialized Replace classifier | Shallow Decision Tree Replace-only specialist; five saved grouped folds; Replace macro F1 0.70884 vs champion 0.75463; hybrid overall macro F1 0.78252 vs 0.79331, every fold lower | Rejected. Paired origin-bootstrap 95% interval for overall macro-F1 delta: -1.38 to -0.79 pp. See targeted report. |
| Further native structure / interactions | Structure-only and gate composition screens lagged; champion-plus-depth add-on failed clean confirmation; simple interactions were previously negative or unstable | No further broad feature expansion justified. |

Earlier experiments report fit/runtime details and exact protocols in `qmutbench_exhaustive_optimization_20261026.md`, `qmutbench_exhaustive_campaign_manifest_20261025.json`, the logistic tuning directory, and the 2026-10-27 improvement campaign directory. Historical scores from different fold counts, training sample sizes, or feature schemas are kept separate above.

## Final targeted Replace-specialist experiment

The remaining plausible mutation-specific hypothesis was tested using a lightweight `DecisionTreeClassifier` with fixed `max_depth=8`, `min_samples_leaf=50`, balanced class weights, and threshold 0.5. `operation` is part of the documented model input schema, so a `Replace` route is available at prediction time when callers supply the required metadata. The model was trained only on Replace rows from each training fold, with fold-local one-hot vocabularies. For non-Replace validation rows, the hybrid retained the champion's saved out-of-fold probabilities.

Across 120,676 Replace records in all 305 origins, the champion's Replace macro F1 was 0.75463 and the specialist scored 0.70884. The specialist was worse on every fold (0.68384 to 0.71713 versus the champion's 0.72899 to 0.77914). The hybrid scored macro F1 0.78252, accuracy 0.78253, and balanced accuracy 0.78377, compared with the champion's 0.79331, 0.79333, and 0.79409. The hybrid's macro-F1 delta was -1.08 pp; paired origin-bootstrap 95% CI was -1.38 to -0.79 pp. The Replace-slice paired interval was -6.31 to -3.01 pp. The candidate is rejected; no follow-up tuning was done.

The complete five-fold fits took 2.82 seconds. Fit-start available RAM ranged from 784 to 798 MiB; after-fit RAM ranged 809 to 817 MiB. A later checkpoint-validation rerun stopped before refitting fold 5 at 679 MiB, below the 700 MiB start guard. The existing complete fold-5 checkpoint was retained; all five outputs were independently checked for exact ID, origin, label, fold, operation, champion score, and saved fit-metric alignment. Aggregate reconstruction used the validated checkpoints without another fit. Details and per-fold metrics are in `qmutbench_replace_specialist_20261027/research_report.md`.

## Latest structural trial and resource status

The additive structure trial used eight features: original/mutant depth, depth delta, both depths per qubit, mutant multi-qubit fraction, multi-qubit fraction delta, and gate-count delta. The fold-1 screen selected the trial for clean confirmation, so confirmation excluded all fold-1 origins from both training and validation. Fold 2 was -0.23 pp, fold 3 +0.06 pp, fold 4 -0.22 pp; pooled macro F1 was 0.786447 candidate vs 0.787772 matched champion. The 95% paired origin-bootstrap interval crosses zero. The fold-5 fit was interrupted when available RAM reached 458 MiB, below the 500 MiB floor; its predictions were never checkpointed. Folds 2-4, feature memmaps, and manifests remain saved. The hypothesis is rejected based on the clean partial evidence and is not worth a risky resume absent new evidence.

At the targeted-task preflight, available physical memory was 756 MiB (801 MiB after scientific-library imports); no Python training process was listed. `psutil` and CatBoost/XGBoost/LightGBM were absent from the default Python environment. The Windows process-inventory WMI call was denied by the environment, so a complete process command-line inventory could not be verified. No dependency install or new large ensemble fit was attempted. The targeted lightweight five-fold decision-tree screen fit in 2.82 seconds; available RAM was 784-798 MiB before fits and 809-817 MiB afterward. Earlier clean structural folds 2-4 required about 1,478 seconds for six fits in total; that runtime and the prior memory-floor event make a full new hyperparameter/operation-specific grouped search unsafe under current headroom.

## Handoff to the subsequent QML phase

Keep the next QML study isolated in `qmutbench_qml/`; do not alter the production bundle or simulator. The task is binary mutation outcome: `WOO`/`OPO` -> 1, `Survivor` -> 0. Use the saved development-only `Origin_program` grouped fold assignments and preserve record IDs. For family comparisons, match QML and classical controls on the pilot's same four fold-selected/scaled scalars (selection/scaling inside each training fold); use matched Extra Trees, Logistic Regression, and RBF SVC controls on identical records. Report the full 23-feature Extra Trees result only as context, with its different feature protocol stated. Use macro F1 primary, plus accuracy, balanced accuracy, per-class results, fold/origin uncertainty, runtime, memory, and circuit resources. The nine-origin pilot and family extension already exist and should not be repeated. The full-feature grouped development baseline is 0.793314 macro F1. Existing QML pilots did not establish superiority; do not claim quantum advantage. The reused frozen test stays out of QML selection.

## Verification and files

Focused tests: `tests/test_qmutbench_master_optimization.py`, `tests/test_qmutbench_inference.py`, and `tests/test_qmutbench_corpus_pipeline.py` - **23 passed in 3.58 seconds**. All four improvement, specialist, and checkpoint-finalizer scripts passed `py_compile`. Consolidated and campaign JSON files parsed successfully; `git diff --check` passed. A full repository suite was not run.

New deliverables for this handoff:
- `qmutbench_final_classical_campaign_20261027.md` - this report.
- `qmutbench_final_classical_campaign_20261027.json` - machine-readable result, candidate ledger, resource state, and QML handoff.

No model, source data, target label, split manifest, historical report, saved prediction, QML artifact, anomaly audit, or quantum simulator was modified. No new frozen-test evaluation or QML training was performed. The current working tree contains prior untracked research artifacts; they were preserved.
