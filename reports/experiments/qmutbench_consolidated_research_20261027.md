# QMutBench consolidated research audit - 2026-10-27

## Executive result

The development-selected candidate remains the existing mutation-aware Extra Trees bundle in `data/qmutbench_ml/mutation_aware_v1/`, fixed threshold 0.5, 23 features (8 existing metadata plus 15 QASM mutation-diff/context features). Its completed five-fold grouped confirmation scored pooled macro F1 **0.7933**, accuracy **0.7933**, balanced accuracy **0.7941**. Its matched baseline on the exact same folds and records scored **0.7734**, **0.7734**, and **0.7747**, respectively. The 85% aspirational macro-F1 target remains unmet.

The candidate was evaluated once on the saved frozen test after the development model and threshold were fixed. Features were extracted in a single process using 1,000-record checkpoints with a 500 MiB available-memory stop floor. The run completed all 144,962 rows in 508.7 seconds; the lowest logged available memory was about 619 MiB at finalization, and available memory later recovered to 904 MiB. Record IDs matched the saved 70-origin manifest exactly. This is a retrospective comparison because the frozen set has prior historical reuse.

## Scope, labels, integrity

- Verified official QMutBench source: 723,079 records, SHA-256 `3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778`.
- Label map: WOO/OPO -> detected (1); Survivor -> undetected (0). Development distribution: **299,704 detected (51.84%)**, **278,413 survivor (48.16%)**, 578,117 records and 305 `Origin_program` groups. The previously defined frozen test has 144,962 rows and 70 origins.
- Saved three- and five-fold manifests keep origins disjoint. Five-fold record hashes and row/group counts are in `data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/summary.json`.
- Mutation-diff extraction uses circuit syntax and mutation metadata only, with no target/origin as features; the cached development extraction covered all rows with zero parse errors. Prior 20-record training/inference parity check found no mismatch. Prefix/suffix alignment can be ambiguous for repetitive or reordered gates.
- No confirmed label error, group leakage, or learned preprocessing outside training folds was identified in audited artifacts. The saved frozen-test diagnostic confirms zero origin overlap with development. Logistic imputation, scaling, and one-hot encoding were fit within each training fold. Feature/model selection is still informed by this corpus; no compatible independent external dataset is verified.

## Development experiments

| Experiment | Matched result | Candidate result | Interpretation |
|---|---|---|---|
| Three-fold OOF: incumbent 8-feature ET vs mutation-aware 23-feature ET | Macro F1 0.7682; accuracy 0.7683; BA 0.7697 | Macro F1 0.7889; accuracy 0.7889; BA 0.7898 | +2.07 pp macro F1, aligned rows/folds; selection-informed. |
| Five-fold grouped confirmation | Macro F1 0.7734; accuracy 0.7734; BA 0.7747 | Macro F1 0.7933; accuracy 0.7933; BA 0.7941 | +1.99 pp pooled macro F1. Fold deltas: 1.56, 1.92, 1.52, 1.75, 3.21 pp; mean 1.99, SD 0.70, positive in all five. Consistent, but five folds are insufficient for strong inferential claims. |
| Nested 5-fold threshold/model/calibration selection | Fixed mutation ET at 0.5: F1 0.7934; acc 0.7935; BA 0.7942 | Nested selection: F1 0.7928; acc 0.7931; BA 0.7927 | No gain from selecting threshold/calibration; retain 0.5. |
| Logistic Regression grid, 7 legacy features, same 3 folds | Balanced liblinear C=1: F1 0.6927; acc 0.6927; BA 0.6935 | lbfgs C=1 unweighted: F1 0.6936; acc 0.6939; BA 0.6938; fold mean 0.6936 +/- 0.0062 | +0.097 pp pooled F1 only; far below mutation ET. No grid rerun warranted. |
| Histogram Boosting mutation screen | Its matched ET baseline F1 0.7685 | F1 0.7772 | Weaker than mutation ET. |
| Drop gate-context ablation | Full matched feature set F1 0.7663 | F1 0.7648 | Ablation hurt. |

Prior hyperparameter, class weight, threshold, calibration, and ablation work already covers the promising classical changes; no repeated experiment was run.

### Errors and difficult slices

At threshold 0.5, three-fold mutation-aware confusion matrix (true rows [survivor 0, detected 1], predicted columns [0,1]): `[[226435, 51978], [70058, 229646]]`. Errors: 51,978 FP and 70,058 FN. Detected F1 **0.7901** and survivor F1 **0.7877**. Matched incumbent errors were 53,311 FP and 80,664 FN; most improvement came from reducing missed detected mutants.

Hard slices (at least 5,000 records): `grover-noancilla` (N=10,095, macro F1 0.483), `vqe` (21,728, 0.652), `ae` (14,289, 0.662), `qwalk-noancilla` (5,417, 0.675), `portfoliovqe` (66,981, 0.680). Dominant-state Yes: N=415,624, F1 0.725; No: N=162,493, F1 0.935. Difficult gates include RXX (N=35,248, F1 0.586), H (31,525, 0.643), SX (29,476, 0.683), RY (32,328, 0.688). Add: N=437,983, F1 0.792; Remove: N=19,458, 0.812; Replace: N=120,676, 0.749. Per-origin accuracy improved in 246, worsened in 58, tied in one. These are descriptive, not grounds for subgroup tuning.

## Metric and protocol discrepancies

The older `qmutbench_oof_validation.json` macro F1 **0.7566** at threshold 0.485 (**0.7548** at 0.5) aggregates a different `master_optimization_v1/oof_predictions.csv` model. It is not the probability set used in the later matched comparison. The 0.7682 incumbent and 0.7889 mutation-aware scores use separate saved predictions from `performance_phase_full_v1` and `exhaustive_campaign_v1`, aligned by record ID, label, and fold. The 0.7928 nested score uses outer grouped folds with inner model/calibration/threshold selection. These figures are not interchangeable.

The comparison baseline is the saved full-development 8-feature Extra Trees model (`learning_curve_v1`), trained on all 578,117 development rows and already predicted on the same 144,962 test IDs. Its saved metrics are macro F1 0.7884, accuracy 0.7884, BA 0.7893. A separate 125,013-row/86-origin subset model scored 0.7346, 0.7349, 0.7371, and is context only. The historical Logistic Regression test comparison is another protocol and its paired macro-F1 interval crosses zero. The frozen test has prior reuse and is not pristine independent evidence.

## QML

No new QML run. The existing nine-origin/72-record pilot and extension remain. Extension macro F1: VQC 0.486 vs matched Logistic 0.494/RBF SVC 0.445; QSVC 0.422 vs RBF SVC 0.445; data re-uploading 0.469; quantum reservoir 0.347. Paired origin-bootstrap intervals were broad and included zero for comparisons with matched controls. VQC is the trainable QNN implementation; QSVC, re-uploading, reservoir are distinct families. The 24-origin expansion was stopped below 450 MiB available and produced no fold metrics. No QML family credibly beat its matched control. No quantum advantage or hardware validation is claimed; full-feature ML is context only because features/sample protocol differ.



## Persistent improvement campaign ? 2026-10-27

### Current leaderboard

| Rank / status | Candidate or reference | Development macro F1 | Development protocol | Frozen-test macro F1 | Decision |
|---|---|---:|---|---:|---|
| Champion | Mutation-aware Extra Trees, 23 features | **0.7933** | 5-fold grouped OOF, 578,117 rows / 305 origins | **0.8184** | Keep; test result is retrospective. |
| Matched incumbent | 8-feature Extra Trees | 0.7734 | Same 5-fold rows and folds | 0.7884 | Control for champion. |
| Nested selection strategy | ET/HGB, fold-local threshold/calibration | 0.7928 | Nested 5-fold grouped evaluation | ? | Did not beat fixed-threshold ET. |
| HGB mutation feature screen | Histogram Gradient Boosting | 0.7772 | Earlier 3-fold grouped screen | ? | Not directly comparable to five-fold champion; weaker in its matched screen. |
| Logistic Regression tuned | lbfgs, C=1, unweighted, 7 legacy features | 0.6936 | Same saved 3-fold folds as LR baseline | 0.7196 | Negligible dev gain; test paired interval crossed zero. |
| Structural add-on challenger | 23 mutation-aware + 8 native depth/complexity features | fold-1 screen: 0.7964; post-screen folds 2?4: 0.7864 | Screen: 1 grouped fold; confirmation subset: 3 folds, excludes screen-fold origins from all fits | ? | Reject for now: post-screen matched delta **?0.13 pp**, origin-bootstrap 95% CI **[?0.55,+0.27] pp**; fold 5 stopped at memory floor. |

Scores from different feature sets, sample coverage, or fold protocols are intentionally shown as context rather than a common ranked contest. The direct current-champion comparison is the five-fold row against the matched incumbent.

### Development uncertainty and origin effects

The 5-fold saved predictions were verified row-by-row aligned by record ID, label, origin, and fold. Candidate minus incumbent paired origin-bootstrap (2,000 resamples, seed 20261027) 95% intervals were **+1.13 to +2.98 pp macro F1**, **+1.13 to +2.99 pp accuracy**, and **+1.08 to +2.87 pp balanced accuracy**. Accuracy improved on 246 origins, worsened on 57, and tied on two. Thus the original mutation-aware gain is not driven by only a handful of origins in this OOF sample, although the candidate family was selected using this corpus and this is not external validation.

### Structural-feature experiment and stopping decision

Hypothesis: native circuit depth and multi-qubit complexity provide information beyond mutation-local edit alignment, especially for difficult algorithms and gates. The add-on was fixed in advance to `qasm_orig_depth`, `qasm_mut_depth`, `qasm_delta_depth`, both depths per qubit, mutant multi-qubit fraction, delta multi-qubit fraction, and delta gate count. All values came from the completed 578,117-row structural cache. The Extra Trees model retained the champion?s 100 trees, depth 22, leaf size 15, sqrt feature sampling, balanced-sub-sample weights, threshold 0.5, and fold seed; category vocabularies were fit on training records only. Fold 1 (screen) improved by 1.85 pp, so the candidate advanced. To prevent screen-label leakage, confirmation folds 2?4 trained without fold-1 origins and each used a freshly fitted matched 23-feature champion on identical training/validation records. Those three folds pooled to macro F1 0.7864 versus 0.7878, delta ?0.13 pp; the paired origin-bootstrap interval includes zero. Per-fold deltas: fold 2 ?0.23 pp, fold 3 +0.06 pp, fold 4 ?0.22 pp. The direction and size do not support promoting the add-on.

Fold 5 was interrupted before a prediction checkpoint when available RAM fell to **458 MiB**, below the 500 MiB stop floor. The active experiment process was interrupted; completed folds 2?4 and feature memmaps remain. At this point the screen had not shown a durable gain, and restarting the last expensive fit was not justified. The partial interval is exploratory and fold 5 remains unfinished. No final test was accessed for this candidate.

### Ranked remaining hypotheses and disposition

1. **Operation-specialized models** (Add / Replace / Remove) could address different mutation mechanisms, especially Replace errors. Extra Trees already captures operation interactions, Remove has much less training data, and three specialized models per fold would add substantial fit cost. Defer until a low-memory grouped screen is available.
2. **More native structure on top of mutation features** was tested above and rejected for now. Earlier structure-only and structure-plus-gate-composition experiments also underperformed older matched baselines, reducing the rationale for another broad structure search.
3. **Extra Trees weights, depth/leaves, and tree count** have prior tuning/screens; current 5-fold confirmation and nested strategy provide no compelling untouched configuration. Do not repeat the completed grid.
4. **HGB / Random Forest** have already been evaluated and did not approach the mutation-aware candidate. **CatBoost, XGBoost, LightGBM, and imbalanced-learn** are absent from the current Python environment; no installation was attempted because prior family evidence and current resource constraints do not justify dependency work.
5. **Thresholding/calibration and simple blending** have already been assessed with grouped/nested development evidence; selection did not beat fixed 0.5 and equal blending underperformed the candidate.
6. **QML** remains a low priority: pilots on nine origins did not beat matched reduced-feature classical controls; the 24-origin run was stopped for memory pressure. No QML run was repeated.

The saved mutation-aware champion remains the best defensible development model. The 85% target remains unmet. The reused frozen test's paired comparison favors this champion, but future untouched-origin evaluation is still needed for an independent generalization claim.

## Frozen-test comparison (retrospective)

The selected candidate was frozen as mutation-aware Extra Trees (SHA-256 `7532be26b6959e7e62277220f0c0b78e648d723a0be35d2b693d68e1c9ebf085`), threshold 0.5. The matched full-development baseline is the saved eight-feature Extra Trees model (SHA-256 `8cd332a309e27ef09cc46c5bf8a8788da35c05aa6ed25ed5de21d26ced076175`), both trained using all 578,117 development records. Both predictions use the same 144,962 IDs, 70 origins, labels, and threshold; ID SHA-256 `bc1d19af8bcd9cad69c308d68f4d1b6c0e8c0231138070d22a0cb202f3d41e87`.

| Model | Accuracy | Balanced accuracy | Macro F1 | Survivor F1 | Detected F1 |
|---|---:|---:|---:|---:|---:|
| Full-development 8-feature Extra Trees | 0.7884 | 0.7893 | 0.7884 | 0.7873 | 0.7896 |
| Mutation-aware 23-feature Extra Trees | **0.8185** | **0.8186** | **0.8184** | **0.8135** | **0.8232** |
| Difference | +3.01 pp | +2.93 pp | +2.99 pp | +2.63 pp | +3.36 pp |

Baseline confusion (true rows [0,1], predicted columns [0,1]): `[[56753,13133],[17536,57540]]`; mutation-aware confusion: `[[57394,12492],[13818,61258]]`. Thus the candidate reduced false positives by 641 and false negatives by 3,718. At label 0, candidate precision/recall/F1 were 0.8060/0.8213/0.8135; at label 1 they were 0.8306/0.8159/0.8232. Baseline per-class metrics appear in the JSON summary.

Paired bootstrap resampling of the 70 origins (1,000 repetitions, seed 20261027) gave candidate-minus-baseline 95% percentile intervals: accuracy **[+1.54,+4.53] pp**, balanced accuracy **[+1.48,+4.41] pp**, macro F1 **[+1.55,+4.52] pp**. The observed improvement is consistent on this reused test set, but must be treated as retrospective, not pristine external confirmation. The full-development candidate test macro F1 is 2.50 pp above the five-fold OOF score; the full-development baseline test score is 1.50 pp above its five-fold OOF score. OOF and final-model test predictions differ in training-row count and are not identical protocols; the test-origin mix can also differ in difficulty. Do not infer that this gap proves leakage.

The bounded runner is `experiments/evaluate_qmutbench_mutation_aware_frozen_bounded.py`. It reused the verified pinned QMutBench checkout and streamed the metadata CSV, parsed one process at a time, cached 23 features by record ID in 168 hashed checkpoints, and stopped if available memory fell below 500 MiB. Smoke checked 64 held-out rows before the full pass. The completed cache manifest, paired predictions, and metrics are under `reports/experiments/qmutbench_frozen_comparison_bounded_20261027/`. All 144,962 feature rows and paired prediction IDs were verified complete and unique; every feature/prediction checkpoint checksum passed. Runtime was 508.7 seconds; minimum observed available RAM during logged checkpoints/finalization was about 619 MiB; after completion it recovered to about 904 MiB. Peak process RSS was not measured.

## Runtime and retained artifacts

Repository state: branch `qmutbench-master-optimization`, HEAD `869aa55034d5016eb4bb4661f5390a7ff38cb8b7` (merge commit `Merge remote-tracking branch origin/qmutbench-master-optimization into qmutbench-master-optimization`). No tracked source data, split manifest, model bundle, anomaly audit, or QML pilot result was changed. New artifacts are this bounded evaluator, the candidate test feature/prediction checkpoints, the paired frozen-test prediction file, and the updated consolidated Markdown/JSON. Existing development predictions/folds/configuration remain in their prior versioned directories. QMutBench tests: 17 focused tests passed in 2.61 seconds; relevant research/inference modules passed `py_compile`; JSON/checkpoint schema and ID/hash validations passed; `git diff --check` passed. The frozen-test reuse caveat remains material: this is not pristine independent validation. A future evaluation on untouched origins is needed for a clean generalization claim.
