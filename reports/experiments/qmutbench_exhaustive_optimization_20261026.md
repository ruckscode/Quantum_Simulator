# QMutBench exhaustive optimization campaign — 2026-10-26

## Result

The selected new development model is the mutation-aware Extra Trees bundle at `data/qmutbench_ml/mutation_aware_v1/`. It adds 15 target-blind QASM mutation-alignment and local-context fields to the incumbent's eight inputs and retains threshold 0.5.

On the saved three-fold origin-grouped OOF, candidate macro F1 is **78.8901%**, balanced accuracy **78.9774%**, and accuracy **78.8908%**. The matched incumbent scores **76.8231%**, **76.9686%**, and **76.8256%**. Macro-F1 gain: **2.07 percentage points**.

An additional five-fold `StratifiedGroupKFold` comparison on identical rows produced **79.3314%** versus **77.3387%** (+**1.99 pp**). All five paired folds improved. This is a development-set stability check, not a fresh external test: feature discovery used development OOF outcomes. The previously examined historical frozen set was excluded.

## Grouped fold results

| Saved split | Baseline macro F1 | Candidate macro F1 | Delta |
|---:|---:|---:|---:|
| 3-fold 1 | 0.7621 | 0.7841 | +2.21 pp |
| 3-fold 2 | 0.7727 | 0.8006 | +2.78 pp |
| 3-fold 3 | 0.7699 | 0.7820 | +1.21 pp |
| 5-fold 1 | 0.7623 | 0.7779 | +1.56 pp |
| 5-fold 2 | 0.7638 | 0.7830 | +1.92 pp |
| 5-fold 3 | 0.7696 | 0.7848 | +1.52 pp |
| 5-fold 4 | 0.7953 | 0.8127 | +1.75 pp |
| 5-fold 5 | 0.7759 | 0.8079 | +3.21 pp |


## Features and errors

QASM feature extraction covered 578,117/578,117 development rows with zero parse errors. It aligns gate-name/wire tuples and excludes measurements and barriers. Exact edit shape and expected gate must agree before an edit is labeled exact; local neighbors are exposed only for small reliable diffs. Exact one-instruction edits account for 88.10%; 11.89% are complex/unaligned. See the JSON feature inventory for each definition, source, coverage, prediction-time availability, and leakage review.

The extractor also computes common-prefix, common-suffix, and a prefix-based relative-position estimate. These three fields are documented in the inventory but excluded from the finalist: the position estimate matched the coarse source position band on only 59.67% of the target-blind pilot.

The candidate yields 51,978 false positives and 70,058 false negatives; the incumbent has 53,311 false positives and 80,664 false negatives. Candidate per-origin accuracy improved on 246 origins, worsened on 58, and tied on one. Subgroup counts, Wilson intervals, high-confidence errors, and weak-origin examples are in [the error-analysis JSON](qmutbench_exhaustive_error_analysis_20261026.json).

## Model comparisons

- The matched 60-tree Extra Trees screen gained 2.23 pp; 100-tree confirmation gained 2.07 pp on the saved three folds.
- The tree depth, leaf size, feature subsampling, and class weighting stayed at the incumbent settings; this campaign compared 60 versus 100 trees rather than launching a broad parameter grid. Prior unweighted Extra Trees and Random Forest screens were reused.
- The prior 60-tree Random Forest screen (0.75891 macro F1) was below its Extra Trees baseline (0.76628).
- Native-categorical HistGradientBoosting with 100 iterations improved its own baseline by 0.87 pp, but its candidate macro F1 (0.7772) remained below the Extra Trees finalist. CatBoost, XGBoost, and LightGBM were unavailable and were not installed.
- Prior QASM structure-only and signed-delta features were negative. Prior gate composition gained only 0.32 pp and lost a fold.
- A matched 60-tree ablation removing the five gate/alignment categories (removed/added gate, alignment class, previous/next gate) scored 0.76475 versus 0.76628 baseline; its fold deltas were +0.38, −0.46, −0.38 pp. The full QASM-diff set scored 0.78856 on those same folds, supporting the categorical mutation-context block as useful.
- A 50/50 incumbent/finalist probability blend scored 0.78528, below the standalone finalist. No ensemble was kept. The threshold remains 0.5; no calibration or new nested threshold study was added.

## Inference and tests

The versioned artifact is `data/qmutbench_ml/mutation_aware_v1/best_model.joblib` with metadata alongside it (SHA-256 `7532be26b6959e7e62277220f0c0b78e648d723a0be35d2b693d68e1c9ebf085`). The incumbent bundle remains unchanged. Inference accepts the existing eight raw fields plus `original_qasm` and `mutant_qasm` OpenQASM 2.0 text, supports single and batch records, and computes the QASM fields without loading the training dataset. An actual development record passed single and two-row batch smoke predictions.

QMutBench regression suite: **88 passed in 17.15s**. Focused feature/inference tests: **15 passed in 2.27s**. The full repository suite was not run. The 85% aspirational target was not reached (five-fold macro F1 79.33%). No result here establishes hardware-noise detection or fault localization.

## Artifacts and interpretation

- [Machine-readable report](qmutbench_exhaustive_optimization_20261026.json)
- [Error and subgroup analysis](qmutbench_exhaustive_error_analysis_20261026.json)
- Model metadata: `data/qmutbench_ml/mutation_aware_v1/best_model_metadata.json`
- Starting commit: `13a45194961fe4293b1b300ca4611df7e4a5900a`

Because feature selection used development outcomes, both grouped validations can be optimistic after selection. No external test was used. Common-prefix/suffix alignment may mark reordered or heavily rewritten circuits complex; such rows do not get an asserted local neighborhood. Probabilities are uncalibrated, and QASM parsing is required at inference.
