# QMutBench master model comparison

## Scope and provenance

This run used `official_all_data_20261015.csv` (723,079 records, 375 `Origin_program` groups; SHA-256 `3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778`). The existing deterministic group holdout leaves 578,117 development records from 305 origins and reserves 144,962 records from 70 origins. The reserved set was not used for any new feature, model, selector, or threshold decision.

All new cross-validation comparisons use the same three `StratifiedGroupKFold` development folds (validation origins: 91, 101, and 113). Each validation origin is excluded from its fitting fold. The selected-v3 chunks were checksum-verified and reused. Previously recorded results are preserved with source-manifest and metrics checksums in `data/qmutbench_ml/master_optimization_v1/reused_historical_evidence.json`.

The prior selected-v3 report's 67.70% macro F1 used its own folds. Recalculating the selected-v3 regime on the common development folds gives 70.09%; the difference is due to different group-fold assignments, not a data or metric mismatch. Historical frozen-test diagnostics remain exploratory and unchanged; this run produced no frozen-test score.

## Findings

The strongest existing evidence remains the tuned, full-development Extra Trees model: 100 trees, depth 22, minimum leaf size 15, square-root feature sampling, and balanced subsample weights. On the fixed development folds it reached 75.48% macro F1, 75.64% balanced accuracy, and 75.49% accuracy. The fitted development-only model was reused rather than refit, with its checksum verified.

| Comparison | Macro F1 | Reading |
|---|---:|---|
| Tuned Extra Trees, all development origins (prior fixed-fold tuning) | 0.7548 | Best supported full-development model |
| Same Extra Trees settings, no class weights (new, 3 folds) | 0.7548 | Weighting made no material difference |
| Unweighted Random Forest (new, 3 folds) | 0.7314 | Below Extra Trees |
| Tuned Extra Trees on selected-v3 rows (new common folds) | 0.7093 | Smaller subset was weaker and variable |
| Existing original Random Forest (prior common folds) | 0.7340 | Below tuned Extra Trees |

On the representative 200k selection, the prevalence-and-coverage selector averaged 0.7345 macro F1, compared with 0.7199 for plain random whole-origin selection. It also exceeded the earlier representative-200k Extra Trees result (0.7235). This screen supports the coverage-aware selector when training cost requires a subset; the full-development model remains the accuracy choice.

The feature screen used the same representative origins and validation folds with 60-tree Extra Trees models:

| Feature set | Mean macro F1 |
|---|---:|
| All seven features | 0.7318 |
| Categorical features only | 0.7303 |
| Remove dominant state | 0.7274 |
| Add `algorithm_group` | 0.7273 |
| Add position field | 0.7200 |
| Remove operation | 0.7020 |
| Remove algorithm | 0.6685 |
| Remove gate | 0.5896 |
| Numeric features only | 0.5140 |

The full feature set is retained. Algorithm and gate were particularly important in this screen; extra `algorithm_group` or position data did not improve the result. Categorical-only performance was close to the full set, but not a consistent improvement.

## Error and threshold analysis

Full-development out-of-fold predictions cover all 578,117 development rows and 305 origins. At the default 0.5 threshold, pooled OOF accuracy was 75.49%, balanced accuracy 75.64%, and macro F1 75.48%. Mean accuracy across origins was 77.50% (median 78.52%), showing substantial origin-to-origin variation. Several supported origins, including graph-state and real-amplitude-random circuits, had much lower accuracy; the complete lowest-origin list and pre-outcome subgroup slices are in `oof_analysis.json`.

Threshold selection used only probabilities from the other two development folds, then measured each selected threshold on the held-out fold. The nested macro-F1 results averaged about 75.57%, versus 75.48% at 0.5. The development-only final threshold selected on all OOF probabilities was 0.485. It is saved in model metadata and used by default by the prediction command. Its same-OOF score is descriptive and is not an unbiased estimate; threshold benefit is modest.

## Artifacts and reproduction

The runner, atomic fold checkpoints, OOF predictions, reuse evidence, metrics, model, and metadata are under the ignored directory `data/qmutbench_ml/master_optimization_v1/`. The manifest records fold membership, source and artifact hashes, compute resources, and run status (`complete`). The winning model was copied from the prior development-only fit; the original model and simulator files were not changed.

To resume or inspect the run:

```powershell
python -m experiments.qmutbench_master_optimization run --phase all
python -m experiments.qmutbench_master_optimization status
```

For inference, supply a CSV containing the seven training features (`algorithm`, `operation`, `gate`, `dominant_state`, `qubits`, `gates`, `Position_int`). Do not include `Killed`:

```powershell
python -m experiments.qmutbench_master_optimization predict --input INPUT.csv --output PREDICTIONS.csv
```

The output contains predicted outcome, detected probability, and `Origin_program` when supplied in the input. The threshold can be overridden with `--threshold`.

Focused tests: `pytest -q tests/test_qmutbench_master_optimization.py` — 4 passed. No frozen-test evaluation was run.
