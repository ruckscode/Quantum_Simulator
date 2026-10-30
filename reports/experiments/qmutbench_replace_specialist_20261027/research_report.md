# QMutBench Replace-specialist screen - 2026-10-27

## Decision

**Reject the specialist and retain the mutation-aware Extra Trees champion.** A predeclared shallow Decision Tree was trained on Replace-only records inside each of the five saved origin-grouped folds. It received no records from validation origins. The deployed-style hybrid routed only records whose `operation` input was `Replace` to the specialist and retained champion probabilities for all other records. Overall macro F1 fell from 0.7933 to 0.7825 (**-1.08 pp**). Every fold was worse. Paired origin-bootstrap intervals exclude zero in the harmful direction, so no second configuration search was run.

## Input and slice audit

`operation` is included in the saved champion's required raw inference schema (`best_model_metadata.json`) and its training feature schema. The pilot therefore routes on this supplied metadata field, not the target label. Users must provide the same operation value at inference. Training and feature preprocessing used only training-fold Replace records and fold-local category vocabularies.

The Replace slice has **120,676** development records, class counts **44,854 survivor** and **75,822 detected**, spanning **305 origins**. Per-origin rows range from 12 to 2524 (median 260); 293 origins contain both labels. The champion's five-fold Replace-slice macro F1 is 0.7546; specialist macro F1 is 0.7088. The paired origin-bootstrap 95% interval for the Replace-slice difference is -6.31 to -3.01 pp.

## Configuration and matched grouped results

One lightweight classifier was fixed before evaluation: `DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, class_weight="balanced", threshold=0.5)`, seed `20261030 + fold`. No threshold or hyperparameter tuning was done. Saved five-fold `Origin_program` assignments, 578,117 development OOF records, record IDs, target labels, and champion probabilities were reused. For each held-out fold, the specialist trained only on Replace rows from the other four folds; the model then replaced champion scores on Replace validation rows only.

| Fold | Replace rows | Origins | Champion Replace F1 | Specialist Replace F1 | Champion overall F1 | Hybrid overall F1 | Hybrid delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24,013 | 60 | 0.7331 | 0.7171 | 0.7779 | 0.7735 | -0.44 pp |
| 2 | 24,565 | 59 | 0.7290 | 0.6838 | 0.7830 | 0.7731 | -0.99 pp |
| 3 | 23,990 | 58 | 0.7596 | 0.7055 | 0.7848 | 0.7714 | -1.33 pp |
| 4 | 23,979 | 69 | 0.7706 | 0.7131 | 0.8127 | 0.8017 | -1.10 pp |
| 5 | 24,129 | 59 | 0.7791 | 0.7164 | 0.8079 | 0.7922 | -1.57 pp |

Pooled results:

| System | Macro F1 | Accuracy | Balanced accuracy |
|---|---:|---:|---:|
| Champion | 0.7933 | 0.7933 | 0.7941 |
| Champion + Replace specialist | 0.7825 | 0.7825 | 0.7838 |

Hybrid confusion matrix (true rows `[0,1]`, predicted columns `[0,1]`): `[[227608, 50805], [74920, 224784]]`. Hybrid per-class F1 is 0.7836 for survivor and 0.7815 for detected. The paired 2,000-resample origin bootstrap for hybrid minus champion yields macro-F1 95% CI **-1.38 to -0.79 pp**, accuracy **-1.37 to -0.80 pp**, and balanced accuracy **-1.34 to -0.73 pp**. Mean fold macro-F1 delta was -1.09 pp (SD 0.42 pp).

## Resource use, checkpoint integrity, and test integrity

The complete five-fold run fitted one shallow tree per fold in 2.82 seconds total. At fit start, available RAM ranged from 784 to 798 MiB; after fits it ranged from 809 to 819 MiB. A subsequent checkpoint verification rerun completed folds 1-4 then stopped before fold 5 at 679 MiB, below the 700 MiB start guard; it did not remove the already completed fold-5 output. All five checkpoints were validated against expected Replace record IDs, labels, origin IDs, folds, operation tokens, champion probabilities, and saved fit-time metrics. The complete result was reconstructed from those validated checkpoints without another fit.

No frozen-test row or label was accessed. The previously reused test remains retrospective context only. This specialist is rejected; the development champion and production model remain unchanged.

## Artifacts

- `experiments/qmutbench_replace_specialist.py` - training and grouped OOF screen.
- `experiments/finalize_qmutbench_replace_specialist.py` - ID- and fold-validated aggregate reconstruction from saved checkpoints.
- `reports/experiments/qmutbench_replace_specialist_20261027/result_summary.json` - complete metrics and fold record.
- `reports/experiments/qmutbench_replace_specialist_20261027/baseline_slice_analysis.json` - operation counts, class balance, and origin distribution.
- `reports/experiments/qmutbench_replace_specialist_20261027/replace_oof_predictions.csv.gz` and `fold_manifest.csv.gz` - reproducible OOF predictions and split/label audit.
- `reports/experiments/qmutbench_replace_specialist_20261027/per_origin_metrics.csv` — paired champion/hybrid metrics and Replace-slice metrics for each origin.
