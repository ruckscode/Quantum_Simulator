# Matched QML vs classical benchmark (QMutBench)

## Purpose and scope

This report compares the Binary VQC with two classical controls on the same bounded QMutBench task. It answers what the QML work completed so far can fairly claim. The task is QMutBench Survivor (label 0) versus WOO/OPO (label 1). It is a mutation-outcome prediction benchmark, not a measure of end-to-end simulator diagnosis accuracy.

## Matched protocol

For each of two sampling seeds (20261028 and 20261029), the three models used the same origin-grouped five-fold split, the same selected sample within each fold, and the same reduced four-feature schema. Each fold used 1,024 training records per class and 1,536 validation records per class. That is 2,048 training and 3,072 validation rows per fold, with 15,360 pooled out-of-fold validation predictions per model. Encoders were fit on that fold's training data only; the frozen test set was not used. The Binary VQC used eight epochs. Classical controls were Extra Trees and Logistic Regression.

Thus, this is a matched-data and matched-evaluation comparison. It is not equal-compute training: VQC optimization requires many quantum-circuit evaluations, whereas the classical controls fit directly. These CPU runs used PennyLane `default.qubit`; Kaggle's T4 GPUs were not used by those executions. A separate, larger run later used PennyLane `lightning.gpu` with the adjoint method through a Kaggle runtime wrapper.

## Results by sampling seed

| Seed | Model | OOF accuracy | OOF macro-F1 | Balanced accuracy | Mean fold accuracy ± SD | Measured fit time |
|---|---|---:|---:|---:|---:|---:|
| 20261028 | Binary VQC | 58.29% | 56.47% | 58.29% | 58.29% ± 1.64 pp | 2,866.19 s (47.8 min) |
| 20261028 | Extra Trees | 58.74% | 56.36% | 58.74% | 58.74% ± 1.44 pp | 0.83 s |
| 20261028 | Logistic Regression | 58.18% | 55.21% | 58.18% | 58.18% ± 1.84 pp | 0.10 s |
| 20261029 | Binary VQC | 58.44% | 55.61% | 58.44% | 58.44% ± 2.04 pp | 2,825.82 s (47.1 min) |
| 20261029 | Extra Trees | 58.55% | 56.53% | 58.55% | 58.55% ± 1.89 pp | 0.81 s |
| 20261029 | Logistic Regression | 58.65% | 54.86% | 58.65% | 58.65% ± 1.09 pp | 0.09 s |

## Two-run descriptive averages

| Model | Mean accuracy | Mean macro-F1 | VQC difference in accuracy | VQC difference in macro-F1 |
|---|---:|---:|---:|---:|
| Binary VQC | 58.37% | 56.04% | — | — |
| Extra Trees | 58.64% | 56.45% | VQC −0.28 pp | VQC −0.41 pp |
| Logistic Regression | 58.41% | 55.03% | VQC −0.05 pp | VQC +1.00 pp |

A balanced random-guess baseline is 50% accuracy. Across the two runs, VQC accuracy is close to both classical controls. VQC does not consistently lead: Extra Trees has the best mean accuracy and macro-F1; Logistic Regression has nearly the same mean accuracy. VQC's mean macro-F1 is about one percentage point above Logistic Regression. Differences are small and no statistical significance claim is made.

## Compute cost

In the second run, VQC optimization took about 2,826 seconds for all five folds, compared with about 0.81 seconds for Extra Trees and 0.094 seconds for Logistic Regression, as measured by the run's trainer. In the first run the respective times were about 2,866, 0.83, and 0.096 seconds. These are implementation-specific wall-clock fit measurements from the recorded run, not a hardware-normalized compute benchmark. They show that this QNN configuration was much slower without delivering a consistent predictive improvement.

## Validity limits

- Each result is sampled development OOF, not full-corpus OOF and not an untouched frozen-test estimate.
- Both seeds reuse the grouped-fold design and the same source/task; the two-run average is descriptive. Do not concatenate the OOF rows or treat the repeats as independent external tests.
- The balanced per-fold cap changes prevalence. The reported accuracy applies to this balanced sample and should be accompanied by macro-F1 and balanced accuracy.
- The classical full-data champion reported elsewhere (79.33% accuracy) uses 578,117 development records and a richer mutation-aware feature schema. It is not a fair direct comparator for this four-feature sampled QNN experiment.
- The QMutBench label is not the simulator's program-fault/hardware-anomaly diagnosis label. These metrics do not establish simulator diagnosis accuracy or imply the QMutBench model is currently called by `diagnose_circuit()`.
- The original all-data QMutBench QNN campaign remains checkpointed and incomplete. This report completely describes the bounded matched experiments completed so far; it does not claim the full campaign is complete.

## Source of recorded metrics

Metrics and run configurations were transcribed from the completed Kaggle notebook outputs into `training_results.json`. The source paths, fold dispersion, confusion matrices, runtime, and device caveats are recorded there. The completed sampled run directories remain in the Kaggle working environment unless separately exported; this report does not claim that their full OOF prediction files were downloaded into this repository.


## Larger GPU-backed matched run (summary pending artifact audit)

A later Kaggle execution completed five origin-grouped folds with seed 20261030, 2,048 training records per class per fold, 1,536 validation records per class per fold, eight epochs, a four-qubit depth-2 angle-reupload VQC, and 15,360 pooled OOF predictions per model. The same reduced four-feature schema and fold-local preprocessing were used for Binary VQC, Extra Trees, and Logistic Regression. The QNN backend smoke check passed and the training wrapper selected PennyLane `lightning.gpu` with `adjoint`; this selection was made by a notebook runtime monkeypatch and is not yet a supported repo configuration option.

| Model | OOF accuracy | OOF macro-F1 | Balanced accuracy | Mean fold accuracy ± SD | Reported fit time |
|---|---:|---:|---:|---:|---:|
| Binary VQC | 58.49% | 58.25% | 58.49% | 58.49% ± 1.96 pp | 2,940.02 s |
| Extra Trees | 59.85% | 58.11% | 59.85% | 59.85% ± 1.06 pp | 0.91 s |
| Logistic Regression | 58.88% | 55.80% | 58.88% | 58.88% ± 1.78 pp | 0.11 s |

Taken at face value, Extra Trees leads VQC by 1.36 percentage points in accuracy, while VQC macro-F1 is 0.14 points above Extra Trees. Those are small descriptive differences and do not establish a QNN advantage. Do not pool this run with the 1,024/class runs: training caps and seed differ.

**Audit required before a firm claim.** The completed notebook summary reports identical fold-2 accuracy, macro-F1, and confusion matrices for all three models. It also reports 2,940 seconds of summed VQC fit time although the whole notebook cell took 2,509 seconds (summary elapsed time: 2,498 seconds). These inconsistencies need reconciliation against `oof_predictions.csv` and fold artifacts. Fold 5 also has a notably low VQC macro-F1 (44.82%). The current local project contains the summary metrics, not the Kaggle OOF prediction artifact, so the row-level check has not yet been completed. The numbers above are therefore transcribed summary results, not a fully artifact-validated benchmark.

The frozen test remained untouched and this is still sampled development OOF. It is not comparable to the 79.33% full-data mutation-aware classical result because the schema and sample scope differ. The Kaggle output directory is `/kaggle/working/Quantum_Simulator/reports/experiments/qmutbench_binary_gpu_matched_2048_20261031_161415`; complete artifacts have not been copied into this repository.
