# QMutBench source-circuit feature extraction and model validation

## Scope and evidence

Dataset SHA-256 `3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778`; 578,117 development records across 305 origins. The historical frozen-test rows and labels were not used. The official checkout is pinned to `01f5e7700c051fe715fc368b307deaa7e6869ba8`.

The local source tree contains 375 originals and 723,079 mutant QASM files. Filename-derived mapping matched all 578,117 development source records to unique mutant files and 305 original circuits. Extraction completed for 578,117/578,117 development rows with 0 parsing failures.

Only pre-outcome source and circuit syntax were parsed. Record IDs trace rows but are not predictors; `Origin_program` is only a group key/diagnostic. `Killed` was not loaded by the extraction process.

## Extracted features

The extractor records original and mutated instruction counts, ASAP per-wire gate-layer depth, depth per qubit, one/two/three-plus-qubit gate counts, multi-qubit fractions, distinct gate names, selected gate-name proportions, and mutant-minus-original count/depth/fraction deltas. Barriers synchronize their wires; measurements are excluded; gate instructions are not decomposed. This is a source-QASM scheduling proxy, not transpiled or hardware depth.

`position` remains a coarse 10-point circuit-percent band; `position_fraction` is its midpoint and is not an exact mutation position. Original/replacement gate-pair reconstruction and local mutation neighborhoods were excluded because the pinned QASM contains compiled/decomposed instructions and does not establish those source-level semantics reliably.

A diagnostic comparison found different gate-count conventions between the metadata `gates` field and parsed QASM for some operations. That match flag was excluded from all model inputs; source metadata was not used as a parser correctness oracle.

## Comparable grouped-fold results

All model comparisons use the same three saved `StratifiedGroupKFold` folds, with 91, 101, and 113 held-out origins; threshold fixed at 0.5. Preprocessing is fit on each training fold. Feature selection and model evaluation use development origins only.

| Model | Macro F1 | Balanced accuracy | Accuracy |
|---|---:|---:|---:|
| Current 100-tree Extra Trees + position fraction | 0.7682 | 0.7697 | 0.7683 |
| Candidate + 12 QASM gate-name fractions and multi-qubit fraction | 0.7714 | 0.7719 | 0.7715 |

Candidate delta: macro F1 +0.32 percentage points, balanced accuracy +0.23 pp, and accuracy +0.32 pp. Fold macro-F1 deltas were -0.27 pp, +0.49 pp, +0.71 pp; 2/3 folds improved. Nested threshold averages were 0.7679 for current and 0.7703 for candidate.
Across 305 origins with both target classes, candidate per-origin macro F1 improved on 135 and worsened on 166.

### Bounded screen and alternative model family

| 60-tree screen (three grouped folds) | Pooled macro F1 | Pooled balanced accuracy | Pooled accuracy | Fold macro F1 |
|---|---:|---:|---:|---|
| Extra Trees matched baseline | 0.7663 | 0.7675 | 0.7663 | 0.7631, 0.7701, 0.7657 |
| QASM structure only | 0.7595 | 0.7607 | 0.7595 | 0.7590, 0.7588, 0.7606 |
| Mutant gate-name proportions + multi-qubit fraction | 0.7711 | 0.7717 | 0.7711 | 0.7591, 0.7786, 0.7755 |
| Signed mutant-minus-original proportions | 0.7640 | 0.7654 | 0.7640 | 0.7625, 0.7649, 0.7644 |
| Random Forest baseline features | 0.7589 | 0.7595 | 0.7590 | 0.7483, 0.7569, 0.7716 |
| Combined structure + composition (stopped after fold 1) | 0.7478 | n/a | n/a | 0.7478 |

The candidate fails the preregistered +0.5 pp materiality criterion and is not consistent across folds; the current model remains selected. Composition-only 60-tree screening was +0.48 pp on average and advanced to confirmation. Signed mutant-minus-original composition was negative against the matched 60-tree baseline. Structure-only features were also lower than the baseline. Combined structure plus composition scored 0.7478 on fold 1 versus the 0.7631 baseline and was stopped. Random Forest baseline feature runs averaged 0.7589 macro F1 on the same folds versus 0.7663 for the 60-tree Extra Trees screen.

The candidate raised detected recall at 0.5 (0.7309 to 0.7592) but reduced Survivor recall (0.8085 to 0.7847); the aggregate macro-F1 gain remained below the selection bar. No candidate was promoted and the inference API remains unchanged.

## Reproducibility and limits

The feature cache is resumable in 58 compressed chunks and carries dataset, QASM checkout, parser, extractor, and saved OOF checksum provenance. Generated feature matrices, predictions, and fitted experiment models remain ignored under `data/qmutbench_ml/`.

The experiment used three fixed outer folds. A new five-fold full-corpus comparison was not practical under available memory and the measured multi-minute fold fits. The 85% aspirational target was not reached. This is historical mutation-outcome prediction only, not hardware-noise detection or fault localization.

Focused tests: `.venv/Scripts/python.exe -m pytest tests/test_qmutbench_qasm_features.py -q` (5 passed). The selected model's real inference API passed single-record and three-record batch smoke checks with finite probabilities and the saved model version/threshold.

Machine-readable discovery inventory: `reports/experiments/qmutbench_qasm_feature_discovery_20261025.json`. Machine-readable model results: `reports/experiments/qmutbench_qasm_model_results_20261025.json`.
