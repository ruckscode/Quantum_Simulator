# Current ML implementation and data-pipeline audit (2026-10-14)

## Scope and result

This is a bounded audit of the current evaluation code on preserved historical feature rows. It does not certify current-source data generation, independent generalization, calibration, deployment, or benchmark performance.

`ml_training.run_preflight()` reports 270 structurally eligible rows in 30 families, balanced at 90 records per class, with five family-stratified folds (216 rows/24 families train and 54 rows/6 families test per fold) and zero family overlap. The five production features have no missing/nonfinite values. The run IDs are unique and no duplicate report fingerprints were found. The preflight is structural and its `ready=true` is not a validation-readiness or independence claim.

The 270 saved report rows are timestamped around 2026-10-07 and have no per-row simulator/diagnosis source hash. Their generic `simulator_version` field cannot identify the exact producing source. The dataset manifest also lacks generation source hashes. Accordingly, their compatibility with the current simulator source cannot be established. This audit does not backfill or rewrite those reports.

## Bounded grouped evaluation

The current fixed Logistic Regression pipeline was evaluated twice, in memory only, after resolving the report-root path handling issue, using the existing five `StratifiedGroupKFold` assignments and production five-feature schema. Both runs produced identical predictions and probabilities. They exactly matched the saved historical OOF predictions/probabilities (maximum probability delta 0). This establishes repeatability and agreement with the saved evaluation artifact on this same historical row set only. A machine-readable snapshot with current source hashes is in `ml_current_pipeline_audit_20261014.json`.

Descriptive re-evaluation: accuracy and balanced accuracy 0.563; macro F1 0.540. Confusion matrix (true rows and predicted columns ordered NORMAL, PROGRAM_FAULT, HARDWARE_ANOMALY): `[[83,0,7],[34,38,18],[47,12,31]]`. Multiclass Brier score 0.558 and log loss 0.908 indicate that the stored probabilities are not established as calibrated. These are not independent or benchmark metrics and must not be used as deployment evidence.

## Integrity and leakage controls

- The production schema contains TVD, Jensen-Shannon divergence, Hellinger distance, entropy, and maximum probability. Current feature extraction also exposes calibration and gate-mismatch fields in alternate schemas. Calibration-derived fields can encode injected hardware condition; `gate_mismatch_fraction` directly reflects program-change construction. They are unsuitable as independent-cause predictors absent a separately justified task and split. They remain excluded from the production five-feature schema.
- The fixed grouped pipeline fits imputation and scaling inside each training fold. Family-level grouping is implemented and fold membership is complete. No hyperparameter tuning was part of this bounded check.
- Scenario labels are assigned by synthetic construction. They are not independent adjudications. Family-held-out folds reduce within-family leakage, but do not create an external test set or independent ground truth.
- Predictions/probabilities repeat exactly with the fixed seed/fold design. That is computational reproducibility on saved rows, not reproducibility of the original row generation.
- The final source compatibility snapshot reports five changed hashes against the old evaluation manifest: the prior runner, diagnosis engine, and simulator changes; relative-path normalization in `ml_evaluation.py`; and future-manifest provenance in `experiments/full_dataset.py`. The last two concern path handling and provenance capture, not model features or fitting. Exact OOF agreement was rechecked under the current code. These source hashes do not repair missing row provenance in historical reports.
- Existing model integration remains development-only, uncalibrated, and unavailable by default. Confidence-based abstention is not supported by calibration evidence.

## Required provenance repair and disposition

Future individual experiment reports now record SHA-256 fingerprints for the declared source set in `reproducibility.source_sha256`. The full-dataset generator now places that source map and Python/NumPy runtime identity in future `dataset_manifest.json` files. Historical rows and manifest remain untouched. The declared set is a scoped implementation manifest, not a whole-repository or data fingerprint. See `ml_source_compatibility_audit_20261014_final.json`.

**Status: partial / blocked for independent generalization.** Further progress requires a new dataset generated from a frozen identified source, independently adjudicated labels, family-separated development/calibration/test partitions, a preregistered evaluation contract, and calibration evidence. No training campaign was run.



## Release-final provenance gate (2026-10-14)

A subsequent audit found the ML training preflight reported the historical dataset structurally ready despite missing source identity. The training entry point now fails closed unless the dataset manifest has generator source/runtime identity, all 270 reports have per-run source hashes, every report matches the manifest, and generator hashes match current code. The historical data now fails preflight with explicit reasons; no additional fitting or predictions were run. See `ml_provenance_preflight_20261014.json`. The refreshed source audit is `ml_source_compatibility_audit_20261014_release_final.json`: six of nine saved evaluation source hashes differ. The previously recorded two-run Logistic Regression repeatability result remains historical evidence from before this final guard; it is not a current-tree result or independent generalization evidence.
