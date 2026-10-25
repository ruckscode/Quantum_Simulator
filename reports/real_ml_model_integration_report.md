# Real ML Model Integration Report

## Dataset used

- `reports/experiments/full_dataset`: controlled execution-derived dataset, 270 reports, 30 circuit-family groups, and 90 examples per class.

## Exact five features

- `total_variation_distance`
- `js_divergence`
- `hellinger_distance`
- `observed_entropy_bits`
- `observed_max_probability`

## Model

- Logistic Regression, using the existing pipeline configuration: median imputation, `StandardScaler`, and `LogisticRegression(max_iter=2000, class_weight='balanced')`.
- The **75.56% accuracy / 75.02% macro-F1** result is for the eight-feature `OBSERVABLE_ONLY` set, which adds `calibration_readout_error_max`, `calibration_t1_min`, and `calibration_t2_min` to the five adapter observables. It does not describe the production adapter's five-feature schema.
- The saved five-feature `OBSERVABLE_ONLY_NO_CALIBRATION` Logistic Regression result is **56.30% accuracy / 53.58% macro-F1** (fold means 0.56296 / 0.53580; standard deviations 0.11234 / 0.13320), from `ml_results/fallback_ml_evaluation/model_comparison.csv`. It uses five-fold `StratifiedGroupKFold`, grouped by `circuit_family_id`, and the median-imputation / scaling / class-weighted Logistic Regression pipeline.
- A separate nested five-feature comparison in `data/qmutbench_ml/final/grouped_model_comparison.md` reports **56.30% accuracy / 54.34% macro-F1** for fixed LR, and **58.52% / 57.56%** for tuned LR. It uses leave-one-family-out outer folds and grouped inner tuning; these results are a different evaluation protocol and are not combined with the fallback evaluation above.

## Training/evaluation methodology

- Existing evaluation: 5-fold `StratifiedGroupKFold`, shuffled with `random_state=42`, holding out circuit-family groups.
- The saved fallback evaluation records fold metrics and group counts, but not row-level fold assignments or out-of-fold predictions. A fresh run that records those artifacts would process all 270 report files; that run was not performed here.
- The prepared exporter creates a **FINAL DEPLOYABLE MODEL** by fitting the unchanged Logistic Regression pipeline on all 270 records after evaluation. It writes only preprocessing parameters and fitted coefficients, with dataset, schema, class, and methodology metadata; it does not persist labels or row metadata.
- No final model training occurred in this environment.

## Artifact path

- Intended portable JSON path: `src/quantum_simulator/models/observable_logistic_regression.json`.
- Artifact created: **No**.

## Loading verification

- The adapter and exporter compile and the adapter imports.
- The expected artifact is absent; the adapter correctly raises `ModelBackendUnavailable`. Genuine artifact loading could not be verified.

## Inference verification

- Real-model probability inference and the real-model end-to-end path remain unverified because there is no genuine artifact.
- The prepared adapter validates the artifact type, exact feature and class ordering, Logistic Regression type, final-model designation, parameter dimensions, and provenance before inference.

## Actual test results

- Adapter/exporter syntax compilation: passed.
- Missing-artifact adapter check: passed; unavailable backend raised as expected.
- Maintained full suite: **358 passed**.
- No model-training or model-inference test passed because no genuine artifact was available to test.

## Environment limitation

- sklearn's native `_predictor` DLL remains blocked by Windows Application Control in the system Python and project `.venv`.
- WSL distribution enumeration returned access denied; `py -0p` found no alternate installed Python runtime. No usable Linux/Kali environment was available through Codex.
