# ML Diagnosis Integration Report

## Files changed

- `src/quantum_simulator/ml_diagnosis.py`
- `tests/test_ml_diagnosis_integration.py`
- `reports/ml_diagnosis_integration_report.md`

## Architecture implemented

- Observable feature extraction is separate from inference and localization.
- `ObservableClassifier` defines an injectable classifier adapter contract. It accepts the exact five approved observable feature names and returns a class plus probabilities.
- Diagnosis routes the classifier class into the existing `diagnose_execution` localization/evidence pipeline. ML identifies the broad class; existing evidence must support a fault location before the final diagnosis is definitive.
- NORMAL skips fault localization. Low confidence abstains as `INSUFFICIENT_EVIDENCE`. Trace deviation alone does not support PROGRAM_FAULT.
- The default backend explicitly raises `ModelBackendUnavailable`; inference does not train or load a fabricated artifact.

## ML model/features used

- No trained model backend or persisted model artifact is present or integrated.
- Feature schema: `total_variation_distance`, `js_divergence`, `hellinger_distance`, `observed_entropy_bits`, `observed_max_probability`.
- Existing validated evaluation results are not used as a model artifact or re-evaluated here.

## Integration behavior

- A genuine trained estimator can be connected through an `ObservableClassifier` adapter without coupling diagnosis/localization to sklearn.
- The tested deterministic mock classifiers exercise the architecture only; they are not trained models or evidence of ML performance.
- Confidence abstention uses the explicit 0.60 operational threshold; this was not tuned on the evaluation data.

## Tests run

- Integration plus Stage 1/2/3 diagnosis/localization targets: **87 passed**.
- Maintained full suite (`pytest tests -q`, workspace-local pytest basetemp): **346 passed**.
- An initial full-suite attempt using the system temp directory yielded **324 passed, 22 setup errors** because Windows denied access to the pytest temp root. The rerun with a workspace-local temp root completed successfully.

## Limitations

- Scikit-learn's native `_predictor` DLL is blocked by Windows Application Control in both the system interpreter and project `.venv`; training/import did not complete.
- No real persisted sklearn model artifact exists, and the real trained model is not connected.
- The adapter still needs to wrap the genuine validated observable-only model when that model artifact is available in an environment where it can be loaded. No real-hardware validation or generalization claim is made.


## Current contract verification - 2026-10-15

The classifier boundary now rejects Boolean, nonnumeric, non-finite, and out-of-range confidence cutoffs and malformed probability values; a missing feature schema raises an explicit ValueError. Focused ML/end-to-end/artifact/statistics/simulator/trace suite: 129 passed in 24.10s. Bounded repository suite: 778 passed, one unchanged frozen 7/45 diagnosis failure, zero skipped, two NO-GO finite-shot tests deselected (781 collected; 223.55s). These are deterministic software tests; no model was trained, and no predictive accuracy or calibration claim is made.


## Missing execution evidence gate - 2026-10-15

The rule-based diagnosis boundary no longer converts direct circuit mismatch into anomaly evidence when measured distributions are unavailable. It preserves the source localization candidate, reports anomaly status as unknown, and abstains. Focused diagnosis/pipeline suite: 85 passed in 33.18s. Bounded suite: 778 passed, one unchanged frozen 7/45 failure, zero skipped, two NO-GO tests deselected (781 collected; 224.61s).


## Causal routing update - 2026-10-15

Injected PROGRAM_FAULT predictions now require rule-based candidates whose source-only counterfactual changes modeled measured outputs; a classifier cannot turn an unrelated anomaly/source mismatch into a fault diagnosis. Source and relevant hardware evidence together remain ambiguous. Candidate formatting no longer presents localization as confirmed fault attribution. Focused suite: 90 passed; bounded suite: 783 passed, one unchanged frozen 7/45 failure, zero skipped, two NO-GO deselections (786 collected; 225.90s). No model fit or evaluation occurred.


Final causal-routing verification: focused suite 90 passed in 28.50s; bounded suite 783 passed, one unchanged frozen 7/45 failure, zero skipped, two NO-GO deselections (786 collected; 226.82s). The classifier cannot promote an unrelated source mismatch into program-fault attribution. Generic model freshness is not asserted; no default timestamp cutoff exists for the simulator's software calibration model.
