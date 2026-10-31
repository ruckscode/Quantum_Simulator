# Classical ML diagnosis integration report

**Updated:** 2026-10-31
**Status:** Integrated as an explicit, opt-in development candidate; not validated for deployment.

## Current workflow

The public integration point is `diagnose_circuit(..., classifier=...)` in `src/quantum_simulator/diagnosis_engine.py`. The optional classifier is passed through `diagnose_execution()` to `diagnose_with_ml()` in `src/quantum_simulator/ml_diagnosis.py`. `PortableLogisticRegressionAdapter` loads the bundled portable JSON model when the caller explicitly constructs it. No model is loaded by default.

The classifier receives five observable-only features: total variation distance, Jensen–Shannon divergence, Hellinger distance, observed entropy, and maximum observed outcome probability. Its broad class prediction is added to `DiagnosisResult` as `ml_predicted_class`, `ml_confidence`, `ml_class_probabilities`, and `ml_features`.

The model does not locate faulty gates. The existing independent diagnosis path checks sampled-output statistics, expected-versus-observed operation differences and their measured-output counterfactual, plus relevant hardware calibration evidence. ML can refine an otherwise ambiguous result only when the corresponding independent evidence uniquely supports it. A conflict with a definitive evidence-led category becomes `AMBIGUOUS`; combined program and hardware evidence is not collapsed into a single-cause ML label. Low-confidence or unsupported evidence remains abstained/ambiguous. The 0.60 confidence cutoff is an operational threshold; probabilities are not calibrated correctness estimates.

## Current model and evidence limits

The bundled logistic-regression artifact is labeled **DEVELOPMENT CANDIDATE; NOT VALIDATED FOR DEPLOYMENT**. Its coefficients were fit on 330 synthetic rows from development families F1 and F2. The evaluation grid contains 900 synthetic configurations across five circuit families, five seeds and three shot counts. It is not physical-hardware evidence, and the families used in the evaluation are no longer a fresh locked holdout for further tuning.

Rescoring the saved Phase 5 outputs through the shared diagnosis evidence gate produced:

- Overall predeclared acceptance: **FAIL** (6 of 7 criteria passed).
- Raw ML accuracy: **80.0% (540/675)** on single-cause rows; class-imbalanced.
- Evidence-adjudicated macro-F1: **0.9363** on the same 675 single-cause rows.
- Program-fault recall: **70.0% (105/150)**, below the predeclared 80% minimum.
- Hardware-anomaly recall: **99.78% (449/450)**; normal recall: **97.33% (73/75)**.
- Exact target localization: **99.83% (599/600)** for the benchmark's labeled program/qubit/gate targets.
- Combined program-plus-hardware ambiguity recall: **94.67% (71/75)**.

These results are specific to the synthetic benchmark grid. They do not imply reliable diagnosis for arbitrary circuits, topologies, fault strengths, shot counts, or live hardware. The detailed scorecard is in the separately maintained run report `reports/experiments/classical_ml_sim_diagnosis_benchmark_20261031/REPORT.md` in the main simulator checkout; its 900 saved case outputs were re-scored without resampling or overwriting them.

## Fixes included in this update

- Added the public optional-classifier route to both `diagnose_circuit()` and `diagnose_execution()` and exposed its prediction details in `DiagnosisResult`.
- Routed the Phase 5 benchmark through the same evidence-fusion implementation used by the public diagnosis API.
- Kept exact gate identity in hardware candidates so calibration for one gate on a qubit tuple is not merged with another gate's calibration on that tuple.
- Made hardware calibration relevance consider both expected and observed circuits, so an omitted program operation cannot hide relevant hardware evidence on the reference circuit.
- Corrected documentation distinguishing raw-count permutation testing from the probability-only finite-shot heuristic. The latter is not a p-value or confidence interval.
- Relabeled the bundled model as a development candidate and documented its non-calibrated probabilities and limited synthetic training/evaluation.

## Verification performed for this update

Focused regression command:

```text
python -m pytest tests/test_ml_diagnosis_integration.py tests/test_hardware_fault_localizer.py -q
41 passed in 0.49s
```

`git diff --check` reported no whitespace errors in this checkout. No QML training, Kaggle runtime, credential, or checkpoint was accessed or modified in this update. Focused tests establish software behavior for their covered fixtures only; they do not establish model accuracy beyond the benchmark summarized above.

## Next release gate

Keep the model opt-in. Before enabling it by default, train and select on development circuit families, calibrate probabilities on a separate calibration split, and evaluate once on a new family-disjoint locked holdout containing broader circuit structures and independently labeled fault cases. Predeclare per-class recall, selective risk, localization and abstention requirements. Physical-device claims require separate real-hardware evaluation.
