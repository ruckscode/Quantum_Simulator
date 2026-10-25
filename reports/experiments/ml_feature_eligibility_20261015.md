# ML feature eligibility and evaluation disposition — 2026-10-15

## Decision

`PRODUCTION_FIVE_OBSERVABLES` is the only primary feature schema. It contains measurements derived from expected and observed output distributions. Calibration evidence and source-circuit comparisons remain separate ablations. `WITH_PROGRAM_COMPARISON` is diagnostic/leakage analysis only and must not support primary performance claims. Historical 270-row results remain ineligible for fitting or current performance claims: generation provenance and per-run source maps are absent, historical row lineage is absent, and nominal family IDs do not establish independent roots. The nine-row pilot is a synthetic pipeline fixture, not model-development evidence. No new model fitting or performance experiment was run.

## Feature eligibility

| Feature | Calculation / input | Available at inference? | Source vs submitted? | Synthetic-label shortcut risk | Detection / cause eligibility | Localization eligibility | Disposition and evidence |
|---|---|---|---|---|---|---|---|
| `total_variation_distance` | TVD between expected and observed distributions | Yes, when both distributions are supplied | No | Low direct leakage; it is an outcome measurement and can be predictive by effect size | Primary candidate for detection; class disposition only with valid independent labels | No | Primary observable. `ml_diagnosis.extract_observable_features`; five-feature runtime contract. |
| `js_divergence` | Jensen–Shannon divergence of the same distributions | Yes | No | Same outcome-derived risk as TVD; redundant signal | Primary candidate for detection/cause disposition with valid labels | No | Primary observable; correlation/redundancy expected and must be assessed within training folds. |
| `hellinger_distance` | Hellinger distance of the same distributions | Yes | No | Same outcome-derived risk as TVD | Primary candidate for detection/cause disposition with valid labels | No | Primary observable; not an independent sample from TVD/JS. |
| `observed_entropy_bits` | Entropy of observed output probabilities | Yes | No | Low direct shortcut risk; can identify intervention strength | Primary candidate for detection/cause disposition with valid labels | No | Primary observable. |
| `observed_max_probability` | Maximum observed output probability | Yes | No | Low direct shortcut risk; can track state/concentration changes | Primary candidate for detection/cause disposition with valid labels | No | Primary observable. |
| `calibration_readout_error_max` | Maximum readout-error value from report hardware parameters | Only if independently measured calibration is available before inference | No | High in current synthetic reports: configuration is coupled to injected hardware-anomaly construction | Separate hardware-evidence ablation only; not output-only primary | No; aggregate maximum loses qubit identity | Excluded from the five-feature primary schema. `ml_evaluation._features` currently reads report configuration, not an independently adjudicated calibration record. |
| `calibration_t1_min` | Minimum T1 value from report hardware parameters | Same condition as above | No | High in current synthetic reports for the same reason | Separate calibration-evidence ablation only | No; aggregate minimum does not identify a qubit | Not primary. Any future use must provenance-link calibration acquisition time/source and show it was available at inference. |
| `calibration_t2_min` | Minimum T2 value from report hardware parameters | Same condition as above | No | High in current synthetic reports for the same reason | Separate calibration-evidence ablation only | No; aggregate minimum does not identify a qubit | Not primary; deduplicated-calibration ablation remains diagnostic. |
| `gate_mismatch_fraction` | Fraction of differing operation signatures between ideal/reference and observed/submitted circuit | Only when both source and submitted circuits are available | **Yes** | **High**: the synthetic `PROGRAM_FAULT` intervention edits the submitted circuit; mismatch can expose the label-generating rule | Only a separately reported diagnostic/leakage ablation; never primary observable-only evidence | No; one fraction does not identify the faulty gate/qubit | Restricted by `_validate_feature_sets` to `WITH_PROGRAM_COMPARISON`; now tagged `diagnostic_leakage_ablation_not_primary_evaluation` in manifests. It must not be presented as an output-only result. |
| `ideal_operation_count` | Number of operations in the reference circuit | Yes if the reference circuit is available | Reference only | Possible family/template proxy; not a direct label field | Separate source-informed ablation only, conditional on family-disjoint evaluation | No | Used only by the historical fallback comparison helper; needs lineage-safe evaluation. |
| `ideal_unique_gate_count` | Unique operation names in the reference circuit | Same as above | Reference only | Possible family/template proxy | Separate source-informed ablation only | No | Same restrictions; never evidence of fault location. |
| `ideal_qubit_count` | Configured/reference circuit width | Same as above | Reference/configuration | Possible family/template proxy and can encode design recipe | Separate source-informed ablation only | No | Same restrictions; keep family roots disjoint. |

### Directly prohibited inputs

The label and post-outcome fields in `ml_evaluation.EXPLICITLY_EXCLUDED`, `ml_training.FORBIDDEN`, and `experiments/ml_data_pilot.FORBIDDEN_PREDICTOR_FIELDS` are not predictors: ground-truth/scenario labels; fault IDs, types, descriptions and operation indices; affected hardware component/condition; diagnosis category/evidence; run/family/lineage IDs; observed-count containers; and configuration objects. They remain in separate provenance/label records where needed. The pilot keeps synthetic scenario labels in `synthetic_labels.jsonl`, separate from predictors and explicitly sets independent labels to null.

## Evaluation and task boundary

- Detection can be evaluated from the five output observables only if the independent unit is a verified recipe root and labels are suitable for that target.
- Cause disposition needs independently established cause truth. Distribution-only models cannot resolve observationally equivalent causes; use an ambiguous/abstain disposition where truth or evidence is insufficient.
- Fault subtype, source/qubit/gate localization are not supported by the five output features. A circuit-difference feature is separate evidence and is not itself localization.
- Calibration features are admissible only in a separate feature-set comparison when calibration is independently measured and demonstrably available before the prediction. Current simulator report configuration does not establish that condition.
- A split by `circuit_family_id` is insufficient when multiple family IDs share a recipe root. The current generator uses six `design_recipe_*` roots for 30 nominal families; saved historical rows lack per-run lineage, so the true split cannot be reconstructed from them.

## Implementation and experiment disposition

The main evaluator has a read-only provenance preflight before fitting. This pass found that the fallback evaluator and full-dataset model exporter lacked that same guard: each could load and fit the historical 270 rows directly. Both now call the shared preflight and stop before loading features or writing outputs when it is not ready. Regression tests assert that no data load, fit, or export occurs after a failed preflight.

No estimator comparison, calibration, threshold selection, or performance metric was run on the historical 270 rows or the nine-row synthetic pilot. This is required by the existing provenance and independent-family blockers, not evidence that any model performs well or poorly. The installed Logistic Regression and tree code remains covered by bounded tests on synthetic unit fixtures only. Existing reports document that Windows Application Control blocks scikit-learn's native `_predictor` DLL in the model export/training path; that restriction was not probed, circumvented, or weakened in this milestone. No package was installed.

## Evidence references

- `ml_evaluation.py`: feature calculations and legacy feature-set definitions.
- `ml_training.py`: five-feature primary schema, group/provenance preflight, fixed estimator and evaluation pipeline.
- `experiments/ml_data_pilot.py`: separate synthetic labels and observable predictors; three-lineage smoke fixture.
- `ml_fallback_training.py` and `experiments/export_observable_model.py`: preflight-before-fit guard added in this milestone.
- `reports/experiments/ml_current_pipeline_audit_20261014_release.json`: historical descriptive rerun and explicit statement that labels are not independently adjudicated and current preflight blocks.
- `reports/experiments/diagnosis_prior_lineage_inventory_20261015.json`: truncated, resource-bounded prior-output scan; no complete new family frame can be established.
- `reports/experiments/diagnosis_family_sampling_frame_20261015.json`: zero eligible fresh roots.
