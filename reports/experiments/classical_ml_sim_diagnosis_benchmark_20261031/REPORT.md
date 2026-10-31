# Classical simulator diagnosis benchmark — 2026-10-31

## Scope and result

The saved Phase 5 run contains **900 synthetic configurations** (60 scenarios × 5 seeds × 3 shot counts) across five four-qubit circuit families. I re-scored the saved outcome distributions through the revised `diagnose_execution(..., classifier=...)` evidence-fusion path, which is now shared by `diagnose_circuit()`. The re-score did not resample circuits, retrain the model, or overwrite any case output.

**Overall acceptance: FAIL.** Six of seven predeclared checks passed. Classification failed because PROGRAM_FAULT recall was **70%**, below the required **80%**. The other checks passed: output sensitivity and specificity at each shot count, exact-target localization, combined-fault abstention recall, seed stability, and deterministic configuration/reference regeneration.

The evaluation includes model-development families F1/F2 as well as family-disjoint F3/F4/F5; therefore aggregate metrics are not a wholly held-out generalization score. F3/F4/F5 were excluded from model fitting, but have now been examined and cannot be treated as fresh locked test data for future tuning.

## Metrics

| Metric | Result | Scope |
|---|---:|---|
| Raw ML model accuracy | 80.0% (540/675) | Single-cause rows; imbalanced classes |
| Evidence-adjudicated macro-F1 | 0.9363 | 675 single-cause rows |
| PROGRAM_FAULT precision / recall / F1 | 1.000 / 0.700 / 0.8235 | 150 rows |
| HARDWARE_ANOMALY precision / recall / F1 | 1.000 / 0.9978 / 0.9989 | 450 rows |
| NORMAL precision / recall / F1 | 1.000 / 0.9733 / 0.9865 | 75 rows |
| Exact target localization | 99.83% (599/600) | 150 program, 225 qubit, 225 gate-tuple cases |
| C15 ambiguous recall | 94.67% (71/75) | Combined program + hardware cases |
| Model confidence coverage at 0.60 | 96.56% (869/900) | Confidence is uncalibrated |
| Adjudicated non-abstain rate | 82.67% (744/900) | All benchmark rows |
| Selective risk | 21.43% (171/798) | Covered single-cause rows; C15 excluded |
| C15 confident raw single-cause predictions | 94.67% (71/75) | Every C15 truth is mixed; these raw 3-class outputs are wrong for the mixed case |

Output-change detection is scored separately from ML class accuracy:

| Shots | Sensitivity | Specificity | Balanced accuracy |
|---:|---:|---:|---:|
| 256 | 98.10% (103/105) | 95.00% (38/40) | 96.55% |
| 1,024 | 100% (105/105) | 100% (40/40) | 100% |
| 4,096 | 100% (105/105) | 100% (40/40) | 100% |

These detection results apply only to this synthetic scenario grid. They do not establish performance for arbitrary circuits or physical hardware.

## Family-level adjudicated recall

The bundled logistic-regression candidate was fitted on 330 synthetic rows from F1/F2. It was not fit on F3/F4/F5. Family-wise results show the transfer limitation:

| Family | NORMAL recall | PROGRAM_FAULT recall | HARDWARE_ANOMALY recall |
|---|---:|---:|---:|
| F1 GHZ chain | 93.33% | 100% | 98.89% |
| F2 Toffoli compute | 100% | 100% | 100% |
| F3 swap routing | 100% | 0% | 100% |
| F4 parity tree | 93.33% | 100% | 100% |
| F5 phase layer | 100% | 50% | 100% |

For C15 combined-fault ambiguity, F4 recalled 11/15 cases; each other family recalled 15/15. All 71 high-confidence raw single-cause predictions were adjudicated as ambiguous by the combined-fault evidence path; the remaining four rows abstained for insufficient confidence. This shows the evidence gate correcting the model's missing mixed-fault class on this synthetic grid, while F4 still misses four cases.

## Pipeline meaning

- **Output anomaly detection in this Phase 5 scorecard** uses the predeclared TVD rule `TVD > max(0.05, 1/sqrt(shots))`. This is a heuristic finite-shot guard plus effect threshold, not a permutation p-value or confidence interval. The high-level `diagnose_circuit()` path has raw counts and separately uses a conditional TVD permutation test plus its practical effect threshold.
- **Class prediction** uses five output-distribution features to predict NORMAL, PROGRAM_FAULT, or HARDWARE_ANOMALY. Its probabilities are uncalibrated.
- **Final attribution** requires independent statistical and localization support. A program diagnosis needs a source/counterfactual link to changed measured behavior; a hardware diagnosis needs relevant calibration evidence. Competing causes remain ambiguous.

The classical model is explicitly supplied to `diagnose_circuit(..., classifier=...)` or to `diagnose_execution()` for already-observed distributions; it is never loaded implicitly. The artifact is labeled **DEVELOPMENT CANDIDATE; NOT VALIDATED FOR DEPLOYMENT**. QMutBench mutation-outcome models have a different target and are not used by this diagnosis workflow.

## Next release gate

These cases are synthetic and four-qubit. They do not cover arbitrary topology, gate families, noise drift, simulator scale, or live backend behavior. The exposed F3/F4/F5 cases must not be reused as fresh test data after tuning. A new candidate needs broader label-compatible circuit fault examples, a new family-disjoint holdout, probability calibration on a separate calibration split, and per-family recall/selective-risk acceptance before any default enablement. Keep the simulator's rule/statistical evidence path available even if ML is disabled.

The 900 raw case output files remain in this directory. No hardware credentials or user API keys are included.
