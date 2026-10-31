# Classical simulator diagnosis benchmark — 2026-10-31

## Scope and acceptance

The Phase 5 scorecard summarizes 900 synthetic configurations: 60 scenarios, five seeds, three shot counts (256, 1,024 and 4,096), and five four-qubit circuit families. Saved outputs were rescored through the shared `diagnose_execution(..., classifier=...)` evidence gate. Rescoring did not resample circuits, retrain the model, or overwrite per-case outputs.

**Overall predeclared acceptance: FAIL.** Six of seven checks passed. Program-fault recall was 70%, below the 80% minimum. The checks for output sensitivity/specificity by shot count, exact-target localization, combined-fault abstention recall, seed stability, and deterministic configuration/reference regeneration passed.

The evaluation includes model-development families F1/F2 as well as family-disjoint F3/F4/F5. It is therefore not a wholly held-out generalization score. The model was fitted only on synthetic F1/F2 rows, but F3/F4/F5 have now been examined and cannot be treated as a fresh locked holdout for tuning.

## Classification and localization

| Metric | Result | Evaluation rows |
|---|---:|---:|
| Raw ML accuracy | 80.0% (540/675) | Single-cause rows; imbalanced classes |
| Evidence-adjudicated macro-F1 | 0.9363 | 675 single-cause rows |
| Program-fault precision / recall / F1 | 1.000 / 0.700 / 0.8235 | 150 |
| Hardware-anomaly precision / recall / F1 | 1.000 / 0.9978 / 0.9989 | 450 |
| Normal precision / recall / F1 | 1.000 / 0.9733 / 0.9865 | 75 |
| Exact target localization | 99.83% (599/600) | Program, qubit, and gate targets |
| Combined-fault ambiguity recall | 94.67% (71/75) | Program plus hardware cases |
| Model confidence coverage at 0.60 | 96.56% (869/900) | Probabilities uncalibrated |
| Adjudicated non-abstain rate | 82.67% (744/900) | All rows |
| Selective risk | 21.43% (171/798) | Covered single-cause rows; mixed cases excluded |

The model predicts only NORMAL, PROGRAM_FAULT, or HARDWARE_ANOMALY; it has no mixed-fault class. All 71 high-confidence raw single-cause predictions on combined-fault rows were adjudicated ambiguous by independent evidence. The other four rows abstained for insufficient confidence. F4 ambiguity recall was 11/15; each other family achieved 15/15.

### Output-change detection (separate from ML class accuracy)

| Shots | Sensitivity | Specificity | Balanced accuracy |
|---:|---:|---:|---:|
| 256 | 98.10% (103/105) | 95.00% (38/40) | 96.55% |
| 1,024 | 100% (105/105) | 100% (40/40) | 100% |
| 4,096 | 100% (105/105) | 100% (40/40) | 100% |

Output detection in this scorecard used the predeclared heuristic `TVD > max(0.05, 1/sqrt(shots))`. It is not a permutation p-value or a confidence interval. The high-level `diagnose_circuit()` route has raw counts and separately uses a conditional TVD permutation test with a practical effect threshold.

## Per-family recall

| Family | Normal | Program fault | Hardware anomaly |
|---|---:|---:|---:|
| F1 GHZ chain | 93.33% | 100% | 98.89% |
| F2 Toffoli compute | 100% | 100% | 100% |
| F3 swap routing | 100% | 0% | 100% |
| F4 parity tree | 93.33% | 100% | 100% |
| F5 phase layer | 100% | 50% | 100% |

These family results show a key limitation: program-fault recall varies substantially with circuit family despite the aggregate macro-F1.

## Interpretation and limitations

- Class prediction uses five output-distribution features. The bundled logistic-regression artifact was fitted on 330 synthetic rows from F1/F2 only.
- The model probabilities are uncalibrated. A confidence of 0.60 is an operational abstention threshold, not an empirical 60% correctness claim.
- A program-fault diagnosis requires independent source/counterfactual evidence linked to changed measured behavior. A hardware diagnosis requires relevant calibration evidence. Competing causes remain ambiguous.
- The results apply only to the synthetic scenario grid. They do not establish performance on arbitrary circuits, topologies, fault strengths, shot counts, or physical devices.
- The raw 900-case execution records are not included in this branch. The scorecard is a report of their saved-output rescore; it should not be treated as an independently reproduced run from this branch alone.

## Release decision

Keep the classifier opt-in and labeled **DEVELOPMENT CANDIDATE; NOT VALIDATED FOR DEPLOYMENT**. Before default enablement, evaluate on a newly generated, family-disjoint locked holdout; calibrate probabilities on a separate split; and predeclare per-class recall, selective-risk, localization, and abstention requirements. Physical-device claims require separate real-hardware validation.
