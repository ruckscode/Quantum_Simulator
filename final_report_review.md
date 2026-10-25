# Final Research Report Review

**Reviewed file:** `final_research_report.md`  
**Source draft:** `final_research_report_draft.md` (read only; preserved unchanged)  
**Review basis:** `project_finalization_audit.md`, `project_finalization_checklist.md`, the cited source/tests, and saved evaluation reports/prediction artifacts. No experiments, tests, training, data generation, package installation, or hardware access were performed.

## Corrections and improvements made

- Reorganized the draft into an academic report structure: abstract; introduction and motivation; problem/objectives; background and related work; architecture; simulator/noise implementation; diagnosis/localization; datasets; ML methodology; results; error analysis; discussion; limitations/threats; conclusion/future work; and references/evidence sources.
- Replaced repeated narrative metric descriptions with separate quantitative tables for the two binary QMutBench partitions and the family-held-out three-class comparison. Denominators, class order, validation threshold selection, and rounded metric precision are stated.
- Kept the two QMutBench binary partitions distinct: the earlier 8,051/1,952 report and later 5,920/2,408/1,675 train/validation/test threshold report. Explained the label construction and zero-feature relationship next to the results.
- Explained family-held-out GroupKFold as a defense against within-family leakage, while limiting the conclusion to transfer among the 30 saved simulator families. Reported the small per-family test size, overlapping outer training sets, feature collisions, and absent significance analysis.
- Improved the diagnosis explanation using source behavior and tests. Added the implementation detail that `AMBIGUOUS` and `INSUFFICIENT_EVIDENCE` enum members share the same serialized enum value; evidence, missing-evidence, and summary fields convey the reason certainty was withheld.
- Separated implemented simulator T1/T2 behavior and prior controlled T1/T2 cases from the proposed first wave, which explicitly excluded T1/T2. Clarified that no physical time-unit validation or real-device validation is evidenced.
- Stated that the 240-report first wave was reviewed as a proposal but blocked at preflight, and that no reports were generated. No planned counts/seeds are presented as experiment observations.
- Described the absent compatible three-class observable model artifact and the incompatibility of the saved binary trace-feature NPZ files. Clarified that mock adapter tests do not validate real artifact inference.
- Added a background/related-work section and a references section without inventing external sources. The report states that no scholarly literature bibliography was found among reviewed project materials; the listed references are verifiable repository sources only.
- Retained negative findings: low NORMAL precision in binary QMutBench results, weak hardware recall in the three-class comparison, inconsistent tuned-family gains, unavailable benchmark ground truth, and missing live hardware/model evidence.

## Quantitative cross-check

All quantitative claims in `final_research_report.md` were checked against the cited saved artifact or, for simple missed-case counts, calculated directly from a saved confusion matrix. Values rounded in the report agree with the source values to the shown precision.

| Report quantities | Supporting artifact and verification |
|---|---|
| Binary dataset: 10,003 total; 9,792 mutant/PROGRAM_FAULT; 211 original/NORMAL; 211 original all-zero; 1,258 mutant all-zero. | `data/qmutbench_ml/final/evaluation_report.json`, `dataset_audit.json`, `zero_feature_diagnostic.json`. Class counts sum to total. |
| Earlier binary split: 8,051 train / 1,952 test; 169 / 42 groups; zero overlap; accuracy 0.8294057; balanced accuracy 0.9128272; macro-F1 0.5529706; class P/R. | `data/qmutbench_ml/final/evaluation_report.json`. Reported rounded metrics match. |
| Later binary split: 5,920 / 2,408 / 1,675 records; 126 / 42 / 43 groups; zero pairwise overlap; threshold 0.8100000000000002; validation-selected and held-out test results. | `data/qmutbench_ml/final/threshold_tuning_report.json`. Confirms group split, threshold selection, 43 NORMAL/1,632 PROGRAM_FAULT test, metrics, and confusion matrix. |
| Later binary matrix `[[1426,206],[0,43]]`; 206 mutant-labeled records predicted NORMAL; recall, precision, accuracy, balanced accuracy, macro-F1. | `threshold_tuning_report.json`; `final_project_summary.json`. Matrix row totals match 1,632 and 43. |
| Three-class dataset: 270 reports, 30 families, 9/family, 90/class; five features; 30 outer folds; 261/9 train/test per fold; 3/class in test; five inner folds. | `data/qmutbench_ml/final/grouped_model_comparison.json`, fields `source` and `evaluation`; grouped Markdown. |
| Grouped comparison scores, class metrics, and confusion matrices for tuned/fixed logistic, RBF-SVM, LDA, Extra Trees, Random Forest, and dummy. | `data/qmutbench_ml/final/grouped_model_comparison.md`, “Out-of-fold results”; detailed numeric data in `.json`. All displayed metrics are rounded to three decimals. |
| Tuned-vs-fixed family comparison 7 better / 18 tied / 5 worse; mean macro-F1 delta 0.037. | `grouped_model_comparison.md`, consistency table; machine-readable comparison in `.json`. |
| Fixed and tuned missed hardware examples: 57 and 52 out of 90. | Derived arithmetically from saved hardware confusion-matrix diagonal values 33 and 38 in `grouped_model_comparison.json`; no new evaluation run. |
| Controlled test fixture: 1 normal, 5 program, 4 hardware cases. | `tests/test_stage3_controlled_diagnosis_evaluation.py`, `CASES`; the test source includes four hardware scenarios, including T1/T2. |
| QMutBench adapter 11/12 executed, 1 unsupported, zero reliable diagnosis-category-compatible GT. | `reports/benchmarks/qmutbench_validation_report.json`; metric fields are null with explicit reason. |
| Bugs4Q focused integration 8 selected cases, 3 NO_ANOMALY and 5 PROGRAM_FAULT. | `README.md`, §15; `reports/bugs4q_focused_integration_report.json`. Described as integration counts only. |
| Proposed first wave planned at 240; T1/T2 excluded; preflight blocked/stop and generated reports zero. | `first_wave_execution_preflight.md/.json`; `data/qmutbench_ml/final/first_wave_spec_v3_review.md`. Planned scenario counts remain described as planned. |
| Historical test totals 280, 89, and 96. | `README.md`; `reports/project_status.json`; `reports/reproducibility_report.json`. Identified as separate snapshots and not combined. |

**Verification outcome:** every quantitative claim included in the final report has a supporting saved artifact or an explicitly described arithmetic derivation. The underlying experiments and predictions were not rerun. This verifies traceability and consistency with the saved record, not independent reproduction of every CSV row.

## Remaining evidence gaps and qualified claims

1. **Related work:** No scholarly literature review or external bibliography was supplied or verified. The final report makes no external comparison or novelty claim. Add a separately sourced literature review if publication requires it.
2. **Real hardware:** No live backend job, physical calibration record, or real-device performance result exists in the reviewed evidence. The IBM report is an offline fixture.
3. **Behavioral labels:** QMutBench labels are record-type-derived, not independently verified behavior labels; the adapter evaluation also lacks diagnosis-category-compatible benchmark ground truth.
4. **Three-class model:** No compatible trained observable three-class artifact is present; real inference through the adapter is unverified. Existing mock tests support interface behavior only.
5. **Three-class reliability:** Scores are modest, hardware recall is weak, family-level tuned gains are inconsistent, and no confidence intervals/significance tests are reported. No reliability/generalization claim is made.
6. **T1/T2:** Simulator code and prior controlled tests include modeled T1/T2 behavior, but first-wave T1/T2 coverage was excluded. Physical units/calibration are not established.
7. **First wave:** The proposed 240-report experiment was blocked and not run; realized same-stream RX skip counts and a safe run/resume protocol remain unresolved.
8. **Fresh verification:** The final report review did not rerun tests or replay all stored predictions. Historical test totals are identified as separate snapshots.

## Claim disposition

- **Supported with direct code/test/saved report evidence:** local statevector execution; configured simulator-side gate/readout/relaxation behaviors; statistical comparison; direct operation candidates; calibration-informed hardware candidates; evidence retention and uncertainty handling in the tested scenarios.
- **Supported only with explicit scope limits:** QMutBench binary classification metrics, controlled family-held-out three-class metrics, adapter integration case counts, simulated T1/T2 test cases.
- **Must remain absent or qualified:** real-hardware validation; calibrated physical relaxation claims; independent QMutBench behavioral ground truth; reliable three-class diagnosis; compatible deployed ML inference; first-wave results; statistical significance or broad family generalization.

The report is academically more structured and traceable, but it is not a substitute for the missing external literature review or unavailable experiments/evidence listed above.
