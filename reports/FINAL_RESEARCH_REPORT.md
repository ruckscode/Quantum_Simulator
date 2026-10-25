# Final Research Report: Quantum Simulator Diagnosis Prototype

**Status date:** 2026-10-16  
**Project status:** Research prototype; current research phase finalized. No new experiments were run for this report.

## 1. Executive summary and final project status

The project implements a local statevector simulator, configurable noise and calibration representations, statistical comparison, program-operation localization, hardware-candidate localization, and a diagnosis workflow that can express ambiguity or insufficient evidence. The work also includes controlled synthetic scenario generation and exploratory grouped cross-validation.

The evidence supports software functionality and exploratory results on locally generated simulator data. It does not support a claim of a validated hardware diagnosis system. The saved five-observable ML evaluation is modest (OOF accuracy 46.30–56.30%), its feature space contains exact cross-label collisions, the frozen diagnosis acceptance evaluation matches 38/45 cases, QMutBench eligibility preflight is blocked, and finite-shot null validation failed its overall registered gate. Independent diagnosis generalization and physical-hardware fault attribution remain unvalidated.

No attempt was made to improve the model or continue experimentation. Existing reports, saved predictions, fixtures, labels, thresholds, and results are preserved.

## 2. Research question, objectives, and intended contributions

The research question was whether observed quantum-program behavior, compared with an expected reference and available calibration context, could support detection and separation of program faults from hardware anomalies while allowing ambiguous or insufficient-evidence outcomes.

The intended contributions were:

- A usable local circuit and statevector simulation API, including selected stochastic noise effects.
- Distribution comparison with finite-shot tolerance.
- Program-operation comparison and hardware calibration candidate localization.
- An integrated evidence-bearing diagnosis result with ambiguity/abstention paths.
- Synthetic scenario evaluation and an exploratory ML baseline.
- Data provenance, lineage, feature-signal, acceptance, and finite-shot audits to identify what the evidence does and does not justify.

## 3. System architecture and implemented components

The README describes the primary flow as circuit representation, ideal or hardware-aware execution, statistical comparison, anomaly detection, localization, and diagnosis.

Implemented components include:

- Circuit builder and statevector simulator with supported gates, measurement, reset, and shot sampling.
- Generic `HardwareModel` for qubit properties, connectivity, gate calibrations, and timestamp provenance.
- Statistical comparison including total variation distance (TVD), distribution metrics, and a shot-dependent tolerance.
- Program fault localization by comparing expected and observed operations.
- Hardware fault localization using circuit-relevant calibration metrics and connectivity.
- `diagnose_circuit()` and lower-level `diagnose_execution()` interfaces that return category, evidence, and missing-evidence information.
- Optional IBM integration and dataset adapters. Their offline fixture checks are not live hardware validation.

The simulator''s noise model is limited and documented: readout flips, probabilistic gate skipping, amplitude damping, and phase-flip dephasing. The README explicitly does not characterize this as a general physical noise model.

## 4. Dataset sources, simulator-generated scenarios, provenance, and labeling

The ML signal audit describes 270 completed reports: 30 generated circuit families × three constructed classes × three seeded repetitions, giving 90 rows per class. The local simulator produced ideal and observed counts, generally at 1,024 shots for this dataset. The labels NORMAL, PROGRAM_FAULT, and HARDWARE_ANOMALY were assigned from the intervention/scenario construction. They are not independent external adjudications.

The production features were TVD, Jensen–Shannon divergence, Hellinger distance, observed entropy, and observed maximum probability. Calibration fields and gate mismatch were explored in separate, non-production schemas. In this synthetic generator, calibration summaries are scenario-associated; gate mismatch is positive for generated program-fault rows and zero for the other two classes, making that field a strong construction proxy.

QMutBench local files include verified local byte hashes and origin-to-mutant grouping for a bounded bundle. This does not establish complete chain of custody, upstream recipe ancestry, independently adjudicated behavioral labels, or independent circuit roots. Dataset and lineage reports are simulator-only where observations are present.

## 5. Experimental methodology and evaluation design

The saved grouped-CV audit used fixed estimator configurations, a five-fold shuffled StratifiedGroupKFold split, seed 42, and `circuit_family_id` as the grouping variable. Repetitions within a family were kept together. The folds therefore test held-out generated families within the same scenario-generation framework. They do not establish independently verified recipe-root separation or generalization to independently sourced families.

The saved OOF predictions were joined and checked by `run_id`. The ML signal audit reports the historical metrics; the later provenance preflight is a separate read-only gate and reports that it initiated no fitting. These facts must not be conflated: the saved OOF scores exist, but their historical dataset is ineligible under the later provenance requirements.

The frozen 45-case diagnosis acceptance set was evaluated with its pre-existing outcomes and thresholds. The seven mismatches remain failures; no fixture, label, threshold, or expected outcome was changed.

## 6. Existing quantitative results

The saved production five-observable OOF evaluation had 270 cases, 90 per synthetic class, no abstentions, and the following accuracies and per-class metrics:

| Model | OOF accuracy | Class | Precision | Recall | F1 | Support |
|---|---:|---|---:|---:|---:|---:|
| Logistic Regression | 56.30% | NORMAL | 50.61% | 92.22% | 65.35% | 90 |
|  |  | PROGRAM_FAULT | 76.00% | 42.22% | 54.29% | 90 |
|  |  | HARDWARE_ANOMALY | 55.36% | 34.44% | 42.47% | 90 |
| Decision Tree | 47.04% | NORMAL | 49.29% | 76.67% | 60.00% | 90 |
|  |  | PROGRAM_FAULT | 54.41% | 41.11% | 46.84% | 90 |
|  |  | HARDWARE_ANOMALY | 33.87% | 23.33% | 27.63% | 90 |
| Random Forest | 46.30% | NORMAL | 47.62% | 66.67% | 55.56% | 90 |
|  |  | PROGRAM_FAULT | 53.75% | 47.78% | 50.59% | 90 |
|  |  | HARDWARE_ANOMALY | 34.38% | 24.44% | 28.57% | 90 |

Confusion matrices below use true classes as rows and predicted classes as columns in the order NORMAL, PROGRAM_FAULT, HARDWARE_ANOMALY.

**Logistic Regression**

| True class | NORMAL | PROGRAM_FAULT | HARDWARE_ANOMALY |
|---|---:|---:|---:|
| NORMAL | 83 | 0 | 7 |
| PROGRAM_FAULT | 34 | 38 | 18 |
| HARDWARE_ANOMALY | 47 | 12 | 31 |

**Decision Tree**

| True class | NORMAL | PROGRAM_FAULT | HARDWARE_ANOMALY |
|---|---:|---:|---:|
| NORMAL | 69 | 7 | 14 |
| PROGRAM_FAULT | 26 | 37 | 27 |
| HARDWARE_ANOMALY | 45 | 24 | 21 |

**Random Forest**

| True class | NORMAL | PROGRAM_FAULT | HARDWARE_ANOMALY |
|---|---:|---:|---:|
| NORMAL | 60 | 10 | 20 |
| PROGRAM_FAULT | 25 | 43 | 22 |
| HARDWARE_ANOMALY | 41 | 27 | 22 |

These are saved exploratory metrics from one simulator-derived, scenario-labeled dataset. They are not current-source validated estimates of independent diagnosis performance. Expanded schemas score higher in the same folds, but calibration fields encode generated profile information and circuit mismatch is a construction proxy; those scores do not establish the performance of the five-observable model.

## 7. Feature overlap and identifiability findings

The read-only candidate-feature audit recomputed the production observables from the saved 270 report distributions, without simulation. It found 223 unique exact feature vectors, 47 duplicate excess rows, and nine cross-class collision groups affecting 46 rows. Those collisions show that the tested representation cannot uniquely recover the constructed label for every affected record. This is a limitation of this representation on this dataset, not a universal impossibility result for diagnosis or other measurements.

The machine-readable audit verifies aggregate group sizes and class counts. It does not include stable record IDs or full five-feature vectors in its collision summary, so exact group identities cannot be cross-checked from that artifact. The nine groups have this verified composition:

| Classes sharing a vector | Groups | Affected rows | Class counts by group |
|---|---:|---:|---|
| HARDWARE_ANOMALY + NORMAL | 1 | 3 | 2 NORMAL, 1 HARDWARE_ANOMALY |
| HARDWARE_ANOMALY + NORMAL + PROGRAM_FAULT | 4 | 33 | (9, 9, 3), (1, 1, 1), (1, 3, 1), (1, 2, 1) for NORMAL, HARDWARE_ANOMALY, PROGRAM_FAULT, respectively |
| HARDWARE_ANOMALY + PROGRAM_FAULT | 2 | 4 | 1 of each class in each group |
| NORMAL + PROGRAM_FAULT | 2 | 6 | 2 NORMAL, 1 PROGRAM_FAULT in each group |

Counts are from `data/qmutbench_ml/final/candidate_feature_audit.json`. The older ML signal-quality audit reports 220 unique vectors and 11 cross-label repeated vectors using its saved-artifact analysis. That result remains in its source report, but the later direct recomputation is used here; the artifacts do not explain the difference.

For the G2/G4 comparison, the saved collision review states that their stored observed counts are identical within each collision group. G6 counts differ in one finite-shot run, but a single run does not establish a stable distinguishing signal. The exact per-group G2/G4/G6 count table was unavailable in the reviewed summary artifacts.

The signal audit also reports strong redundancy: TVD correlates with Hellinger at 0.990 and JS divergence at 0.944; JS and Hellinger correlate at 0.969; observed entropy and maximum probability correlate at −0.918. Thus, five columns do not represent five independent signal dimensions.

## 8. Diagnosis engine, ambiguity handling, abstention, and metric reporting

The implemented diagnosis workflow returns categories including NO_ANOMALY, PROGRAM_FAULT, HARDWARE_ANOMALY, AMBIGUOUS, and INSUFFICIENT_EVIDENCE, along with evidence and missing-evidence details. The program localizer can produce operation-level candidates; the hardware localizer can surface calibration-relevant candidates. The system can abstain from a specific cause when evidence is ambiguous or insufficient.

The README and implementation reports distinguish anomaly detection from causal attribution: a detected distribution change alone does not identify its cause. A source-level mismatch may be localized even if observed distributions agree, and an observationally equivalent program change cannot be identified from distribution comparison alone.

Metric reporting improved through explicit class ordering, per-class precision/recall/F1/support, confusion matrices, saved OOF joins by run identifier, grouped-fold manifests, signal-quality audits, and provenance preflights. However, the historical ML metrics record zero abstentions, so they do not show a validated selective-prediction benefit. Implemented ambiguity/abstention behavior is a software capability, not evidence of calibrated real-world abstention performance. The saved focused diagnosis checkpoint records the forensic mismatch-explanation test passing, while the generalization-outcomes test remains failing on the same seven cases (one pass, one fail in the focused file run). This verifies the recorded explanation against those fixtures; it does not make the acceptance criterion pass.

## 9. Frozen acceptance evaluation

The frozen acceptance evaluation matched **38/45** cases. Seven mismatches remain, so the acceptance criterion is unmet.

| Frozen case | Ideal TVD | Sampled TVD at 256 shots | Evidence-backed explanation |
|---|---:|---:|---|
| qft_like_3q_program_fault_11 | 0.0078559 | 0.01953125 | Expected-positive fault falls below unchanged 0.05 threshold and 0.0625 finite-shot guard; source target found. |
| qft_like_3q_program_fault_29 | 0.0012597 | 0 | Same sub-threshold limitation; source target found. |
| qft_like_3q_program_fault_47 | 0.0112994 | 0.01953125 | Same sub-threshold limitation; source target found. |
| ring_cluster_4q_correct_11 | 0 | 0.0859375 | Equal ideal distributions yielded a sampled false positive at the registered seed. |
| ring_cluster_4q_correct_29 | 0 | 0.125 | Same false-positive mechanism. |
| ring_cluster_4q_equivalent_11 | 0 | 0.0859375 | Commuting CZ reorder preserves output distribution; sampling false positive and syntactic operation candidates produce a PROGRAM_FAULT category. |
| ring_cluster_4q_equivalent_29 | 0 | 0.125 | Same observational-equivalence and finite-shot false-positive mechanism. |

The current rule flags an anomaly when `TVD > max(threshold, 1/sqrt(shots))`; the default threshold is 0.05, so at 256 shots the effective cutoff is 0.0625. The reference and submitted circuits are sampled on separate streams, allowing nonzero empirical TVD even when ideal distributions match. The recorded explanations are simulator diagnostics, not proof of physical causation. Frozen fixtures, labels, thresholds, and outcomes remain unchanged.

## 10. QMutBench and lineage limitations

The primary ML data-foundation preflight reports **0/270 eligible rows**. All 270 lack a required circuit lineage ID and per-run source hash; the dataset manifest lacks generation provenance and saved generator hashes do not match current generation code. The preflight states that no fitting, prediction, or model export started and that it performed no writes.

The QMutBench evidence-readiness report is BLOCKED. It identifies unresolved archive and mapping provenance, complete prior-artifact reconciliation, source integrity/chain of custody, independently adjudicated behavioral labels, execution measurement provenance, and independent reviewer authentication. The lineage audit finds zero eligible recipe roots; only one original circuit group is represented in the bounded local bundle/run, and exact MQT Bench recipe ancestry is unresolved.

Historical QMutBench or related 270-row reports do not establish independent diagnosis generalization. Their labels and observations are not independently adjudicated; recipe ancestry and independent roots are not verified; source/runtime provenance is incomplete. Family-grouped folds are not equivalent to independently verified recipe-root separation.

## 11. Finite-shot validation findings and confirmation status

The completed synthetic two-sample TVD characterization used a fixed detector, `TVD > max(0.05, 1/sqrt(shots))`, and a preregistered null upper-bound gate of 0.06 at each shot count. It tested 2,000 null pairs per shot count and 500 alternatives per effect/shot cell.

| Shots | Null detections / 2,000 | Null rate | One-sided 95% upper bound | Gate |
|---:|---:|---:|---:|---|
| 256 | 680 | 34.00% | 0.3578 | Fail |
| 1,024 | 114 | 5.70% | 0.0663 | Fail |
| 4,096 | 0 | 0.00% | 0.0015 | Pass |

The overall preregistered decision was **FAIL**, because the first two null upper bounds exceed 0.06. For the true TVD 0.025 alternative, detection rates were 0.44, 0.214, and 0.02 at 256, 1,024, and 4,096 shots respectively. This illustrates the fixed minimum cutoff''s behavior as sampling variation contracts; it does not establish performance on circuits, independent families, or hardware.

A 38,228-pair confirmation design is a proposal, not completed validation. Its confirmation experiment was not run. The separate trust-gate report says execution remains NO-GO: required security and execution controls were not established, including a provisioned external reviewer identity/key and signed approval, protected trust-root/ACL evidence, and deployment controls. This report does not treat sample-size calculations or readiness proposals as experiment results, and does not resume or redesign that process.

## 12. What worked, what failed, and evidence-supported root causes

| Research component | Work completed | Evidence available | Current status | Remaining limitation |
|---|---|---|---|---|
| Simulator and circuit API | Statevector execution, gates, measurements, selected noise trajectories | README, source modules, saved local validation artifacts | Implemented prototype | Simulator evidence is not physical-device fidelity |
| Diagnosis workflow | Statistical comparison, program/hardware candidate localization, ambiguity and insufficient-evidence outputs | README, diagnosis reports and frozen fixtures | Implemented; partial frozen acceptance | 38/45 match; causal attribution unvalidated on hardware |
| Synthetic dataset | 270 rows across 30 families and three labels | Dataset reports, manifest, signal audit | Completed historical dataset | Scenario labels and simulator-only observations |
| Five-feature ML evaluation | Three estimators with saved grouped-fold OOF results | OOF predictions, metrics, confusion matrices and manifest | Exploratory result preserved | Weak scores; later provenance gate rejects all rows |
| Feature identifiability | Exact vector collision and correlation audits | ML signal-quality summary | Overlap demonstrated in tested data | Fine-grained collision listing unavailable in reviewed artifacts |
| QMutBench readiness | Local bytes/groupings audited; eligibility gates evaluated | Lineage and evidence-readiness reports | Blocked; zero eligible roots | Chain of custody, ancestry, labels and independent review missing |
| Finite-shot characterization | Fixed-detector synthetic distribution study | Preregistration and result report | Overall gate failed | Does not validate circuit or hardware diagnosis |
| Finite-shot confirmation | Prospective design and trust/readiness review | Preregistration and security/readiness reports | Not executed; NO-GO | Security/execution controls not established |
| Physical-hardware diagnosis | Optional IBM adapter and offline fixtures exist | IBM offline validation report and README | Not established | No verified live hardware evidence or adjudicated fault labels |

Evidence-supported explanations for failure are limited to the recorded facts: weak or overlapping output-distribution summaries; exact cross-label feature collisions; high feature redundancy; low fault-class recall; sub-threshold ideal differences in three frozen positive cases; and finite-shot false positives in four frozen correct/equivalent cases at the registered seeds. The provenance gates explain why historical ML and QMutBench artifacts cannot support stronger independent claims. Causes beyond these documented mechanisms remain unverified.

## 13. Limitations and threats to validity

- Synthetic labels are scenario assignments, not independent fault adjudications.
- Simulator-generated data cannot establish physical hardware behavior, device fidelity, or real fault attribution.
- Thirty generated families provide a limited grouping frame; family-held-out folds do not verify recipe-root independence.
- The five observables are redundant summaries of output distributions and have demonstrated exact cross-label collisions.
- The saved evaluation is historical; its 270 records fail later provenance and lineage gates.
- Fixed estimator results are descriptive and were not selected through nested evaluation.
- The acceptance cases are a frozen constructed set and currently fail seven expected outcomes.
- The finite-shot characterization uses synthetic multinomial distributions and fails its overall null gate.
- QMutBench's small local bundle and metadata do not establish comprehensive provenance or independent labels.
- No physical hardware fault dataset with verified measurement lineage and independent adjudication was established.
- Stable IDs and exact five-feature vectors for individual collision groups, and a row-level G2/G4/G6 observed-count table, remain unavailable.

## 14. Conclusions: supported and unsupported claims

**Supported by existing evidence:** the repository implements a simulator-oriented diagnosis prototype; saved local-simulator synthetic evaluations exist; grouped-family OOF accuracies for the three fixed ML estimators are 56.30%, 47.04%, and 46.30%; the five-feature representation has cross-label collisions and substantial redundancy; the frozen acceptance result is 38/45; the QMutBench and primary ML eligibility gates are blocked; and the completed finite-shot null study failed at 256 and 1,024 shots while passing the 4,096-shot cell.

**Not supported:** that the system is a fully validated hardware diagnosis system; that it independently generalizes to new recipe roots or real quantum programs; that it reliably distinguishes program faults from hardware anomalies on physical devices; that historical QMutBench rows provide independent diagnosis ground truth; that G2/G4 are stably distinguishable using the observed-count difference seen for G6; or that the unrun confirmation design validates the detector.

## 15. Future work required before stronger scientific claims are justified

Before stronger claims are justified, the evidence requirements are:

1. Establish source integrity, full execution provenance, and independently verified recipe-root lineage for eligible diagnosis cases; family-level separation within the synthetic generator does not satisfy this requirement.
2. Obtain independently adjudicated behavioral labels and define a locked, recipe-root-separated evaluation with a held-out test set. Report per-class performance and uncertainty, including abstention behavior.
3. For hardware-attribution claims, collect physical-device observations with documented circuits, settings, raw counts, contemporaneous calibration, and independent fault adjudication. Simulator-only noise profiles and offline adapter fixtures are insufficient.
4. Validate finite-shot detection and attribution on eligible independent cases under a prospectively fixed protocol. The existing synthetic null study failed its overall gate, and its proposed confirmation remains unexecuted and NO-GO.

These are evidence prerequisites identified by existing audits; no such work was performed as part of this finalization.

## Evidence reviewed

Primary evidence included `README.md`; `ml_results/grouped_cv_audit_v2/ML_SIGNAL_QUALITY_AUDIT.md`, saved OOF predictions, metrics, confusion matrices, fold assignments and evaluation manifest; `data/qmutbench_ml/final/candidate_feature_audit.json`; `data/qmutbench_ml/final/training_data_label_audit.json`; `reports/independent_research_handoff_20261015.md`; `reports/experiments/heldout_diagnosis_root_cause_20261014.json`; `reports/experiments/ml_data_foundation_preflight_20261015.json`; `reports/experiments/qmutbench_external_evidence_readiness_20261015.json`; `reports/qmutbench_lineage_audit_20261015.json`; `reports/experiments/finite_shot_synthetic_characterization_report_20261015.md`; and the finite-shot null-confirmation preregistration, trust-gate, and implementation-readiness reports.

The stable identities and exact five-feature vectors for the nine collision groups, and the full G2/G4/G6 per-group observed-count table, remain unavailable. The audited aggregate class composition is included above. G2/G4 observed-count equality and the single-run G6 difference remain qualitative findings without recoverable row-level tables.

