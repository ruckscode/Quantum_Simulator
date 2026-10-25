# Independent Research Handoff: Finite-Shot Diagnosis and ML Validation

**Status:** frozen-baseline handoff with planning-only research gates. A separately authorized bounded QMutBench acquisition/ingestion smoke milestone is recorded in §6; it did not authorize model fitting, a confirmatory evaluation, full ingestion, or hardware work.  
**Baseline:** frozen and unchanged. The existing 45-case evaluation remains a historical acceptance set, not a development or calibration set.

## 1. Frozen baseline

### Reproduction record

The finalization checkpoint records these exact commands and current-tree results:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_reference_diagnosis_generalization.py -q --tb=short --basetemp .t-finalization-focused
# 1 failed, 1 passed in 66.59s

.\.venv\Scripts\python.exe -m pytest tests -q --tb=short -k "not bounded_recipe_comparison_120_templates_and_finite_shot_subset and not independent_calibration_fixture_and_permutation_comparison" --basetemp .t-finalization-suite
# 794 passed, 1 failed, 0 skipped, 2 deselected (797 collected) in 259.71s
```

The focused run's passing test is `test_frozen_heldout_mismatches_are_explained_by_distribution_resolution`. Its failing test is `test_heldout_generalization_outcomes_and_targets`. The bounded suite's only failure is the same frozen acceptance test. These are saved checkpoint outcomes; this handoff did not rerun them. The `reports/project_status.json` legacy `verification` object still says 89 tests passed and is an older snapshot. Its later `finalization_milestone` object records the current 794/1 result. Do not read the legacy count as current.

### Seven unchanged mismatches

| Case | Ideal TVD | Sampled TVD at 256 shots | Existing explanation |
|---|---:|---:|---|
| `qft_like_3q_program_fault_11` | 0.0078559 | 0.01953125 | Frozen expected-positive fault is below the 0.05 threshold and 0.0625 finite-shot guard; source target is found. |
| `qft_like_3q_program_fault_29` | 0.0012597 | 0 | Same sub-threshold limitation; source target is found. |
| `qft_like_3q_program_fault_47` | 0.0112994 | 0.01953125 | Same sub-threshold limitation; source target is found. |
| `ring_cluster_4q_correct_11` | 0 | 0.0859375 | Identical ideal distributions yield a sampled false positive at this registered seed. |
| `ring_cluster_4q_correct_29` | 0 | 0.125 | Same false-positive mechanism. |
| `ring_cluster_4q_equivalent_11` | 0 | 0.0859375 | Commuting CZ reorder preserves output distribution; sampling false positive plus syntactic operation candidates produces an ambiguous diagnosis. |
| `ring_cluster_4q_equivalent_29` | 0 | 0.125 | Same observational-equivalence and finite-shot false-positive mechanism. |

All source-target checks pass; no hardware target was expected in these cases. The diagnosis test measures detection, category, abstention, and localization on 45 constructed cases while withholding scenario and mutation metadata from `diagnose_circuit`. The exact/sampled values are simulator diagnostics, not independent proof of physical causation. The frozen fixture, thresholds, outcomes, and acceptance test remain untouched. Sources: `tests/test_reference_diagnosis_generalization.py`, `tests/fixtures/reference_diagnosis_generalization_cases.json`, and `reports/experiments/heldout_diagnosis_root_cause_20261014.json`.

### Current rule and exclusions

`statistics.compare_distributions` computes TVD and flags an anomaly only when `TVD > max(threshold, 1/sqrt(shots))`; its default configured threshold is 0.05. At 256 shots the effective cutoff is 0.0625; at 1,024 and 4,096 shots it is 0.05. In the reference-circuit path, `diagnose_circuit` samples the expected and submitted circuits in separate streams (`seed` and `seed + 1`), then compares the two empirical distributions. Therefore even equal underlying distributions can have nonzero sampled TVD. The current rule is a heuristic cutoff, not a calibrated hypothesis test or uncertainty interval. Abstention at final diagnosis is not the same field as `anomaly_detected`.

The bounded command deselects:

- `test_bounded_recipe_comparison_120_templates_and_finite_shot_subset`: computes exact comparisons for 120 templates and finite-shot comparisons over a selected subset and shot grid.
- `test_independent_calibration_fixture_and_permutation_comparison`: evaluates the registered small finite-shot fixture and a permutation comparison.

They remain excluded under the registered finite-shot NO-GO decision, which says the current 252-family proposal is not authorized and cannot support its combined detection/localization/hardware-attribution claim. Existing protocol blockers include no verified independent family frame, no independent hardware evidence/adjudication, unresolved condition/endpoint/estimand details, and no defensible paired power/resource plan. Saved calibration reports are historical simulator evidence; they do not authorize repeating the comparisons or establish physical validity. The prior calibration diagnostic reports 30/60 registered decisions passing and serious current-rule false positives in low-shot negative controls; see `reports/experiments/finite_shot_distribution_calibration_diagnostic.md` and `reports/experiments/finite_shot_blocker_status.json`.

### Evidence limits

- Current lineage reconciliation reports zero eligible fresh independent roots; 5,186 prior-artifact circuit records have unknown or unreconciled lineage. Historical 270-row three-class ML reports lack per-run source hashes/recipe ancestry and independent adjudications.
- QMutBench-derived NORMAL/PROGRAM_FAULT labels follow original/mutant record type, not independently adjudicated behavioral effect. Synthetic scenario labels and simulator-injected conditions are not independent ground truth.
- The observable ML adapter has a tested contract, but no compatible trained three-class artifact has been evaluated on eligible independent data. Historical ML scores are not evidence of independent generalization.
- No physical-device execution, matched independently sourced calibration/telemetry, or externally adjudicated hardware-fault labels are established. Calibration timestamps are caller-reported; generic freshness requires an explicit age policy and does not authenticate provenance.

Sources: `reports/experiments/diagnosis_family_sampling_frame_20261015.json`, `reports/experiments/ml_feature_eligibility_20261015.md`, `reports/experiments/ml_independent_evaluation_protocol_20261015.md`, and `reports/experiments/independent_diagnosis_protocol_package_20261015.json`.

## 2. Separate finite-shot follow-up

### Question and separation

On a newly sampled, independently lineage-verified set of simulator circuit families, can a prespecified comparison procedure reduce false alarms on no-change/equivalent controls while retaining sensitivity to independently defined subtle output-changing faults at bounded shot counts? This is initially a **simulator-only statistical-method question**. It does not test hardware causation. The 45 frozen cases and all descendants/known families from their recipe roots must be excluded from method construction, threshold selection, and confirmatory testing.

For a software-only study, “truth” may be defined from exact ideal simulator distributions, computed separately from finite-shot observations and hidden from the detection procedure. It must be labelled synthetic exact-simulator truth. Independent reviewers should verify circuit/recipe lineage and labels. If the study requires empirical device references, defer it to the separate hardware protocol below.

The independent unit is a verified recipe/family root. Seeds, shots, repeated batches, variants, and mutations derived from one root are nested observations, not additional independent families. Partition entire roots into development (if any), calibration, and sealed test groups before looking at outcomes. Deduplicate exact and near-duplicate roots across prior artifacts and all partitions.

### Candidate hypotheses (none selected)

All candidates are compared with the unchanged current TVD rule on the same new families, using paired family-level results. Implementations and thresholds must be frozen before opening the sealed test set. Do not use any of the 45 known cases to choose a candidate or cutoff.

#### A. Independent reference distribution

- **Hypothesis / failure targeted:** using a trusted exact ideal distribution or a reference sample collected independently from the observed sample may reduce two-sample Monte Carlo noise that creates false positives between identical circuits.
- **Inputs / assumptions:** exact statevector probabilities for covered simulator circuits, or independently generated reference counts with declared exchangeability, shot count, seed/session lineage, and no shared sampling stream with the observed counts. A high-shot reference is still estimated and its uncertainty must be included. This tests deviation from that reference; it cannot distinguish causes with identical output distributions.
- **Groups:** controls include unchanged circuits and semantically equivalent transformations; positives include independently adjudicated subtle, moderate, and large output-changing program faults, stratified by a predeclared effect-size rule.
- **Leakage risk:** using the target submitted circuit's exact distribution, known scenario name, or family identity as an input leaks label/construction information. Exact distributions may define simulator truth and test scoring only; freeze reference generation before test labels are opened.
- **Metrics / uncertainty:** control FPR; sensitivity by effect stratum; localization only when separate source truth exists; family-clustered intervals and paired differences. Report reference-estimation uncertainty separately.
- **Success gate:** meet the common provisional gates below on the sealed roots; additionally, the paired 95% family-cluster interval must show lower control FPR than current TVD without breaching the predeclared sensitivity non-inferiority margin. No claim about equivalent causal attribution.
- **Resources:** exact probabilities are cheap for covered small circuits but scale exponentially in qubit count; empirical reference collection adds a second shot stream. Freeze circuit-width and total shot/execution/storage caps after the eligible frame is known.

#### B. Repeated independent measurement batches

- **Hypothesis / failure targeted:** repeated batches may distinguish a stable output shift from a single noisy TVD excursion and allow an explicit sequential/aggregate decision.
- **Inputs / assumptions:** fixed number of independently seeded batches per circuit, same declared device/simulator configuration, a prespecified aggregate or sequential rule, and no optional stopping outside the rule. Batches reduce uncertainty only for that circuit; they do not create new family units.
- **Groups:** unchanged and equivalent controls receive the same batch schedule as subtle/moderate/large output-changing faults.
- **Leakage risk:** selecting number of repeats, stopping time, or fault strata after seeing the outcomes inflates error; using each batch as independent sample size inflates precision. Freeze multiplicity/alpha spending and aggregate at family level.
- **Metrics / uncertainty:** FPR, stratum sensitivity, abstention, coverage, selective risk, executions per case, and family-clustered intervals; report paired within-family differences to the current rule.
- **Success gate:** common provisional gates below, with family-level FPR/sensitivity intervals and a predeclared maximum number of batches. If the resource cap is reached without a decision, report indeterminate/abstain.
- **Resources:** multiplies shot executions by the fixed batch count; cap families, batches, shots, elapsed time, peak memory, and output size before any run.

#### C. Uncertainty-aware two-sample test / threshold

- **Hypothesis / failure targeted:** a calibrated multinomial or resampling-based test may control false alarms more directly than the current fixed TVD cutoff while preserving power at a known effect size.
- **Inputs / assumptions:** finite count vectors with declared support; exact or sampled reference handling; exchangeability under the null; test statistic, resampling count, significance level, support handling, multiplicity, and power model frozen in advance. A nominal p-value is not evidence of cause.
- **Groups:** unchanged and equivalent controls estimate the null; independently adjudicated subtle/moderate/large output-changing faults estimate sensitivity at matching shots and support sizes.
- **Leakage risk:** selecting alpha, statistic, support, or permutation count on the sealed test or frozen 45 cases. Use calibration roots only for selection and a disjoint sealed family test once.
- **Metrics / uncertainty:** empirical type-I/FPR, sensitivity by stratum, calibration/coverage of any confidence sets, computation cost, and family-clustered intervals. Apply the registered family-wise error policy across all primary strata.
- **Success gate:** common gates below; additionally, the test must pass null controls at its declared error level and retain required sensitivity on the subtle stratum in sealed families. A failed null or power gate is a failed method, not permission to retune.
- **Resources:** computation grows with count support and resampling repetitions; define a bounded resampling and runtime cap during protocol design. Do not execute the existing excluded comparison as a substitute.

#### D. Explicit uncertainty-based abstention

- **Hypothesis / failure targeted:** an indifference zone can abstain when the uncertainty interval spans both “no meaningful change” and “meaningful change,” reducing confident false positives without hiding uncertainty as a negative result.
- **Inputs / assumptions:** predeclared no-change and minimum meaningful-effect margins, valid interval coverage for the sampling design, and a rule separating anomaly detection from source/hardware attribution. A syntactic source diff alone cannot turn an abstention into a confirmed fault.
- **Groups:** unchanged/equivalent controls and the same independently adjudicated fault-positive strata; include difficult near-margin examples sampled independently of the frozen cases.
- **Leakage risk:** choosing indifference margins or reporting only covered cases after observing test results. Set margins from the operational use and obtain independent approval before sampling; score all abstentions in the denominator.
- **Metrics / uncertainty:** FPR and sensitivity, abstention rate, coverage, selective risk among non-abstained cases, risk-versus-coverage curve, and family-clustered uncertainty. Report both unconditional outcomes and selective outcomes.
- **Success gate:** common provisional gates below, with selective-risk upper interval no greater than the predeclared risk limit and coverage at least the predeclared minimum. Missing these gates means the method is inconclusive/failed for the proposed use.
- **Resources:** adds interval calculation and labels but need not add circuit executions if derived from existing counts. The data still require independent families and fixed shot schedules.

### Common provisional decision contract

These are **planning targets only**, adapted from prior registered design targets rather than estimated from the frozen 45 cases. They require intended-use/cost review and written approval before preregistration; if those owners reject them, replace them before data collection, never after test access.

1. On no-change and equivalent controls, the one-sided 95% upper family-cluster confidence bound for FPR is at most 5%.
2. On the independently defined subtle-fault stratum, the one-sided 95% lower family-cluster bound for sensitivity is at least 90%.
3. Abstention is at most 20% overall and coverage is at least 80%; the one-sided 95% upper bound on selective error is at most 5%.
4. For a candidate claimed to improve on current TVD, use a paired family-level comparison: the 95% interval must support lower FPR and sensitivity must be non-inferior by a margin no worse than 5 percentage points. Control the family-wise error across primary comparisons/strata under a prespecified hierarchy.
5. Count each independent family once in inferential denominators; use cluster bootstrap or another prespecified family-valid interval method. Do not use shots, seeds, mutations, or batches as independent n. Determine sample size only after the eligible frame and defensible family dependence assumptions are known; no n is asserted here.

An experiment succeeds only if all applicable gates pass on the one sealed test evaluation. Otherwise report the failed gate and stop. Do not select a winner from the four candidates using sealed-test outcomes. The selection/calibration families and final test families must be disjoint and all prior-artifact ancestry reconciled.

## 3. ML validation protocol

### Target and eligible evidence

Begin with an output-only **detection** target only if independent behavioral labels and family lineage can be established. Cause classification, hardware attribution, and source/qubit/gate localization are separate targets requiring their own independently adjudicated truth; five output-summary features cannot resolve observational equivalence. Mark each result synthetic-only if simulator-generated. No eligible dataset is currently established.

Required per-record provenance includes: stable record/run ID; canonical source and submitted circuit hashes; complete generator/source revision and recipe/family/ancestor lineage; protocol/preregistration hash; configuration and observation hashes; deterministic seed derivation, shots, noise/settings and simulator/runtime/library versions; immutable label evidence references; independent reviewer tokens, blinding and disagreement resolution; and partition assignment. Labels must be decided independently of predictions and hidden from feature construction. Provenance validators can check structure but cannot authenticate an external reviewer or prove the family population is representative.

### Partition and model comparison

- Split at the complete recipe-lineage root. All mutants, variants, random seeds, repeated batches, and topology descendants of one root stay in one partition. Reject missing ancestry and duplicates across development, calibration, and test.
- Keep train/development families for model construction, disjoint calibration families for thresholds/probability calibration, and sealed test families opened once after code, features, thresholds, and analysis are frozen. Keep the frozen 45-case set outside all three partitions.
- Primary features, if eligible: the five declared observables (`total_variation_distance`, `js_divergence`, `hellinger_distance`, `observed_entropy_bits`, `observed_max_probability`). Exclude labels/scenario names, fault IDs, target indices/components, diagnosis outputs/evidence, family/lineage/run IDs, configuration objects and post-outcome fields. Calibration features are a separate ablation only if independently measured before inference. `gate_mismatch_fraction` and source/program comparisons are leakage diagnostics, not the output-only primary model.
- Compare majority-class and stratified dummy baselines with a fixed prespecified candidate. Scaling, feature selection, threshold setting, and hyperparameter selection occur inside grouped development/calibration partitions only. No training begins until the data eligibility gate passes.
- Report class support and confusion matrix, FPR/specificity, sensitivity/recall by class and effect stratum, precision, balanced accuracy and macro-F1, abstention, coverage, selective risk, unsupported/execution-failure rates, and calibrated-probability metrics only if calibration is explicitly evaluated. Provide family-clustered intervals and paired family-level comparisons; avoid row-level binomial intervals when rows share roots.

### Conditions that block fitting

Block model fitting, threshold tuning, artifact export, and performance claims if any of these hold: incomplete or conflicting lineage; zero independently verified eligible roots; labels derived only from mutation/scenario names or simulator condition IDs; no blinded independent behavioral adjudication; family overlap/duplicates across splits; feature schema includes label/post-outcome proxies; missing immutable source/observation hashes or execution config; unapproved endpoint/margins/power/resource caps; or failed provenance/data-flow audit. Current evidence already triggers the block: the family frame lists zero eligible roots, historical 270-row records have missing ancestry/source hashes, and the nine-row pilot labels are synthetic and unresolved.

## 4. Separate future physical-hardware protocol

Do not merge this with simulator finite-shot validation or ML training. Before any hardware run, require separate authorization, a named device/backend and use scope, preflight/credential approval, immutable job IDs and raw counts, backend/software versions, circuit/source hashes, acquisition timestamps with independently verified calibration provenance, and matched control sessions. Group inference by device/session/time window to account for drift; distinguish measured calibration covariates available before inference from post-outcome fields. Obtain blinded independent adjudication of fault/no-fault/ambiguous status from evidence independent of model output. No intentional hardware fault injection without explicit authorization. Set execution/shot/cost caps, outage and drift stop rules, and device-held-out analysis before job submission. Simulator-only results cannot satisfy this protocol.

## 5. Prioritized dependencies and stop rules

### Necessary next actions

1. **Data gate (blocking):** produce a complete independently sourced circuit/recipe-root sampling frame, canonical/near-duplicate reconciliation against prior artifacts, and verifiable source/lineage hashes. Existing reconciliation reports zero eligible roots and 5,186 records with unknown/unreconciled lineage; do not manufacture replacements in code.
2. Obtain independent behavioral label evidence and a blinded reviewer/resolver process for no-change, equivalent, subtle-fault, ambiguous, and unresolved cases. Keep labels physically/logically separate from predictors.
3. Specify one intended use, meaningful-effect margin, costs, primary candidate comparison, and whether the first study is simulator-only. External methods review must establish a sufficiently distinct research question before a novelty claim.
4. After (1)-(3), freeze candidate algorithms, partition roots, target/acceptance criteria, family-cluster uncertainty/power assumptions, multiplicity, resource caps, analysis code hash, and failure rules; get protocol and execution authorization separately.
5. Only after those gates, run one bounded calibration study, freeze the selected method without viewing the test set, then evaluate once on sealed families. Report a null/failed/inconclusive outcome as such. Physical-device validation requires its own separately approved protocol.

### Stop conditions

- Stop before any evaluative or confirmatory simulation/data generation if the independently verified family frame or behavioral labels are absent or conflicting. The bounded ingestion smoke in §6 is pipeline validation only and does not satisfy this gate.
- Stop on any lineage overlap, schema/data-flow failure, invalid ground truth, or resource cap breach; preserve the partial ledger and do not resume under changed hashes.
- Stop model fitting if the independent evidence gate fails; do not substitute synthetic labels, old report metrics, or a source-code counterfactual.
- Stop after one sealed evaluation. A failed criterion requires a new research question, new independent data and a newly reviewed preregistration, not another round of tuning against the same test.
- The frozen 45-case test remains read-only and never becomes a development set. The two finite-shot NO-GO tests remain excluded until a separately approved protocol resolves the current blockers.
- No automatic backlog selection, implementation, experiment launch, model training, or hardware work follows from this handoff.

**Single most important next action:** obtain and independently verify a complete new family/recipe-lineage sampling frame with trusted source hashes. Until it exists, the independent train/calibration/test partitions cannot be formed and both the statistical follow-up and ML validation remain blocked.

## Evidence index

- Frozen baseline and exact run record: `final_research_report.md` §15; `reports/experiments/heldout_diagnosis_root_cause_20261014.json`.
- Threshold implementation and reference sampling: `src/quantum_simulator/statistics.py::compare_distributions`; `src/quantum_simulator/diagnosis_engine.py::diagnose_circuit`.
- Finite-shot NO-GO and historical calibration limits: `reports/experiments/finite_shot_blocker_status.json`; `reports/experiments/finite_shot_blocker_resolution.md`; `reports/experiments/finite_shot_distribution_calibration_diagnostic.md`.
- ML eligibility and independent protocol blockers: `reports/experiments/ml_feature_eligibility_20261015.md`; `reports/experiments/ml_independent_evaluation_protocol_20261015.md`; `reports/experiments/independent_diagnosis_protocol_package_20261015.json`; `reports/experiments/diagnosis_family_sampling_frame_20261015.json`.

## 6. Bounded QMutBench data-pipeline milestone

A later authorized integration task acquired a pinned official metadata CSV and a small QASM subset, then implemented a separate streaming/resumable ingestion path. Details, file hashes, schema, exact command, smoke counts, focused and bounded test results, and limitations are in `reports/qmutbench_data_pipeline_20261015.md`.

The official source metadata contains 723,079 mutant records and 375 distinct `Origin_program` IDs; this does **not** establish 375 independent recipe roots. The downloaded source copy has SHA-256 `3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778` and is preserved separately from the existing workspace copy. The prior size/hash discrepancy is resolved as LF versus CRLF line endings: all parsed rows and columns match. The source release acknowledges MQT Bench as an initial circuit source, while the complete upstream recipe-root map remains unavailable.

The bounded smoke processed 12 QMutBench mutants from one documented GHZ original: 12 parsed/supported and simulated; four changed and eight were identical in the exact computational-basis distribution; no failures. This is a simulator-only pipeline check, not diagnosis truth or an ML evaluation. QMutBench's raw `Killed` field is retained but not interpreted as a project fault label. The preflight remains blocked with zero eligible independent roots, no independent behavioral labels, no preregistered minimum family requirement, and no locked splits.

The bounded suite was rerun with the two finite-shot NO-GO tests deselected: 803 passed, 1 failed, 2 deselected. The only failure is the unchanged frozen 45-case acceptance test with the same seven mismatches documented in §1. No frozen fixture, threshold, expected outcome, or acceptance-test source changed during the integration task.

**Current stop:** do not run full QMutBench ingestion or fit a model. First obtain a versioned, externally reviewable origin/recipe-lineage attestation for the full corpus, resolve shared MQT Bench ancestry, and reconcile it against the prior artifact inventory. Independent labels, a protocol-approved minimum root count, leakage-safe primary features, and locked train/calibration/test splits are still required.

### QMutBench lineage and mutation-detection eligibility audit (2026-10-15)

The bounded deterministic audit is recorded in `reports/qmutbench_lineage_audit_20261015.json` and implemented by `qmutbench_lineage_audit.py`. It recomputes hashes for the 24 QASM artifacts listed in the pinned source manifest against locally staged bytes and hashes the separately downloaded official metadata CSV. These checks verify byte identity relative to the local manifest. They do not authenticate the unsigned upstream commit, source custody, or historical completeness. In the bounded smoke records, `Origin_program` establishes a parent/group association; it does not establish an independently generated recipe root. QMutBench's statement that MQT Bench was an initial source remains documented upstream context, with no per-origin MQT recipe mapping locally verified.

The audit distinguishes original circuit identity (the QMutBench origin ID and original QASM hash), source release (repository, pinned commit, metadata/QASM hashes), recipe root (unresolved), mutant ancestry (parent association documented; generator/run ancestry not attested), observation configuration (local ideal/no-noise simulator observations at the recorded shots/seed), and behavioral label provenance (none independently adjudicated). Raw `Killed` values, IDs, and simulator outcomes are not labels. Exact artifact mapping is bounded to the 24 locally staged QASM files and one GHZ original group; other corpus origins lack local source files in this bundle. No conflicting hashes were found among checked staged artifacts; absent full-corpus source/lineage and incomplete prior-inventory reconciliation remain explicit gaps.

**Mutation-detection decision:** zero independently eligible recipe roots; no train/validation/test evaluation manifest is justified. The required next evidence is a custodian-supplied, versioned complete origin-to-recipe-root/ancestor manifest including MQT Bench links and artifact checksums, independent reconciliation against a complete prior-artifact inventory, plus independent behavioral labels for the specified observable endpoint (blinded reviewers and resolver) and compatible immutable observation/configuration provenance. This output-only program-mutation detection question remains separate from the project's three-way normal/program-fault/hardware-fault diagnosis task. The frozen 45-case result is not changed.

The focused audit tests pass (3 passed). On the requested bounded suite with the two named exclusions, the current run reports **805 passed, 2 failed, 2 deselected**. One failure is the frozen 45-case test and reproduces its same seven mismatches. The other is `test_incomplete_file_is_invalid_and_preserved_for_recovery`: its recovery step raises `FileNotFoundError` because the expected nested `reports/experiments/.../calibration/...` parent does not exist under the new pytest basetemp. No fixture or threshold was changed.

**Recovery-test follow-up:** diagnosis showed the long repository-style output path plus pytest's fresh nested temp path exceeded the effective Windows path limit. The test now supplies the same calibration specification with a short temporary output root, preserving its incomplete-file invalidation, separate recovery artifact, and original-byte preservation assertions. The create-only writer uses a short staging filename. Recovery and lineage focused tests pass (11 passed); the bounded suite reports **807 passed, 1 failed, 2 deselected**. Its sole failure is the frozen acceptance test with the same seven mismatches. The earlier extra `FileNotFoundError` is resolved; no frozen fixture or criterion changed.
