# Current claim reconciliation ? 2026-10-14

## Scope and rule

This is a current-tree addendum to older report snapshots. It does not replace, revise, or erase their original evidence. Use the canonical completion report and stabilization status for release readiness. Treat older reports as historical evidence at their recorded date and source state.

Reviewed: `README.md`, `final_research_report.md`, `final_report_review.md`, `project_finalization_audit.md`, and `reports/experiments/project_wide_completion_report.md`.

## Claims that match current evidence

| Topic | Current disposition | Evidence |
|---|---|---|
| Simulator behavior | Describe supported local statevector, measurement/reset, configured noise, and circuit behavior within tested scope. Do not generalize to a complete physical noise model. | Targeted mathematical, measurement/reset, noise-channel and gate-oracle tests; project completion report. |
| Diagnosis | Describe evidence candidates and abstention as implemented; do not claim broad detection/classification reliability. The unchanged family-held-out contract still fails in 7/45 cases. | `heldout_diagnosis_root_cause_20261014.json`; `clean_environment_validation_20261014.json`. |
| Finite-shot rule | Keep the 0.05 threshold and `max(0.05, 1/sqrt(shots))` guard as frozen evidence. The current registered calibration has 30/60 incorrect decisions and the proposed 252-family run is NO-GO/unauthorized. | `finite_shot_blocker_status.json`; current completion report. |
| ML | Saved results are historical/development evidence. The dataset fingerprint matches, but seven of nine saved evaluation-source hashes differ from current source; historical rows lack generation hashes; probabilities are uncalibrated and current-source generalization/deployment is not established. | `ml_source_compatibility_audit_20261014_release_final.json`; `ml_current_pipeline_audit_20261014_release.json`; `ml_provenance_preflight_20261014.json`; held-out failure report. |
| Benchmarks | Report adapter parsing/execution coverage only. Do not state diagnosis accuracy without independent, compatible ground truth and provenance. | `benchmark_provenance_static_audit_20261014.json`; current completion report. |
| Hardware | Distinguish simulator fixtures/calibration models from physical hardware. No live job was run or authorized. | IBM offline fixture artifacts and current status. |
| Readiness | Project is NOT READY for broad diagnosis, benchmark accuracy, ML deployment, publication, or real-hardware claims. | `project_wide_completion_report.md`; `autonomous_stabilization_status.json`. |

## Historical documents that must not be treated as current release statements

- `project_finalization_audit.md` is dated 2026-10-12 and begins with **FINALIZE WITH EXISTING EVIDENCE**. Its detailed implementation snapshot says `diagnose_circuit` does not accept a distinct reference circuit. The current README/API and tests do support `reference_circuit`; that statement is stale. The old decision also predates the current 7/45 held-out failure and source-hash reconciliation. Preserve it as a historical audit only.
- `final_research_report.md` says its evidence was reviewed through 2026-10-12. Its saved ML metric descriptions may be cited only with their historical dataset/protocol/source context. They are not current-source model validation and must not be presented as deployment or general diagnosis performance.
- `final_report_review.md` and related review/checklist files are earlier editorial snapshots. They do not supersede current status, later blockers, or the frozen held-out failure.
- Earlier README test totals are retained as dated historical checkpoints. The current README now distinguishes the prior primary-environment run from the latest isolated-environment result.

## Remaining editorial gate

As of the 2026-10-15 update, the current README and completion report state the latest bounded suite result, frozen held-out failure, finite-shot no-go, ML compatibility boundary, benchmark truth limits, and hardware status. This reviewed subset is reconciled. The broader archive contains historical reports that are intentionally preserved and not all were line-by-line reissued; therefore the project-wide claims-freeze gate remains **partial**. Before external publication, either keep those artifacts explicitly dated/historical in the publication bundle or perform a publication-specific archive review. No new scientific claim or accuracy result is authorized by this addendum.

## Continuation checkpoint (historical, 2026-10-14)

That checkpoint's bounded suite collected 651 items: 648 passed, one unchanged frozen held-out diagnosis acceptance failed (7/45 outcomes), and two registered finite-shot comparisons were deselected under NO-GO. A long pytest base path first caused four Windows path failures; the short workspace-local rerun passed those tests. No acceptance threshold, expected outcome, or simulator behavior was altered for the held-out cases.

Qiskit is an optional gate-reference oracle declared by the `oracle` project extra (`qiskit>=2.5,<3`), not a core/runtime dependency. The primary oracle suite passed 33 tests; the existing clean environment's combined oracle/core gate subset passed 37 with 4 optional skips. No environment was changed. The README now documents the separate `ibm` extra (`qiskit>=2.5,<3`, `qiskit-ibm-runtime>=0.50,<0.51`) and offline fixture test command; these local tests do not establish live service compatibility.

The current 270-row ML audit is a bounded repeatability check on historical synthetic reports: repeated grouped evaluation exactly matched saved OOF output, but reports and manifest lack row-level generation hashes and labels are not independent. New experiment rows now carry scoped source hashes. A separate path-normalization defect in `load_dataset()` was fixed and regression-tested. The release-final preflight now rejects missing, stale, or internally inconsistent generator and per-report source provenance before standard training; historical rows fail closed. These changes do not clear ML generalization, benchmark truth, or deployment gates. See `ml_current_pipeline_audit_20261014_release.json`, `ml_source_compatibility_audit_20261014_release_final.json`, `ml_provenance_preflight_20261014.json`, and `diagnosis_method_feasibility_design_20261014.md`.


## Final provenance-gate addendum

The final source audit finds six of nine saved evaluation source hashes changed. The historical 270-row data fingerprint still matches, but its manifest lacks generator provenance and every report lacks a per-run source map. `ml_training.py` now fails closed before fitting unless generator, dataset-manifest, and report hashes agree with the current source; direct preflight on historical data returns three provenance blockers and performs no fitting or writes. Previously recorded metrics remain historical, synthetic repeatability evidence. The latest bounded suite is 648 passed, one unchanged 7/45 held-out failure, two NO-GO exclusions (651 collected). The independent-review design now specifies a versioned family sampling frame and adjudication requirements, but the actual external frame, labels, reviewer, and authorization remain unavailable.


## ML data foundation continuation (2026-10-14)

The historical report-level `ground_truth_label` values are synthetic scenario-construction metadata. Their preservation and use as simulator fixtures do not establish independently adjudicated diagnosis truth. The production five distribution features exclude direct configuration shortcuts; calibration-profile candidates and the program-mismatch ablation have explicit synthetic-construction leakage. Family-only folds were insufficient because the generator has recurring recipe roots; future data and fold construction now carry/group those roots. The historical 270 rows omit these roots and source hashes, so the training gate rejects them. See `ml_data_foundation_audit_20261014.md` and `ml_data_foundation_preflight_20261014.json`. No current ML generalization, calibration, benchmark accuracy, or deployment claim is supported.


The README and current-claim summary now refer to the post-lineage audit: the release-final source snapshot had six changed hashes, and the later current-tree snapshot has seven after lineage edits. The earlier grouped-OOF repeatability result is explicitly historical; current strict preflight rejects all 270 old rows.


The final focused ML data-lineage regression (including a test that lineage IDs are forbidden predictors) passed 29 tests in 5.17s. This remains bounded engineering validation only; the historical data gate remains failed.


The future dataset provenance scope was expanded to hash transitive simulator statistics/localization code, package entry/circuit files and root import shims. Its focused regression passed 29 tests in 5.88s. Historical report source compatibility remains unprovable.


Clarification: strict preflight reads and validates the historical report rows, then blocks before estimator fitting, prediction, export or training-output writes. No claim is made that the preflight avoids row reads.


## Bounded ML pilot and final regression (2026-10-14)

The current compatibility snapshot records seven changed hashes among the nine saved evaluation sources; the earlier six were implementation-version changes, and the later `experiment_config.py` difference followed lineage propagation. All seven historical row-generation attributions remain unresolvable because source hashes were not persisted per row or in the dataset manifest. The historical data fingerprint still matches; the refreshed strict preflight rejects all 270 rows and performs no fitting or writes. Historical scores remain historical, synthetic, and uncertified for current-source reproduction.

The nine-row pilot is simulator-derived pipeline evidence only. Its provenance and leakage checks pass, while its nine synthetic labels have no independent adjudication; independent truth remains unavailable. The pilot cannot support ML training, calibration, generalization, benchmark accuracy, deployment, or real-hardware claims. The protocol review draft lists the external frame, reviewers, class/cluster/ICC inputs, use-specific acceptance criteria, and resource cap needed before any evaluation or power calculation.

The latest bounded project virtual-environment suite is 660 passed, one unchanged frozen held-out failure (7/45), zero skipped, and two deselected campaign-like finite-shot tests (663 collected). Project state remains **NOT READY** for scientific validity, ML generalization/deployment, benchmark accuracy, publication readiness or hardware validation.


Final exact-current-tree rerun on 2026-10-15: 660 passed, one unchanged frozen 7/45 held-out diagnosis acceptance failure, and two NO-GO finite-shot tests deselected (663 collected). Focused ML provenance/pilot/leakage regression passed 55 tests. ML pilot labels remain synthetic and blocked from fitting pending independent adjudication; no generalization claim is supported.


## Post-pilot evidence reconciliation ? 2026-10-15

The latest nine-run pilot manifest was revalidated against the current source/runtime. Structural and provenance checks pass; the data remain simulator-derived with synthetic labels and zero independent adjudications, so they support pipeline integrity only, not model fitting, independent truth, generalization or deployment. The historical 270-row dataset remains ineligible under the previously recorded fail-closed preflight, and the seven historical evaluation-source differences remain unresolved at row-generation attribution level.

The fresh bounded regression result is 660 passed, one unchanged frozen 7/45 held-out acceptance failure, and two deselected finite-shot NO-GO tests (663 collected). No diagnosis threshold or expected outcome changed. Overall claims remain NOT READY for broad diagnosis accuracy, ML generalization/deployment, benchmark accuracy, publication readiness, or real hardware.


## Verified final bounded run ? 2026-10-15

The single bounded repository run for this closure collected 663 tests: **660 passed, 1 failed, 0 skipped and 2 deselected in 168.65s**. The sole failure is the unchanged frozen held-out diagnosis acceptance test (7/45); no thresholds or expected outcomes changed. The local engineering evidence supports tested simulator semantics, recovery, data-integrity guards and optional offline adapters. The evidence does not support broad diagnosis validity, ML generalization, benchmark accuracy, publication readiness or hardware claims.


## Research milestone protocol update (2026-10-15)

The diagnosis design is a conditional proposal only. Its candidate detector and evidence router have not been implemented or evaluated. The added record contract and focused tests support feature/label separation, family/lineage integrity, explicit ambiguous/equivalent labels, and fail-closed missing prerequisites; they do not establish independent truth, statistical power, method accuracy, or readiness. The broad repository suite was not rerun after these protocol-only additions; the latest full-suite result remains the pre-addition 660/1/0/2 result, including the frozen 7/45 acceptance failure. The current task ledger is 40 total: 26 verified, 6 partial, 6 blocked, and 2 deferred.


Final focused protocol rerun after separating single-cause attribution from ambiguous/equivalent disposition: **47 passed in 35.93s** across protocol, pilot lineage and model-artifact checks. The full repository suite was not rerun after adding the contract module; its pre-addition result is recorded separately and remains one frozen 7/45 acceptance failure.


## Current-tree ML evidence update (2026-10-15)

The ML evidence status remains blocked for generalization/deployment. Current strict preflight rejects all 270 historical rows (0 eligible) for absent per-row source hashes, absent generation provenance and source incompatibility. Two unguarded entry points were corrected to stop before fitting/export on preflight failure. Feature review keeps the five output observables as the primary schema, configuration-derived variables as calibration-only, and `gate_mismatch_fraction` as a diagnostic leakage ablation only. Focused safeguard tests passed 40/40; the bounded suite passed 709, failed the unchanged frozen diagnosis contract once, skipped 0 and deselected 2 NO-GO tests (712 collected). No historical metric was reproduced and no ML fitting occurred. See `ml_feature_eligibility_20261015.md` and the current canonical status.


## 2026-10-15 ML record recovery and evaluation controls

The historical 270 report rows have been reconciled individually in `ml_historical_record_recovery_20261015.json`: all preserve some run configuration/runtime/seed metadata, but 270/270 lack source hashes and recipe ancestry, the manifest lacks generation provenance, and there are zero independent adjudications and zero hardware observations. All rows remain ineligible for fitting, development/model selection, calibration and held-out evaluation. The current source comparison remains 7/9 different from the saved evaluator manifest.

QMutBench's local ML data answer a mutant/original behavioral question and its manifest source-CSV hash does not match the local source file. Bugs4Q has no compatible local ML data for the three-way target; the lineage scan remains incomplete. No eligible independent dataset was found. The current evaluation infrastructure adds registered schemas, fixed dummy/model comparators, family-disjoint three-way checks, lineage-cluster uncertainty reporting, synthetic-only evidence labels and non-overwriting v3 output. It was tested on software fixtures only; no model was fit.

Focused suite: 41 passed in 10.52s. Bounded repository suite: 719 passed, the unchanged frozen diagnosis acceptance failed at 7/45, 0 skipped and 2 deselected (722 collected; 216.10s). Backlog is 42 tasks: 28 verified, 6 partial, 6 blocked and 2 deferred. These results support engineering controls, not ML accuracy, generalization, benchmark accuracy, or hardware claims.


## Final 2026-10-15 verification

The v2 record-level table preserves per-field recorded/inferred/recomputed/missing status and was regenerated deterministically without overwriting the earlier same-date artifact. Focused validation passed 41 tests in 15.39s. Final bounded suite: 719 passed, the unchanged frozen diagnosis gate failed at 7/45, 0 skipped, 2 NO-GO finite-shot tests deselected (722 collected; 219.00s). Backlog totals 42 (28 verified, 6 partial, 6 blocked, 2 deferred). The evidence supports engineering safeguards and ineligibility disposition only; no current ML accuracy, generalization, deployment, benchmark or hardware claim is supported.


## Independent-label ingestion status (2026-10-15)

The ML loader/preflight currently supports controlled synthetic scenario labels only. Although the diagnosis protocol validator separates truth from observations, ML ingestion does not yet join independent adjudication records. This integration remains blocked until independent reviewers approve a blinded record schema and provide provenance-complete examples with family/recipe roots. Backlog task P1-4 records the dependency. Final task counts are 43 total: 28 verified, 6 partial, 7 blocked, 2 deferred.

## ML evaluation and diagnosis abstention update - 2026-10-15

The saved seven-case analysis remains authoritative; no forensic study was repeated. It identifies three sub-threshold QFT-like expected positives, two healthy ring finite-shot false positives, and two observationally equivalent ring false positives where syntax alone does not establish a behavioral cause. A separate existing equivalence regression demonstrated a decision-path defect: direct `diagnose_execution` and the ML adapter could classify a source-operation diff as a behavioral program fault when observed distributions matched. The paths now retain localization while abstaining on behavioral cause. This fix is locally verified and is not accuracy evidence; the frozen 7/45 contract still fails unchanged.

ML outer folds are explicitly validated for row coverage and family/lineage isolation before estimators are constructed or fit. Focused validation passed 92 tests; the final bounded suite passed 722, failed only the frozen diagnosis acceptance, skipped 0 and deselected 2 NO-GO finite-shot tests (725 collected; 224.68s). No model was fit. The 270 historical rows and nine-row synthetic pilot remain ineligible. The injected classifier interface has no validated probability calibration basis; no calibration result or performance claim is supported. Current backlog: **44 tasks: 29 verified, 6 partial, 7 blocked, 2 deferred**.

The independently reviewed population/frame, blinded adjudication labels, protocol acceptance decisions and family-level power/resource assumptions remain external prerequisites. Diagnosis generalization, ML calibration/generalization, benchmark accuracy and real-hardware validity remain unestablished. Overall status remains **NOT READY** for these claims.
Final-tree rerun after the probability-semantics manifest guard: focused diagnosis/ML suite **92 passed in 16.89s**; bounded repository suite **722 passed, 1 failed, 0 skipped, 2 deselected (725 collected; 221.49s)**. The one failure is still the unchanged frozen diagnosis contract at 7/45, and the two NO-GO finite-shot tests remain deselected. No calibration, model fitting, or performance result is claimed.

## Diagnosis validation protocol evidence update (2026-10-15)

**Verified within bounded engineering scope:** the protocol record validator rejects malformed or missing per-case generation/source/lineage/circuit/configuration/seed/noise/runtime/reproduction provenance and no longer accepts approval strings embedded in the protocol as independent review. Focused protocol/candidate/preflight suite: 54 passed in 0.82s.

**Blocked:** actual independent review requires an external trust-policy verification of a protocol-bound reviewer attestation. The sampling frame, independent adjudicators, use-specific acceptance criteria, family-level power/resource assumptions, and execution authorization remain absent. The validator and protocol package establish neither actual reviewer independence nor scientific validity. Frozen diagnosis acceptance remains 7/45 failed; finite-shot NO-GO evidence is preserved.


## Final validation evidence (2026-10-15)

Focused protocol/candidate/preflight checks passed 54/54. Bounded repository suite: **730 passed, one failed, zero skipped, two deselected (733 collected; 222.36s)**. The sole failure is the unchanged frozen diagnosis test (7/45). The two finite-shot NO-GO tests were not run. A first broad run with a longer temporary path had a Windows path-length failure in one recovery test; the isolated short-path test passed and the final short-path full suite had no recovery failure. Protocol structure/provenance checks are engineering verification only; independent scientific validation remains blocked.


## Provenance and adjudication gate update (2026-10-15)

**Verified engineering scope:** a deterministic, capped, evidence-linked reconciliation utility and fail-closed blinded-adjudication/preflight gate are implemented. Focused tests passed 64/64. The report retains 6 candidate roots, 3 fingerprint-conflicted roots, 3 roots inferred from source declarations only, 5,186 records with unknown lineage, 457 exact duplicate artifacts, 30 nominal synthetic ML families/270 rows without ancestry and run-source hashes, and zero eligible families. It does not promote inference into verified lineage.

**Still blocked:** inventory completeness and lineage conflicts require external reconciliation; independent reviewers must supply protocol-bound identity/blinding/independence and judgment attestations; use-specific criteria, family-level power/resource assumptions, and separate execution authorization are absent. Simulator labels remain synthetic. The frozen diagnosis acceptance remains 7/45 failed; finite-shot NO-GO stays in force.


## Final bounded validation - provenance/adjudication phase (2026-10-15)

Focused reconciliation, adjudication, study-preflight, and protocol tests: **64 passed in 3.32s**. Final bounded repository suite: **747 passed, 1 failed, 0 skipped, 2 deselected (750 collected; 226.33s)**. The sole failure is the unchanged frozen diagnosis acceptance at 7/45; both finite-shot NO-GO tests remained deselected. No ML fitting, dataset regeneration, experiment, package installation, or hardware job occurred. The canonical backlog now contains **46 tasks: 31 verified, 6 partial, 7 blocked, 2 deferred**. External family reconciliation, independent adjudication/review, criteria/power/resource decisions, and authorization remain blockers.


## Continuation update - 2026-10-15

A hash-bound provenance conflict review ledger was added and tested. It improves evidence-custodian workflow but does not resolve any saved conflict, establish inventory completeness, or alter research readiness. The current bounded suite records 752 passed, the unchanged frozen 7/45 diagnosis acceptance failure, and two NO-GO finite-shot tests deselected. No scientific performance, ML generalization, calibration, benchmark accuracy, or hardware-validity claim is supported by this tooling change.


## Simulator correctness continuation - 2026-10-15

The finite calibration and distribution-validation changes are software correctness improvements. They do not alter the frozen diagnosis threshold or expectations. The final bounded suite has 767 passes, the unchanged frozen 7/45 diagnosis acceptance failure, and two finite-shot NO-GO deselections. No scientific performance, independent ML generalization, calibration, or physical-hardware claim is newly supported.


## ML inference boundary update (2026-10-15)

Deterministic contract tests verify fail-closed handling for invalid confidence cutoffs, malformed classifier scores, and absent feature schemas. This changes no model, threshold, frozen diagnosis result, or scientific claim. The bounded suite remains 778 passed, the unchanged frozen 7/45 failure, zero skipped, and two finite-shot NO-GO tests deselected. No ML accuracy, generalization, calibration, or hardware claim is supported by this software fix.


## Missing-observation attribution safeguard - 2026-10-15

Source localization without execution distributions no longer implies behavioral anomaly or program fault. It remains a source candidate with unknown anomaly status and an abstaining final category. Focused suite: 85 passed; bounded suite: 778 passed, the unchanged frozen 7/45 failure, zero skipped, and two NO-GO deselections. This software safeguard supports no scientific accuracy or generalization claim.


## Causal evidence gate - 2026-10-15

A rule-based program attribution now requires modeled measured-output impact from the source-only counterfactual; source candidates that are disconnected from measured qubits or behaviorally equivalent are retained without causal attribution. Simultaneous source and relevant hardware evidence stays ambiguous. The bounded run passed 783 tests, with the same frozen 7/45 acceptance failure and two NO-GO deselections. This verifies software behavior only; independent causal ground truth and hardware calibration freshness are not established.


Validation and freshness limitation - 2026-10-15: final bounded suite passed 783 tests with the same frozen 7/45 failure and two NO-GO deselections. A generic timestamp rejection was not retained because software fixtures use the declared synthetic-config marker and no freshness cutoff is defined for generic HardwareModel diagnosis. IBM live-execution preflight uses a separate explicit freshness check. Hardware validity claims remain unsupported.


## Calibration provenance and age semantics - 2026-10-15

The generic hardware model now records optional caller-reported calibration acquisition time and provenance separately from snapshot `created_at`. Generic freshness has no universal cutoff: callers can supply a maximum age, and only then can a reported timezone-aware timestamp become `FRESH` or `STALE`. Missing/unknown provenance, malformed or timezone-naive timestamps, and future timestamps do not support current hardware attribution under a policy. Synthetic configuration remains usable as simulator configuration. This does not authenticate timestamp sources or establish physical-hardware validity. IBM preflight retains its separate explicit age contract.

Focused hardware/diagnosis/IBM validation: 108 passed. Final short-path bounded suite: 793 passed, one unchanged frozen 7/45 failure, zero skipped, and two finite-shot NO-GO tests deselected (796 collected). No expected result or threshold changed.


## Generic calibration provenance/freshness and bounded follow-up - 2026-10-15

The generic `HardwareModel` freshness contract now distinguishes reported acquisition time, synthetic configuration, unknown legacy provenance, and snapshot creation time. An explicit caller maximum age is required to label a reported calibration fresh or stale; stale/unknown/invalid/future calibration cannot support current hardware attribution under that policy. JSON snapshot roundtrip preserves provenance, and synthetic-config remains allowed. Focused hardware/diagnosis/IBM preflight suite: 109 passed. Final bounded suite: 794 passed, one unchanged frozen 7/45 failure, zero skipped, two finite-shot NO-GO deselections (797 collected; 227.40s). A first longer-basetemp run hit a Windows temporary-directory setup error; the short-path run had none.

Two existing safe backlog items were also revalidated without code changes: two-qubit parameterized-gate canonicalization (8 passed) and bounded Bugs4Q discovery (1 passed, 11 deselected). The latter's default temp-root attempt was denied by Windows; its workspace-local rerun passed. No training, hardware execution, installation, or large benchmark occurred.
