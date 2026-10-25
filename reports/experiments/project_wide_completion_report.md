# Project-wide completion report

Updated: 2026-10-15

## Current verdict

NOT READY for broad diagnosis, benchmark, ML deployment, or real-hardware claims. The current routine test suite completes, but one locked held-out diagnosis acceptance fails. The core statevector/gate/measurement trajectory subset has strong targeted evidence; dependent inference and research claims remain blocked by calibration failure, identifiability, label/provenance and source-compatibility gaps.

## Current release gates

| Gate | Status | Evidence summary |
|---|---|---|
| core mathematical simulator correctness | pass | Targeted analytical/Kraus and Qiskit references are recorded; broad current-tree suite includes current gate/circuit regressions. |
| measurement reset noise correctness | pass | Readout collapse and entangled reset trajectory analytical tests are recorded; supported trajectory scope only. |
| circuit parsing execution compatibility | partial | The 12-case QMutBench QASM fixture and Bugs4Q source integration pass; broad QASM compatibility and unsupported constructs remain outside the adapter. |
| independent validation regression | partial | Latest bounded repository suite after ML and diagnosis fixes: 722 passed, 1 frozen held-out failure, 0 skipped, 2 deselected (725 collected; 221.49s). The unchanged failure is the frozen diagnosis acceptance at 7/45. |
| diagnosis attribution abstention localization | blocked | Frozen held-out outcomes fail in 7/45 cases; finite-shot blocker status is NO-GO; output distributions do not identify observationally equivalent causes. |
| ml dataset integrity reproducibility generalization | blocked | Historical 270 synthetic rows remain ineligible: current strict preflight returns ready=false and 0 eligible rows due to absent per-run source maps, absent circuit lineage, missing generation provenance and incompatible generator hashes. Fallback training and observable-model export enforce preflight before fitting/output. A record-level recovery table confirms these gaps for 270/270 rows. The nine-row pilot and Phase 5/QMutBench ML data do not supply independent labels for the three-way diagnosis task. Feature eligibility and split/comparison limits are documented in `ml_feature_eligibility_20261015.md` and `ml_generalization_recovery_checkpoint_20261015.md`. |
| benchmark methodology | blocked | Reviewed benchmark snapshots lack compatible independent truth/provenance; no diagnosis accuracy claim. |
| performance resource recovery | partial | Atomic writes/checkpoints and opt-in modelled array guard are tested; broad resource envelope is not established. |
| integration interfaces environment reliability | partial | Optional Qiskit oracle contract is declared/documented and 33 oracle tests pass in the primary environment; the held-out acceptance failure remains. |
| documentation claims demonstration readiness | partial | Current README/report include the latest suite, oracle strategy, ML evidence limits and blockers; historical reports remain archived for publication-specific review. |
| live hardware evidence | deferred_with_reason | No live quantum job, backend access, or real-device evidence is authorized or available. |

## Implemented and verified in current stabilization records

- Corrected U2 and RCCX conventions against Qiskit; seeded composed-circuit differential comparisons cover circuits from one to four qubits.
- Corrected measurement/readout and reset trajectory semantics; compared representative behavior to analytic/Kraus references.
- Rejected invalid gate parameters, malformed/nonunitary custom matrices, malformed indices and duplicate operands before mutation.
- Validated positive integral shot counts and generator replay consistency.
- Added opt-in modeled state-array memory preflight and atomic writes for experiment, benchmark, checkpoint and ML model artifacts.
- Preserved unsupported benchmark inputs as explicit unsupported/failure dispositions.
- Preserved source-operation localization while abstaining on behavioral program-fault attribution when observed distributions match; ML diagnosis applies the same safeguard.
- Moved grouped-fold family/lineage validation before estimator construction and fitting.

## Validation evidence and current-tree outcome

The latest bounded primary-environment suite collected 725 items and completed **722 passed, 1 failed, 0 skipped, 2 deselected in 221.49s** with a workspace-local pytest base directory. The failure is `tests/test_reference_diagnosis_generalization.py::test_heldout_generalization_outcomes_and_targets`: the same seven of 45 frozen detection, classification or abstention outcomes differ. Source/hardware target localization assertions passed. No thresholds, held-out labels, splits or expected outcomes were changed. Earlier dated suite results and the resolved long-base-path temporary-directory issue are retained below and in the journal.

Two calibration comparison tests were deselected: `test_bounded_recipe_comparison_120_templates_and_finite_shot_subset` and `test_independent_calibration_fixture_and_permutation_comparison`. The registered finite-shot study is NO-GO, and further calibration execution requires a separately justified protocol. A 12-case QMutBench QASM fixture did run as normal adapter validation; no full data corpus or benchmark accuracy analysis ran.


## Scientific evidence boundaries

- Finite-shot preregistered calibration currently fails 30 of 60 registered decisions; the proposed 252-family run is explicitly NO-GO and unauthorized.
- Program faults and modeled hardware changes can be observationally equivalent from output counts; do not claim causal attribution from distributions alone.
- The saved 270-row dataset fingerprint matches, but its report rows and dataset manifest have no per-row source hashes. The latest source compatibility snapshot differs on seven of nine evaluator files; historical row-to-source attribution remains unresolved. These differences do not alter the saved historical evaluation, which is not current-source generalization evidence. Prior grouped evaluation is repeatability on the same synthetic rows only, not independent validation. `gate_mismatch_fraction` remains excluded from the production schema. The nine-row provenance pilot is integrity evidence only and remains fit-ineligible.
- QMutBench/Bugs4Q reports lack current-compatible independent truth for performance claims.
- No real quantum hardware jobs were performed; no model is deployment validated; probabilities are not calibrated. Current IBM layout/freshness mock readiness and optional dependency instructions are recorded in `hardware_adapter_local_readiness_20261014.md`; the focused contract and adapter suite passed 45 tests. Live backend selection/authorization remains outstanding. Qiskit is available only through the optional `oracle` extra (`qiskit>=2.5,<3`); its 33-case suite passed in the primary environment.
- Resource profiles are bounded local synthetic simulator observations, not a portable capacity guarantee.

## Current project-wide state

The remaining research gates are blocked or partial. The current-tree held-out outcome contract fails; the registered finite-shot experiment is NO-GO; benchmark truth/provenance and ML source compatibility/calibration remain unresolved; live hardware was not authorized. The new diagnosis method feasibility memo proposes a family-level protocol but does not clear or authorize the no-go. See the canonical backlog for criteria, dependencies and evidence.

## Reproduction and artifacts

- Run targeted tests from the repository root with `.venv/Scripts/python.exe -m pytest <test modules> -q --basetemp <new temp directory>`.
- Current status: `autonomous_stabilization_status.json`
- Journal: `autonomous_stabilization_journal.md`
- Backlog: `project_wide_completion_backlog.json`
- Evidence index: `comprehensive_project_inventory.json`, `benchmark_provenance_static_audit_20261014.json`, `ml_source_compatibility_audit_20261014_release_final.json`, `ml_current_pipeline_audit_20261014_release.json`, `ml_provenance_preflight_20261014.json`, `ml_source_compatibility_audit_20261014_final.json`, `ml_current_pipeline_audit_20261014.json`, `ml_source_compatibility_audit_20261014.json`, `ml_source_compatibility_audit_20261014_rerun.json`, `ml_source_compatibility_audit_20261014_post_provenance.json`, `ml_current_pipeline_audit_20261014.md`, `diagnosis_method_feasibility_design_20261014.md`, `heldout_diagnosis_root_cause_20261014.json`, `clean_environment_validation_20261014.json`, `project_claims_reconciliation_20261014.md`, `hardware_adapter_local_readiness_20261014.md`, `finite_shot_blocker_status.json`, and resource profile reports.


## Backlog totals

The reconciled backlog contains **44 tasks** at this checkpoint.

| Status | Tasks |
|---|---:|
| blocked | 7 |
| deferred_with_reason | 2 |
| partial | 6 |
| verified | 29 |

| Severity | Tasks |
|---|---:|
| critical | 1 |
| high | 15 |
| medium | 27 |
| low | 1 |

## Final evidence boundaries

The 12-case QMutBench adapter check is benchmark-derived local data and verifies parsing/execution integration only; it provides no supported accuracy claim. Controlled diagnosis, finite-shot profiles and the bounded ML comparison are synthetic/local. Qiskit comparisons validate gate behavior against a software oracle. No actual hardware job was run. Saved benchmark/ML artifacts remain historical where source/data fingerprints differ.


## Held-out failure root cause (current-tree review)

The seven frozen mismatches were reproduced and measured without editing the fixture or acceptance criteria. Three QFT-like fault cases have exact distribution TVD below the fixed 0.05 threshold (0.00126 to 0.01130), so the expected positive decisions are unsupported by the current rule. Four ring correct/equivalent cases have exact TVD 0 but sampled TVD 0.08594 or 0.125 at 256 shots, above the fixed 0.0625 finite-shot guard. The equivalent CZ reordering is distribution-preserving; syntactic source candidates are present, and the false-positive anomaly decision changes the resulting category. No implementation defect was identified in the seven outcomes; the current method has insufficient sensitivity for the first cases and poor specificity for the latter. The frozen acceptance still fails 7/45, and diagnosis remains blocked. A separate forensic regression passed 1 test in 6.62s; it does not replace the acceptance test. Full per-case evidence is in `heldout_diagnosis_root_cause_20261014.json`.

## Historical ML source-hash reconciliation (superseded by final audit below)

The saved 270-row dataset fingerprint still matches. Recomputed live hashes confirm three source changes: `experiments/run_experiment.py`, `src/quantum_simulator/diagnosis_engine.py`, and `src/quantum_simulator/quantum_simulator.py`. The hash changes correspond to substantive runner/provenance, diagnosis, and simulator behavior changes; saved grouped-CV remains historical and cannot validate the current tree. The prior audit's current hash for `diagnosis_engine.py` was stale; it is preserved and superseded by `ml_source_compatibility_audit_20261014_rerun.json`. No model training, fitting, or grouped-CV rerun occurred.

Earlier checkpoint note (historical): at that checkpoint the backlog had one new verified evidence-freshness task (P2-13) and 35 tasks; the subsequent 2026-10-14 checkpoint had 37 tasks. That earlier 2026-10-15 checkpoint had 39 tasks; the later protocol milestone increased the canonical backlog to 40 tasks. Overall release verdict remains **NOT READY**. Historical statuses and reports are preserved; finite-shot and held-out gates remain blocked, as do independent benchmark truth and live-hardware requirements.


## Earlier resource and validation checkpoint (historical)

Bounded local H/CX profiles now cover 15-20 qubits. Widths 19 and 20 completed in 6.205s and 11.669s with norm error 1.11e-16; sampled process RSS peaked at 52,260,864 B and 69,079,040 B. After width 20, host available memory was 419,151,872 B, below the 512 MiB stop floor, so profiling stopped. This remains a bounded session-specific smoke profile, not a capacity guarantee (`resource_envelope_profile_20261014_checkpoint2.json`).

The clean isolated environment and its `pip freeze` are recorded in `clean_environment_validation_20261014.json`. The later IBM dependency contract uses a separately declared optional extra; the existing clean environment was not modified. Clean routine validation reproduced the held-out failure and omitted 33 optional Qiskit reference tests because Qiskit is not a declared dependency. At that earlier checkpoint, no calibration threshold tuning, ML fitting, full QMutBench processing, or hardware jobs had occurred. Later bounded ML reevaluation is summarized in the latest checkpoint below.


## Current claims and historical report boundary

The README now reports the isolated clean-environment outcome and links the current completion report. A read-only reconciliation found earlier finalization/research reports dated 2026-10-12; one contains a stale API statement and a finalize verdict made before current blockers were established. Historical artifacts remain unchanged and are not current release claims. See `project_claims_reconciliation_20261014.md`. The documentation gate remains partial until an external publication bundle is scoped against the preserved archive.


## Latest checkpoint: feasibility, provenance, and final review

The seven held-out mismatch investigation was not repeated. Its frozen acceptance remains failed 7/45, and no threshold or expectation changed. `diagnosis_method_feasibility_design_20261014.md` separates distribution-change detection from cause attribution and source/hardware localization. Output-count methods can test a declared distributional null only with an appropriately calibrated sampling design; observationally equivalent causes cannot be distinguished from those outputs, regardless of shot count. The proposed study remains no-go until independent truth, a family-level sampling frame, development-only method validation, joint power/resource design, and independent protocol review are in place.

The ML audit found 270 historical rows across 30 families, five disjoint family folds, and no missing/nonfinite production features. Two current fixed-pipeline runs agreed exactly and matched historical OOF predictions/probabilities, but the rows have no per-report generation hash and labels are synthetic by construction. This does not establish current-source generalization, independent benchmark performance, calibration, or deployment. Future individual experiment reports and dataset manifests now carry scoped source/runtime hashes; historical rows/manifests were preserved. A relative-path loader defect found during preflight was fixed and regression-tested.

Qiskit is an optional software oracle, not a runtime requirement. `pyproject.toml` exposes the `oracle` extra and README documents its install/test path. The primary oracle passed 33 cases; the existing clean environment's combined gate run passed 37 tests with four optional skips. No environment was modified.

Benchmark review remains limited to case ingestion/execution: QMutBench snapshots conflict and provide zero diagnosis-compatible labels; the 42-case Bugs4Q source audit has zero diagnosis-category or independent localization labels. Simulated/injected conditions remain distinct from independent benchmark truth and real hardware evidence. Full QMutBench, large-scale training, campaign-like finite-shot studies, and hardware jobs were not run.

The final bounded repository suite after ML provenance gating and the IBM dependency-contract regression reported **648 passed, one known frozen held-out failure, and two deselected** (651 collected). The IBM dependency/layout/freshness contract plus adapter fixtures passed 45 focused tests. The newly discovered ML preflight fail-open defect is fixed and regression-tested. Other remaining partial/blocked tasks are tied to the frozen acceptance, missing independent evidence, target resource envelope, or declared publication scope. Overall verdict remains **NOT READY**; remaining gates require new scientific evidence, external data provenance, or hardware authorization/credentials rather than additional local engineering.


## Release-final ML provenance gate

The structural/group-aware audit had reported the historical dataset as structurally ready while lacking source identity. That was insufficient for the standard training entry point. The current preflight now requires a `dataset_manifest.json` generation source map and Python/NumPy versions, each run's source map, exact report-to-manifest agreement, and agreement with current generator source hashes. `ml_training.py` stops before row loading or fitting if these requirements fail. On the preserved 270-row dataset, preflight is **not ready**: the manifest has no `generation_provenance`, all 270 reports lack per-run hashes, and eight declared generator source paths are absent or stale. Historical bytes were not rewritten. The six-of-nine ML evaluation source hash differences are separately recorded in `ml_source_compatibility_audit_20261014_release_final.json`. Prior two-run Logistic Regression metrics remain historical repeatability evidence and were not rerun because the dataset fails provenance gating.

The guard and malformed-provenance handling passed 29 ML data/preflight/artifact tests. In the existing clean environment, 45 IBM adapter and optional-dependency contract tests passed without installing Qiskit or IBM Runtime. The full current-tree suite is rerun below after these changes.


## Latest checkpoint: ML provenance fail-closed and final regression

A release audit found the training preflight was permissive about missing source identity. The current entry point now requires manifest generation source hashes and Python/NumPy runtime, per-run source maps matching the manifest, and generation hashes matching current code. It blocks before loading/fitting when those checks fail. The preserved historical dataset is blocked: its manifest has no generation_provenance, all 270 reports lack per-run source hashes, and eight generator source paths are absent or stale. No historical data were modified and no fitting occurred. The row loader now safely records null/malformed reproducibility as missing provenance instead of crashing. Regression tests cover missing/stale/mismatched provenance and pre-fit blocking.

Fresh full bounded suite after these changes: **648 passed, 1 failed, 2 deselected (651 collected, 151.10s)**. The sole failure is the same frozen 7/45 held-out outcome acceptance. No investigation was repeated, thresholds were not tuned, and expected outcomes were not changed. Current source compatibility is **six of nine** hashes changed; the historical data fingerprint still matches, but is not traceable to its generator. See `ml_source_compatibility_audit_20261014_release_final.json`, `ml_current_pipeline_audit_20261014_release.json`, and `ml_provenance_preflight_20261014.json`.

Reconciled backlog: **37 tasks: 23 verified, 6 partial, 6 blocked, 2 deferred**. P2-18 is verified for its future-only manifest acceptance; P2-20 is verified for fail-closed ML provenance. Diagnosis protocol materials now specify a versioned family sampling frame, lineage clustering, independent truth, alternative explanations, and review requirements, but no independent population/labels/reviewer or authorized study exists. IBM offline adapter checks passed 45 tests in the existing clean environment without installing optional packages; live access remains unverified. Overall verdict remains **NOT READY** for broad diagnosis, ML deployment, benchmark accuracy, publication, or hardware claims.


## ML data foundation and lineage checkpoint (2026-10-14)

The six saved ML source-hash mismatches were reconciled in `ml_data_foundation_audit_20261014.md`. At file-change level they correspond to current implementation changes, but none proves which version generated the historical rows: the saved 270-row dataset lacks per-run source maps and its manifest lacks generation provenance. The dataset byte fingerprint still matches. The initial 6/9 mismatch snapshot is preserved; after this local lineage work, a new snapshot records 7/9 changed paths. Historical saved scores were not rerun or certified.

A leakage review found the `gate_mismatch_fraction` ablation deterministically identifies all synthetic PROGRAM_FAULT rows (90/90 nonzero vs 0/180 otherwise), while calibration-named candidates are read from configured synthetic hardware profiles. Both remain excluded from production features. Exact duplicate report fingerprints are zero, but an exploratory circuit-sequence screen found 51/435 pairs at similarity >=0.8; the generator's six recurring recipe roots are now explicit `circuit_lineage_id` values and the ML fold builder groups and validates on those roots. Historical rows lack lineage and fail closed.

Prospective evaluation now has a documented lineage-separated training/development, calibration, and untouched test protocol, with class-wise, proper-probability and root-cluster uncertainty metrics. No performance floors/power values are fabricated: independent review must freeze them before evaluation. Current data have only six synthetic roots and no independent adjudication, so the protocol cannot yet be run. Current strict preflight reports zero eligible historical rows; no fitting or regeneration occurred.

Focused regression: `python -m pytest tests/test_ml_preflight.py tests/test_full_dataset_provenance.py tests/test_experiment_runner.py tests/test_ml_evaluation_audit.py -q --tb=short --basetemp .pytest-tmp-ml-data-foundation-final`; 28 passed in 5.30s. Backlog now has 38 tasks: 24 verified, 6 partial, 6 blocked, 2 deferred. Overall ML readiness remains blocked; project release verdict remains NOT READY.


Current-tree lineage propagation was rerun after adding the explicit report-metadata regression: 28 passed in 5.35s (`.pytest-tmp-ml-data-foundation-final2`). This is the latest focused result. The previous 5.30s result is retained as an earlier run.


Final regression after adding a guard against using circuit_lineage_id as a predictor: `python -m pytest tests/test_ml_preflight.py tests/test_full_dataset_provenance.py tests/test_experiment_runner.py tests/test_ml_evaluation_audit.py -q --tb=short --basetemp .pytest-tmp-ml-data-foundation-final3`; 29 passed in 5.17s.


Expanded source-scope check: row fingerprints now include the transitive package statistics/localization modules, package entry/circuit, and root import shims used by experiment execution. The regression suite including dependency-map assertions passed 29 tests in 5.88s (`.pytest-tmp-ml-data-foundation-provenance-final`). The refreshed current preflight still blocks all 270 historical rows; the newer complete source scope further confirms their manifest cannot match this current generator.


Preflight wording correction: `run_preflight()` reads and inspects source reports to validate provenance and row integrity. The `main()` gate prevents the subsequent training-stage load and estimator fitting when that check fails; no prediction, export, or training-output write follows. Earlier shorthand describing the guard as operating before all row reads should be read with this correction.


## ML data pilot and final validation checkpoint (2026-10-15)

The six previously reconciled saved source-hash differences remain attributable to later file-level code changes, but historical row-generation attribution is unresolved for all six; the post-lineage current-tree comparison has seven of nine changed hashes, with two still matching. A fresh `ml_source_compatibility_audit_20261014_pilot_final.json` confirms the historical dataset fingerprint matches saved bytes, while `ml_data_foundation_preflight_20261014_pilot_final.json` rejects all 270 rows (0 eligible) for missing generation provenance, per-run source hashes, and current generator compatibility. No old metric was regenerated or treated as current.

Added a separate, deterministic smoke pilot at `reports/experiments/ml_data_foundation_pilot_20261014_final/`: nine local-simulator cases, three families/recipe lineages, 32 shots per run. It captures scoped generation, execution, feature-extraction and validator hashes, Git commit/worktree identity, complete configurations/digest, Python/NumPy/platform runtime, seeds, lineages, artifact digests and deterministic run fingerprints. Predictors and synthetic labels are separate files; raw diagnosis reports are marked internal-only. The five predictors exclude `gate_mismatch_fraction`, configured calibration proxies, labels and diagnosis fields. All nine construction labels remain synthetic, causal attribution is unresolved, and independent adjudication count is zero. The pilot validates structurally and fails closed before model fitting.

Review materials in `ml_independent_evaluation_protocol_20261014.md` specify target populations and a finite family-level sampling frame, blinded adjudication, lineage-separated development/calibration/untouched-test splits, classwise and probabilistic/calibration/abstention metrics, root-cluster confidence intervals, and family-level power inputs. The frame, prevalence, cluster sizes/ICC, independent reviewers, use-specific numeric acceptance floors and resource cap are missing; therefore no numeric sample size or power claim is made and no study is authorized.

The focused provenance/lineage/pilot/artifact suite passed **54 tests in 35.52s**. Final project virtual-environment suite: **659 passed, 1 failed, 2 deselected (662 collected) in 192.07s**. The only failure remains the same frozen held-out diagnosis test with 7/45 outcome mismatches. The two campaign-like finite-shot tests remain deselected. A system-Python bounded run also preserved the same sole failure (623 passed, 4 skipped, 2 deselected). The longer temporary root caused a known Windows path-length failure during an intermediate run; the final `.pwb17` rerun passed that recovery test.

Revised backlog: **39 tasks: 25 verified, 6 partial, 6 blocked, 2 deferred**. The new local pilot integrity task is verified; ML generalization/deployment, independent diagnosis validity and overall release remain **NOT READY**.


### Exact-current-tree rerun (2026-10-15)

After adding explicit stale-configuration and internal-run-artifact tampering regressions, the focused suite passed **55 tests in 39.77s**. The final bounded `.venv` suite passed **660**, retained **one** frozen held-out diagnosis acceptance failure (7/45), and deselected **two** registered campaign-like finite-shot tests (**663 collected; 185.31s**). This exact-current-tree result supersedes the immediately previous checkpoint above. Backlog remains **39 tasks: 25 verified, 6 partial, 6 blocked, 2 deferred**. Overall verdict remains **NOT READY**.


## Post-pilot blocker closure ? 2026-10-15

A fresh focused integration run passed **344 tests in 69.10s**. It covered the simulator, supported gate semantics, measurement/reset, noise, circuit execution, checkpoint/artifact recovery, dataset provenance and schemas, ML pilot/preflight/artifacts, and optional benchmark/IBM adapters. The existing `ml_data_foundation_pilot_20261015_v3` manifest also revalidated read-only: 9 observations/labels, 3 families and lineages, no near-duplicate family pairs, valid current source/runtime/config provenance. All labels remain synthetic; with zero independent adjudications, the pilot is not eligible for fitting or independent evaluation.

The bounded repository suite completed with **660 passed, 1 failed, 2 deselected (663 collected) in 179.93s**. Its only failure is the unchanged frozen held-out diagnosis acceptance (7/45). The two registered campaign-like finite-shot tests remain deselected under NO-GO. No threshold, label, expected outcome or acceptance rule was changed, and the previously documented forensic explanation was not re-investigated.

The 39-task backlog was recounted from its records; the stale summary count was corrected to **25 verified, 6 partial, 6 blocked, 2 deferred**. No additional independently demonstrated local defect was found. Scientific validity, ML generalization, benchmark accuracy, publication readiness and live hardware validation remain unsupported/blocked; the project is **NOT READY** for those claims.


## Evidence-based capability disposition (2026-10-15)

| Capability | Evidence status | Strongest supported claim and limit |
|---|---|---|
| Simulator correctness | Verified within a bounded tested scope | Supported statevector operations and registered gate matrices pass analytical and Qiskit oracle tests; this does not establish a complete physical noise model. |
| Measurement, reset and configured noise | Verified within a bounded tested scope | Forced readout, sampled reset trajectory and tested noise-channel semantics pass bounded reference checks; arbitrary channels and density-matrix evolution are outside scope. |
| Circuit compatibility | Partially verified with explicit limitations | Supported instruction/QASM fixtures and Bugs4Q/QMutBench adapters execute; broad QASM and all benchmark constructs are not covered. |
| Fault diagnosis and detection | Blocked by independent evidence or research validation | The unchanged held-out gate fails 7/45. Existing evidence supports a method/observation-resolution limitation; no broad detection or attribution accuracy claim. |
| Source/hardware localization | Verified within a bounded tested scope | Localization target checks pass in current fixtures; no real fault-cause or physical hardware localization claim follows. |
| ML provenance and pipeline guards | Verified within a bounded tested scope | New pilot manifests, lineage separation and fail-closed validators pass focused tests; this establishes data-pipeline controls only. |
| ML generalization, calibration and deployment | Blocked by independent evidence or research validation | The historical 270 rows are rejected; the nine pilot labels are synthetic with zero adjudications; no training or deployment claim is supported. |
| Benchmark accuracy | Blocked by independent evidence or research validation | Local adapters demonstrate parsing/execution only; version-linked authoritative independent case labels are absent. |
| Resource capacity | Partially verified with explicit limitations | Guard and bounded profiles are tested through 20 qubits on one host/session; no portable maximum or total-process memory cap is established. |
| Reproducibility | Verified within a bounded tested scope | Deterministic pilot regeneration and artifact/source manifests are checked; historical row-generation provenance remains unresolved. |
| Optional integrations | Partially verified with explicit limitations | Qiskit oracle is optional and its suite passes in the primary environment; offline IBM fixture behavior is tested, but clean optional installation and live service behavior are not claimed. |
| Publication claims | Partially verified with explicit limitations | Current README/status/report claims are scoped to evidence; a publication-specific archive review requires a declared publication bundle. |
| Real hardware | Deferred pending external resources or authorization | No backend identity, credentials or explicit execution authorization; no hardware result exists. |
| Research protocol record integrity | Verified within a bounded tested scope | The case contract blocks hidden-label leakage, family/lineage overlap and incomplete confirmatory prerequisites; it does not verify independent truth, power or detector performance. |


## Independent diagnosis research milestone (2026-10-15)

The existing diagnosis feasibility design now specifies a conditional staged candidate: a count-based distribution screen calibrated under its declared sampling model, followed by separate source/hardware evidence routing and explicit ambiguous/equivalent/unresolved dispositions. This is a proposal, not a production change or validated method. The interface currently exposes sampled distributions and localization candidates, but its fixed TVD guard is uncalibrated and its program/hardware candidate scores are not causal probabilities. No simulator or diagnosis decision rule changed.

Added `experiments/diagnosis_evaluation_protocol.py` and `tests/test_diagnosis_evaluation_protocol.py`. The contract rejects hidden truth/identity/internal-state fields in observable inputs; inconsistent count/shot records; duplicate case IDs and exact duplicate family fingerprints across roots; and family/lineage partition crossing. It keeps independent adjudication metadata separate, requires explicit cause-set semantics for ambiguous/equivalent cases, and blocks recorded confirmatory prerequisites when independent endpoint truth, provenance, hashed preregistration/criteria/power/resources, or review is missing. This verifies schema behavior only; it cannot certify external adjudication, population sampling, power or method performance.

Latest focused protocol, pilot, lineage and model-artifact validation: **47 passed in 35.93s** (supersedes an intermediate 46-pass run). An initial authoring run had **40 passed and 3 failed** because three new assertions expected different diagnostic wording; those assertions were corrected before the final run. The last full repository suite remains **660 passed, 1 frozen 7/45 held-out failure, 0 skipped, 2 deselected (663 collected; 168.65s)** and predates this protocol-only module/test addition; no post-addition full suite was run.

The canonical backlog is now **40 tasks: 26 verified, 6 partial, 6 blocked, 2 deferred**. The new verified task is the protocol record contract, not research validation. No confirmatory experiment is authorized or ready: a named target population, independent family frame and blinded adjudicators, independent review, use-specific acceptance criteria, family-level power analysis, and reviewed resource/stop plan are still required.

## Development-only diagnosis candidate advancement (2026-10-15)

The saved finite-shot comparison provides bounded synthetic calibration evidence for targeting low-shot healthy-control false alarms: the fixed TVD guard flagged all 120/120 controls at 256 shots and 104-110/120 at 1,024; its pooled-label permutation comparator flagged 2-8/120 in the reported calibration control cells. The same comparator detected only 14/120 subtle-effect calibration cases at 256 shots and 36/120 at 1,024. These are single-recipe repeated-template records, not independent families; the saved validation split has already been inspected and is not fresh validation. The three-family generalization fixture is synthetic engineering data. The frozen 45-case evaluation remained excluded.

An isolated seeded permutation p-value helper was added in experiments/diagnosis_candidate_methods.py, with seven input/determinism contract tests. Focused result: 7 passed in 0.74s. It is not wired into the default detector and has not been performance-evaluated in this pass. The p-value is not a fault probability and its null requires independent, exchangeable shot outcomes. No threshold, criterion or historical output changed; no campaign was run.

The next possible method evaluation needs new family-disjoint simulator development and validation sets, predeclared sampling/effect/control strata and a bounded resource plan. That may support only simulator behavior over its sampled recipe frame. Confirmatory claims still require a named target/use, independent family frame and adjudication, independent review, frozen acceptance/multiplicity rules, family-level power/resource analysis and separate authorization. Backlog remains 40 tasks: 26 verified, 6 partial, 6 blocked, 2 deferred. At this earlier checkpoint, the last full suite was 660 passed, one frozen 7/45 failure, zero skipped, two deselected (663 collected); it was superseded by the short-path suite recorded below.

## Family-disjoint development validation checkpoint (2026-10-15)

**Study disposition: not run; blocked before generation.** The finite-shot artifacts identify their primary recipe but do not provide a complete per-template lineage/fingerprint and descendant inventory. The three-family reference fixture is synthetic and used by the frozen outcome test. Some controlled-fault result records include family IDs but no complete ancestry-root metadata. The existing full-dataset generator assigns 30 nominal family IDs to only six lineage recipe roots, so its 30 IDs cannot be counted as independent families. The proposed finite sampling frame and deterministic sampler are also not materialized or versioned. These gaps prevent certifying that new validation families would be unrelated to prior inspected families and descendants.

Created a machine-readable plan at reports/experiments/diagnosis_development_validation_plan_20261015.json with proposed eight development and eight validation family roots, healthy/subtle/moderate/strong simulator intervention strata, 256/1,024 shot levels, two repeated measures per cell, baseline and permutation procedures, exploratory family-clustered uncertainty, provisional directional pilot categories, and caps of 512 circuit executions, 327,680 shot-circuit executions, 180 seconds, 256 MiB peak memory, and 5 MiB outputs. The plan is expressly blocked, not execution authorization. No family was sampled, no data generated, and no candidate settings selected.

Added fail-closed lineage/fingerprint and partition preflight, deterministic condition-manifest seed derivation, and atomic resource budget checks in experiments/diagnosis_development_study.py. The initial focused candidate/preflight/protocol checks passed 34 tests in 0.80s; after adding the descendant-root regression, the final focused suite passed 35 in 0.85s. These are engineering contract tests and do not provide detector performance evidence. The final bounded repository suite passed 695, failed only the unchanged frozen diagnosis acceptance (7/45), skipped 0 and deselected 2 NO-GO campaign tests (698 collected; 187.28s). A first longer-basetemp run also hit a Windows path-length failure; the failing recovery test passed in isolation with a short temp path, and the final short-path suite had no path-length failure.

Backlog remains 40 tasks: 26 verified, 6 partial, 6 blocked, 2 deferred. Diagnosis method feasibility remains partial and research validation remains blocked on a complete independently reconciled prior lineage inventory, verified new family frame, independent protocol review, and separate authorization. The frozen 7/45 diagnosis failure is unchanged.

## Lineage and family-frame milestone — 2026-10-15

**Disposition: inventory and frame materialized; design eligibility preflight does not pass.** The scanner at `experiments/diagnosis_lineage_inventory.py` generated a deterministic inventory over 5,431 files from `experiments`, `tests/fixtures`, and `reports/experiments`. It records 5,366 circuit-like JSON records, 5 exact duplicate-content artifacts, and 5,186 circuit-like records without reconciled lineage. Some explicit roots exist in artifacts, but none may be deemed fresh from this prior-artifact scan. The inventory deliberately remains `complete: false`: this scan cannot establish that all prior inspected retained temporary and external artifacts are represented, and many historical artifacts lack per-template roots/fingerprints. Inventory hash: `0338cb2d6ee8fbebb3afa1882cb8457dd4a010461dc1f8a23ff4969dd195ca32`.

A deterministic sampler artifact at `reports/experiments/diagnosis_family_sampling_frame_20261015.json` is bound to that inventory and versioned as `sha256-ranked-family-frame-v1`. It has zero eligible roots and preserves the planned eight-development/eight-validation shortfall. Frame hash: `6a96efd133d1e1f2fb9d19299b3206101499f22083493781afc1a4bc5c841e9a`. The fail-closed preflight now checks inventory/frame hashes, sampler version, inventory identity/completeness and required partition counts. Thus materialization is verified, but design eligibility is blocked. Missing prerequisites are independent reconciliation of all prior artifacts, resolution or conservative exclusion of missing ancestry/fingerprints, and proof of at least 16 genuinely fresh independent roots. No circuits, shots, thresholds, ML training, or hardware jobs ran.

Validation: focused suite 40 passed in 0.73s. Bounded repository suite 699 passed, 1 failed, 0 skipped, 2 deselected (702 collected; 165.26s). The sole failure is the frozen `test_heldout_generalization_outcomes_and_targets`, still 7/45 mismatches. The finite-shot NO-GO campaign tests remain deselected. Backlog remains 40 tasks: 26 verified, 6 partial, 6 blocked, 2 deferred; P1-4 remains partial.

Final verification addendum (2026-10-15): after preserving six cross-artifact family/root review flags without inferring a resolution, the focused suite passed 40/40. The bounded repository suite then completed with 700 passed, 1 failed, 0 skipped, and 2 deselected out of 703 collected (228.74s). Sole failure is the unchanged frozen diagnosis acceptance test (7/45). Final inventory hash is `0338cb2d6ee8fbebb3afa1882cb8457dd4a010461dc1f8a23ff4969dd195ca32`; final frame hash is `6a96efd133d1e1f2fb9d19299b3206101499f22083493781afc1a4bc5c841e9a`. The frame has zero eligible families. Historical intermediate test counts earlier in this report remain as checkpoints, not current results.

Final canonical checkpoint (2026-10-15): the preflight rejects self-declared inventory completeness. It requires zero unresolved and conflicting records, complete scan roots, and a source-hash-bound reconciliation attestation with at least two independent reviewers. Inventory/frame hashes after this code change are `8c952df9165b11b94fe68f07aede5a5d6a7a340f2e1ed10a11e0156cae5bec17` and `f808b8b19c477df0675b26f01a05f0392bb6f49e8560842c8624d60ee9cdb719`. Focused tests: 41 passed in 0.85s. Final broad suite: 701 passed, 1 failed, 0 skipped, 2 deselected out of 704 collected (223.99s). The unchanged frozen 7/45 diagnosis failure remains the sole failure; no study was run and all circuit/shot resources remain unused.

Final inventory scope correction (2026-10-15): retained prior temp-output directories and data/ML roots are now discovered and included under deterministic limits. The inventory scanned 20,000 artifacts across 108 roots (124,638,914 bytes), reached its 20,000-file cap, and skipped five files larger than the per-file cap. It therefore remains truncated/incomplete. Of 5,366 circuit-like records, 5,186 have unknown lineage; six observed cross-artifact root/family relationships remain review flags. Three recipe roots appear in saved rows; `experiments/full_dataset.py` declares six possible roots through its modulo-six expression, while only three occur in scanned saved rows. The finite frame has zero eligible roots and an 8+8 shortfall. Hashes: inventory `ec39aeccbac38407d7cb9aa7891cd388d681b056332a0aa4197dfd74400f5c37`; frame `ccef18762d9c87412e0f88e6d2b84ebdbbfd8b9d4edbc3ff0d22c69c03c0031b`.

Final validation after all source changes: focused suite 46 passed in 0.62s; bounded repository suite 706 passed, 1 failed, 0 skipped, 2 deselected (709 collected; 248.38s). The single failure remains the frozen 7/45 diagnosis acceptance test. No circuit generation, shot execution, threshold selection, model training, or hardware job occurred.

Deduplication detail: the expanded scan records 457 exact duplicate-content links (19,543 distinct content hashes in the 20,000 scanned artifact records), retaining every path and marking duplicate relationships. The earlier five-duplicate count applied only to the initial primary-tree scan and is superseded for the expanded inventory. It also identifies 1,671 family-bearing artifacts with no explicit lineage root. These rows remain unresolved/excluded; the frame still contains zero eligible roots.

Final exact-tree suite record (2026-10-15): 706 passed, 1 failed, 0 skipped, and 2 deselected out of 709 collected (217.46s). The sole failure is the unchanged frozen held-out diagnosis test (7/45). The focused lineage suite passed 46 tests in 0.62s. Inventory and frame hashes validate and repeat generation matches; the frame contains no eligible roots. No study execution occurred.


## 2026-10-15 current ML safeguards checkpoint

Feature eligibility review is recorded in `ml_feature_eligibility_20261015.md`. The production predictor remains the five output-distribution observables. Configuration-derived calibration variables are secondary calibration-only inputs, and `gate_mismatch_fraction` is explicitly limited to a diagnostic leakage ablation; feature-schema validation rejects it elsewhere. `ml_fallback_training.py` and `experiments/export_observable_model.py` now enforce strict provenance preflight before loading training features or fitting/writing. The current historical dataset check remains `ready=false`, with 0 eligible of 270 rows, so no training, score reproduction or model export was run.

Focused suite: **40 passed in 50.33s**. Bounded repository suite: **709 passed, 1 failed, 0 skipped, 2 deselected (712 collected; 217.11s)**. The sole failure remains the unchanged frozen diagnosis acceptance with 7/45 mismatches. Backlog counts remain **40 tasks: 26 verified, 6 partial, 6 blocked and 2 deferred**. The implementation safeguards are verified; ML calibration, independent-truth validity, generalization and deployment remain blocked.


## 2026-10-15 ML recovery and evaluation infrastructure checkpoint

The bounded recovery audit reconciles the row table for the historical 270-run dataset with its saved reports: 270 unique run IDs, 30 nominal families, 90 labels per class, and 3,765,323 report bytes read under a 64 MiB cap. Row reports retain IDs, configurations, seeds, shots, Python/NumPy runtime, generic simulator string, and synthetic labels. All 270 lack per-run source hashes and circuit ancestry; the dataset manifest lacks generation provenance; no row has independent adjudication or hardware observation. Saved-byte hashes were not interpreted as evidence of the code that generated those bytes. The existing recovery table reproduces exactly from all 270 current reports.

No alternate local source is eligible for the current three-way diagnosis target. The QMutBench ML artifact is a distinct mutant/original task, and its saved source CSV SHA-256 does not match the local CSV; the pilot, Phase 5 data and finite-shot evidence are synthetic. The prior-lineage scan remains capped/truncated after 20,000 files across 108 roots. Five oversized entries were streamed for hashes under a 320 MiB cap (315,670,588 bytes total), which verifies local bytes but does not complete that inventory.

Evaluation infrastructure now includes a primary registered observable schema, explicit ablation names, dummy baselines, fixed scaled Linear SVM and existing fixed classifiers, single-worker limits, three-way family/lineage split validation, lineage-cluster percentile intervals with unsupported class intervals withheld, synthetic-only claim metadata, and a new non-overwriting v3 output path. These are software controls only; no model was fit.

Focused recovery/evaluation/preflight/fallback tests: **41 passed in 10.52s**. Latest bounded repository suite: **719 passed, 1 failed, 0 skipped, 2 deselected (722 collected; 216.10s)**. The sole failure remains the preserved frozen diagnosis test at **7/45**. Backlog: **42 tasks ? 28 verified, 6 partial, 6 blocked, 2 deferred**. The independent family sampling frame, adjudicated truth, study power inputs, and external review are still required before ML generalization results can be reported.


### Final rerun after recovery-table schema v2

The v2 record-level recovery table now explicitly labels per-field evidence as recorded, inferred, recomputed from current saved bytes, or missing. It was generated as a new file; the earlier same-date report was preserved. Its deterministic rebuild matches the saved v2 artifact exactly. Final focused recovery/evaluation/preflight/fallback tests passed **41 in 15.39s**. The final bounded repository suite completed **719 passed, 1 failed, 0 skipped, 2 deselected (722 collected; 219.00s)**. The sole failure remains the frozen diagnosis acceptance at 7/45. Backlog counts remain **42: 28 verified, 6 partial, 6 blocked, 2 deferred**. No estimator was trained.


### External integration prerequisite identified

The current ML loader/preflight consumes simulator scenario-construction labels only. The diagnosis protocol separates adjudicator truth, but those independent records are not wired into the ML fitting input contract. No local independent labels or reviewed adjudication schema exists. Added blocked backlog task P1-4 for this bridge; it requires a reviewed blinded adjudication schema and provenance-complete observations with independent recipe roots. Current backlog is **43 tasks: 28 verified, 6 partial, 7 blocked, 2 deferred**.

## ML evaluation and diagnosis abstention milestone - 2026-10-15

The current family-grouped evaluator now checks every generated outer fold for exact row coverage and family/lineage disjointness before estimator construction or fitting. A deliberately leaking split is rejected before model construction. The primary five-observable schema, fixed comparison partitions and estimators, class-ordered confusion/per-class metrics, lineage-cluster bootstrap intervals, and synthetic-only scope remain in force. Calibration-quality metrics are not reported: there are no eligible independent calibration families or labels. No model was fit or scored in this milestone, and the historical 270 rows and nine-row pilot remain ineligible.

An independently available equivalence fixture exposed a cause-attribution gap in the direct `diagnose_execution` and ML integration paths. They could promote a syntactic operation difference despite identical observed distributions. They now preserve the source localization candidate and abstain on behavioral program-fault attribution when no distribution anomaly is observed. Existing mixed evidence remains ambiguous. This is a conservative decision-path correction, not an accuracy result. The frozen acceptance was not modified and still fails in 7/45 cases.

Focused validation passed **92 tests in 16.89s**. The final bounded repository suite collected **725 tests: 722 passed, 1 failed, 0 skipped, 2 deselected in 221.49s**. The sole failure remains the frozen held-out diagnosis acceptance; the two registered finite-shot NO-GO tests remain deselected. No fitting, dataset generation, installation, benchmark campaign or hardware job occurred.

The canonical backlog is **44 tasks: 29 verified, 6 partial, 7 blocked and 2 deferred**. The implementation work is verified within its regression scope. Diagnosis performance/finite-shot validity, independent family data, independent blinded labels/review, ML calibration/generalization, benchmark truth, and hardware evidence remain blocked or deferred. The project remains **NOT READY** for those claims.
Case-level map from the preserved root-cause record: `qft_like_3q_program_fault_11`, `_29`, and `_47` are subtle program-fault detection misses with exact TVD below 0.05; `ring_cluster_4q_correct_11` and `_29` are healthy-control finite-shot false positives; `ring_cluster_4q_equivalent_11` and `_29` are observationally equivalent CZ reorderings with finite-shot false positives and syntactic localization that is not proof of behavioral cause. The source/hardware localization target assertions passed. No case details were inferred beyond the saved record.

The existing diagnosis review checklist is in `diagnosis_method_feasibility_design_20261014.md`; the ML review protocol is in `ml_independent_evaluation_protocol_20261015.md`. Both specify the target population/frame, independent lineage roots, blinded adjudication, endpoints and uncertainty, use-specific acceptance decisions, power inputs, and resource/stop limits. Numerical performance floors, power, and family counts remain unset pending external review and frame evidence.

## Independent diagnosis protocol readiness update (2026-10-15)

A machine-readable protocol package is recorded at `reports/experiments/independent_diagnosis_protocol_package_20261015.json`. The diagnosis record validator now requires source/generation/lineage/circuit/configuration/seed/noise/runtime/reproduction provenance and rejects protocol self-attestation as proof of independent review. Focused protocol, candidate-contract, and development-preflight tests passed 54/54 in 0.82s. The package remains NOT YET APPROVED and the confirmatory study remains BLOCKED; no study or new diagnosis observations were generated. The frozen 7/45 acceptance failure and finite-shot NO-GO exclusions are unchanged.


## 2026-10-15 final bounded validation addendum

The diagnosis protocol package and fail-closed provenance/review contract were tested with 54 passing focused tests. Final bounded repository suite: **730 passed, 1 failed, 0 skipped, 2 deselected of 733 collected (222.36s)**. The sole failure is the unchanged frozen held-out diagnosis contract at 7/45. A longer-base-directory first run encountered a Windows path-length failure in one recovery test; that test passed alone with a short base directory and the short-basetemp full rerun had no such failure. The two registered finite-shot NO-GO tests remain deselected. Backlog remains **44 tasks: 29 verified, 6 partial, 7 blocked, and 2 deferred**. Research approval, family frame, independent adjudicators/truth, criteria, power/resource assumptions, and execution authorization remain blocked.


## Provenance reconciliation and adjudication gate (2026-10-15)

The bounded deterministic reconciliation report is `reports/experiments/diagnosis_sampling_frame_reconciliation_20261015.json`. It reused saved inventory/recovery artifacts only. It records six possible roots (three conflicted and three source-inferred), six unresolved inventory conflicts, 5,186 unknown-lineage circuit records, 457 exact duplicate artifacts in the inventory, 30 nominal ML family IDs / 270 rows without recipe ancestry or per-run source hashes, no sampling-frame candidates, and zero eligible independent families. Scan completeness remains false.

A privacy-aware independent-adjudication contract and confirmatory preflight now require opaque reviewer tokens, external case/protocol/blinding/independence attestations, evidence-backed separate judgments and resolution, execution provenance, exact frame-to-train/calibration/test lineage mapping, and the existing protocol review/criteria/power/resource gates. Focused evidence/preflight/protocol tests: **64 passed in 3.32s**. These are engineering tests only; no real reviewers, independent labels, external attestations, or study execution exist. The bounded repository suite is the remaining validation step for this milestone.


## Final bounded validation - provenance/adjudication phase (2026-10-15)

Focused reconciliation, adjudication, study-preflight, and protocol tests: **64 passed in 3.32s**. Final bounded repository suite: **747 passed, 1 failed, 0 skipped, 2 deselected (750 collected; 226.33s)**. The sole failure is the unchanged frozen diagnosis acceptance at 7/45; both finite-shot NO-GO tests remained deselected. No ML fitting, dataset regeneration, experiment, package installation, or hardware job occurred. The canonical backlog now contains **46 tasks: 31 verified, 6 partial, 7 blocked, 2 deferred**. External family reconciliation, independent adjudication/review, criteria/power/resource decisions, and authorization remain blockers.


## Hash-bound conflict review ledger - 2026-10-15

The evidence reconciliation tool now accepts an optional custodian review ledger. Each review binds to one exact conflict hash and requires a rationale, opaque reviewer token, and evidence references. Stale/duplicate records are rejected. Reviewer classifications are not treated as verified facts, and the ledger never resolves lineage conflicts by itself. External attestation requires a matching independently verified record through the trust boundary.

Focused tests: **21 passed**. Bounded repository suite: **752 passed, 1 failed, 0 skipped, 2 deselected (755 collected; 232.80s)**. Sole failure: unchanged frozen diagnosis acceptance at 7/45. Both finite-shot NO-GO tests remained deselected. Backlog: **47 tasks (32 verified, 6 partial, 7 blocked, 2 deferred)**. Research remains blocked by incomplete lineage inventory, absent independent evidence/review, and missing acceptance/power inputs.


## Simulator correctness and observation validation - 2026-10-15

Fixed two verified input/model defects. Hardware calibration time constants and durations must now be finite and nonnegative. Constructor-level default gate error/duration values now apply when gate-specific calibration is absent and are preserved across calibration snapshot serialization. Distribution comparisons reject negative, Boolean, or non-finite masses, invalid thresholds, ambiguous stringified outcome keys, and multidimensional arrays. Max-scaled normalization avoids overflow for very large finite values. A diagnosis boundary regression confirms NaN observations raise rather than returning NO_ANOMALY. A separate seeded parity fixture confirms traced and untraced final states match under gate-error, T1/T2, and readout noise; its focused suite passed 6/6.

Focused validation: **149 passed in 4.45s**. Bounded suite: **768 passed, 1 failed, 0 skipped, 2 deselected (771 collected; 224.48s)**. The only failure remains the frozen 45-case diagnosis acceptance at 7/45; both finite-shot NO-GO tests remain deselected. These tests establish software-input and model behavior only. They do not establish diagnosis accuracy, ML generalization, calibration, or hardware validity.


## ML inference contract validation - 2026-10-15

The integrated observable classifier boundary now rejects invalid confidence cutoffs (including NaN, infinity, and booleans), invalid probability values (including booleans, strings, and non-finite values), and classifiers that omit the declared feature schema. No threshold, frozen case, or diagnosis expectation changed. Focused pipeline suite: 129 passed in 24.10s. Bounded repository suite: 778 passed, 1 failed, 0 skipped, 2 deselected (781 collected; 223.55s). The sole failure remains the frozen 7/45 diagnosis acceptance; both finite-shot NO-GO tests stayed deselected. No model fitting or evaluation occurred; this is software contract verification only.


## Diagnosis evidence gate - 2026-10-15

When expected and observed distributions are absent, a syntactic source mismatch is now retained for localization but does not set anomaly_detected or support PROGRAM_FAULT attribution. Focused tests: 85 passed in 33.18s. Bounded suite: 778 passed, 1 failed, 0 skipped, 2 deselected (781 collected; 224.61s); the only failure remains the frozen 7/45 acceptance. Both finite-shot NO-GO tests were deselected. No frozen target or threshold changed.


## Causal evidence gating - 2026-10-15

Program fault attribution now requires both a direct source candidate and a source-only counterfactual output change that reaches measured qubits. Disconnected unmeasured changes and behaviorally equivalent gate substitutions do not explain an observed anomaly. Relevant hardware evidence alongside a source candidate remains ambiguous. The fault formatter labels non-definitive cases as source mismatch candidates and omits definitive correction language. Focused diagnosis/pipeline tests: 90 passed. Bounded suite: 783 passed, 1 failed, 0 skipped, 2 deselected (786 collected; 225.90s). The frozen acceptance remains the same 7/45 mismatches; both finite-shot NO-GO tests stayed deselected. No scientific accuracy claim is made.


Final causal-gating validation: focused suite 90 passed in 28.50s. Bounded suite 783 passed, one unchanged frozen 7/45 diagnosis failure, zero skipped, two NO-GO finite-shot deselections (786 collected; 226.82s). A temporary generic timestamp gate was removed after it caused four failures in software experiment fixtures using the intentional synthetic-config timestamp marker. Generic hardware-model freshness remains unresolved because no max-age policy is defined; no physical-hardware validity claim is made.


## Calibration provenance and age semantics - 2026-10-15

The generic hardware model now records optional caller-reported calibration acquisition time and provenance separately from snapshot `created_at`. Generic freshness has no universal cutoff: callers can supply a maximum age, and only then can a reported timezone-aware timestamp become `FRESH` or `STALE`. Missing/unknown provenance, malformed or timezone-naive timestamps, and future timestamps do not support current hardware attribution under a policy. Synthetic configuration remains usable as simulator configuration. This does not authenticate timestamp sources or establish physical-hardware validity. IBM preflight retains its separate explicit age contract.

Focused hardware/diagnosis/IBM validation: 108 passed. Final short-path bounded suite: 793 passed, one unchanged frozen 7/45 failure, zero skipped, and two finite-shot NO-GO tests deselected (796 collected). No expected result or threshold changed.


## Generic calibration provenance/freshness and bounded follow-up - 2026-10-15

The generic `HardwareModel` freshness contract now distinguishes reported acquisition time, synthetic configuration, unknown legacy provenance, and snapshot creation time. An explicit caller maximum age is required to label a reported calibration fresh or stale; stale/unknown/invalid/future calibration cannot support current hardware attribution under that policy. JSON snapshot roundtrip preserves provenance, and synthetic-config remains allowed. Focused hardware/diagnosis/IBM preflight suite: 109 passed. Final bounded suite: 794 passed, one unchanged frozen 7/45 failure, zero skipped, two finite-shot NO-GO deselections (797 collected; 227.40s). A first longer-basetemp run hit a Windows temporary-directory setup error; the short-path run had none.

Two existing safe backlog items were also revalidated without code changes: two-qubit parameterized-gate canonicalization (8 passed) and bounded Bugs4Q discovery (1 passed, 11 deselected). The latter's default temp-root attempt was denied by Windows; its workspace-local rerun passed. No training, hardware execution, installation, or large benchmark occurred.


Safe discovery-boundary follow-up (2026-10-15): current-tree Bugs4Q conventional-root scan test passed (1 passed, 11 deselected) and QMutBench conventional-root scan test passed (1 passed, 23 deselected). Both implementations were already present; stale backlog priority markers were reconciled. No benchmark data were regenerated or broadly scanned.
