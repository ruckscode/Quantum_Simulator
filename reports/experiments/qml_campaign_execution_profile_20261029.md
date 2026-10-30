# QML campaign execution profile — 2026-10-29

## Overall state

The campaign is **not complete**. Independent preprocessing, grouped QNN pilot, candidate sweep, and artifact-validation work recorded below is complete or has a documented data/reuse blocker. The QMutBench binary full-development QNN remains eligible and resumable, but its last verified checkpoint is paused because available physical RAM is below the measured startup gate. No frozen-test data was read by the QMutBench development run or the additional Qolumbina development experiments.

The latest verified state is 371 committed QMutBench chunks, 5,936 Adam updates, 189,952 training-record exposures, fold 1 / epoch 0 / next batch 5,936, and zero completed folds. All 371 checkpoints pass the progress audit. Recorded run-wide minimum available RAM was 1,050.4 MiB; maximum checkpointed process RSS was 471.7 MiB. The latest post-validation probe was 1,901.1 MiB, below the 2,016 MiB startup requirement; the runner also rejected a launch at 1,962 MiB before training. The 1,280 MiB checkpoint-boundary pause threshold remains active.

## Task matrix

| # | Campaign task | Preprocessing / semantic status | QNN / evaluation status | Remaining action |
|---:|---|---|---|---|
| 1 | Bugs4Q official framework / source-state | Validated, 62 rows / 31 groups; grouped split and feature-duplicate checks passed. | Existing grouped pilot and candidate sweeps complete. | None identified in current plan. |
| 2 | Bugs4Q-NA / source-state | Validated, 62 / 31; grouped split and feature-duplicate checks passed. | Existing grouped pilot and candidate sweeps complete. | None identified in current plan. |
| 3 | Bugs4Q pooled integration | Validated, 124 / 62; source distinction retained. | Existing grouped pilot and candidate sweeps complete. | None identified in current plan. |
| 4 | Bugs4Q core-only transfer | Source-restricted task is validated. | Valid source-restricted resumable pilot and candidate sweep complete; older mixed-source result is invalid for transfer and preserved. | None identified in current plan. |
| 5 | Bugs4Q-NA-only transfer | Source-restricted task is validated. | Valid source-restricted resumable pilot and candidate sweep complete. | None identified in current plan. |
| 6 | QBugs HornBro subset | Original feature/label formulation blocked by exact input collisions with contradictory labels. Duplicate-aware QASM variant validated on 1,496 rows / 748 groups. | Duplicate-aware QASM grouped pilot and two-architecture QNN candidate sweep complete. Amplitude: 68.38% accuracy / balanced accuracy, 68.30% macro-F1; angle re-upload: 73.26% accuracy / balanced accuracy, 73.22% macro-F1. OOF and checkpoint hashes verified. | Original ambiguous formulation remains blocked; variant is exploratory grouped validation. |
| 7 | Quantum platform issue-comment task | Original comment task blocked by repeated comments across folds. Duplicate-aware variant validated on 319 rows / 304 groups. | Duplicate-aware grouped pilot complete. | Original formulation needs component-aware split or a defensible context redesign. |
| 8 | Qolumbina ideal-backend seeded-mutant detection | Validated, 182 / 26. | Grouped QNN pilot and two-architecture candidate sweep complete. | None identified in current plan. |
| 9 | LintQ TP/FP warning triage | Warning-only formulation blocked by repeated features with conflicting labels. Context-v2 is exploratory, with three conflicts in an identical-context component. | Context-v2 pilots and the separately versioned conflict-excluded QNN variant complete; the original 290-row task remains label-ambiguous. The variant rejects all three rows with byte-identical inputs and conflicting labels, retains 287 rows / 216 groups, and scores 74.91% accuracy, 45.40% macro-F1, 51.14% balanced accuracy on grouped OOF. Artifact: reports/experiments/lintq_context_v2_conflict_excluded_20261029/run_manifest.json. | Resolve label ambiguity with justified context or an explicitly redesigned split/target. |
| 10 | Bugs4Q-Robust execution prediction | Validated, 2,220 / 37; feature collisions documented. | Grouped QNN pilot and six-configuration, 20-epoch sweep complete and audited. | None identified in current plan. |
| 11 | QSimBench backend fingerprinting | Validated, 300 / 150; cross-setting descriptor collisions limit generalization claims. | Grouped QNN pilot and six-configuration, 20-epoch sweep complete and audited. | Keep conclusions exploratory and setting-specific. |
| 12 | QMutBench three-class mutation outcome | Validated historical mutation-outcome dataset, 205 / 12. | Recovered development OOF and nine optimizer/RNG checkpoints; reports retain matched exploratory metrics. | Separate from hardware fault diagnosis. |
| 13 | Mute-Alg source-state task | Validated main source task, 9,140 / 812. A separate compatible subset is too narrow (5,440 / 11 groups); source snapshot lacks local license/commit evidence. | Main grouped QNN pilot complete. | Do not generalize from the narrow compatibility subset; preserve source-rights/provenance limitation. |
| 14 | Bug-Pattern | Ineligible: nine local notebooks, category leakage in titles/markdown, no clean negatives or adequate independent groups. | Not trained. | Obtain defensible row-level outcomes and independent negative examples. |
| 15 | QChecker | Ineligible: no verified per-file detector outcomes or clean negatives; severe imbalance and source naming/syntax issues. | Not trained. | Establish verified outcomes and a leakage-safe supervised sample. |
| 16 | QML platform workbooks | Blocked: heterogeneous/shifted schema, malformed/unlabeled rows, repository/title concentration, and no reuse license found. | Not trained. | Resolve reuse rights, normalize labels, and establish a defensible split. |
| 17 | QMutBench binary Survivor vs WOO/OPO | Streaming index validated on 578,117 development records / 305 origins; frozen test excluded. | Partial QNN: 371 verified chunks, 5,936 updates, 189,952 exposures; no completed folds or OOF metrics. | Resume the existing run when the calibrated preflight passes; latest measured available RAM is 1,901.1 MiB. |
| 18 | Nine-source grouped binary screen | Nine validated adapters; 5,055 rows / 1,606 task-local groups. | Existing classical screen verified (180 model/task summaries, 900 fold/model metrics); this is not a matched QNN comparison. | External Kaggle notebook remains unexecuted; separate from current QML scope. |

The task rows above are based on the current [source preprocessing matrix](qml_source_preprocessing_20261029/source_preprocessing_matrix.md), with new Qolumbina multi-backend work verified below. The original six manifests (2,595 accepted rows) were reused rather than rebuilt.

## New Qolumbina multi-backend execution

The 546-row manifest, 26 original-program groups, five-fold split, labels, and prior pilot OOF predictions were checked before training. IDs were unique and identical across the three artifacts; all labels and fold assignments matched; no group crossed folds. Target: DOSS detection of seeded Qolumbina mutants across ideal, fake_algiers, and fake_brooklyn. This is empirical mutation outcome, not hardware diagnosis.

The QNN-only two-architecture grouped sweep completed with 1,092 OOF predictions and ten checkpoint files. Both configurations predicted the positive class for every row: accuracy 92.674%, macro-F1 48.099%, balanced accuracy 50.0%. This is not useful minority-class detection under 40 negative / 506 positive rows.

A development-only angle-reupload QNN follow-up used inverse-frequency class weights computed separately from each training fold (`n_train / (2 * n_class)`), four epochs, batch size 64, Adam, seed 28102026, and fold-local min-max scaling. It completed five folds in 4.04 seconds, produced 546 OOF predictions, and reached accuracy 59.34%, macro-F1 41.07%, balanced accuracy 43.53%. The weighted run did not improve balanced accuracy. Its summary, OOF, and five checkpoint hashes were verified. No classical model was trained for this follow-up.

Artifacts: [candidate sweep manifest](qolumbina_multibackend_candidate_sweep_20261029/run_manifest.json), [weighted follow-up manifest](qolumbina_multibackend_candidate_sweep_20261029/weighted_loss_followup/run_manifest.json).

## QMutBench binary resume state and command

The preserved run is `reports/experiments/qmutbench_binary_qnn_full_development_chunk16_20261029`. Its [progress audit](qmutbench_binary_qnn_full_development_chunk16_20261029/progress_audit.json) reports `VERIFIED`, zero issues, source hash `3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778`, QASM manifest hash `284188822b6b846a12bcf791fc5122146d3ed78be29588b97b39595534abda01`, and reference archive hash `bfc1affdd40ac1032b65e417548075ce066407297fc603085ba6519403f56367`. Frozen-test access is false.

Resume from the repository root when RAM preflight permits:

```powershell
.venv\Scripts\python.exe -m qmutbench_qml.training.run_binary_development --mode full-development --output reports/experiments/qmutbench_binary_qnn_full_development_chunk16_20261029 --chunk-batches 16 --max-chunks-per-invocation 50 --resume
.venv\Scripts\python.exe experiments\audit_qmutbench_full_development_progress.py
```

After every bounded invocation, retain the output and audit before starting another. The runner now pauses at a durable chunk boundary below 1,280 MiB. Startup requires the calibrated 2,016 MiB estimate; the hard runtime floor remains 1,024 MiB. Do not lower these limits to force execution. Two bounded QMutBench resumes advanced from checkpoint 271 to 321 and then 371, adding 100 chunks (1,600 optimizer updates, 51,200 record exposures). Both invocation audits passed with zero issues; all committed chunks have gradient and parameter-update evidence. Frozen-test access remains false. The latest live probe is 2,012.3 MiB, below the 2,016 MiB startup gate. The QBugs candidate run completed one architecture; the second stopped before training under the 700 MiB gate. The attempt log is [resume_attempt_history.json](qmutbench_binary_qnn_full_development_chunk16_20261029/resume_attempt_history.json).

At checkpoint 371, the observed-rate estimate is 134.75 hours of QNN updates for 578,136 updates; it excludes streaming/preprocessing and fold evaluation and is an extrapolation, not a completion guarantee. No binary QNN OOF metrics exist yet because no fold has completed.

## Tests and acceptance issues

Latest focused run on 2026-10-30: `42 passed, 1 skipped` across source-isolated preprocessing, QMutBench binary checkpoint/gradient/resume, QMutBench full-streaming resume, QBugs candidate sweep, Qolumbina multi-backend sweep, and LintQ conflict-excluded QNN tests. Earlier relevant run: `43 passed, 1 skipped` across QMutBench binary checkpoint, streaming, resume, and Qolumbina sweep audit tests. The skipped test is the real PennyLane smoke test because available RAM was below its unchanged 2,048 MiB requirement. The resume command also passed `--validate-config` without loading data. No relevant test failed in the latest run.

The frozen simulator diagnosis mismatch remains a separate acceptance issue: **7 mismatches among 45** in the existing acceptance record. QML mutation-outcome experiments do not address quantum hardware fault attribution; no quantum advantage or successful hardware diagnosis is claimed.

## Completion boundary

All independent, currently executable QNN work identified in the campaign matrix has been advanced or verified. Remaining eligible planned work is QMutBench binary full-development. QMutBench is not complete. At the latest probe, available RAM was 1,901.1 MiB, below the 2,016 MiB QMutBench gate. The active fold-1 checkpoint is 371, with 5,936 optimizer updates committed and no fold complete. Resume with the command above when the preflight passes. Re-run the relevant tests after any code changes and continue until all five folds and evaluation artifacts complete.
