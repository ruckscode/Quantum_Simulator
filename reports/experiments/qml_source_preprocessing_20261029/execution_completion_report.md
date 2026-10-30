# Source validation and resumable QML pilot â€” 2026-10-29

## Reconciled state

- The source inventory has 19 entries and the current campaign task CSV has 18 rows, matching the previous state report. The matrix also tracks four separately versioned task variants.
- The six previously completed source tasks remain checksummed at 2,595 accepted rows. The 12 task rows beyond these are now represented as validated, blocked, ineligible, or an explicitly unexecuted external bundle in the [task matrix](source_preprocessing_matrix.md).
- The semantic audit verified 17 processed task datasets/indexes: the six prior manifests, validated source-specific task variants including QBugs QASM and the Mute-Alg compatibility subset, the QMutBench three-class development sample, and its binary development index.
- Source SHA-256, fold SHA-256, partition SHA-256, row totals, unique record IDs, grouped fold assignments, and contract target mappings passed for all 16 table-backed tasks. Binary QMutBench index SHA-256 passed; 578,117 development records and 305 origins were indexed, with zero frozen-test rows.
- The existing candidate sweep artifact audit is `verified`; recomputation matched all 90 of 90 recorded checkpoint hashes. The pilot's processed source, fold, contract, partitions, per-fold checkpoints, histories, and OOF predictions are also checksummed in its run artifacts.

## Semantic readiness and blockers

`READY_FOR_TRAINING` applies to the validated Bugs4Q core, NA, pooled, and transfer tasks; Bugs4Q-Robust harness outcomes; Qolumbina ideal seeded-mutant detection; and the sampled QMutBench three-class historical mutation-outcome task. QMutBench labels describe mutation outcomes and do not diagnose hardware faults.

- **LintQ:** blocked from new training. The warning-message-only input has 115 rows in duplicate fingerprints spanning folds; eight fingerprints carry conflicting TP/FP labels.
- **LintQ context variant:** a separate 290-row task uses the source-provided rule, warning, and code context. Original issue groups and identical complete model-inputs are connected into 217 components before assigning folds; no exact input fingerprint crosses folds. Three records in one identical-context component still have conflicting labels, so the task and both pilot variants remain exploratory.
- **QBugs:** blocked from new training. Its declared input text/metadata has 701 rows in duplicate fingerprints spanning folds; 34 fingerprints have conflicting correct/buggy labels.
- **Quantum platform comments:** the original 319-row split has 15 repeated comment rows crossing folds and is blocked. A separate v2 unions issue IDs and exact comment text into 304 components; all 319 rows are accepted, exact duplicate comments stay within folds, and a balanced QNN pilot completed. This is curated software issue-comment annotation, not hardware fault diagnosis.
- **QSimBench:** exploratory only. Quantized output descriptors have 171 cross-fold repeated rows and 15 conflicting-label fingerprints. Its target is backend identity, not fault attribution.
- **QMutBench binary:** preprocessing ready, training blocked. A fresh preflight measured 1,294 MiB available; sampled pilot requires 2,048 MiB and full-development requires 2,076 MiB while preserving the 1,024 MiB floor.
- **Mute-Alg:** the full task remains blocked. In the restricted Qiskit 1.4.5 scan, 77/169 files loaded and 92 failed with `state is not a dictionary`. A separately processed compatible subset has 5,440 accepted rows, but 1,720 rows (860 pairs) were rejected for exact-QASM label conflicts and only 11 duplicate-aware groups remain; that subset is marked ineligible for representative training.
- **Bug-Pattern and QChecker:** ineligible until a supervised task and label mapping are validated.
- **QML platform workbooks:** all 39 extracted workbooks parse as OOXML, but exports have heterogeneous or shifted columns. The integrated Conclusion workbook has 481 issue rows: 339 unique rows have clean 0/1 quantum-related labels (220/119), 140 rows are unlabeled or malformed, and two duplicate issue rows were removed. Ten repositories collapse to nine repository/title components, one containing 264 rows. The source inventory found no reuse license, so training remains blocked pending rights clarification and a defensible held-out split.
- **Nine binary source-task bundle:** prepared but its external 14-model and imbalance-method suite remains unexecuted.

For Bugs4Q-Robust, the first preprocessing attempt is preserved: the contract incorrectly treated the run-index field as a single fixed value, rejecting 2,146 of 2,220 rows. Inspection showed it contains run numbers. The corrected bounded numeric contract produced v2 with 2,220/2,220 accepted and five issue-group folds. The rejected v1 output was not overwritten.

## Resumable pilot

Seven end-to-end development runs use the registered resumable trainer: Bugs4Q-NA and core source-state tasks (62 rows/31 issue groups each), Bugs4Q-Robust harness outcomes (2,220 rows/37 issue groups), two LintQ context variants (290 rows/217 duplicate-aware components), a balanced Quantum platform comment task (319 rows/304 issue-and-comment components), and QBugs OpenQASM correct/buggy classification (1,496 rows/748 complete pair groups). The Bugs4Q tasks reuse established five-fold splits; LintQ uses a new stratified group split that unions each original issue with identical complete model-input contexts. All use deterministic 8-dimensional hashed-code features, fold-local train-only min-max scaling, a 3-qubit depth-1 amplitude QNN, Adam, seed 28102026, batch size 64, and four epochs. The balanced LintQ variant derives class weights only from each training fold. Every classical control uses the same eligible features, split, and fold-local scaler. Robust predicts observed harness pass/fail, not a universal bug label.

| Task/model | Accuracy / balanced accuracy | Macro F1 | Confusion matrix |
|---|---:|---:|---|
| Bugs4Q-NA QNN | 0.4677 / 0.4677 | 0.4609 | [[11,20],[13,18]] |
| Bugs4Q-NA logistic regression | 0.5000 / 0.5000 | 0.4999 | [[15,16],[15,16]] |
| Bugs4Q core QNN | 0.4516 / 0.4516 | 0.4493 | [[12,19],[15,16]] |
| Bugs4Q core logistic regression | 0.4677 / 0.4677 | 0.4609 | [[13,18],[15,16]] |
| Bugs4Q-Robust QNN | 0.5270 / 0.5270 | 0.5269 | [[570,540],[510,600]] |
| Bugs4Q-Robust logistic regression | 0.6081 / 0.6081 | 0.6075 | [[630,480],[390,720]] |
| LintQ context QNN, unweighted | 0.7448 / 0.5000 | 0.4269 | [[0,74],[0,216]] |
| LintQ context QNN, fold-balanced loss | 0.4379 / 0.4583 | 0.4185 | [[37,37],[126,90]] |
| LintQ context logistic regression, balanced | 0.5966 / 0.5648 | 0.5433 | [[37,37],[80,136]] |

For Bugs4Q-NA, QNN minus classical macro-F1 is -0.0381, with paired 1,000-resample group-bootstrap 95% interval [-0.1134, 0.0274]. For core, the difference is -0.0115, interval [-0.0840, 0.0734]. For Robust, it is -0.0777, interval [-0.2011, 0.0536]. For the LintQ context task, unweighted QNN predicted every record as TP (balanced accuracy 0.50). Applying class-balanced loss computed from training-fold labels changed the confusion matrix but reduced macro-F1 to 0.4185; its paired interval against the balanced logistic baseline is [-0.2063, -0.0389]. QNN/classical comparisons are development grouped OOF, not confirmation evidence. Each run deliberately stopped at epoch 2, resumed, and completed all folds at epoch 4. All verified optimizer and RNG continuity against uninterrupted references and passed gradient checks. Invocation runtime was 0.535 seconds for NA, 0.494 seconds for core, 1.974 seconds for Robust, 0.839 seconds for unweighted LintQ, 0.852 seconds for balanced LintQ, and 0.961 seconds for the platform task. Platform available RAM was 712.2 MiB at start and 702.5 MiB at end; all stayed above the configured 500 MiB floor.

Artifacts: `reports/experiments/resumable_qnn_pilot_bugs4q_na_20261029/`, `reports/experiments/resumable_qnn_pilot_bugs4q_core_20261029/`, `reports/experiments/resumable_qnn_pilot_bugs4q_robust_20261029/`, both `reports/experiments/resumable_qnn_pilot_lintq_context_v2*_20261029/` runs, and `reports/experiments/resumable_qnn_pilot_quantum_platform_v2_balanced_20261029/`. Each contains run manifests, metrics, matched classical baseline manifests, OOF predictions, per-fold checkpoints, and histories. Twelve checkpoint/history/prediction hashes matched their recorded values for each run.

## Additional grouped classical screen

After confirming source eligibility and 1,341 MiB available at startup, I ran the prepared classical suite locally on five nonblocked sources: Bugs4Q core, Bugs4Q-NA, pooled Bugs4Q, Qolumbina ideal detection, and Bugs4Q-Robust. It completed 20 model variants per task, with **100 aggregate OOF rows and 500 fold/model metrics**. It used five-fold GroupKFold by issue/program group; each fold fit TF-IDF, four-component SVD, min-max scaling, and any sampling using training data only. Sampling strategies were random over/under-sampling and continuous-space SMOTE. The process ended with 1,240.6 MiB available and sampled peak process RSS was 145.6â€“153.1 MiB.

The highest descriptive OOF macro-F1 model per task was HistGradientBoosting for core (0.5321), KNN for NA (0.5617), KNN for pooled Bugs4Q (0.5080), Extra Trees with random oversampling for Qolumbina (0.5345), and KNN for Robust (0.5810). These maxima are descriptive after looking across models and are not independent model-selection estimates. This suite uses TF-IDF/SVD features and its own GroupKFold assignments, so these scores are not matched comparisons with the hashed-feature QNN pilots. Four duplicate-risk tasks were excluded; the separate external Kaggle bundle remains unexecuted. The [suite audit](../local_grouped_classical_suite_eligible_20261029/report.md) replayed all OOF predictions, targets, group assignments, and aggregate metrics without issues.

The exact source/task/hash configuration is [local_classical_suite_config.json](local_classical_suite_config.json). The suite finished in one invocation; it has no checkpoint-resume mode. To reproduce into a fresh output directory, set `QML_CLASSICAL_CONFIG_PATH` to that config and override `QML_CLASSICAL_OUTPUT` to a new, empty directory before running `experiments/run_grouped_classical_suite.py`. The runner refuses to overwrite an existing output directory and checks source hashes.

## Tests and limits

- `tests/test_resumable_qnn_pilot.py`, `tests/test_source_isolated_preprocessing.py`, and `tests/test_qmutbench_streaming.py`: **16 passed**. Pilot tests validate the seven registered pilot task configurations, task separation, fold integrity, fold-local class weights, and rejection of unregistered tasks.
- Python compilation passed for the modified runner, semantic audit, status builder, streaming adapter, QMutBench indexer, pilot, and training modules.
- `git diff --check` passed (Git emitted only existing CRLF conversion notices).
- A broader combined run also attempted the QML foundation/multiclass training tests. Three tests stopped at the model's 1,024 MiB safety floor because the machine had less memory; those tests are not recorded as passing here. Earlier generic source-runner tests in that same attempt exposed the `None` regression, then passed after the fix.
- QMutBench's frozen test was not opened or used for preprocessing, feature selection, tuning, or model selection. The simulator acceptance mismatch remains unchanged at 7 of 45; QML is not claimed to resolve hardware fault attribution.

## Resume commands

The exact source-task commands and their established chunk sizes are listed in [the task matrix](source_preprocessing_matrix.md). To reconstruct duplicate-aware variants first, run `experiments\prepare_lintq_context_v2.py` or `experiments\prepare_quantum_platform_comment_v2.py`, then pass the matching contract to `prepare_source_isolated_task.py`. Rebuild the QMutBench development index with:

```powershell
.venv\Scripts\python.exe experiments\prepare_qmutbench_binary_development_index.py
```

If the pilot is interrupted before completion, resume it with:

```powershell
.venv\Scripts\python.exe experiments\run_resumable_qnn_pilot.py --resume --epochs 4
```

The equivalent Bugs4Q core pilot command includes `--task bugs4q_core_source_state` before `--resume`. For task-specific runs, pass the task identifier before `--resume`; the QBugs QASM command is shown below.

The pilot is complete now, so this command is only for a future interrupted NA run. For core or Robust, pass the task identifier before `--resume`:

```powershell
.venv\Scripts\python.exe experiments\run_resumable_qnn_pilot.py --task bugs4q_core_source_state --resume --epochs 4
.venv\Scripts\python.exe experiments\run_resumable_qnn_pilot.py --task bugs4q_robust_execution --resume --epochs 4
```

For the QBugs QASM pilot, resume with:

```powershell
.venv\Scripts\python.exe experiments\run_resumable_qnn_pilot.py --task qbugs_qasm_v2_source_state --resume --epochs 4
```

For QMutBench full-development training, only after the memory preflight passes, run one bounded chunk per invocation and add `--resume` on later invocations:

```powershell
.venv\Scripts\python.exe -m qmutbench_qml.training.run_binary_development --mode full-development --max-chunks-per-invocation 1
.venv\Scripts\python.exe -m qmutbench_qml.training.run_binary_development --mode full-development --max-chunks-per-invocation 1 --resume
```


## Duplicate-aware platform pilot

The original Quantum platform comment task remains blocked because exact comment text repeats across issue folds. The independent v2 task created issue/comment connected components before a stratified five-fold split: 319 accepted, 304 groups, 309 exact comment fingerprints, 17 duplicate rows, zero contradictory labels among identical comments, and zero exact comment fingerprints crossing folds. The pilot stopped after epoch 2 and resumed to epoch 4 across all five folds; optimizer, RNG, and model-state parity matched the uninterrupted reference. It did not open the frozen test set.

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| QNN, training-fold balanced loss | 0.6207 | 0.6071 | 0.5894 |
| Balanced logistic regression | 0.8527 | 0.8768 | 0.8393 |

QNN minus classical macro-F1 was -0.2500, with paired group-bootstrap 95% interval [-0.3163, -0.1836]. The pilot took 0.961 seconds, with available memory 712.2 MiB at start and 702.5 MiB at end (500 MiB floor). This is a development OOF result for curated issue comments.


## QBugs OpenQASM variant and resumable pilot

The original 1,500-row QBugs feature representation omitted gate parameters. Analysis found 703 simplified token sequences shared by both labels, and pair-plus-token connected components collapsed all rows into contradictory-label groups. OpenQASM 2 preserves the gate parameters: after serialization, only two exact-QASM fingerprints remained contradictory, affecting four rows in two complete source pairs. Those pairs were explicitly rejected with reasons; the remaining 1,496 rows form 748 balanced correct/buggy pair groups. Five stratified grouped folds contain 149 or 150 groups each, with no exact QASM crossing folds. The source pickle hashes, derived input, fold file, and rejection list all verified.

The development pilot stopped after epoch 2 and resumed to epoch 4 across all five folds. Checkpointed parameters, Adam state, and RNG state matched the uninterrupted reference; frozen test access was false. On the same five splits and 8 hashed-QASM features, QNN accuracy/balanced accuracy/macro F1 were all 0.6845. Balanced logistic regression scored 0.7467/0.7467/0.7464. QNN minus baseline macro F1 was -0.0622 (paired group-bootstrap 95% interval [-0.0843, -0.0394]). The completion invocation took 2.25 seconds; available RAM was 1,295.9 MiB at start and 1,281.4 MiB at end, above the 500 MiB floor. The final manifest records this invocation's resource usage; it does not retain combined wall time for the earlier paused invocation.

A limited 100-row preprocessing run first exposed a greedy split allocator that left two folds empty. That development output is preserved and marked `INVALID_DEVELOPMENT_SPLIT_FOLD_COVERAGE`. The corrected allocator was verified on a second 100-row sample with all five folds populated before processing the complete source. The splitter now rejects any empty or single-class fold. The generic source-runner manifest also had stale boilerplate claiming that these task-specific folds were reused from QNN OOF files; a hash-bound provenance correction sidecar records the actual generated split source without overwriting the processed manifest.


## Mute-Alg compatibility and semantic audit

A restricted Qiskit 1.4.5 scan read all 169 source files and recorded each raw file hash. Seventy-seven files across five algorithm directories loaded as even `QuantumCircuit` lists; 92 failed with `UnpicklingError: state is not a dictionary`. The [official HornBro README](https://github.com/JanusQ/HornBro) defines the source-pair order: originals in the first half and buggy circuits in the second half. Using that documented target, OpenQASM 2 serialization, pair grouping, and exact-QASM component checks produced a separate compatibility-selected task variant. Of 7,160 rows from loadable files, 1,720 rows in 860 complete pairs were rejected because five exact QASM fingerprints had contradictory labels. The 5,440 retained rows form only 11 exact-QASM/pair components across five folds. The data and splits pass checksums and row accounting, but this low diversity and compatibility-based selection make the subset **ineligible for representative model training**. The full 169-file task remains blocked until compatible deserialization and broader independent circuits are available.

One limited QNN-family sample and a five-file sample failed before output because all examples collapsed to fewer than five duplicate-safe groups; no training was run on this task. The complete partial-source compatibility scan, unreadable-file ledger, rejected-pair ledger, and source-isolated partitions are preserved.


The most recent read-only QMutBench memory preflight found 1,294 MiB available, versus 2,048 MiB required for the sampled pilot and 2,076 MiB for full-development. The established memory guard therefore remains active; no training job was started.


## QML platform workbook audit

The OOXML schema audit parsed all 39 extracted workbooks (one sheet each); the source package also contains two nested archives. The integrated `Classified/Conclusion.xlsx` identifies a manual target of 0=not quantum and 1=quantum-related, distinct from its separate agreement field. Of 481 data rows, 339 unique issue URLs have valid targets (220 class 0, 119 class 1); 137 labels are blank, two cells contain `Algorithm`, one contains `User`, and two repeated issue rows were deduplicated after confirming identical titles and labels. Exact duplicate issue titles were grouped with repository IDs; ten repositories collapse to nine components, with one 264-row component. This provides a concrete target/schema audit, but the local inventory records no data reuse license, and the repository-level grouping is too concentrated for a convincing cross-repository evaluation. The task remains blocked from training. Detailed evidence is in `qml_platform_workbook_schema_audit.json` and `qml_platform_conclusion_label_audit.json`.

## 2026-10-29 addendum: QMutBench three-class checkpoint recovery

The 2026-10-27 three-class experiment remains historically OOF-only and had no saved model parameters. A separate versioned 12-epoch recovery now has nine Adam/RNG checkpoints for three configurations over the same 205-row / 12-origin / three-fold development split. All 615 OOF probability rows replayed within 2e-7. The companion `training_artifact_audit.json` preserves the original finding and records this recovery independently.

The matched fold-local controls were Logistic Regression (46.34% accuracy / 44.07% macro F1), RBF SVC (43.41% / 42.46%), and Extra Trees (48.78% / 46.08%). Best recovered QNN was angle re-upload (45.85% / 44.49%). Its macro-F1 difference versus Extra Trees was -1.59 points (12-origin bootstrap 95% interval -12.08 to +6.72 points). This remains exploratory historical mutation-outcome classification, separate from binary QMutBench and hardware fault diagnosis. Full binary QMutBench training remains behind the unmet memory preflight.


## Addendum: QMutBench binary full-development streaming status

The source matrix now records the stream training as partial and resumable. The separate chunk-16 run contains 51 audited checkpoints in fold 1 (26,112 train-record exposures, 816 updates), but no complete fold or OOF metrics yet. A later resume was stopped by the runner before data loading at 262 MiB available vs 2,076 MiB required. The runner's 1,024 MiB in-run guard remains unchanged. See the [execution follow-up](execution_followup_20261029.md) and the versioned run's `progress_audit.json`, `resume_attempt_history.json`, and `full_campaign_manifest.json`.
