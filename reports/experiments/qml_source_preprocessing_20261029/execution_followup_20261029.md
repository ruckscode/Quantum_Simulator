# Execution follow-up — 2026-10-29

## Reconciled state

The current campaign matrix has 18 task rows across 19 inventory entries. The semantic source-task audit reports 17 manifests/indexes verified, with zero review-required entries. The six original completed preprocessing manifests still record 2,595 accepted rows; their artifacts were preserved. Seven existing deterministic QNN pilots retain verified interruption/resume parity. No pilot was started in this follow-up.

The original campaign matrix and the additional task-variant table are preserved in `source_preprocessing_matrix.md`. They remain the authoritative row-by-row status record; this follow-up adds only new bounded-validation evidence.

## New bounded Mute-Alg check

The restricted Qiskit 1.4.5 compatibility loader now has narrowly scoped handlers for three observed legacy pickle states: `OuterCircuitScopeInterface`, `CircuitInstruction`, and the removed `ParameterTable` class. Compilation succeeded. A five-pickle sample stopped after duplicate/contradictory-QASM filtering left no accepted pairs. A 25-pickle sample reached fold construction but failed the five-fold group requirement: only one duplicate-safe connected component remained. Therefore this shim has not established full-source compatibility or Mute-Alg training eligibility. The existing partial 77/169-file manifest remains untouched and is still ineligible (5,440 rows, 11 groups); no new source task or model run is claimed.

The 25-file sample command was:

```powershell
.qbugs1_env\Scripts\python.exe experiments\prepare_mute_alg_compatible_subset.py --limit-pickles 25
```

It exited at `assign_folds` with `RuntimeError: Only 1 duplicate-safe groups; five folds require at least five`. The five-file diagnostic command was:

```powershell
.qbugs1_env\Scripts\python.exe experiments\prepare_mute_alg_compatible_subset.py --algorithm ae --limit-pickles 5
```

It exited because no structurally valid pairs remained after semantic contradiction filtering. Neither failed sample wrote a processed dataset or replaced the existing partial artifacts.

## QMutBench and training state

The established QMutBench binary development stream/index remains validated at 578,117 development rows. The frozen test was not opened. Current `memory_preflight(DEFAULT_CONFIG)` fails closed at 721 MiB available, below its 2,048 MiB estimated requirement and 1,024 MiB safety floor. No QMutBench training or full-data expansion was started.

The existing QNN pilot and matched-classical artifacts remain the current completed end-to-end pilots; their per-task metrics and resume status are listed in `source_preprocessing_matrix.md`. No new matched comparison is claimed for this follow-up.

## Verification

- Semantic audit: 17 tasks verified; 0 review required.
- Focused tests: 16 passed (`test_resumable_qnn_pilot.py`, `test_source_isolated_preprocessing.py`, `test_qmutbench_streaming.py`).
- Mute compatibility script `py_compile`: passed.
- `git diff --check`: passed; Git emitted only existing LF-to-CRLF notices.
- QMutBench memory preflight: failed safely as expected at 721 MiB available / 2,048 MiB required.

## Resume and next actions

- Retry the QMutBench full-development stream only after the preflight passes; use one bounded chunk and `--resume` on subsequent invocations:

```powershell
.venv\Scripts\python.exe -m qmutbench_qml.training.run_binary_development --mode full-development --max-chunks-per-invocation 1
.venv\Scripts\python.exe -m qmutbench_qml.training.run_binary_development --mode full-development --max-chunks-per-invocation 1 --resume
```

- Rebuild the existing Mute partial subset with the original command only into a new versioned output path; do not overwrite the preserved v1 artifacts. Before doing so, resolve whether QASM-equivalent programs have a source-supported shared target or whether the task is intrinsically underdetermined after duplicate-safe grouping.
- Continue the unprocessed source rows independently, preserving source/task contracts, frozen splits, and the simulator diagnosis acceptance issue (7 mismatches among 45 outcomes).

## Additional local source validation

A read-only validator now records source hashes and per-row dispositions for the remaining local Bug-Pattern and QChecker examples in `bug_pattern_qchecker_source_audit.json`.

- Bug-Pattern: 9 local notebooks across four categories. All 9 remain unresolved for the proposed supervised classification task because titles/markdown expose category context, local source has no clean negatives, and independent category examples are too sparse for a defensible grouped holdout. README-linked external issue examples are not local training rows.
- QChecker: 31 local instance files across nine directory labels; 30 parse, and `DO/case2.py` is rejected for a syntax error. The other 30 remain unresolved because the source gives directory taxonomy but no verified detector result per row or clean negatives; class coverage is sparse, and the README's `MI` code conflicts with the local `IM` directory.
- The matrix builder now includes these specific reasons. Both tasks remain `INELIGIBLE_UNVALIDATED`; no labels were inferred from execution and no source files were run.

Commands:

```powershell
.venv\Scripts\python.exe experiments\audit_bug_pattern_qchecker_sources.py
.venv\Scripts\python.exe experiments\build_qml_preprocessing_status.py
```

The audit script compiled and ran successfully; `git diff --check` passed with only existing line-ending notices.

## Full Mute-Alg source validation and resumable training pilot

The Qiskit 1.4.5 restricted compatibility loader was expanded only for three observed legacy object states, then run against the complete HornBro Mute-Alg source. It loaded and normalized all 169/169 source pickle files (all source-file hashes were rechecked). It produced 16,360 pair records. Exact OpenQASM fingerprints with contradictory original/buggy labels caused complete source pairs to be rejected: 7,220 records in total. The retained 9,140 rows have balanced classes (4,570 each), 812 pair/QASM-connected groups, and five deterministic group-exclusive folds. The old partial 5,440-row, 11-group subset remains untouched and ineligible.

The source-isolated runner wrote 72 checksummed partitions (9,140 rows total), a checksummed split manifest, and a resumable processed manifest. It completed at chunk size 128 in 2.328 s; sampled process RSS peaked at 29 MiB and system available RAM moved from 1,130.2 to 1,046.1 MiB. The semantic audit reports this task VERIFIED with no exact feature duplicates crossing folds and no group crossing folds.

A deterministic five-fold QNN pilot then trained for two epochs, checkpointed, and resumed to four epochs. Resume verification passed for all five folds, including parameter, Adam state, and RNG parity against uninterrupted references. All checkpoint/history hashes matched their receipts. It used 8-dimensional deterministic hashed OpenQASM tokens, training-fold-only scaling, a 3-qubit amplitude model, Adam, batch size 64, seed 28102026, and 4 total epochs. Frozen-test access was false.

On identical development OOF IDs, labels, group folds, and eight hashed features, the QNN achieved accuracy/balanced accuracy 0.5071 and macro F1 0.4894. Balanced logistic regression achieved 0.5002 and macro F1 0.4997. Group-bootstrap QNN minus classical macro-F1 difference was -0.0115 (95% percentile interval -0.0694 to 0.0516 over 812 groups); this pilot does not show a reliable QNN advantage. The QNN gradient check passed with absolute error 2.63e-12 in each fold. Final resume invocation elapsed 9.085 s; available RAM changed from 988.0 to 926.9 MiB, with 927.4 MiB the lowest recorded per-fold available RAM.

Reproduce the scan/partition path in a fresh versioned output, or resume the completed pilot if interrupted:

```powershell
.qbugs1_env\Scripts\python.exe experiments\prepare_mute_alg_compatible_subset.py
.venv\Scripts\python.exe experiments\prepare_source_isolated_task.py --contract experiments\source_task_contracts\mute_alg_compatible_subset_qasm_v2_qiskit_compat_v2.json --chunk-size 128
.venv\Scripts\python.exe experiments\run_resumable_qnn_pilot.py --task mute_alg_hornbro_qasm_v2 --stop-after-epochs 2
.venv\Scripts\python.exe experiments\run_resumable_qnn_pilot.py --task mute_alg_hornbro_qasm_v2 --resume --epochs 4
```

The preprocessing command refuses to overwrite its completed nonempty output; use a new version tag for a fresh source scan. The verified run artifacts are in `reports/experiments/resumable_qnn_pilot_mute_alg_qasm_v2_20261029/`. The QMutBench binary preflight still fails safely at 1,097 MiB available versus 2,048 MiB estimated required, so its full-development QNN run remains unstarted. The simulator's 7/45 diagnosis mismatch remains a separate unresolved issue.

## Latest post-pilot verification

The refreshed source matrix still reconciles 19 inventory entries and 18 original task rows, and now records Mute-Alg v2 as 9,140 rows / 812 groups / `READY_FOR_TRAINING` with a completed resumable pilot. The older Mute v1 subset remains listed separately as ineligible. The semantic audit reports 18/18 task manifests verified and zero review-required tasks. Rechecked all 169 raw Mute pickle hashes, all 72 processed partition hashes, the input/fold hashes, 9,140 matching QNN/classical OOF IDs, groups, labels, and folds, and all ten checkpoint/history hash receipts. Resume parity covers model parameters, Adam moments, and RNG state.

After registering Mute-Alg in the pilot runner and test suite, the focused command `.\venv\Scripts\python.exe -m pytest tests\test_resumable_qnn_pilot.py tests\test_source_isolated_preprocessing.py tests\test_qmutbench_streaming.py -q` passed (16 tests). Compilation passed for the modified preparation, pilot, audit, and matrix scripts. `git diff --check` passed with only existing line-ending notices. The latest QMutBench preflight failed closed at 1,097 MiB available versus 2,048 MiB estimated required; no QMutBench training was launched.

Upstream provenance check: GitHub's official HornBro repository page lists an MIT license and the README confirms qcs.pkl pair order. The local snapshot omits the root LICENSE, README, and exact commit ID; source file hashes pin the imported data. This limitation and the upstream link are recorded in source_provenance.json.

## Nine-source classical screen refresh and resource stop

A new input-builder reads the nine validated source-isolated task contracts and manifests, verifies input and partition hashes, and writes compact per-task adapters with only record ID, source group, target, and permitted model text. The adapters cover 5,055 rows in these nine contracts: Bugs4Q core, Bugs4Q-NA, pooled Bugs4Q, QBugs OpenQASM v2, quantum-platform comments v2, Qolumbina ideal detection, exploratory LintQ context v2, Bugs4Q-Robust execution v2, and exploratory QSimBench fingerprinting. It preserves the LintQ same-context label conflict and QSimBench descriptor-collision caveats. Inputs and config are in `classical_nine_source_v2_inputs/`.

The broad 20-model/five-fold suite was attempted with this config. Its preflight stopped before creating output because only 641.7 MiB was available; the suite requires 1,200 MiB to preserve its existing 500 MiB floor. No suite output was written and no guard was changed. Existing results for the prior eligible five-task suite remain intact.

Retry when the machine has enough free memory:

```powershell
$env:QML_CLASSICAL_CONFIG_PATH = 'reports/experiments/qml_source_preprocessing_20261029/classical_nine_source_v2_inputs/classical_nine_source_v2_config.json'
.venv\Scripts\python.exe experiments\run_grouped_classical_suite.py
```

The status matrix now records the validated nine-task adapters and this exact memory blocker.

## Additional resumable Qolumbina pilot

Registered `qolumbina_ideal_seeded_mutant_detection` in the source-specific pilot runner with its existing source contract, 182-row processed manifest, and 26 source-program groups. The pre-training loader verified input, partition, and saved split hashes, all five folds, and binary class coverage (32 label 0; 150 label 1). The task predicts DOSS majority detection on ideal-backend report seeds; it is not a repair ground truth or hardware diagnosis.

The 3-qubit amplitude QNN used eight hashed text features, training-fold-only scaling, Adam, batch size 64, seed 28102026, and four epochs. It checkpointed after epoch 2 and resumed to completion with one resume event. All five checkpoint/history hashes and model/optimizer/RNG parity receipts verified. Frozen-test access was false.

On the same 182 OOF IDs, targets, groups, saved folds, and features, QNN accuracy was 0.8242, balanced accuracy 0.5000, and macro F1 0.4518. Balanced logistic regression accuracy was 0.5220, balanced accuracy 0.4519, and macro F1 0.4303. Group bootstrap difference in macro F1 was +0.0252 (95% interval -0.0579 to +0.1157 across 26 groups): the result does not establish a reliable advantage. The high raw QNN accuracy alongside chance balanced accuracy reflects the imbalanced target and must not be read as strong detection.

The resumed invocation took 0.634 s; available RAM moved from 874.8 to 837.4 MiB against the unchanged 500 MiB pilot floor. Resume command:

```powershell
.venv\Scripts\python.exe experiments\run_resumable_qnn_pilot.py --task qolumbina_ideal_seeded_mutant_detection --resume --epochs 4
```

The task is included in the refreshed pilot matrix. The semantic audit again reports 18/18 task manifests verified and zero review-required; focused tests passed (16).

## Latest classical-suite retry observation

After the Qolumbina pilot, available RAM briefly measured 1,214.6 MiB, but the nine-task classical runner's immediate preflight still failed its 1,200 MiB gate before creating outputs. A post-failure sample measured 1,174.7 MiB, and the current check is below 1,000 MiB; Windows memory compression, OneDrive, Chrome, and editor processes occupy much of the system memory. The failed retry does not justify changing the guard. The validated 5,055-row adapters and their hashes remain ready for a later run.


## Nine-source classical screen completed and audited

The previous memory-limited attempts stopped before output creation. On a later run, the same configured 1,200 MiB startup gate passed at 1,857.4 MiB, and the unchanged 500 MiB in-run floor was retained. The nine-source suite completed all 9 tasks, producing 180 model/task summaries and 900 fold-model rows. Its audit manifest verifies 9 task directories and reports no issues. The independently replayed grouped OOF audit is `VERIFIED`. The run took 748.05 seconds summed across tasks; maximum sampled task-process RSS was 154.97 MiB; system-available RAM was 2,034.4 MiB at runner completion. Frozen tests were not accessed.

Multiple MLP fits emitted `ConvergenceWarning` at the configured 200 iterations. Those results were retained without post-hoc tuning. The complete raw console stream was not saved, so the exact warning count is unavailable. The execution record documents this limitation. The output is a grouped classical screen, not a matched QNN comparison.

Artifacts: `reports/experiments/local_grouped_classical_suite_nine_source_v2_20261029/` (including `execution_record.json`, `audit_manifest.json`, aggregate result and fold CSVs, per-task OOF files, and report). Audit command: `.venv\Scripts\python.exe experiments\audit_nine_source_classical_suite.py`.

## Source-restricted Bugs4Q transfer pilots verified

The generic mixed-source transfer run is explicitly invalidated by `task_validity_correction.json` and excluded from valid transfer summaries. New versioned pilots now enforce training on only the declared source in every fold, apply preprocessing only to permitted training rows, and evaluate all held-out groups while reporting the other source separately. The transfer audit verifies 62 groups, identical paired OOF IDs/targets/groups/folds for QNN and logistic regression, source-only training in every fold, checkpoint hashes, resume status, and no frozen-test access. Audit status is `VERIFIED` with no issues.

- Core-only training ? Bugs4Q-NA transfer: QNN macro F1 0.5479, accuracy/balanced accuracy 0.5484; logistic macro F1 0.4988, accuracy/balanced accuracy 0.5000. Paired 31-group bootstrap QNN-minus-classical macro-F1 difference +0.0512, 95% interval [-0.0480, +0.1516].
- NA-only training ? Bugs4Q-core transfer: QNN macro F1 0.5156, accuracy/balanced accuracy 0.5161; logistic macro F1 0.5141, accuracy/balanced accuracy 0.5161. Paired 31-group bootstrap difference +0.0007, 95% interval [-0.0650, +0.0684].

Both used the deterministic 2-epoch stop and resume to 4 epochs; parameter, optimizer, and RNG continuity were verified. Exact resume commands:

```powershell
.venv\Scripts\python.exe experiments\run_resumable_qnn_pilot.py --task bugs4q_core_only_transfer_source_restricted_v2 --resume --epochs 4
.venv\Scripts\python.exe experiments\run_resumable_qnn_pilot.py --task bugs4q_na_only_transfer_source_restricted_v2 --resume --epochs 4
.venv\Scripts\python.exe experiments\audit_resumable_qnn_transfer_pilots.py
```

## QMutBench and final focused validation

QMutBench's streaming infrastructure remains intact; no full development run was started because the latest checked preflight was below its unchanged 2,048 MiB startup estimate and 1,024 MiB in-run floor (full-development estimate 2,076 MiB). At the next run, recheck the preflight; do not force past the memory gate. QMutBench's historical mutation outcome remains separate from hardware diagnosis, and its frozen test was not accessed.

The focused command `.venv\Scripts\python.exe -m pytest tests\test_resumable_qnn_pilot.py tests\test_source_isolated_preprocessing.py tests\test_qmutbench_streaming.py -q` passed: **18 passed in 3.45 s**. Current available memory at the follow-up sample was 790.6 MiB. The simulator's 7/45 diagnosis mismatch remains a separate unresolved acceptance issue.


A later explicit QMutBench runner preflight sampled 1,612 MiB available and failed closed: sampled-pilot estimate 2,048 MiB (1,024 MiB required floor), full-development estimate 2,076 MiB. No QMutBench QNN job launched. The runner's documented continuation command, after preflight passes, is `.venv\Scripts\python.exe -m qmutbench_qml.training.run_binary_development --mode full-development --resume`; it checkpoints chunks and can be bounded with `--max-chunks-per-invocation N`.


## Qolumbina multi-backend extension completed

The original 18-row matrix keeps its ideal-backend task unchanged. A separately versioned extension uses the existing 546-row source manifest for ideal, fake_algiers, and fake_brooklyn. It verifies all 546 labels against the source's majority-of-report-seeds rule, has 26 original-program groups, and reuses the existing five-fold program assignment; all variants and backend observations for one program remain in the same fold. The source-isolated runner accepted 546/546 rows, wrote nine checksummed partitions and a checksummed 546-row split manifest, and found no duplicate IDs or cross-fold groups. The empirically measured target is seeded-mutant detection, not semantic repair truth or hardware diagnosis. fake_brooklyn has only one negative row; fake_algiers has seven, so per-backend estimates are highly imprecise.

Two deterministic 3-qubit amplitude pilots trained for four epochs in batch size 64, each with a two-epoch stop and verified resume. Both have 26 groups, matched logistic OOF on the same rows, features, and folds, and pass the independent audit (`pilot_audit.json` and `pilot_audit_balanced.json`). Frozen-test access is false. The backend name is included as a pre-run hashed text feature; report outputs and detection rates are excluded. Scalers fit on training folds only.

| Variant | QNN accuracy / balanced accuracy / macro F1 | Matched balanced logistic accuracy / balanced accuracy / macro F1 | Paired 26-group bootstrap QNN?classical macro-F1 difference (95% interval) |
|---|---|---|---|
| Unweighted BCE | 0.9267 / 0.5000 / 0.4810 | 0.4231 / 0.4700 / 0.3445 | +0.1408 [ +0.0495, +0.2375 ] |
| Fold-local class-balanced BCE | 0.5568 / 0.5076 / 0.4161 | 0.4231 / 0.4700 / 0.3445 | +0.0741 [ ?0.0474, +0.1893 ] |

The unweighted QNN predicts every row positive; its high accuracy is an imbalance artifact and its balanced accuracy is 0.50. The balanced-loss run raises balanced accuracy only slightly above chance and its bootstrap interval includes zero. These development results do not establish a dependable advantage or backend fault attribution. They remain development experiments, not confirmation results.

Resume commands:

```powershell
.venv\Scripts\python.exe experiments\run_resumable_qnn_pilot.py --task qolumbina_multibackend_seeded_mutant_detection --resume --epochs 4
.venv\Scripts\python.exe experiments\run_resumable_qnn_pilot.py --task qolumbina_multibackend_seeded_mutant_detection_balanced --resume --epochs 4
.venv\Scripts\python.exe experiments\audit_qolumbina_multibackend_pilot.py
.venv\Scripts\python.exe experiments\audit_qolumbina_multibackend_pilot.py --run-dir reports/experiments/resumable_qnn_pilot_qolumbina_multibackend_balanced_20261029
```

The resume invocations recorded 705.8?693.2 MiB available for the unweighted run and 643.2?629.3 MiB for the balanced run; both stayed above the unchanged 500 MiB pilot floor. Epoch-history samples show a minimum of 597.8 MiB and 511.4 MiB, respectively. The final resume invocation took 0.784 s per metrics record; four-epoch training histories and checkpoint hashes are in each run directory. Compilation passed. Focused tests now pass **19 tests** including registration and validation of both multi-backend tasks. The candidate-sweep audit verified 9 tasks, 90 checkpoints, and 90 gradient checks; the matched comparison audit verified 18 QNN/classical result rows.


## Repository-wide training artifact reconciliation

The repository-wide existing-training audit completed for nine prior QNN runs with zero issues (`reports/experiments/qmutbench_multisource_qml_campaign_20261028/training_artifact_audit.json`). It rescored saved OOF outputs and reproduced QNN checkpoint predictions; it reports `frozen_test_opened: false` and `production_modified: false`. Its QMutBench three-class section explicitly says OOF training artifacts exist but there is **no real-data model parameter checkpoint**. The refreshed matrix now records that row as `OOF_RESULTS_PRESENT_CHECKPOINT_UNVERIFIED`, preserving the reported OOF metrics without claiming reproducible QNN training completion.

The latest QMutBench QNN preflight after these local pilots sampled 771 MiB available and failed closed against the 2,048 MiB sampled-pilot requirement and 2,076 MiB full-development requirement, both with the 1,024 MiB reserved floor. The full QMutBench streaming QNN campaign remains unrun.


## Full repository test suite after the path fix

The first full run exposed a Windows-only path separator mismatch: `ml_evaluation.load_dataset` emitted backslashes where the test and persisted report schema expect POSIX-relative paths. The same normalization issue existed in the disposition loader. Both now use POSIX separators for relative and absolute report paths. The failing module passed 6/6 tests after the fix, focused QML tests passed 19/19, and the rerun of the full suite completed **954 passed, 2 skipped in 871.87 s**. The earlier failing run was 953 passed, 2 skipped, 1 failed; that failure is resolved and did not involve QML training behavior. `git diff --check` reports only existing LF/CRLF notices.

Immediately after the full suite, the QMutBench memory preflight still failed closed: 1,315 MiB available versus 2,048 MiB sampled-pilot estimate and 2,076 MiB full-development estimate, reserving the unchanged 1,024 MiB floor. No full QMutBench training was launched.


Final preflight recheck sampled 1,680.5 MiB available: still below both QMutBench gates (2,048 MiB sampled-pilot, 2,076 MiB full-development, with 1,024 MiB reserved).

## Recovered resumable architecture sweeps (2026-10-29 continuation)

The historical two-task 20-epoch sweep status was reconciled without overwriting its stopped attempt. A versioned runner completed six QNN configurations over five GroupKFold folds on Bugs4Q-Robust (2,220 rows/37 groups) and QSimBench (300 rows/150 groups). The independent audit reports `VERIFIED`, no issues, and confirms source/split hashes, OOF coverage, gradient checks, checkpoints, matched controls, and no frozen-test access.

Matched fold-local balanced LogisticRegression baselines use the same hashed features and train-only scaling. Bugs4Q-Robust (8-feature control) reached accuracy/macro-F1/balanced accuracy 60.81%/60.75%/60.81%; its best QNN was amplitude 3q depth1 at 54.05%/53.75%/54.05%. The 4-feature control was 58.11%/58.04%/58.11%. QSimBench (8-feature control) reached 65.33% on all three metrics; the best QNN was amplitude 3q depth1 at 67.33% on all three. Its 4-feature control was 65.00% on all three. The six matched comparisons and 1,000-draw group bootstrap intervals are in each task's `matched_classical_comparison.json`; they remain exploratory development OOF.

Robust created 21 new checkpoints with optimizer and RNG state. Nine old epoch-20 weights were reused for inference only because their format lacked optimizer/RNG state; the legacy epoch-17 fold was retrained from scratch. One epoch-6 history loss is missing after a transient OneDrive file-replacement lock; `training_history_gap.json` records the gap, while the checkpoint with optimizer/RNG state was saved and resumed. QSimBench created 30 new resumable checkpoints. Runtime was 164.95 s and 58.39 s, respectively; recorded fold-level available RAM ranged 1,983.7–2,111.9 MiB and 2,040.7–2,121.1 MiB, respectively, with the existing 500 MiB runner floor unchanged.

The consolidated benchmark now has 56 candidate rows across 12 tasks, and its generated report contains these sweeps. Status JSON preserves the original 473.4 MiB stop as historical evidence and records the completed versioned runs separately. Verification after consolidation: audit `VERIFIED`; Python compilation passed; focused `tests/test_resumable_qnn_pilot.py` **7 passed**. The earlier repository-wide suite remains **954 passed, 2 skipped** from the previously recorded run.

Resume/recompute commands:

```powershell
.venv\Scripts\python.exe experiments\run_resumable_qnn_hyperparameter_sweep.py --task bugs4q_robust_execution --resume
.venv\Scripts\python.exe experiments\run_resumable_qnn_hyperparameter_sweep.py --task qsimbench_backend_fingerprint --resume
.venv\Scripts\python.exe experiments\evaluate_resumable_qnn_hyperparameter_sweep_classical.py --task all
.venv\Scripts\python.exe experiments\audit_resumable_qnn_hyperparameter_sweep.py --task all
.venv\Scripts\python.exe experiments\build_qml_accuracy_benchmark.py
```

The QMutBench binary QNN gate remains unmet: latest recorded availability was 1,680.5 MiB, below both the 2,048 MiB sampled pilot and 2,076 MiB full-development estimate with a 1,024 MiB reserve. No full QMutBench QNN training was launched. The existing 7/45 simulator diagnosis mismatch remains separate and unresolved.

## QMutBench three-class resumable checkpoint recovery

The prior three-class campaign's OOF archive had no reloadable checkpoint. To close that reproducibility gap, a separate versioned 12-epoch run retrained its three QNN configurations on the same 205-row, 12-origin, three-fold development sample. It wrote nine checkpoints containing Adam moments and RNG state. Training completed in 9.66 s; recorded available system RAM ranged 1,542.2–1,733.4 MiB across epoch samples. The frozen test was not opened.

The recovery audit verified the source, split, fold-local feature manifest, all nine checkpoint schemas/configurations/epoch counts, and exact replay of 615 QNN OOF probability rows (tolerance 2e-7). A new matched evaluation fitted Logistic Regression, RBF SVC, and Extra Trees on the same four fold-local inputs, train-only scaling, origins, and validation IDs. Extra Trees scored 48.78% accuracy / 46.08% macro F1 / 47.97% balanced accuracy. Logistic Regression scored 46.34% / 44.07% / 45.72%; RBF SVC scored 43.41% / 42.46% / 43.12%.

The best recovered QNN, angle re-upload, scored 45.85% / 44.49% / 45.41%. Its macro-F1 difference from Extra Trees was -1.59 points (12-origin paired group-bootstrap 95% interval -12.08 to +6.72 points). QNN variants and controls remain development OOF; no confirmation or frozen-test claim follows.

The refreshed source matrix now distinguishes the original campaign's missing checkpoint from the verified versioned recovery. `training_artifact_audit.json` retains the old campaign finding and includes the supplemental recovery audit; all nine pre-existing QNN model runs still replayed successfully. Python compilation passed; `qmutbench_qml/tests/test_foundations_and_multiclass.py` passed **5 tests**.

Resume an interrupted recovery run with:

```powershell
$env:QUT_OUT_DIR='qmutbench_threeclass_resumable_recovery_20261029'
$env:QUT_EPOCHS='12'
$env:QUT_RESUME='1'
.venv\Scripts\python.exe experiments\train_qmutbench_threeclass_light_qnn.py
.venv\Scripts\python.exe experiments\audit_qmutbench_threeclass_resumable_recovery.py
.venv\Scripts\python.exe experiments\evaluate_qmutbench_threeclass_resumable_recovery_classical.py
```

The matched classical rerun used 0.379 s total, with approximately 0.029 s Logistic Regression, 0.014 s RBF SVC, and 0.225 s Extra Trees fit time. Available RAM changed from 1,920.8 to 1,914.4 MiB; process RSS sampled from 144.3 to 147.8 MiB. Bootstrap calculation was vectorized while retaining 2,000 paired origin resamples; the updated matched-control audit replays and checks all result metrics and row/fold/label alignment.


## QMutBench binary full-development streaming pilot (2026-10-29 continuation)

The existing stream/index/split infrastructure was used; no new full-dataset in-memory load was added. The default 128-batch interval first failed its startup gate at 2,070 MiB vs 2,076 MiB. A subsequent invocation passed narrowly, then hit the existing 1,024 MiB in-run floor during epoch 1 before any chunk checkpoint. That failure committed zero optimizer updates. Windows cleanup also raised a secondary memmap file-lock error. I fixed explicit NumPy memmap closure on training failure and checkpoint pause, added tolerant temporary-directory cleanup, and added a regression test for releasing mapped-file handles.

A separate versioned run at `reports/experiments/qmutbench_binary_qnn_full_development_chunk16_20261029` uses 16-batch checkpoint intervals. It has 51 validated chunks in fold 1, 26,112 training-record exposures, and 816 optimizer updates. All checkpoints passed the circuit gradient and parameter-update checks; progress audit status is `VERIFIED`. The latest saved checkpoint is chunk 51 at epoch 0 / next batch 816. Minimum available RAM recorded at checkpoint boundaries was 1,747.96 MiB; maximum process RSS recorded in checkpoints was 468.03 MiB. Cumulative QNN training time at chunk 51 was 837.31 s; the latest checkpoint recorded 1,929.84 MiB available. No validation metrics exist yet; OOF generation requires completing the training path. Frozen test access remains false.

The runner has resumed from chunk 1 once and reached chunk 51. A later resume attempt failed before training: system availability was 262 MiB against the 2,076 MiB full-development start requirement (a separate process sample observed 224 MiB). The 51 checkpoints remain intact. The Windows process list showed VS Code/Pylance and memory compression as major system memory consumers; I left unrelated processes untouched. The task matrix now marks this task `PARTIAL_FULL_DEVELOPMENT_QNN_RESUMABLE`, not complete.

Resume only after preflight passes:

```powershell
.venv\Scripts\python.exe -m qmutbench_qml.training.run_binary_development --mode full-development --output reports/experiments/qmutbench_binary_qnn_full_development_chunk16_20261029 --chunk-batches 16 --max-chunks-per-invocation 50 --resume
.venv\Scripts\python.exe experiments\audit_qmutbench_full_development_progress.py
```

The failed default attempt is preserved under `qmutbench_binary_qnn_full_development` with zero committed chunks and status `stopped_memory_guard`. Focused Windows memmap cleanup test passed (**1 passed**); the broader related focused set passed **17 tests** before the latest memory pressure. Python compilation and `git diff --check` passed (only line-ending notices).
`progress_audit.json` now includes a computation estimate: the observed 1.026 s/optimizer update extrapolates to about 164.8 hours of QNN training for all five configured folds and eight epochs, excluding preprocessing and evaluation. This is a rate-based estimate from the first fold's initial 816 updates, not a measured completion time; throughput may change.`n
A later focused test rerun under degraded system memory produced **7 passed, 3 stopped at the test suite's existing 1,024 MiB QML floor**. The affected tests were the real three-class fit, trainable-readout fit, and exact full-batch/chunked-gradient fit. Their exceptions were `MemoryError` from that guard before a completed fit; the source-isolated tests passed. A post-run system sample showed 827.2 MiB available. Do not treat those three as code regressions or as passing; rerun them only when the guard can be met.

Latest read-only full-development preflight on 2026-10-29 measured 973 MiB available and failed closed against the 2,076 MiB requirement. The focused three-class tests likewise cannot be rerun until availability exceeds their 1,024 MiB in-test floor. No training was started by this preflight probe.
