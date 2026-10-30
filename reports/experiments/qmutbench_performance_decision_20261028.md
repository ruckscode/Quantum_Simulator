# QMutBench performance decision — 2026-10-28

## Decision

Retain the verified mutation-aware 23-feature Extra Trees classifier as the development incumbent. Its five-fold, origin-grouped OOF macro F1 is **0.793314**. This is **+1.993 percentage points** over the matched legacy 8-feature Extra Trees result (0.773387). No evaluated alternative has a stronger justified result. The 0.85 target was not reached.

This is a model-selection conclusion from existing saved campaign outputs, not a new full-corpus training run. The campaign’s model choice and feature work used development data only. The historically inspected frozen test is not pristine and must not be interpreted as independent confirmation.

## Verified incumbent

- Task mapping: WOO/OPO = 1; Survivor = 0.
- Data: 578,117 development rows, 305 origins; five saved grouped folds.
- Model: ExtraTreesClassifier, 23 mutation-aware features, 100 estimators, max depth 22, minimum leaf 15, `max_features='sqrt'`, `class_weight='balanced_subsample'`, seed 20261025, threshold 0.5, one job in the saved champion metadata.
- Saved model SHA-256: `7532be26b6959e7e62277220f0c0b78e648d723a0be35d2b693d68e1c9ebf085`.
- Pooled OOF: macro F1 0.793314; accuracy 0.793329; balanced accuracy 0.794087; ROC AUC 0.883281; Brier score 0.144080.
- Confusion matrix (true rows 0/1, predicted columns 0/1): `[[226818, 51595], [67885, 231819]]`.
- Class 0 F1 0.791526 (precision 0.769649, recall 0.814682); class 1 F1 0.795102 (precision 0.817952, recall 0.773493).
- Paired origin bootstrap for macro-F1 improvement over legacy 8-feature Extra Trees: +1.13 to +2.98 pp (2,000 resamples). This supports a development-corpus improvement, not external generalization.

The OOF numeric archive was independently pooled and its five fold counts and standard metrics matched the saved champion metadata exactly. Only numeric arrays were loaded for this verification; the object-valued origin array was not deserialized. Source and model hashes are recorded in the existing campaign JSON and model metadata.

## Strong candidate comparison

| Candidate | Grouped development macro F1 | Evidence / disposition |
|---|---:|---|
| Mutation-aware Extra Trees, 23 features | **0.793314** | Five-fold pooled OOF; incumbent. |
| Legacy Extra Trees, 8 features | 0.773387 | Five-fold matched reference; incumbent gain +1.993 pp. |
| Nested model/threshold/calibration strategy | 0.792801 | Below fixed 0.5 reference; rejected. |
| Histogram Gradient Boosting | 0.777170 | Earlier three-fold screen; beat its own matched control but below champion. Not a five-fold paired comparison. |
| Random Forest | 0.758910 | Earlier grouped screen, below matched Extra Trees (0.766280). |
| Logistic Regression tuned | 0.693626 | Earlier three-fold screen; only +0.097 pp versus its matched baseline. |
| Champion/legacy probability blend | 0.785275 | Below champion; rejected. |
| Champion plus eight structure features | Partial clean folds pooled 0.786447 vs matched 0.787772 | Folds 2–4; fold 5 interrupted at low RAM. Rejected; incomplete. |
| Replace-only shallow-tree hybrid | 0.782522 | Complete five-fold saved predictions; below incumbent in every fold. Rejected. |

Values from different protocols are labelled as such and are not pooled as if they were one experiment. Additional mutation-feature ablations are in `qmutbench_final_classical_campaign_20261027.md`. The selected champion OOF archive is authoritative; `data/qmutbench_ml/master_optimization_v1/oof_predictions.csv` is a different three-fold artifact and does not reproduce the five-fold champion, so it was not used for the main result.

## QNN comparison

The completed QNN family screen is a matched, development-only pilot on 72 records from 9 origins, with three GroupKFold splits and four fold-selected/scaled inputs. It is not directly comparable to the 23-feature, 578,117-row champion. On the pilot, VQC/trainable QNN macro F1 was 0.4857 versus matched Logistic Regression at 0.4938 (3-fold mean fold macro F1 0.4637 ± 0.0937 vs 0.4892 ± 0.1397). Data re-uploading scored 0.4688, QSVC 0.4215, and quantum reservoir 0.3466. The pilot did not demonstrate a QNN advantage. The current binary QNN remains untrained on the full dataset, and no full-data quantum gradient or parameter update is claimed.

## Resource decision and training command

At inspection, available physical RAM was **1,695 MiB of 7,488 MiB** (77.4% used); a later pre-final snapshot fell to **1,410 MiB** (81.2% used). Both are below the binary QNN startup gate of 2,048 MiB. No training was launched. Keep the QNN thresholds unchanged: 2,048 MiB at startup and 1,024 MiB in-run floor. The machine has previously reached 458 MiB during a structural trial. Do not repeat blocked training attempts at this memory level.

The QNN campaign has no validated checkpoint to resume (its output directory is empty). When the startup gate passes, run the guarded five-fold development campaign as follows; it will refuse to start below 2,048 MiB and preserves the 1,024 MiB in-run floor:

```powershell
$freeMiB = [int](& .\.venv\Scripts\python.exe -c "import psutil; print(psutil.virtual_memory().available // 1048576)")
if ($freeMiB -ge 2048) {
    .\.venv\Scripts\python.exe -m qmutbench_qml.training.run_binary_development --output reports/experiments/qmutbench_binary_qnn_development
} else {
    Write-Output "Not started: available RAM is $freeMiB MiB; required 2048 MiB."
}
```

If a future run has valid epoch/fold checkpoints, add `--resume` to that same runner command. For the classical campaign runner, resumable status can be inspected with `.\.venv\Scripts\python.exe -m experiments.qmutbench_master_optimization status`; do not start `run --phase all` under the present resource condition. The existing runner has a 450 MiB stage gate and is less conservative than the QNN startup policy, so that runner gate alone is not sufficient authorization for a new full-corpus search.

The existing optimization runner uses atomic result/manifest writes and per-fold outputs. Its 450 MiB internal stage guard is less conservative than the QNN’s fixed 2,048/1,024 MiB gates; it is not justification to launch a new full-corpus run under current headroom. Model profiling indicates the main cost driver is fitting many trees on the full 578k-row, fold-expanded one-hot matrix; bounded one-hot encoding, single-thread fitting, and saved fold predictions are already in use. Any subsampling or smaller estimator budget would change the selected configuration and needs grouped confirmation before adoption.

## Work and remaining limits

- Completed: inspected existing reports, candidate leaderboard, saved champion metadata and OOF results; independently confirmed the pooled champion metrics; checked current memory and runner resource/atomic checkpoint design; reviewed the matched QNN results and dependency availability recorded in the campaign.
- Historical focused validation: 23 targeted tests passed in 3.58 s; relevant experiment scripts passed `py_compile`; campaign JSON and `git diff --check` passed. These are existing recorded outcomes, not tests rerun for this decision note. No full repository test suite was run.
- Not repeated: classical full-corpus fits, nested search, or QNN training, because prior campaigns already evaluated the viable candidates and current RAM is below the QNN gate. No new candidate result is claimed.
- Not verified as independent evidence: the historical frozen test was reused in earlier research; no pristine untouched-origin holdout is available.
- Remote backup: the dedicated QNN backup branch was verified in the earlier recovery audit; no performance files were pushed or committed for this task.
- Workspace source, models, datasets, recovery bundles, and existing untracked entries were not modified or deleted. This report is the only new file for this performance decision.

Primary records: `reports/experiments/qmutbench_final_classical_campaign_20261027.md/.json`, `reports/experiments/qmutbench_qml_advantage_campaign_20261027/research_report.md`, and `reports/experiments/qmutbench_replace_specialist_20261027/research_report.md`.


