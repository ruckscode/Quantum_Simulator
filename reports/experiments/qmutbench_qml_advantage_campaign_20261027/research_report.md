# QMutBench QML advantage campaign - 2026-10-27

## Verdict

**No predictive QML advantage was established.** The highest QML macro F1 was VQC/trainable QNN at 0.4857 on the existing exploratory pilot. It remained below matched Logistic Regression at 0.4938. Data re-uploading reached 0.4688, QSVC 0.4215, and the quantum reservoir 0.3466. A fixed 50:50 re-uploading/Logistic blend was worse at 0.4178. No QML model or hybrid is promoted. This is a negative finding for the completed small pilot, not proof that all QML approaches are incapable of improvement.

The task is the repository's documented **binary** mutation outcome: WOO and OPO map to detected (1), Survivor maps to undetected (0). This follows the production model metadata and QMutBench README; the QML pilot did not collapse a specified three-class task. The 85% target is not a QML stopping criterion.

## Protocol, data, and integrity

The existing pilot uses a fixed seeded sample of 9 development origins and 8 records per origin: 72 records total, 30 survivor and 42 detected. The sample/split seed is 20261026. Three-fold `GroupKFold` holds out 3 origins/24 records in each fold and trains on 6 origins/48 records; fold-specific model seeds use the saved seed plus fold number. The same record IDs, labels, origins, folds, four transformed inputs, and prediction-time source features were used for each quantum family and matched classical control. OOF IDs, labels, and origins were checked against the verified development-only 578,117-record/305-origin manifest; all saved model predictions align exactly with the 72 held-out IDs. Origin overlap is zero in each fold. Feature selection by mutual information, categorical encoding, and scaling were fit within each training fold. `Killed`, `Origin_program`, and row IDs were not predictors.

The 23 feature incumbent has four scalar inputs selected independently within each pilot training fold:

| Fold | Selected encoded inputs |
|---:|---|
| 1 | qasm_diff_removed_arity, qasm_diff_orig_span, qubits, qasm_diff_expected_gate_match |
| 2 | qasm_diff_is_local, gates, algorithm_qgan dummy, dominant_state_no dummy |
| 3 | qasm_diff_exact_single_edit, algorithm_qgan dummy, qasm_diff_added_arity, gates |

Selection is fold-local, but this three-fold pilot has only nine groups and 72 records. The extension reused the VQC, QSVC, and control predictions from the same pilot, and ran only the two additional circuit families. Neither pilot read frozen-test records or labels. The historically reused 144,962-record test was not accessed for this campaign and remains retrospective only.

## Matched results

Pooled three-fold out-of-fold performance on the identical 72 records:

| Model | Macro F1 | Accuracy | Balanced accuracy | F1 class 0 / class 1 | Fold macro F1 mean +/- SD |
|---|---:|---:|---:|---:|---:|
| vqc | 0.4857 | 0.5000 | 0.4857 | 0.400/0.571 | 0.4637 +/- 0.0937 |
| qsvc | 0.4215 | 0.4306 | 0.4214 | 0.349/0.494 | 0.4103 +/- 0.1261 |
| data_reupload | 0.4688 | 0.4861 | 0.5310 | 0.565/0.373 | 0.4547 +/- 0.2072 |
| quantum_reservoir | 0.3466 | 0.3750 | 0.4214 | 0.483/0.211 | 0.3231 +/- 0.0843 |
| logistic_4f | 0.4938 | 0.5000 | 0.4952 | 0.438/0.550 | 0.4892 +/- 0.1397 |
| rbf_svc_4f | 0.4448 | 0.4722 | 0.4476 | 0.321/0.568 | 0.4455 +/- 0.1220 |
| extra_trees_4f | 0.3461 | 0.3472 | 0.3643 | 0.373/0.319 | 0.3146 +/- 0.0588 |

Logistic Regression is the strongest matched classical control at macro F1 0.4938. QML minus Logistic paired origin-bootstrap 95% intervals (500 resamples of only 9 validation origins) were:

| QML family | Delta macro F1 | 95% interval |
|---|---:|---:|
| VQC / trainable QNN | -0.80 pp | [-23.65, +20.83] pp |
| QSVC | -7.22 pp | [-18.53, +2.05] pp |
| Data re-uploading | -2.50 pp | [-25.57, +19.37] pp |
| Quantum reservoir | -14.71 pp | [-29.71, +2.00] pp |

Every interval includes zero and is very broad. The re-uploading classifier did beat the weak four-feature Extra Trees control (macro F1 0.4688 vs 0.3461; paired interval +1.24 to +24.83 pp), but Logistic Regression was the stronger matched control and re-uploading did not beat it. This is not evidence of quantum advantage.

The established full-feature Extra Trees has five-fold grouped OOF macro F1 0.7933, accuracy 0.7933, and balanced accuracy 0.7941 on all 578,117 development rows. Its saved OOF predictions on the same 72 pilot record IDs give macro F1 0.7469, accuracy 0.7500, and balanced accuracy 0.7524. That subset score is context only: those full-feature models used many more training records and the existing five-fold assignments, unlike the 48-record, 3-fold pilot training protocol. It cannot isolate model-family effects. On full development, QML was far below the project champion.

## Complementarity and fixed hybrid check

On the same 72 predictions, VQC and matched Logistic Regression were each correct on 19 records, each uniquely correct on 17, and both wrong on 19. Re-uploading and Logistic were both correct on 19, uniquely correct on 16 vs 17, and both wrong on 20. These tiny counts do not show a clear complementary advantage.

A single equal-weight mean of the saved re-uploading and Logistic probabilities was evaluated with fixed 0.5 weights and threshold (no fitted blend parameters). It scored macro F1 0.4178, accuracy 0.4583, balanced accuracy 0.5167; Logistic alone scored 0.4938, 0.5000, and 0.4952. The paired origin-bootstrap 95% macro-F1 interval for the fixed blend difference was -29.85 to +11.69 pp. It is exploratory on the already examined pilot and not independent confirmation. No stacking, learned weighting, threshold search, or other hybrid was attempted.

## Model families, compute, and resource decision

| Family | Existing pilot status | Cost / configuration |
|---|---|---|
| VQC / trainable QNN | Completed, not selected | PennyLane default.qubit; 4 qubits, depth 2, 20 Adam updates/fold, analytic backprop; 2,880 train and 72 inference circuit executions; 72.29 s train. Loss decreased in 3/3 folds, without validation advantage. |
| QSVC | Completed, not selected | 4-qubit analytic fidelity kernel, depth 2; 6,912 train and 3,456 inference circuit executions; 20.55 s train + 10.17 s inference. |
| Data re-uploading | Completed, not selected | Distinct repeated-encoding trainable architecture, 4 qubits, depth 2; 2,880 train and 72 inference circuit executions; 74.05 s train. |
| Quantum reservoir | Completed, not selected | Fixed 4-qubit depth-2 map plus Logistic readout; 144 train and 72 inference circuit executions; 0.56 s circuit/readout training. |
| Torch/Keras QNN wrappers | Excluded | Same VQC/trainable-QNN family; Torch and TensorFlow absent from isolated env. |
| Hardware/finite-shot QML | Excluded | Only local analytic CPU simulator was used; no hardware jobs, credentials, or hardware validation. |
| Quantum convolutional/deeper statevector models | Not attempted | New architecture and larger compute with no promising pilot evidence; statevector/device cost grows exponentially with qubits. |
| Learned hybrid/stacking | Rejected after fixed diagnostic | Fixed blend loses to Logistic; no independent grouped data to learn weights without nested evaluation. |

The original nine-origin pilot took 105.1 seconds (RSS 185.5 MiB before and 747.4 MiB after; peak unavailable). The family extension took 115.3 seconds total. Its process RSS changed from 185.1 MiB before to 573.1 MiB after; peak RSS was not recorded. An earlier 24-origin expansion stopped when available memory fell below 450 MiB and produced no fold metrics. At this campaign start, available RAM was about 752 MiB on a 7.5 GiB machine, with 31.2 GiB disk free. A guarded focused QML test subset passed 8 tests and deselected the full-development data-loader test; minimum available RAM during that test was 653 MiB. The simulator ran on CPU; the isolated environment had PennyLane 0.45.1 and no Torch/TensorFlow. No QML model was trained in this campaign.

## Search disposition and remaining limits

The completed families cover trainable VQC/QNN, a quantum kernel QSVC, data re-uploading, and quantum reservoir features. VQC and trainable QNN are one family, not counted twice. Alternative kernels/entanglers would be configurations of existing families. The simple hybrid diagnostic was negative. Further encoding, ansatz, seed, or regularization tuning would reuse the same nine-origin validation results after all four families already failed to beat the strongest matched control. A larger 24-origin attempt previously hit the 450 MiB stop; current headroom is still below 1 GiB. Therefore a five-fold, 305-origin QML run or additional circuit optimization was not justified or safe. No frozen-test evaluation was attempted.

The pilot results were not confirmed independently of model-family review: the same nine-origin sample was used for family comparison and the fixed-blend diagnostic, and only one seed was used. The 500-resample intervals cover just nine validation origins. This does not resolve whether a larger QML design might perform differently. A future QML study would need materially safer memory, a predeclared larger origin sample, nested or untouched-origin confirmation, matched classical controls on identical records and training rows, and evidence that a candidate can beat matched Logistic/RBF controls before scaling. Any comparison to the 23-feature champion must also disclose training-size and feature-schema differences. A separate pristine untouched-origin test remains unavailable. These simulator results establish neither real-hardware performance nor quantum computational speedup.

## Artifacts and verification

- `integrity_and_metrics.json` - label, split, prediction, and metric audit plus full champion same-record context.
- `candidate_leaderboard.csv` - all QML candidates, matched controls, fixed blend, uncertainty, and full-feature context.
- `experiment_ledger.json` and `progress_manifest.json` - completed/rejected/incomplete experiments and continuation criteria.
- `matched_error_overlap.csv`, `per_origin_metrics.csv`, and `exploratory_reupload_logistic_blend.csv.gz` - prediction complementarity, per-origin errors, and the fixed blend outputs.
- `resource_log.json` - environment, runtime, memory, and stop decisions.
- `experiments/qmutbench_qml_campaign_audit.py` - reproducible artifact audit and saved-prediction analysis.

Focused tests: 8 QML model/split/encoder tests passed in 6.61 seconds; the full development-data adapter test was deselected under the memory guard. The audit script passed `py_compile`; QML JSON manifests and split/prediction IDs passed validation. The existing production model, source data, split manifests, historical QML results, anomaly audit, and quantum simulator were not modified. No IBM jobs or real-hardware claims were made.
