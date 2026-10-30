# QMutBench three-class supervised QNN campaign — 2026-10-27

## Executive result

A genuine end-to-end supervised three-class QNN was implemented, mathematically checked, trained on real development records, and tested against matched classical controls. It learned the toy task, but none of the three QNN encoding/readout variants beat the matched four-feature classical models. The outcome supports no quantum advantage claim and does not establish that all QML designs fail.

The project champion predicts a different binary target (WOO/OPO merged versus Survivor). It remains unchanged. Its 79.33% five-fold OOF macro F1 is not directly comparable to this three-class study. The raw source contains three labels, but the established production target intentionally merges WOO and OPO. This campaign follows the directive’s separate three-class research objective without changing the incumbent.

## Verified labels and development sample

Raw development label distribution: {'WOO': 294802, 'Survivor': 278413, 'OPO': 4902}. Mapping: Survivor=0, WOO=1, OPO=2. OPO occurs in only 48 of the 305 development origins.
For class coverage, a seeded sample selected 12 OPO-bearing origins and up to six records per class within each origin. It contains {'0': 72, '1': 68, '2': 65} across 205 rows. This deliberately class-balanced, rare-origin sample is not prevalence-representative and estimates transfer only among these OPO-bearing origins. It is not the earlier nine-origin QML pilot.

Three-fold StratifiedGroupKFold split the 12 origins into disjoint groups; each validation fold covered all three labels. Feature selection and scaling were fitted separately inside each training fold. All QNNs and classical controls used identical transformed four-dimensional inputs and held-out record IDs/origins/folds. No frozen-test rows were used.

## Supervised QNN foundations and learning loop

For a one-qubit state |0>, RY(theta) produces cos(theta/2)|0> + sin(theta/2)|1>. Therefore P(0)=cos²(theta/2), P(1)=sin²(theta/2), <Z>=cos(theta), and d<Z>/dtheta=-sin(theta). PennyLane statevectors, probabilities, expectation values, and autodiff gradients matched these analytic identities at theta=0, pi/3, and pi/2. Hadamard/CNOT entanglement behavior is covered by the circuit validation tests.

The QNN applies angle or amplitude feature encoding, trainable Rot gates and CNOT entanglers, measures three Pauli expectations as logits, applies softmax, and optimizes weighted three-class cross-entropy with PennyLane Adam/backprop. The hybrid readout variant jointly learns an affine 3x3 head and bias. Circuit parameters and, where enabled, head parameters demonstrably changed. The toy task loss fell from 1.304 to 0.716 and classified 12/12 examples.

All reported real-data circuits use ideal analytic PennyLane `default.qubit` statevector simulation with no finite shots or noise. This says nothing about shot noise, noisy devices, hardware performance, or computational speedup. A finite-shot Pauli estimate would have standard error sqrt((1-<Z>²)/N); it was not simulated.

## Matched development results

Macro F1 is primary; pooled metrics are computed across identical paired validation records. Fold standard deviations are across three grouped folds.

| Model | Pooled macro F1 | Accuracy | Balanced accuracy | Fold macro F1 mean +/- SD |
|---|---:|---:|---:|---:|
| three_class_vqc | 0.2998 | 0.3220 | 0.3180 | 0.2370 +/- 0.0376 |
| three_class_vqc_amplitude | 0.3086 | 0.3366 | 0.3445 | 0.2901 +/- 0.0496 |
| three_class_vqc_angle_linear | 0.3218 | 0.3561 | 0.3649 | 0.2860 +/- 0.0623 |
| extra_trees_4f | 0.4669 | 0.4927 | 0.4846 | 0.4583 +/- 0.0286 |
| logistic_4f | 0.4407 | 0.4634 | 0.4572 | 0.4076 +/- 0.0789 |
| rbf_svc_4f | 0.4306 | 0.4390 | 0.4348 | 0.3617 +/- 0.1737 |

### Per-class results and errors

QNN per-class labels are {'0': 'Survivor', '1': 'WOO', '2': 'OPO'}. Pooled QNN confusions, per-class precision/recall/F1/support, validation log loss, and per-origin metrics are saved in `qnn_pooled_metrics.json` and `per_origin_metrics.csv`. The angle QNN had fold-specific class collapse: fold 1 predicted no Survivor; fold 2 never predicted OPO. Its training loss barely moved in fold 1 (1.083 to 1.078) and improved more in folds 2-3, but validation macro F1 remained poor. The trainable readout lifted pooled macro F1 only to 0.3218; the amplitude encoding variant reached 0.3086. No model showed stable three-class separation.

### Paired origin uncertainty

The table below reports candidate minus control macro F1, with percentile intervals from 2,000 paired bootstrap resamples of the 12 validation origins. The small, class-balanced rare-origin sample means these intervals describe only this exploratory sample.

| QNN | Control | Delta (pp) | 95% origin-bootstrap interval (pp) |
|---|---|---:|---:|
| three_class_vqc | extra_trees_4f | -16.71 | [-25.63, -8.53] |
| three_class_vqc | logistic_4f | -14.09 | [-22.99, -3.76] |
| three_class_vqc | rbf_svc_4f | -13.09 | [-22.73, -4.02] |
| three_class_vqc_amplitude | extra_trees_4f | -15.83 | [-22.80, -9.30] |
| three_class_vqc_amplitude | logistic_4f | -13.21 | [-21.99, -1.01] |
| three_class_vqc_amplitude | rbf_svc_4f | -12.21 | [-23.19, +1.52] |
| three_class_vqc_angle_linear | extra_trees_4f | -14.51 | [-22.63, -7.86] |
| three_class_vqc_angle_linear | logistic_4f | -11.89 | [-18.43, -5.54] |
| three_class_vqc_angle_linear | rbf_svc_4f | -10.88 | [-20.57, -0.52] |

Every QNN point estimate is below every matched classical control; paired origin intervals and variation are in `paired_origin_bootstrap.csv`. No observed improvement is statistically convincing.

## Encoding and circuit costs

| Variant | Encoding/readout | Qubits | Ansatz depth | Trainable params | Notes |
|---|---|---:|---:|---:|---|
| angle_reupload | RY(pi*x) per qubit and repeated before each trainable layer / three Pauli-Z expectations as softmax logits; weighted multiclass cross-entropy | 4 | 12 (2 layers) | 24 | ideal analytic statevector |
| amplitude | normalized four-value amplitude state; global input norm discarded; state-preparation gates costed separately as an explicit limitation / Z on each wire plus ZZ expectation as three softmax logits | 2 | 8 (2 layers) | 12 | ideal analytic statevector |
| angle_linear | RY(pi*x) per qubit and repeated before each trainable layer / three Pauli-Z expectations; trainable affine 3x3 readout and bias then softmax | 4 | 12 (2 layers) | 36 | ideal analytic statevector |

Angle re-uploading uses four feature wires, two trainable layers, 24 parameters, and repeats RY(pi*x) before each layer. PennyLane reports compiled circuit depth 12 (18 logical operations in the high-level resource listing). The amplitude variant uses two wires to encode four selected values as a normalized state, 12 parameters; a decomposed circuit has depth 8 and 11 gates including a three-RY/two-CNOT state preparation, so preparation is not treated as free. Normalization discards overall input norm. The trainable-readout angle variant adds a 3x3 affine head and bias (36 total parameters). Circuit execution counts and per-fold runtime/RSS snapshots are in `all_fold_metrics.csv` and `resource_log.json`.

Binary project champion context: on the same 205 sampled record IDs, saved five-fold OOF binary predictions can be compared after collapsing WOO/OPO. Champion binary macro F1 was 0.923 on this unusually balanced, OPO-origin subset; three-class QNN predictions collapsed to binary scores between 0.431 and 0.501. This is context only: target mapping, 23 features, fold scheme and training size differ. It is not a matched three-class baseline and does not enter the leaderboard; same-record binary metrics are in `consolidated_summary.json`.

## Rejected approaches and remaining limits

- Basis encoding was not applied because the four fold-selected continuous/dummy values would require discretization and lose numerical information; it would be a distinct encoding hypothesis with its own matched controls.
- Three-class quantum-kernel QSVC and fixed quantum-reservoir readout were not rerun: the available implementations/pilots target the established binary label, were weak against matched controls there, and the primary three-class QNN families remained well below controls. No evidence justified spending additional circuit runs on these distinct families under the present resource limit.
- The input sample is class-balanced by design and includes 12 of 48 OPO-bearing origins. It is not representative of the 305-origin deployment population. This was the largest safely justified QNN comparison after preflight found about 1.1 GiB free RAM (the angle run began with about 800 MiB available) and an earlier 24-origin expansion stopped below its 450 MiB floor.
- Three encoding/readout QNN variants failed to approach even the strongest matched four-feature classical control, while each QNN took about 68-141 seconds and classical fits took under one second. Scaling to all development rows or adding further architecture/optimizer sweeps is not justified by this evidence.
- A predeclared confirmation fit using the 12-origin training sample and all 36 remaining OPO-bearing development origins was attempted but stopped when available RAM crossed below the 500 MiB floor; it emitted no predictions or metrics. No retry was made. On resume, I replaced the second loader pass with a single union load and added exact full-batch gradient accumulation in chunks. A six-row, one-update toy check matched full-batch parameters (circuit-weight max error 2.4e-12; readout 7e-18; loss 2.2e-16) with the same six circuit executions. The optimized real-data confirmation was not launched: the latest raw RAM probe was 853 MiB before importing PennyLane/scikit-learn, while the runner requires 700 MiB remaining after imports; safe post-import headroom was not established. No new QMutBench metrics or parameter checkpoint resulted. No 85% target was optimized, no frozen-test values were used for model development in this continuation, and no production model, source data, saved split manifest, anomaly audit, or simulator was modified. See the data-access audit caveat below for the older adapter's parsing behavior.
- A stronger generalization study needs a larger safe-memory environment, a predeclared origin-representative sample, coverage of non-OPO origins, multiple seeds, and held-out origins untouched during architecture selection. Those 36 origins remain unmeasured by a completed candidate.

## Artifacts and verification

Campaign outputs are stored under this directory, including the source experiment runner, separate predictions/configuration/fold manifests, three training-curve sets, per-origin metrics, bootstrap comparisons, resource log, foundations report, persistent ledger, progress manifest, and model prediction artifacts. No real-data model parameter checkpoint was saved because the selected-candidate confirmation fit hit the RAM floor before completion.

The focused suite had an earlier 4-test QNN/foundations pass. The final broader rerun passed 10 tests and deselected 1 full-development adapter test; 2 QNN fit tests were stopped by the explicit 500 MiB RAM guard. On 2026-10-28, the two analytic-foundation tests passed again; the exact-gradient chunking pytest reached the same RAM guard before its fit, although the separately recorded six-row direct equivalence smoke passed with loss difference 2.22e-16. The default.qubit backend passed a forward/gradient smoke (0.018 s for the measured circuit); lightning.qubit could not load because Windows Application Control blocked its binary, and lightning.gpu is not installed. The new export and Kaggle scripts pass syntax checks. No real-data confirmation training, checkpoint, or frozen-test evaluation was produced.

## Resume status: development-origin confirmation and Kaggle packaging (2026-10-28)

The exact reserved-origin manifest is now recorded in `confirmation_origin_manifest.json`: 48 OPO-bearing development origins were identified from the saved development OOF IDs, 12 match the completed exploration sample, and 36 are reserved with zero overlap. The scan accessed `Killed` only for saved development IDs; the frozen-test target column was not used. Development labels remain Survivor=278,413, WOO=294,802, OPO=4,902.

**Data-access audit caveat:** historical versions of `run_three_class_campaign.py` and `data/loader.py` used during the completed exploratory pilot read the raw `Killed` column into bounded pandas chunks before filtering by development IDs. No frozen-test values were retained, summarized, used in feature selection, fitting, model choice, or reported metrics, but the old path did parse them in memory; therefore the older claim of literally zero frozen-label access is too strong. The current origin-manifest scan and corrected loaders inspect row IDs first and only access target/features for IDs in the saved development OOF manifest. No new frozen-test labels were used in this continuation. The earlier frozen-model retrospective comparison remains labeled as such in its own report.

`kaggle_confirmation/` now contains a portable backend smoke and confirmation runner; `training/export_confirmation_data.py` writes the fixed 12/36 development-only compact sample for private user upload after conservative memory checks. The workflow uses matched 4-feature Logistic Regression, RBF SVC, Extra Trees, and the selected trainable-readout QNN, with fold-local encoder fitting, per-class metrics, group bootstrap intervals, predictions, parameter checkpoint, circuit counts, runtime, and memory snapshots. It requires the user to upload the compact data privately and enable a Kaggle accelerator in the UI. No credentials are embedded or inspected.

Local backend evidence is saved in `backend_smoke_local.json`: default.qubit forward and gradient calculations passed. The optional CPU lightning plugin was unavailable because a DLL was blocked by Windows Application Control; the lightning.gpu plugin is absent and no Kaggle GPU session was accessible. The targeted test rerun passed the two foundation tests but the exact-gradient fit test stopped at the existing 500 MiB safety floor before a model update. The compact-data exporter was run and stopped before importing data: system headroom was 667 MiB against its conservative 900 MiB pre-import requirement, so it created no output and loaded no source rows. An earlier probe was 788.8 MiB. A pandas extension DLL was also blocked by Windows policy during an initial local CLI help attempt; lazy imports now let CLI help run without those binary modules. Real-data QNN confirmation remains blocked locally; no metric or checkpoint is claimed. No frozen-test values were used for selection or metrics in this continuation; see the preceding data-access caveat for historical source parsing.
