# Separate QMutBench QML research pilot

## Chunk checkpointing and full-development mode

The binary VQC runner now commits an immutable, checksum-validated checkpoint after each configured batch chunk (`--chunk-batches`, default 8 for the sampled pilot and 128 for full mode). Checkpoints preserve the deterministic epoch permutation reference and next batch, PennyLane Adam moments/step, RNG state, weights/readout, partial-epoch and chunk loss, processed-record ID receipt, elapsed time, and RAM readings. A fold manifest tracks pending/running/completed chunks. Resume validates the source/data/configuration fingerprint, reconciles the manifest from valid checkpoint files, and continues from the next batch. An interruption can repeat work only inside the not-yet-committed chunk.

Run `python -m qmutbench_qml.training.run_binary_development --mode sampled-pilot` for the capped pilot or use `--mode full-development` for all 578,117 development records. Full mode scans the verified metadata CSV sequentially, retains only compact reference arrays and a disk-backed SQLite index, joins QASM cache chunks by ID, and writes fold features/labels/IDs to temporary memory-mapped files. Preprocessing is fitted on a deterministic, class-balanced sample (up to 5,000 rows) drawn only from that fold's training partition; the QNN still trains on every training record in that fold. Validation predictions are generated in mini-batches and written incrementally. At completion, unique held-out IDs are reconciled against the complete development reference. The frozen test is not used.

Hosted full-development runs use `--max-chunks-per-invocation` to stop after a bounded number of committed chunks, save an invocation summary, upload the verified checkpoint artifact, then dispatch the next run with the completed run ID and attempt. The next run restores that artifact and resumes at the checkpoint's `next_batch`; completed chunks are not repeated. A full run can exceed a single hosted job's time limit, so each invocation finishes cleanly before handing off to the next.

Full mode requires the exact source metadata, saved reference CSV/NPZ, and complete QASM-diff cache. It reports per-fold record/class/origin counts, chunk receipts, selected features, resource measurements, and feature/data fingerprints. Temporary memory maps are recreated after interruption; committed QNN chunks remain in the output directory or downloaded workflow artifact. GitHub artifact retention is 30 days; select a prior run ID and attempt to resume. Hosted standard-runner time limits can require multiple manual dispatches. Do not switch to full-development mode unless the workflow artifact contains the previous full-mode output and all required inputs revalidate. The guarded PennyLane smoke and full-mode preflight keep the 2,048 MiB startup reserve and 1,024 MiB in-run floor unchanged.

This package is isolated from `qmutbench_ml/` and `src/quantum_simulator/`. It reuses the verified metadata CSV and QASM mutation-diff cache by read-only access; it does not copy the dataset or alter the existing model, simulator, or historical research artifacts.

## Research target and data adapter

The prediction task matches the incumbent: `WOO` and `OPO` map to detected (`1`), and `Survivor` maps to undetected (`0`). `data/loader.py` checks the raw dataset SHA-256 against the synchronized research report, maps source labels, aligns `record_id`/origin/labels to the saved 578,117-row development OOF manifest, and verifies the cached QASM features are one-to-one with those records. It scans source IDs and reads target/features only after an ID matches the saved development manifest; frozen-test rows are not retained for feature assembly, metrics, model selection, prediction, or evaluation. Historical adapter code used in earlier pilot runs parsed the target column in bounded chunks before filtering IDs; those test labels were not retained or used, but the old access path was not strict row-level isolation. This was corrected for future runs and is disclosed in the campaign report.

The full incumbent input contains 23 fields: eight QMutBench metadata-derived features plus 15 QASM mutation-diff features. Categorical values use fold-local one-hot encoding; continuous values use fold-local min-max scaling. One-hot expansion of categories such as algorithm and gate would require many more input dimensions than the four-wire pilot allows. Raw variable-length QASM is also unsuitable as a direct four-wire input, so the pilot uses the same cached QASM mutation-diff features as the incumbent. The pilot ranks all encoded candidates by mutual information using only each training fold, chooses four scalars for four input wires, then rescales them to `[0, 1]` for angle encoding. `results/feature_parity.csv` documents all 23 model inputs and the excluded target/group/row identifiers and raw QASM sources; the run manifest records the selected columns per fold. The same four transformed columns and the same train/validation records are used by the VQC, QSVC, Extra Trees, Logistic Regression, and RBF SVC. Consequently, the controls test model-family differences under the same representation; the full-feature incumbent is listed separately as protocol context.

## Model families

- **VQC:** PennyLane `default.qubit`, analytic statevector, 4 wires, 2 layers of trainable `Rot` gates plus a ring of CNOTs. Four selected features are encoded with `RY(pi*x)`. The measured `Z` expectation on wire 0 maps to detected probability `(1 - <Z>)/2`. Class-balanced binary cross-entropy and PennyLane Adam optimize circuit parameters using simulator backpropagation. Fixed seed, 20 updates, no shots.
- **QSVC:** a PennyLane analytic statevector fidelity kernel from an entangling RY/RZ feature map; scikit-learn `SVC(kernel="precomputed", C=1)`. Pairwise kernel circuits are run explicitly and counted. This is a quantum-simulator kernel experiment, not evidence that quantum kernels offer an advantage.
- **QNN:** a quantum neural network is a model family, not an additional independent classifier here: the trainable VQC above is the PennyLane QNode QNN implementation, optimized by autograd/backprop. PennyLane `TorchLayer` and `KerasLayer` wrappers are not separately benchmarked because Torch and TensorFlow are absent from the isolated environment; adding those stacks would change its locked dependencies. A second implementation of the same trainable QNN would be a configuration, not a distinct model family.
- **Data re-uploading classifier:** a distinct VQC circuit architecture that re-encodes all four inputs before each variational layer. It uses the same optimizer and output rule, so differences test repeated encoding rather than a different target or representation.
- **Quantum reservoir classifier:** a seeded, fixed two-layer quantum circuit maps the same four inputs to four Pauli-Z expectations; only a balanced classical Logistic Regression readout is fitted. It avoids circuit-gradient training and is a distinct fixed-quantum-feature/classical-readout family.
- **Classical controls:** 100-tree Extra Trees, balanced Logistic Regression, and balanced RBF SVC (`C=1`), all using the exact same four selected inputs per fold.

Other quantum classifier ideas are not double-counted as new families: alternate VQC optimizers/entanglers and alternate QSVC kernels are configurations of those families. Quantum convolutional networks and higher-dimensional statevector models are technically feasible at small scale but are excluded from this bounded comparison because they would need a separate architecture and tuning budget. Hardware QNNs and finite-shot comparisons are excluded because the environment has only a local analytic simulator and no quantum hardware backend or credentials.

## Validation and run size

The bounded pilot chooses nine development origins and at most eight records per origin using a fixed seeded random sample that does not stratify by target. `GroupKFold(3)` evaluates disjoint groups. Feature selection and scaling are refit inside each training fold. This pilot therefore uses at most 72 of 578,117 development records (actual counts are emitted), from 9 of 305 development origins. Its three-fold validation is exploratory and small. It is not the existing full-development grouped result.

The frozen test remains untouched. No model/hyperparameter is selected for a final test run by this pilot, and no frozen-test result is claimed. The staged next action, if desired after reviewing this pilot, is a larger development-only grouped run with a predeclared compute budget, followed by freezing decisions before any one-time frozen-test evaluation.

## Install and run in isolation

Python 3.11 or newer is required. PennyLane 0.45.1 documents Python >=3.11; `default.qubit` supports statevector execution and backpropagation. No Torch or hardware credentials are needed.

From the repository root:

```powershell
python -m venv qmutbench_qml/.venv
qmutbench_qml/.venv/Scripts/python.exe -m pip install -r qmutbench_qml/requirements-lock.txt
qmutbench_qml/.venv/Scripts/python.exe -m qmutbench_qml.training.run_pilot
qmutbench_qml/.venv/Scripts/python.exe -m qmutbench_qml.training.run_expansion
qmutbench_qml/.venv/Scripts/python.exe -m qmutbench_qml.training.run_family_extension
qmutbench_qml/.venv/Scripts/python.exe -m pytest qmutbench_qml/tests -q
```

Dependencies live in the ignored `qmutbench_qml/.venv`, not in the repository or existing environments. `requirements-lock.txt` records the verified environment versions. The original nine-origin pilot remains in `results/`. A completed extension under `results/family_extension_9origins_20261027/` reuses its VQC, QSVC, and matched-control predictions and benchmarks data re-uploading and quantum reservoir models on the same folds. A larger 24-origin attempt was stopped when available system memory fell below 450 MiB; it wrote no metrics. The expansion entry point remains available to rerun when adequate memory is available. Results include run manifests, per-record split manifests, out-of-fold predictions, per-class metrics, model comparisons, paired metric differences, feature parity, timings, circuit counts, RSS snapshots, and a human-readable report. Neither completed QML run uses frozen-test rows or labels.

## Separate three-class supervised QNN campaign

The raw development source contains three labels (`Survivor`, `WOO`, `OPO`), while the established production model deliberately merges WOO and OPO into one binary detected class. A separate research campaign evaluated the three-class task without changing that model or the original binary QML pilot. It implements and tests a trainable PennyLane QNN, angle re-uploading and amplitude encodings, a trainable quantum-to-classical readout, matched grouped classical controls, and a reserved-origin confirmation attempt. Results, metrics, training curves, folds, uncertainty, and resource decisions are under [the campaign report](../reports/experiments/qmutbench_three_class_qnn_campaign_20261027/research_report.md). No QML variant beat its matched controls; the confirmation fit stopped at the 500 MiB memory floor. The report describes class-aware sampling and the distinction from the binary champion.

## Interpretation limits

The report separates metrics by protocol. Full-feature mutation-aware Extra Trees and nested grouped development scores are historical full-data context only; they are not directly comparable with the nine-origin pilot. QML minus matched-classical deltas are reported in score points with an origin-level paired bootstrap interval. Nine groups yield broad uncertainty. RSS entries are before/after process snapshots rather than peak memory. Analytic shots omit finite-shot sampling noise. The QASM mutation alignment has the incumbent's documented common-prefix/common-suffix ambiguity for repeated operations. This experiment does not establish quantum advantage, hardware-noise detection, or fault localization.

## Kaggle backend validation bundle

`kaggle_confirmation/` contains a data-free, development-only workflow for the predeclared three-class confirmation. `training/write_confirmation_origin_manifest.py` records the 36 unused OPO-bearing origins separately from the 12-origin exploration sample. `training/export_confirmation_data.py` exports only compact sampled development features when a conservative RAM preflight passes; it does not package source rows or test data. See `kaggle_confirmation/README.md` for private-dataset and notebook UI steps. The runner compares the selected QNN with matched 4-feature Logistic Regression, RBF SVC, and Extra Trees, and writes metrics, predictions, feature names, origin bootstrap intervals, and model parameters.

The backend smoke checks forward and gradient agreement before fitting. `default.qubit` is the local supported path. The Kaggle GPU path is optional, uses adjoint differentiation, and only proceeds when the installed Lightning-GPU backend passes parity; the local Windows environment did not have the GPU plugin and could not load the CPU Lightning binary. The 2026-10-28 campaign resume has the exact origin manifest and resource/test results in the linked campaign report. No confirmation fit or Kaggle GPU run is claimed. The frozen test remains untouched.

## GitHub Actions development run

`.github/workflows/qmutbench-qml-development.yml` runs the existing bounded binary QNN campaign on `ubuntu-24.04`, checks the real circuit gradient/update and checkpoint recovery path, regenerates the development QASM-difference cache from the pinned public QMutBench commit, then saves campaign checkpoints, predictions, fold metrics, and resource snapshots as a workflow artifact. It does not evaluate the frozen test.

The established five-fold development assignment is project-generated and is not in the public QMutBench repository. Export only its development records with:

```powershell
qmutbench_qml/.venv/Scripts/python.exe -m qmutbench_qml.training.export_reference_bundle --output reports/experiments/qmutbench_qml_reference_dev_bundle.zip
```

Upload that small ZIP as a GitHub Release asset named `qmutbench_qml_reference_dev_bundle.zip`. Start **Actions → QMutBench binary QML development → Run workflow**, providing the release tag and the SHA-256 printed by the exporter. To resume, provide the previous run ID (and its attempt number if that run was retried); the workflow restores validated QNN epoch/fold checkpoints and the chunked QASM cache from that run before continuing. The bundle contains only development IDs, origin groups, binary labels, and fold assignments, not the source dataset or frozen-test rows.
