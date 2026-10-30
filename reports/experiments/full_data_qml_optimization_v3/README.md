# Full-data QML optimization v3 — development-only extraction

The full local QMutBench metadata has 723,079 records, but the existing evaluation protocol reserves 70 origins / 144,962 records as a frozen test set. The streaming extractor therefore uses only the **578,117 record IDs from the saved grouped-development OOF file**. It never reads the metadata `Killed` column and never opens QASM files for records outside that development ID list.

## Run feature extraction

Run from the repository root in PowerShell:

```powershell
python experiments/qmutbench_full_corpus_streaming.py
```

Expected local inputs:

- Metadata: `data/qmutbench_ingestion/raw/official_all_data_20261015.csv`
- QASM root: `data/qmutbench_full/data`
- Development record IDs: `data/qmutbench_ml/master_optimization_v1/oof_predictions.csv`
- Output/checkpoints: `data/qmutbench_ml/qmutbench_dev_qasm_stream_v2`

For smaller, more frequently checkpointed chunks:

```powershell
python experiments/qmutbench_full_corpus_streaming.py --chunk-rows 250
```

For a synthetic one-chunk smoke run, pass a tiny metadata file, matching OOF ID file, and QASM root to the Python `extract(..., expected_dev_rows=...)` API; do not point a smoke run at the production output directory.

The extractor emits compressed feature chunks, per-chunk error logs, and a JSON manifest with input/parser/extractor hashes, chunk checksums, counts, available RAM, and status. It resumes only when the input and configuration hashes match and all prior chunk checksums validate. Keep the 1,024 MiB RAM safety floor in place.

Feature chunks exclude labels. `record_id` is a lineage key and `origin_program` is a grouping key, not a predictive feature. The saved OOF file is read using `record_id` only; its labels, predictions, fold metrics, and probabilities are not used for feature extraction. The metadata reader explicitly selects columns that exclude `Killed`.

## Validation

```powershell
python -m pytest tests/test_qmutbench_full_corpus_streaming.py -q
```

The synthetic integration test includes a second row that represents a frozen-test record and deliberately has no corresponding QASM file. The test verifies that it is neither parsed nor reported as an extraction error.

## Status semantics

- `partial`: the full metadata scan has not completed.
- `paused_resource_guard`: paused before starting a chunk that would breach the RAM floor.
- `complete_with_errors`: every development record was attempted, but one or more feature rows failed.
- `complete`: all 578,117 development IDs were found, features were extracted for each, and no path/parser errors occurred.

Feature extraction completion is not training completion. The subsequent QML stage must join labels separately by `record_id`, preserve origin-grouped splits, and keep the frozen test untouched.

## Existing local run notes retained

# Full-data QML optimization campaign v3

The full QMutBench development QASM extraction already existed in the local repository and has now been independently verified: 578,117 development records / 305 origins, 58 structural chunks and 58 mutation-diff chunks, zero parse errors, and matching metadata values across the official checkout and ingestion snapshot. The frozen-test records were excluded. See `feature_extraction_audit.json` and `label_leakage_audit.json`.

No full-data QNN training was started in this continuation because available RAM was below the 1 GiB floor. Existing three-fold feature screens did not demonstrate a consistent QASM gain, and their folds differ from the five-fold 79.33% macro-F1 historical champion. See `qasm_feature_ablation_report.md` and `blocked_tasks.json`.

No frozen-test outcomes were read. The required extractor-specific test file and script named by the prompt are absent from this checkout; equivalent existing QASM and corpus tests passed. See `test_report.md`.


## Safe recovery follow-up

See `safe_recovery_report.md` and `safe_recovery_audit.json` for current HEAD, current RAM, QML entry-point scope, Git ACL diagnosis, frozen-test checksum status, and exact resume commands. The QML runner checks RAM before importing the QML/data stack and enforces a 1 GiB in-run floor with 200 MiB startup headroom. No training was launched.


## Binary QNN implementation

A dedicated sampled, five-fold, origin-grouped binary QNN trainer with matched classical controls is implemented. Its eight pure-Python checkpoint/sampling tests pass. Training was stopped by the 2 GiB startup estimate (724 MiB available); details and exact resume commands are in `binary_qnn_implementation_report.md` and the configuration is in `binary_qnn_default_config.json`.


The dedicated sampled binary QNN trainer is implemented but has not trained: available RAM was 724 MiB against a 2 GiB startup estimate. Eight pure-Python tests passed, and the exact historical five-fold grouping was verified without loading object arrays. See `binary_qnn_implementation_report.md`, `binary_qnn_default_config.json`, and `binary_reference_fold_audit.json`.
