# Full-data QML optimization v3 — execution entry point

This directory tracks the full-data campaign's audit and reproducible execution commands. The streaming extractor and its tests are maintained in the repository so they can be run against the full, local-only QMutBench checkout without committing the corpus.

## Full-corpus QASM feature extraction

Run from the repository root in PowerShell:

```powershell
python experiments/qmutbench_full_corpus_streaming.py
```

The defaults expect:

- Metadata: `data/qmutbench_ingestion/raw/official_all_data_20261015.csv`
- QASM root: `data/qmutbench_full/data`
- Output/checkpoints: `data/qmutbench_ml/full_corpus_qasm_stream_v1`

To reduce peak per-chunk memory and make interruption recovery finer-grained:

```powershell
python experiments/qmutbench_full_corpus_streaming.py --chunk-rows 250
```

For an explicit one-chunk smoke test before a full run:

```powershell
python experiments/qmutbench_full_corpus_streaming.py --chunk-rows 10 --max-chunks 1 --output-dir data/qmutbench_ml/full_corpus_qasm_smoke
```

The smoke test writes to a separate directory and cannot be mistaken for a complete corpus extraction.

The extractor writes compressed feature chunks, per-chunk error logs, and a JSON manifest with source/parser/extractor hashes, counts, checksums, resource observations, and status. It resumes only when the metadata, parser, extractor, configuration, and completed chunk hashes match. A changed input requires a new output directory.

Feature chunks intentionally exclude `Killed` and all other target labels. `record_id` is a lineage key and `origin_program` is retained for group splitting only; neither is a predictive feature. Join the target labels separately by `record_id` in a later training step, after validating the source mapping.

The default RAM safety floor is 1,024 MiB. If available memory drops below it, the run pauses and records `paused_resource_guard`; do not lower the floor to force progress. Resume the same command after freeing memory.

## Validation

Run the synthetic-data tests:

```powershell
python -m pytest tests/test_qmutbench_full_corpus_streaming.py -q
```

These tests cover path construction and path-safety checks, the RAM guard, target-free feature output, one-chunk extraction, and checkpoint reuse. They do not claim that the complete local corpus has been parsed or trained.

## Completion semantics

- `partial`: not all expected metadata rows have been processed.
- `paused_resource_guard`: paused before starting work that would breach the memory floor.
- `complete_with_errors`: all expected rows were attempted, but one or more records failed extraction.
- `complete`: all 723,079 expected metadata rows were attempted and no parse/path errors were recorded.

A `complete` extraction is a data-preparation milestone, not a QML training result. Grouped development evaluation, label semantics, and frozen-test separation must be handled by the subsequent training pipeline.
