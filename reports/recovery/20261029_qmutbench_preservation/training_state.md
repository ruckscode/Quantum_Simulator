# Verified training state

- Run: `reports/experiments/qmutbench_binary_qnn_full_development_chunk16_20261029`
- Process check: no `python`, `pythonw`, or `py` process was present in `Get-Process` at inventory time. WMI process command-line inspection returned Access denied, so this is a process-name check rather than full command-line proof.
- Status: `paused_checkpointed`; frozen test access is false.
- Fold progress: 1 of 5 active; no completed folds. Fold 1 is at epoch 0, next batch 816, next chunk 52.
- Verified QNN checkpoints: 51, numbered continuously 1 through 51.
- Verified optimizer updates: 816; processed records: 26,112. The latest checkpoint's Adam `t` is 816.
- Latest checkpoint: `fold_checkpoints/fold_01_qnn_chunk_00000051.json`.
- Every checkpoint passed the runner's read-only payload validator. Each file hash matched its chunk manifest SHA-256; the chunk sequence, 16-batch ranges, record counters, progress counters, gradient checks, and parameter update flags were consistent.
- Each checkpoint contains the model parameters, Adam first and second moments, Adam step count, RNG and shuffle RNG states, current epoch/batch counters, and processed-record receipt. There is no separate scheduler state in this run configuration (optimizer is Adam; no scheduler is configured).
- Configuration: full-development; 5 folds; 8 epochs; batch size 32; chunk size 16 batches; 4 qubits; depth 2; angle re-upload encoding; learning rate 0.02; seed 20261028; per-origin class cap 32; train/validation caps 256 per class per fold.
- The five checkpointed source fingerprints (streaming loader, feature loader, feature encoder, model, full trainer) all match current source files. Python 3.12.10 matches the checkpoint identity.
- Fold/campaign manifests, progress audit, summary, and resume attempt history are present. The latest recorded resume attempt stopped at the memory guard before training; no further attempt is recorded.
- The run directory contains 56 files totaling 1,877,393 bytes. It contains no separate training log, prediction artifact, optimizer file, or scheduler file; resume state is in the checkpoint JSONs and progress manifests.
