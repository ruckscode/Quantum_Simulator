"""Run a guarded real-circuit gradient/update/chunk-checkpoint recovery smoke."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import psutil


def run_smoke(output: Path | None = None) -> dict:
    available_mib = int(psutil.virtual_memory().available // (1024 * 1024))
    required_mib = 2048
    if available_mib < required_mib:
        raise MemoryError(f"Smoke requires {required_mib} MiB available; found {available_mib} MiB")
    # Defer PennyLane/NumPy imports until after the smoke's guarded preflight.
    import numpy as np

    from qmutbench_qml.models.binary_vqc import BinaryVQC, RAM_FLOOR_MIB
    from qmutbench_qml.training.run_binary_development import (
        DEFAULT_CONFIG, attach_chunk_record_receipt, load_latest_qnn_chunk_checkpoint,
        save_qnn_chunk_checkpoint,
    )
    x = np.array([[.05, .15], [.8, .9], [.2, .7], [.95, .35]], dtype=float)
    y = np.array([0, 1, 0, 1], dtype=int)
    config = {**DEFAULT_CONFIG, "qubits": 2, "depth": 1, "epochs": 2,
              "batch_size": 2, "max_train_per_class_per_fold": 2,
              "max_validation_per_class_per_fold": 2, "per_origin_class_cap": 2}
    fingerprint = "binary-qnn-real-circuit-smoke-v1"
    train_ids = np.arange(len(y), dtype=np.int64)
    if output is None:
        temp = tempfile.TemporaryDirectory(prefix="qmutbench-qnn-smoke-")
        checkpoint_dir = Path(temp.name)
    else:
        temp = None
        checkpoint_dir = output.resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = BinaryVQC(n_qubits=2, depth=1, epochs=2, learning_rate=.02,
                      batch_size=2, seed=318, optimizer="adam")
    saved_first_chunk = False

    class StopAfterFirstChunk(Exception):
        pass

    def stop_at_first_chunk(state: dict) -> None:
        nonlocal saved_first_chunk
        attach_chunk_record_receipt(state, None, train_ids, 2)
        save_qnn_chunk_checkpoint(checkpoint_dir, 1, fingerprint, config, state)
        saved_first_chunk = True
        raise StopAfterFirstChunk

    try:
        model.fit(x, y, checkpoint_callback=stop_at_first_chunk, max_batches_per_chunk=1)
    except StopAfterFirstChunk:
        pass
    if not saved_first_chunk:
        raise AssertionError("Real QNN did not save the first completed chunk")
    chunk1 = load_latest_qnn_chunk_checkpoint(checkpoint_dir, 1, fingerprint, config)
    if chunk1["next_batch"] != 1 or chunk1["processed_records"] != 2:
        raise AssertionError("Chunk checkpoint did not preserve the next mini-batch boundary")
    resumed = BinaryVQC(n_qubits=2, depth=1, epochs=2, learning_rate=.02,
                        batch_size=2, seed=318, optimizer="adam")
    state = chunk1
    previous_state = [chunk1]
    while True:
        def persist(value: dict) -> None:
            attach_chunk_record_receipt(value, previous_state[0], train_ids, 2)
            save_qnn_chunk_checkpoint(checkpoint_dir, 1, fingerprint, config, value)
            previous_state[0] = value

        resumed.fit(x, y, resume_state=state,
                    checkpoint_callback=persist,
                    max_batches_per_chunk=1)
        if resumed.training_complete:
            break
        state = resumed.resume_state
    latest = load_latest_qnn_chunk_checkpoint(checkpoint_dir, 1, fingerprint, config)
    probabilities = resumed.predict_proba(x)
    if latest["completed_epochs"] != 2 or not latest["loss_history"]:
        raise AssertionError("Checkpoint recovery did not complete epoch 2")
    if np.allclose(np.asarray(latest["weights"]), np.asarray(chunk1["weights"])):
        raise AssertionError("QNN circuit parameters did not change after resumed training")
    if not np.isfinite(probabilities).all() or probabilities.shape != (len(y),):
        raise AssertionError("Resumed QNN predictions are invalid")
    result = {
        "status": "passed_real_binary_qnn_training_smoke",
        "records": len(y), "smoke_only_not_benchmark": True,
        "gradient_check": resumed.gradient_check_result,
        "parameter_update_verified": True,
        "checkpoint_recovery_verified": True,
        "completed_epochs": latest["completed_epochs"],
        "completed_chunks": latest["chunk_index"],
        "loss_history": latest["loss_history"],
        "training_executions": resumed.training_executions,
        "inference_executions": resumed.inference_executions,
        "checkpoint_directory": str(checkpoint_dir),
        "available_ram_mib_at_start": available_mib,
        "ram_floor_mib": RAM_FLOOR_MIB,
        "configuration": resumed.configuration(),
        "predictions": probabilities.tolist(),
    }
    if temp is not None:
        temp.cleanup()
    return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = run_smoke(args.output)
    except MemoryError as exc:
        print(json.dumps({"status": "blocked_memory_guard", "error": str(exc)}, indent=2))
        return 3
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
