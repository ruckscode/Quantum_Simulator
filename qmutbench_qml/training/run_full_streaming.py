"""Full-development, disk-backed QMutBench binary VQC campaign."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from qmutbench_qml.data.streaming import DevelopmentStream
from qmutbench_qml.training import run_binary_development as checkpointing


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def _fingerprint_arrays(x_path: Path, y_path: Path, id_path: Path, rows: int,
                        columns: int, chunk_rows: int = 8192) -> str:
    import numpy as np
    digest = hashlib.sha256()
    x = np.memmap(x_path, mode="r", dtype=np.float64, shape=(rows, columns))
    y = np.memmap(y_path, mode="r", dtype=np.int64, shape=(rows,))
    ids = np.memmap(id_path, mode="r", dtype=np.int64, shape=(rows,))
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        digest.update(np.asarray(x[start:stop], dtype=np.float32).tobytes())
        digest.update(np.asarray(y[start:stop], dtype=np.int8).tobytes())
        digest.update(np.asarray(ids[start:stop], dtype=np.int64).tobytes())
    del x, y, ids
    return digest.hexdigest()


def _fit_encoder(stream: DevelopmentStream, fold: int, n_features: int, seed: int,
                 fit_rows: int):
    import numpy as np
    import pandas as pd
    from qmutbench_qml.data.loader import FEATURES
    from qmutbench_qml.features.reduction import ReducedFeatureEncoder
    frames, labels, per_class = [], [], {0: 0, 1: 0}
    per_class_limit = math.ceil(fit_rows / 2)
    for frame in stream.iter_fold(fold, "training"):
        for label in (0, 1):
            remaining = per_class_limit - per_class[label]
            if remaining <= 0:
                continue
            selected = frame[frame.y == label].head(remaining)
            if len(selected):
                frames.append(selected[FEATURES].copy())
                labels.append(selected.y.to_numpy(dtype=np.int8))
                per_class[label] += len(selected)
        if min(per_class.values()) >= per_class_limit:
            break
    total = sum(per_class.values())
    if not total:
        raise ValueError(f"Fold {fold} has no training rows for preprocessing")
    sample = pd.concat(frames, ignore_index=True)
    y = np.concatenate(labels)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Fold {fold} preprocessing prefix lacks both training classes")
    encoder = ReducedFeatureEncoder(n_features=n_features, seed=seed).fit(sample, y)
    return encoder, total


def _write_memmaps(stream: DevelopmentStream, fold: int, role: str,
                   encoder: ReducedFeatureEncoder, directory: Path, stem: str,
                   rows: int, qubits: int):
    import numpy as np
    from qmutbench_qml.data.loader import FEATURES
    x_path, y_path, id_path, origin_path = [directory / f"{stem}_{suffix}.dat"
        for suffix in ("x", "y", "ids", "origin")]
    x_map = np.memmap(x_path, mode="w+", dtype=np.float64, shape=(rows, qubits))
    y_map = np.memmap(y_path, mode="w+", dtype=np.int64, shape=(rows,))
    id_map = np.memmap(id_path, mode="w+", dtype=np.int64, shape=(rows,))
    origin_map = np.memmap(origin_path, mode="w+", dtype=np.int16, shape=(rows,))
    offset = 0
    origins = set()
    labels = {0: 0, 1: 0}
    for frame in stream.iter_fold(fold, role):
        count = len(frame)
        stop = offset + count
        if stop > rows:
            raise ValueError(f"Fold {fold} {role} stream exceeded its verified row count")
        x_map[offset:stop] = encoder.transform(frame[FEATURES])
        y_map[offset:stop] = frame.y.to_numpy(dtype=np.int64)
        ids = frame.record_id.to_numpy(dtype=np.int64)
        id_map[offset:stop] = ids
        origin_map[offset:stop] = stream._origin_by_id[ids]
        origins.update(frame.origin_program.astype(str).unique())
        for label, count_label in frame.y.value_counts().items():
            labels[int(label)] += int(count_label)
        offset = stop
    if offset != rows:
        raise ValueError(f"Fold {fold} {role} stream ended at {offset}, expected {rows}")
    x_map.flush(); y_map.flush(); id_map.flush(); origin_map.flush()
    del x_map, y_map, id_map, origin_map
    audit = dict(stream.last_stream_audit)
    audit.update({"rows_written": offset, "unique_ids": offset,
                  "origins": len(origins), "class_counts": {str(k): v for k, v in labels.items()}})
    return x_path, y_path, id_path, origin_path, audit


def _fold_metrics(confusion: np.ndarray, brier_sum: float, records: int) -> dict:
    import numpy as np
    tn, fp = map(int, confusion[0])
    fn, tp = map(int, confusion[1])
    per_class = {}
    for label, tp_c, fp_c, fn_c in ((0, tn, fn, fp), (1, tp, fp, fn)):
        precision = tp_c / (tp_c + fp_c) if tp_c + fp_c else 0.0
        recall = tp_c / (tp_c + fn_c) if tp_c + fn_c else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(label)] = {"precision": precision, "recall": recall,
                                 "f1": f1, "support": tp_c + fn_c}
    accuracy = (tn + tp) / records if records else 0.0
    return {"records": records, "accuracy": accuracy,
        "macro_f1": (per_class["0"]["f1"] + per_class["1"]["f1"]) / 2,
        "balanced_accuracy": (per_class["0"]["recall"] + per_class["1"]["recall"]) / 2,
        "per_class": per_class, "confusion_matrix_labels_0_1": [[tn, fp], [fn, tp]],
        "brier_score": brier_sum / records if records else 0.0}


def _pool_metrics(folds: list[dict[str, Any]]) -> dict:
    import numpy as np
    matrix = np.sum([np.asarray(item["confusion_matrix_labels_0_1"], dtype=np.int64)
                     for item in folds], axis=0)
    n = int(matrix.sum())
    brier = sum(item["brier_score"] * item["records"] for item in folds) / max(1, n)
    out = _fold_metrics(matrix, brier * n, n)
    for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
        values = [float(item[metric]) for item in folds]
        out[f"fold_{metric}_mean"] = statistics.mean(values)
        out[f"fold_{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return out


def _validate_saved_oof(path: Path, fold: int, stream: DevelopmentStream,
                       metric: dict[str, Any], expected_rows: int) -> bool:
    import numpy as np
    if not path.is_file() or checkpointing.file_sha256(path) != metric.get("oof_predictions_sha256"):
        return False
    seen = np.zeros_like(stream._seen_reference)
    confusion = np.zeros((2, 2), dtype=np.int64)
    brier_sum, rows = 0.0, 0
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"record_id", "origin_program", "fold", "y_true", "model", "y_pred", "prob_0", "prob_1"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            return False
        for row in reader:
            rid, actual, assigned_fold = int(row["record_id"]), int(row["y_true"]), int(row["fold"])
            if (rid < 0 or rid >= len(seen) or not stream._seen_reference[rid] or seen[rid] or
                    assigned_fold != fold or int(stream._fold_by_id[rid]) != fold or
                    actual != int(stream._y_by_id[rid]) or
                    row["origin_program"] != stream._origin_names[int(stream._origin_by_id[rid])] or
                    row["model"] != "binary_vqc"):
                return False
            p1 = float(row["prob_1"]); p0 = float(row["prob_0"]); predicted = int(row["y_pred"])
            if not (math.isfinite(p0) and math.isfinite(p1) and 0 <= p0 <= 1 and 0 <= p1 <= 1 and
                    math.isclose(p0 + p1, 1.0, abs_tol=1e-6) and predicted == int(p1 > .5)):
                return False
            seen[rid] = True; rows += 1
            confusion[actual, predicted] += 1
            brier_sum += (p1 - actual) ** 2
    return (rows == expected_rows and int(seen.sum()) == expected_rows and
        confusion.tolist() == metric.get("confusion_matrix_labels_0_1") and
        math.isclose(brier_sum / max(1, rows), float(metric.get("brier_score", -1)), rel_tol=1e-9, abs_tol=1e-12))


def run_full_streaming_campaign(root: Path, output: Path, config: dict[str, Any],
                                resume: bool = False, feature_fit_rows: int = 5000,
                                source_chunk_rows: int = 10000,
                                feature_chunk_rows: int = 10000,
                                max_chunks_per_invocation: int | None = None) -> dict:
    """Train five grouped folds using SQLite join + disk-backed NumPy arrays."""
    checkpointing.validate_config(config)
    if feature_fit_rows < 100 or source_chunk_rows < 1 or feature_chunk_rows < 1:
        raise ValueError("Full-streaming preprocessing and chunk settings are invalid")
    if max_chunks_per_invocation is not None and max_chunks_per_invocation < 1:
        raise ValueError("max_chunks_per_invocation must be positive when specified")
    preflight = checkpointing.memory_preflight(config, mode="full-development")
    stream_config = {"metadata_chunk_rows": source_chunk_rows,
        "qasm_feature_chunk_rows": feature_chunk_rows,
        "preprocessor_fit_row_limit": feature_fit_rows,
        "preprocessor_sampling": "deterministic train-only class-balanced first-in-cache-order"}
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Full-development output exists; pass --resume to continue validated checkpoints")
    output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import pandas as pd
    import psutil
    from qmutbench_qml.data.loader import FEATURES
    from qmutbench_qml.features.reduction import ReducedFeatureEncoder
    from qmutbench_qml.models.binary_vqc import BinaryVQC
    checkpoint_dir = output / "fold_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    global_manifest_path = output / "full_campaign_manifest.json"
    progress = {"schema": "qmutbench-full-stream-campaign-v1", "status": "running",
        "expected_development_rows": checkpointing.DEVELOPMENT_ROWS,
        "completed_folds": [], "frozen_test_accessed": False,
        "configuration": config, "streaming_configuration": stream_config,
        "chunks_committed_this_invocation": 0,
        "records_processed_this_invocation": 0,
        "optimizer_updates_this_invocation": 0,
        "total_chunks_committed": 0,
        "total_qnn_records_processed": 0,
        "total_optimizer_updates": 0,
        "resume_events": 0}
    prior_progress = None
    if global_manifest_path.exists() and resume:
        prior = json.loads(global_manifest_path.read_text(encoding="utf-8"))
        if prior.get("expected_development_rows") != checkpointing.DEVELOPMENT_ROWS or prior.get("frozen_test_accessed") is not False:
            raise ValueError("Full-streaming campaign manifest does not match this development-only run")
        if prior.get("configuration") != config:
            raise ValueError("Full-streaming campaign configuration changed during resume")
        if prior.get("streaming_configuration") != stream_config:
            raise ValueError("Full-streaming preprocessing/chunk configuration changed during resume")
        prior_progress = prior
        progress["completed_folds"] = prior.get("completed_folds", [])
        progress["fold_metrics"] = prior.get("fold_metrics", [])
        progress["fold_stream_audits"] = prior.get("fold_stream_audits", [])
        progress["total_chunks_committed"] = int(prior.get("total_chunks_committed", 0))
        progress["total_qnn_records_processed"] = int(prior.get("total_qnn_records_processed", 0))
        progress["total_optimizer_updates"] = int(prior.get("total_optimizer_updates", 0))
        progress["resume_events"] = int(prior.get("resume_events", 0)) + 1
    _atomic_json(global_manifest_path, progress)

    def pause_report(active_fold: int, state: dict[str, Any] | None = None) -> dict:
        progress.update({"status": "paused_checkpointed", "active_fold": active_fold,
            "active_chunk": int(state["chunk_index"]) if state else 0,
            "next_chunk": int(state["chunk_index"]) + 1 if state else 1,
            "next_epoch": int(state["epoch"]) if state else 0,
            "next_batch": int(state["next_batch"]) if state else 0,
            "last_checkpoint_validated": True})
        report = {"schema": "qmutbench-binary-qnn-full-development-v1",
            "status": "paused_checkpointed", "mode": "full-development",
            "development_only": True, "frozen_test_accessed": False,
            "expected_development_records": checkpointing.DEVELOPMENT_ROWS,
            "completed_folds": list(progress.get("completed_folds", [])),
            "chunks_committed_this_invocation": int(progress["chunks_committed_this_invocation"]),
            "records_processed_this_invocation": int(progress["records_processed_this_invocation"]),
            "optimizer_updates_this_invocation": int(progress["optimizer_updates_this_invocation"]),
            "total_chunks_committed": int(progress["total_chunks_committed"]),
            "total_qnn_records_processed": int(progress["total_qnn_records_processed"]),
            "total_optimizer_updates": int(progress["total_optimizer_updates"]),
            "resume_events": int(progress["resume_events"]),
            "active_fold": active_fold,
            "next_chunk": int(progress["next_chunk"]),
            "next_epoch": int(progress["next_epoch"]),
            "next_batch": int(progress["next_batch"]),
            "genuine_quantum_gradient_update_occurred": bool(
                progress.get("genuine_quantum_gradient_update_occurred", False)),
            "gradient_check": progress.get("gradient_check"),
            "parameter_update_verified": bool(progress.get("parameter_update_verified", False)),
            "checkpoint_dir": "fold_checkpoints", "configuration": config,
            "streaming_configuration": stream_config}
        _atomic_json(global_manifest_path, progress)
        _atomic_json(output / "summary.json", report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return report

    with tempfile.TemporaryDirectory(prefix="qmutbench-full-stream-") as temp_dir:
        temp = Path(temp_dir)
        stream = DevelopmentStream(root, temp / "development.sqlite",
            metadata_chunk_rows=source_chunk_rows, feature_chunk_rows=feature_chunk_rows)
        data_audit = stream.build_index()
        if prior_progress and prior_progress.get("stream_audit") not in (None, data_audit):
            raise ValueError("Full-streaming dataset/split/cache fingerprints changed during resume")
        progress["stream_audit"] = data_audit
        package_root = Path(__file__).resolve().parents[1]
        source_fingerprints = {
            "streaming_loader_sha256": checkpointing.file_sha256(package_root / "data" / "streaming.py"),
            "feature_loader_sha256": checkpointing.file_sha256(package_root / "data" / "loader.py"),
            "feature_encoder_sha256": checkpointing.file_sha256(package_root / "features" / "reduction.py"),
            "qnn_model_sha256": checkpointing.file_sha256(package_root / "models" / "binary_vqc.py"),
            "full_trainer_sha256": checkpointing.file_sha256(Path(__file__).resolve()),
            "source_metadata_sha256": data_audit["source_sha256"],
            "reference_csv_sha256": data_audit["reference_sha256"],
            "reference_archive_sha256": data_audit["reference_archive_sha256"],
            "qasm_manifest_sha256": data_audit["qasm_manifest_sha256"]}
        fold_metrics = list(prior_progress.get("fold_metrics", [])) if prior_progress else []
        fold_audits = list(prior_progress.get("fold_stream_audits", [])) if prior_progress else []
        fold_rows = data_audit["fold_record_counts"]
        for fold in range(1, config["fold_count"] + 1):
            if (max_chunks_per_invocation is not None and
                    int(progress["chunks_committed_this_invocation"]) >= max_chunks_per_invocation and
                    fold not in progress.get("completed_folds", [])):
                return pause_report(fold)
            checkpointing.require_memory_floor(psutil, checkpointing.RAM_FLOOR_MIB,
                                                f"Full fold {fold} before preprocessing")
            train_rows = checkpointing.DEVELOPMENT_ROWS - int(fold_rows[str(fold)])
            valid_rows = int(fold_rows[str(fold)])
            versions = {key: value for key, value in checkpointing._software_versions().items()
                        if key != "platform"}
            prior_metric = next((item for item in fold_metrics if int(item.get("fold", 0)) == fold), None)
            prior_prediction = output / f"oof_fold_{fold:02d}.csv"
            if (prior_metric and fold in progress.get("completed_folds", []) and
                    prior_metric.get("parameter_update_verified") and
                    prior_metric.get("gradient_check", {}).get("passed") and
                    _validate_saved_oof(prior_prediction, fold, stream, prior_metric, valid_rows)):
                prior_fold_config = {**config, "mode": "full-development", "fold": fold,
                    "feature_schema": FEATURES,
                    "selected_features": prior_metric["selected_features"],
                    "preprocessor_fit_rows": prior_metric["preprocessor_fit_rows"],
                    "streaming_configuration": stream_config,
                    "software_versions": versions,
                    "source_fingerprints": source_fingerprints,
                    "checkpoint_schema": checkpointing.CHUNK_CHECKPOINT_SCHEMA}
                latest_done = checkpointing.load_latest_qnn_chunk_checkpoint(
                    checkpoint_dir, fold, prior_metric["feature_fingerprint"], prior_fold_config)
                if (latest_done and int(latest_done["completed_epochs"]) == config["epochs"] and
                        int(latest_done["processed_records"]) == train_rows * config["epochs"]):
                    fold_audit = next((item for item in fold_audits if int(item.get("fold", 0)) == fold), None)
                    if fold_audit:
                        print(f"[full-stream fold {fold}/5] verified prior checkpoint and OOF artifact", flush=True)
                        continue
            encoder, fit_sample_count = _fit_encoder(stream, fold, config["qubits"],
                                                       config["seed"] + fold, feature_fit_rows)
            work = temp / f"fold_{fold:02d}"
            work.mkdir()
            train_paths = _write_memmaps(stream, fold, "training", encoder, work, "train",
                                         train_rows, config["qubits"])
            valid_paths = _write_memmaps(stream, fold, "validation", encoder, work, "valid",
                                         valid_rows, config["qubits"])
            train_xp, train_yp, train_idp, _, train_audit = train_paths
            valid_xp, valid_yp, valid_idp, valid_op, valid_audit = valid_paths
            x_train = np.memmap(train_xp, mode="r", dtype=np.float64, shape=(train_rows, config["qubits"]))
            y_train = np.memmap(train_yp, mode="r", dtype=np.int64, shape=(train_rows,))
            train_ids = np.memmap(train_idp, mode="r", dtype=np.int64, shape=(train_rows,))
            x_valid = np.memmap(valid_xp, mode="r", dtype=np.float64, shape=(valid_rows, config["qubits"]))
            y_valid = np.memmap(valid_yp, mode="r", dtype=np.int64, shape=(valid_rows,))
            ids_valid = np.memmap(valid_idp, mode="r", dtype=np.int64, shape=(valid_rows,))
            origins_valid = np.memmap(valid_op, mode="r", dtype=np.int16, shape=(valid_rows,))
            feature_hash = hashlib.sha256()
            feature_hash.update(_fingerprint_arrays(train_xp, train_yp, train_idp, train_rows, config["qubits"]).encode())
            feature_hash.update(_fingerprint_arrays(valid_xp, valid_yp, valid_idp, valid_rows, config["qubits"]).encode())
            fingerprint = checkpointing.canonical_hash({"feature_hash": feature_hash.hexdigest(),
                "configuration": config, "fold": fold, "feature_schema": FEATURES,
                "selected_features": encoder.selected_names, "preprocessor_fit_rows": fit_sample_count,
                "streaming_configuration": stream_config,
                "source_sha256": data_audit["source_sha256"],
                "qasm_manifest_sha256": data_audit["qasm_manifest_sha256"],
                "reference_sha256": data_audit["reference_sha256"],
                "reference_archive_sha256": data_audit["reference_archive_sha256"],
                "source_code_fingerprints": source_fingerprints})
            fold_config = {**config, "mode": "full-development", "fold": fold,
                "feature_schema": FEATURES, "selected_features": encoder.selected_names,
                "preprocessor_fit_rows": fit_sample_count, "software_versions": versions,
                "streaming_configuration": stream_config,
                "source_fingerprints": source_fingerprints,
                "checkpoint_schema": checkpointing.CHUNK_CHECKPOINT_SCHEMA}
            latest = checkpointing.load_latest_qnn_chunk_checkpoint(
                checkpoint_dir, fold, fingerprint, fold_config) if resume else None
            batches_per_epoch = math.ceil(train_rows / config["batch_size"])
            total_chunks = config["epochs"] * math.ceil(batches_per_epoch / config["chunk_batches"])
            checkpointing.initialize_qnn_chunk_manifest(checkpoint_dir, fold, fingerprint, total_chunks)
            qnn = BinaryVQC(n_qubits=config["qubits"], depth=config["depth"],
                epochs=config["epochs"], learning_rate=config["learning_rate"],
                batch_size=config["batch_size"], seed=config["seed"] + fold,
                optimizer=config["optimizer"], encoding=config["encoding"])
            initial_weights = np.random.default_rng(config["seed"] + fold).normal(
                0, .12, (config["depth"], config["qubits"], 3))
            qnn_start = time.perf_counter()
            state = latest
            prior_chunk_state = [state]

            def persist_chunk(value: dict) -> None:
                chunk_updates = math.ceil(int(value["chunk_records"]) / config["batch_size"])
                checkpointing.attach_chunk_record_receipt(value, prior_chunk_state[0],
                                                          train_ids, config["batch_size"])
                checkpointing.save_qnn_chunk_checkpoint(checkpoint_dir, fold, fingerprint,
                                                        fold_config, value)
                prior_chunk_state[0] = value
                progress["chunks_committed_this_invocation"] += 1
                progress["records_processed_this_invocation"] += int(value["chunk_records"])
                progress["optimizer_updates_this_invocation"] += chunk_updates
                progress["total_chunks_committed"] += 1
                progress["total_qnn_records_processed"] += int(value["chunk_records"])
                progress["total_optimizer_updates"] += chunk_updates
                progress.update({"status": "running", "active_fold": fold,
                    "active_chunk": int(value["chunk_index"]),
                    "active_epoch": int(value["epoch"]),
                    "qnn_records_processed_in_active_fold": int(value["processed_records"]),
                    "genuine_quantum_gradient_update_occurred": bool(
                        value.get("gradient_check_result", {}).get("passed") and
                        not np.allclose(np.asarray(value["weights"]), initial_weights)),
                    "gradient_check": value.get("gradient_check_result"),
                    "parameter_update_verified": not np.allclose(
                        np.asarray(value["weights"]), initial_weights)})
                _atomic_json(global_manifest_path, progress)
                print(f"[full-stream fold {fold}/5] checkpoint validated; "
                      f"chunk={value['chunk_index']} epoch={value['epoch']} "
                      f"next_batch={value['next_batch']} chunk_records={value['chunk_records']} "
                      f"optimizer_updates={chunk_updates} "
                      f"committed_this_run={progress['chunks_committed_this_invocation']}/"
                      f"{max_chunks_per_invocation or 'unbounded'} "
                      f"gradient={bool(value.get('gradient_check_result', {}).get('passed'))} "
                      f"parameter_update={progress['parameter_update_verified']} "
                      f"available_ram_mib={psutil.virtual_memory().available // 2**20}", flush=True)

            while True:
                if not (state and int(state.get("completed_epochs", 0)) >= config["epochs"]):
                    next_chunk = int(state.get("chunk_index", 0)) + 1 if state else 1
                    checkpointing.mark_qnn_chunk_running(checkpoint_dir, fold, fingerprint, next_chunk)
                    progress.update({"status": "running", "active_fold": fold,
                                     "active_chunk": next_chunk})
                    _atomic_json(global_manifest_path, progress)
                qnn.fit(x_train, y_train, resume_state=state, checkpoint_callback=persist_chunk,
                        max_batches_per_chunk=config["chunk_batches"], compact_batch_order=True)
                if qnn.training_complete:
                    break
                state = qnn.resume_state
                if (max_chunks_per_invocation is not None and
                        int(progress["chunks_committed_this_invocation"]) >= max_chunks_per_invocation):
                    return pause_report(fold, state)
            qnn_seconds = time.perf_counter() - qnn_start
            parameter_update_verified = not np.allclose(np.asarray(qnn.weights), initial_weights)
            if not qnn.gradient_check_result.get("passed") or not parameter_update_verified:
                raise ArithmeticError(f"Fold {fold} failed real quantum gradient/update verification")
            prediction_path = output / f"oof_fold_{fold:02d}.csv"
            temp_prediction = prediction_path.with_suffix(".csv.tmp")
            confusion = np.zeros((2, 2), dtype=np.int64)
            brier_sum = 0.0
            infer_start = time.perf_counter()
            with temp_prediction.open("w", encoding="utf-8", newline="") as out:
                fields = ["record_id", "origin_program", "fold", "y_true", "model", "y_pred", "prob_0", "prob_1"]
                writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader()
                for start in range(0, valid_rows, config["batch_size"]):
                    stop = min(valid_rows, start + config["batch_size"])
                    checkpointing.require_memory_floor(psutil, checkpointing.RAM_FLOOR_MIB,
                                                        f"Full fold {fold} validation batch")
                    p1 = qnn.predict_proba(x_valid[start:stop])
                    truth = np.asarray(y_valid[start:stop], dtype=np.int8)
                    pred = (p1 > 0.5).astype(np.int8)
                    for actual, predicted in zip(truth, pred):
                        confusion[int(actual), int(predicted)] += 1
                    brier_sum += float(np.sum((p1 - truth) ** 2))
                    codes = np.asarray(origins_valid[start:stop], dtype=np.int16)
                    for rid, code, actual, probability, predicted in zip(
                            ids_valid[start:stop], codes, truth, p1, pred):
                        writer.writerow({"record_id": int(rid), "origin_program": stream._origin_names[int(code)],
                            "fold": fold, "y_true": int(actual), "model": "binary_vqc",
                            "y_pred": int(predicted), "prob_0": float(1 - probability), "prob_1": float(probability)})
                out.flush(); os.fsync(out.fileno())
            os.replace(temp_prediction, prediction_path)
            metrics = _fold_metrics(confusion, brier_sum, valid_rows)
            metrics.update({"fold": fold, "train_records": train_rows,
                "validation_records": valid_rows, "train_origins": train_audit["origins"],
                "validation_origins": valid_audit["origins"],
                "train_class_counts": train_audit["class_counts"],
                "validation_class_counts": valid_audit["class_counts"],
                "train_records_seen": train_audit["records_seen"],
                "validation_records_seen": valid_audit["records_seen"],
                "preprocessor_fit_rows": fit_sample_count,
                "selected_features": encoder.selected_names,
                "feature_fingerprint": fingerprint, "qnn_training_seconds": qnn_seconds,
                "oof_predictions_sha256": checkpointing.file_sha256(prediction_path),
                "software_versions": versions,
                "source_fingerprints": source_fingerprints,
                "validation_seconds": time.perf_counter() - infer_start,
                "peak_rss_bytes": int(qnn.peak_rss_bytes),
                "available_ram_bytes_after": int(psutil.virtual_memory().available),
                "training_executions": int(qnn.training_executions),
                "parameter_update_verified": parameter_update_verified,
                "qnn_records_processed": train_rows * config["epochs"],
                "completed_chunks": int(qnn.resume_state["chunk_index"]),
                "gradient_check": qnn.gradient_check_result,
                "checkpoint_manifest": f"fold_checkpoints/fold_{fold:02d}_campaign_manifest.json"})
            fold_metrics.append(metrics)
            fold_audits.append({"fold": fold, "training": train_audit, "validation": valid_audit})
            progress["completed_folds"] = [item["fold"] for item in fold_metrics]
            progress["records_seen_in_completed_validation_folds"] = sum(
                item["validation_records_seen"] for item in fold_metrics)
            progress["fold_metrics"] = fold_metrics
            progress["fold_stream_audits"] = fold_audits
            _atomic_json(global_manifest_path, progress)
            del x_train, y_train, train_ids, x_valid, y_valid, ids_valid, origins_valid
            print(f"[full-stream fold {fold}/5] train={train_rows} valid={valid_rows} "
                  f"chunks={total_chunks} RAM={psutil.virtual_memory().available // 2**20} MiB", flush=True)

        # Merge fold predictions incrementally and reconcile exact one-time OOF coverage.
        all_ids = np.zeros(len(stream._seen_reference), dtype=np.bool_)
        merged = output / "oof_predictions.csv"
        merged_tmp = merged.with_suffix(".csv.tmp")
        columns = ["record_id", "origin_program", "fold", "y_true", "model", "y_pred", "prob_0", "prob_1"]
        total_predictions = 0
        with merged_tmp.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=columns); writer.writeheader()
            for fold in range(1, 6):
                fold_path = output / f"oof_fold_{fold:02d}.csv"
                with fold_path.open("r", encoding="utf-8", newline="") as src:
                    reader = csv.DictReader(src)
                    for row in reader:
                        rid = int(row["record_id"])
                        if rid < 0 or rid >= len(all_ids) or not stream._seen_reference[rid] or all_ids[rid]:
                            raise ValueError(f"Duplicate or unexpected OOF development record: {rid}")
                        if int(row["fold"]) != int(stream._fold_by_id[rid]):
                            raise ValueError(f"OOF validation fold mismatch for record {rid}")
                        all_ids[rid] = True; total_predictions += 1; writer.writerow(row)
            target.flush(); os.fsync(target.fileno())
        if total_predictions != checkpointing.DEVELOPMENT_ROWS or not np.array_equal(all_ids, stream._seen_reference):
            raise ValueError("Full-development OOF prediction coverage did not reconcile to 578,117 rows")
        os.replace(merged_tmp, merged)
        pooled = _pool_metrics(fold_metrics)
        report = {"schema": "qmutbench-binary-qnn-full-development-v1",
            "status": "completed_full_development_oof", "mode": "full-development",
            "development_only": True, "frozen_test_accessed": False,
            "genuine_quantum_gradient_update_occurred": all(
                item["gradient_check"]["passed"] and item["parameter_update_verified"]
                for item in fold_metrics),
            "expected_development_records": checkpointing.DEVELOPMENT_ROWS,
            "unique_validation_records": total_predictions,
            "training_record_exposures": sum(item["train_records"] for item in fold_metrics),
            "qnn_records_processed_total": sum(item["qnn_records_processed"] for item in fold_metrics),
            "total_chunks_committed": int(progress["total_chunks_committed"]),
            "total_optimizer_updates": int(progress["total_optimizer_updates"]),
            "folds_completed": 5, "fold_metrics": fold_metrics,
            "pooled_oof_metrics": pooled, "stream_audit": data_audit,
            "fold_stream_audits": fold_audits, "preflight": preflight,
            "prediction_file": merged.name, "checkpoint_dir": checkpoint_dir.name,
            "configuration": config, "streaming_configuration": stream_config}
        _atomic_json(output / "summary.json", report)
        progress.update({"status": report["status"], "unique_validation_records": total_predictions})
        _atomic_json(global_manifest_path, progress)
        return report
