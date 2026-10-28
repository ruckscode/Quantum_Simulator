"""Development-only, sampled full-corpus binary QMutBench QNN campaign.

The reference split is taken from the numeric arrays in the saved five-fold
incumbent OOF archive. The archive's object-valued origin array is never loaded.
Scientific dependencies are imported only after the RAM preflight succeeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

SCHEMA = "qmutbench-binary-qnn-development-v1"
EPOCH_CHECKPOINT_SCHEMA = "qmutbench-binary-qnn-epoch-v1"
LABEL_MAP = {"Survivor": 0, "WOO": 1, "OPO": 1}
DEVELOPMENT_ROWS = 578_117
DEVELOPMENT_ORIGINS = 305
N_METADATA_ROWS = 723_079
RAM_FLOOR_MIB = 1024
STARTUP_FLOOR_MIB = 2048
DEFAULT_CONFIG = {
    "seed": 20261028,
    "fold_count": 5,
    "qubits": 4,
    "depth": 2,
    "optimizer": "adam",
    "learning_rate": 0.02,
    "epochs": 8,
    "batch_size": 32,
    "max_train_per_class_per_fold": 256,
    "max_validation_per_class_per_fold": 256,
    "per_origin_class_cap": 32,
    "encoding": "angle_reupload",
    "classical_estimators": 100,
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_label(label: str) -> int:
    try:
        return LABEL_MAP[label]
    except KeyError as exc:
        raise ValueError(f"Unexpected QMutBench binary label: {label!r}") from exc


def validate_group_assignments(rows: list[dict[str, Any]], folds: int = 5) -> dict[str, int]:
    """Check origin disjointness and binary label support for reference OOF rows."""
    group_fold: dict[str, int] = {}
    for row in rows:
        origin, fold, label = str(row["origin_program"]), int(row["fold"]), int(row["y"])
        if not origin or fold < 1 or fold > folds or label not in (0, 1):
            raise ValueError("Invalid origin, fold, or binary label in development manifest")
        previous = group_fold.setdefault(origin, fold)
        if previous != fold:
            raise ValueError(f"Origin {origin!r} crosses reference folds {previous} and {fold}")
    return group_fold


def stratified_origin_sample(rows_by_stratum: dict[tuple[int, int, str], list[int]],
                             fold: int, role: str, per_origin_cap: int,
                             per_class_cap: int, seed: int) -> list[int]:
    """Deterministically cap records per origin/class, then per class."""
    if role not in {"train", "validation"}:
        raise ValueError("role must be train or validation")
    by_class: dict[int, list[int]] = {0: [], 1: []}
    for (row_fold, label, _origin), ids in rows_by_stratum.items():
        eligible = row_fold == fold if role == "validation" else row_fold != fold
        if eligible:
            rng = random.Random(seed + row_fold * 1009 + label * 9176 +
                                int(hashlib.sha256(_origin.encode()).hexdigest()[:8], 16))
            chosen = list(ids)
            rng.shuffle(chosen)
            by_class[label].extend(chosen[:per_origin_cap])
    selected: list[int] = []
    for label in (0, 1):
        candidates = by_class[label]
        rng = random.Random(seed + fold * 65537 + label * 97 + (1 if role == "validation" else 0))
        rng.shuffle(candidates)
        selected.extend(candidates[:per_class_cap])
    if len(selected) != len(set(selected)):
        raise ValueError("Sampling produced duplicate record IDs")
    return selected


def save_completed_fold(output: Path, fold: int, fingerprint: str,
                        configuration: dict[str, Any], payload: dict[str, Any]) -> Path:
    """Atomically persist one self-checksummed completed fold; never replace one."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"fold_{fold:02d}.json"
    if path.exists():
        raise FileExistsError(f"Fold checkpoint already exists: {path}")
    body = {"schema": SCHEMA, "fold": fold, "fingerprint": fingerprint,
            "configuration": configuration, "payload": payload}
    body["payload_sha256"] = canonical_hash(body)
    raw = (json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=output, prefix=f"fold_{fold:02d}_", suffix=".tmp", delete=False) as tmp:
        temp_path = Path(tmp.name)
        tmp.write(raw)
        tmp.flush()
        os.fsync(tmp.fileno())
    try:
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return path


def load_completed_fold(output: Path, fold: int, fingerprint: str,
                        configuration: dict[str, Any]) -> dict[str, Any]:
    path = output / f"fold_{fold:02d}.json"
    if not path.is_file():
        raise ValueError(f"Incomplete fold checkpoint: fold {fold}")
    raw = path.read_bytes()
    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable fold checkpoint: {path}") from exc
    if body.get("schema") != SCHEMA or body.get("fold") != fold:
        raise ValueError(f"Incompatible fold checkpoint: {path}")
    expected_digest = body.pop("payload_sha256", None)
    if not isinstance(expected_digest, str) or canonical_hash(body) != expected_digest:
        raise ValueError(f"Corrupt or incomplete fold checkpoint: {path}")
    if body.get("configuration") != configuration or body.get("fingerprint") != fingerprint:
        raise ValueError(f"Stale fold checkpoint rejected: {path}")
    if not isinstance(body.get("payload"), dict) or not body["payload"].get("predictions"):
        raise ValueError(f"Incomplete fold results in checkpoint: {path}")
    return body


def save_qnn_epoch_checkpoint(output: Path, fold: int, fingerprint: str,
                              configuration: dict[str, Any], state: dict[str, Any]) -> Path:
    """Atomically create one immutable checkpoint after a completed QNN epoch."""
    epoch = int(state.get("completed_epochs", 0))
    if epoch < 1 or len(state.get("loss_history", [])) != epoch:
        raise ValueError("Epoch checkpoint must contain a complete loss history through its boundary")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"fold_{fold:02d}_qnn_epoch_{epoch:04d}.json"
    if path.exists():
        raise FileExistsError(f"QNN epoch checkpoint already exists: {path}")
    body = {"schema": EPOCH_CHECKPOINT_SCHEMA, "fold": fold, "epoch": epoch,
            "fingerprint": fingerprint, "configuration": configuration, "state": state}
    body["payload_sha256"] = canonical_hash(body)
    raw = (json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=output, prefix=f"fold_{fold:02d}_epoch_{epoch:04d}_",
                                     suffix=".tmp", delete=False) as tmp:
        temp_path = Path(tmp.name)
        tmp.write(raw)
        tmp.flush()
        os.fsync(tmp.fileno())
    try:
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return path


def load_qnn_epoch_checkpoint(path: Path, fold: int, epoch: int, fingerprint: str,
                              configuration: dict[str, Any]) -> dict[str, Any]:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable QNN epoch checkpoint: {path}") from exc
    if body.get("schema") != EPOCH_CHECKPOINT_SCHEMA or body.get("fold") != fold or body.get("epoch") != epoch:
        raise ValueError(f"Incompatible QNN epoch checkpoint: {path}")
    expected = body.pop("payload_sha256", None)
    if not isinstance(expected, str) or canonical_hash(body) != expected:
        raise ValueError(f"Corrupt or incomplete QNN epoch checkpoint: {path}")
    if body.get("fingerprint") != fingerprint or body.get("configuration") != configuration:
        raise ValueError(f"Stale QNN epoch checkpoint rejected: {path}")
    state = body.get("state")
    required = {"completed_epochs", "weights", "readout_scale", "readout_bias", "loss_history",
                "rng_state", "optimizer_accumulation", "training_seconds",
                "training_executions", "peak_rss_bytes"}
    if (not isinstance(state, dict) or not required <= set(state) or
            state["completed_epochs"] != epoch or len(state["loss_history"]) != epoch or
            not isinstance(state["weights"], list) or not state["weights"] or
            not isinstance(state["rng_state"], dict)):
        raise ValueError(f"Incomplete QNN epoch state in checkpoint: {path}")
    if epoch > int(configuration["epochs"]):
        raise ValueError(f"QNN epoch checkpoint exceeds configured epoch budget: {path}")
    try:
        weights = state["weights"]
        valid_weights = (len(weights) == int(configuration["depth"]) and
            all(len(layer) == int(configuration["qubits"]) and
                all(len(rotation) == 3 and all(math.isfinite(float(value)) for value in rotation)
                    for rotation in layer) for layer in weights))
        valid_history = all(math.isfinite(float(value)) for value in state["loss_history"])
        valid_readout = math.isfinite(float(state["readout_scale"])) and math.isfinite(float(state["readout_bias"]))
    except (TypeError, ValueError, OverflowError):
        valid_weights = valid_history = valid_readout = False
    if not (valid_weights and valid_history and valid_readout):
        raise ValueError(f"Invalid numeric QNN epoch state in checkpoint: {path}")
    accumulation = state["optimizer_accumulation"]
    if configuration["optimizer"] == "adam":
        if (not isinstance(accumulation, dict) or not {"fm", "sm", "t"} <= set(accumulation) or
                len(accumulation["fm"]) != 3 or len(accumulation["sm"]) != 3 or
                int(accumulation["t"]) < 1):
            raise ValueError(f"Missing Adam optimizer state in QNN epoch checkpoint: {path}")
    elif accumulation is not None:
        raise ValueError(f"Unexpected optimizer state in QNN epoch checkpoint: {path}")
    return body


def load_latest_qnn_epoch_checkpoint(output: Path, fold: int, fingerprint: str,
                                     configuration: dict[str, Any]) -> dict[str, Any] | None:
    """Validate the consecutive immutable epoch history and return its latest state."""
    prefix = f"fold_{fold:02d}_qnn_epoch_"
    paths = sorted(output.glob(f"{prefix}*.json"))
    if not paths:
        return None
    epochs = []
    for path in paths:
        try:
            epochs.append(int(path.stem.removeprefix(prefix)))
        except ValueError as exc:
            raise ValueError(f"Malformed QNN epoch checkpoint name: {path}") from exc
    if epochs != list(range(1, len(epochs) + 1)):
        raise ValueError(f"QNN epoch checkpoint history has a gap or duplicate: {output}")
    loaded = [load_qnn_epoch_checkpoint(path, fold, epoch, fingerprint, configuration)
              for path, epoch in zip(paths, epochs)]
    for previous, current in zip(loaded, loaded[1:]):
        if current["state"]["loss_history"][:previous["epoch"]] != previous["state"]["loss_history"]:
            raise ValueError("QNN epoch checkpoint loss history is not a continuation of the prior boundary")
    return loaded[-1]["state"]


def load_or_train_fold(output: Path, fold: int, fingerprint: str,
                       configuration: dict[str, Any], trainer: Callable[[], dict[str, Any]],
                       resume: bool) -> tuple[dict[str, Any], bool]:
    path = output / f"fold_{fold:02d}.json"
    if path.exists():
        if not resume:
            raise FileExistsError("Fold outputs exist; pass --resume to validate and reuse them")
        return load_completed_fold(output, fold, fingerprint, configuration), True
    payload = trainer()
    save_completed_fold(output, fold, fingerprint, configuration, payload)
    return {"payload": payload}, False


def validate_config(config: dict[str, Any]) -> None:
    required = set(DEFAULT_CONFIG)
    if set(config) != required:
        raise ValueError(f"Configuration keys differ from schema; missing={required-set(config)}, extra={set(config)-required}")
    if config["fold_count"] != 5:
        raise ValueError("This trainer requires the established five-fold reference split")
    if not 2 <= config["qubits"] <= 8:
        raise ValueError("qubits must be between 2 and 8")
    if config["depth"] < 1 or config["epochs"] < 1 or config["batch_size"] < 1:
        raise ValueError("depth, epochs, and batch_size must be positive")
    if config["optimizer"] not in {"adam", "gd"}:
        raise ValueError("optimizer must be adam or gd")
    if config["encoding"] != "angle_reupload":
        raise ValueError("Only validated angle_reupload encoding is currently implemented")
    if not math.isfinite(float(config["learning_rate"])) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    for key in ("max_train_per_class_per_fold", "max_validation_per_class_per_fold",
                "per_origin_class_cap", "classical_estimators"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")


def memory_preflight(config: dict[str, Any]) -> dict[str, int]:
    """Estimate working memory and require the 1 GiB floor plus headroom."""
    validate_config(config)
    import psutil
    avail = int(psutil.virtual_memory().available // (1024 * 1024))
    sampled_rows = config["fold_count"] * 2 * (
        config["max_train_per_class_per_fold"] + config["max_validation_per_class_per_fold"])
    # Complex128 statevector/autodiff tape bound: one state plus an estimated
    # intermediate state per parameterized/entangling gate in a circuit batch.
    circuit_states = 1 + config["depth"] * (4 * config["qubits"] + 1)
    quantum_working_set = math.ceil(16 * (2 ** config["qubits"]) * circuit_states *
                                    config["batch_size"] / (1024 * 1024))
    parameter_working_set = math.ceil((config["depth"] * config["qubits"] * 3 + 2) * 8 * 6 /
                                     (1024 * 1024))
    # Reserve 700 MiB for Python/scientific runtimes and bounded source join,
    # plus sampled dense arrays, the estimated QNN state/tape, and the 1 GiB floor.
    sample_working_set = math.ceil(sampled_rows * 64 / (1024 * 1024))
    estimated_peak = (RAM_FLOOR_MIB + 700 + sample_working_set + quantum_working_set +
                      parameter_working_set)
    required = max(STARTUP_FLOOR_MIB, estimated_peak)
    if avail < required:
        raise MemoryError(f"Only {avail} MiB available; estimated peak requires {required} MiB "
                          f"to preserve the {RAM_FLOOR_MIB} MiB floor")
    return {"available_mib": avail, "estimated_peak_mib": estimated_peak,
            "required_start_mib": required, "sampled_rows_upper_bound": sampled_rows,
            "estimated_sample_working_set_mib": sample_working_set,
            "estimated_quantum_state_and_gradient_working_set_mib": quantum_working_set,
            "estimated_optimizer_parameter_working_set_mib": parameter_working_set}


def require_memory_floor(psutil_module, minimum_mib: int, context: str) -> int:
    """Raise before work proceeds if available RAM is below a configured floor."""
    available = int(psutil_module.virtual_memory().available // (1024 * 1024))
    if available < minimum_mib:
        raise MemoryError(f"{context}: available RAM is {available} MiB, below the {minimum_mib} MiB floor")
    return available


def _load_scientific_stack():
    """Called only after memory_preflight; keep imports out of module import path."""
    import numpy as np
    import pandas as pd
    import psutil
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 f1_score, precision_recall_fscore_support)
    from qmutbench_qml.data.loader import FEATURES, TARGET_MAP, load_development_records
    from qmutbench_qml.features.reduction import ReducedFeatureEncoder
    from qmutbench_qml.models.binary_vqc import BinaryVQC
    return locals()


def _software_versions() -> dict[str, str]:
    versions = {"Python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "scikit-learn", "PennyLane", "autograd"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _reference_rows(root: Path, np) -> dict[str, Any]:
    """Validate numeric 5-fold archive fields against dev OOF; skip its object array."""
    archive = root / "data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/mutation_diff_oof.npz"
    oof_path = root / "data/qmutbench_ml/master_optimization_v1/oof_predictions.csv"
    with np.load(archive, allow_pickle=False) as saved:
        required = {"record_id", "y", "fold"}
        if not required <= set(saved.files):
            raise ValueError(f"Reference OOF is missing numeric arrays: {required - set(saved.files)}")
        record_id = np.asarray(saved["record_id"], dtype=np.int64)
        y_ref = np.asarray(saved["y"], dtype=np.int8)
        fold_ref = np.asarray(saved["fold"], dtype=np.int8)
    if not (record_id.ndim == y_ref.ndim == fold_ref.ndim == 1 and
            len(record_id) == len(y_ref) == len(fold_ref) == DEVELOPMENT_ROWS):
        raise ValueError("Reference five-fold OOF arrays have invalid dimensions or row count")
    if np.any(record_id < 0) or np.any(record_id >= N_METADATA_ROWS) or len(np.unique(record_id)) != DEVELOPMENT_ROWS:
        raise ValueError("Reference five-fold record IDs are out of range or duplicated")
    if not set(np.unique(y_ref)) <= {0, 1} or set(np.unique(fold_ref)) != {1, 2, 3, 4, 5}:
        raise ValueError("Reference five-fold labels or fold IDs are invalid")
    fold_by_id = np.zeros(N_METADATA_ROWS, dtype=np.int8)
    y_by_id = np.full(N_METADATA_ROWS, -1, dtype=np.int8)
    fold_by_id[record_id] = fold_ref
    y_by_id[record_id] = y_ref
    row_count = 0
    stratum_ids: dict[tuple[int, int, str], list[int]] = {}
    origins: dict[str, int] = {}
    seen = bytearray(N_METADATA_ROWS)
    with oof_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line in csv.DictReader(stream):
            rid = int(line["record_id"])
            y = int(line["y_true"])
            origin = str(line["origin_program"])
            if rid < 0 or rid >= N_METADATA_ROWS or seen[rid] or fold_by_id[rid] == 0:
                raise ValueError(f"Invalid/duplicate/missing reference OOF record ID: {rid}")
            if y_by_id[rid] != y or y not in (0, 1) or not origin:
                raise ValueError(f"Reference OOF label/group mismatch at record {rid}")
            seen[rid] = 1
            fold = int(fold_by_id[rid])
            row_count += 1
            stratum_ids.setdefault((fold, y, origin), []).append(rid)
            origins.setdefault(origin, fold)
            if origins[origin] != fold:
                raise ValueError(f"Origin {origin!r} crosses established five-fold assignments")
    if row_count != DEVELOPMENT_ROWS or sum(seen) != DEVELOPMENT_ROWS or len(origins) != DEVELOPMENT_ORIGINS:
        raise ValueError("Reference OOF IDs/origins do not match the verified development contract")
    return {"fold_by_id": fold_by_id, "strata": stratum_ids,
                  "fold_counts": {str(f): int(np.sum(fold_ref == f)) for f in range(1, 6)},
                  "origins": len(origins), "rows": row_count,
                  "archive": str(archive.relative_to(root)).replace("\\", "/"),
                  "archive_sha256": file_sha256(archive)}


def _metrics(y, probabilities, stack):
    np = stack["np"]
    accuracy_score = stack["accuracy_score"]
    balanced_accuracy_score = stack["balanced_accuracy_score"]
    confusion_matrix = stack["confusion_matrix"]
    f1_score = stack["f1_score"]
    precision_recall_fscore_support = stack["precision_recall_fscore_support"]
    pred = np.asarray(probabilities).argmax(axis=1).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0)
    return pred, {"accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "per_class": {str(c): {"precision": float(precision[c]), "recall": float(recall[c]),
            "f1": float(f1[c]), "support": int(support[c])} for c in (0, 1)},
        "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def _summary_metrics(all_predictions: list[dict[str, Any]], fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    np = __import__("numpy")
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in all_predictions:
        by_model.setdefault(row["model"], []).append(row)
    summaries = {}
    for model, rows in by_model.items():
        y = np.asarray([r["y_true"] for r in rows], dtype=int)
        pred = np.asarray([r["y_pred"] for r in rows], dtype=int)
        p1 = np.asarray([r["prob_1"] for r in rows], dtype=float)
        tp = int(np.sum((y == 1) & (pred == 1)))
        fp = int(np.sum((y == 0) & (pred == 1)))
        fn = int(np.sum((y == 1) & (pred == 0)))
        tn = int(np.sum((y == 0) & (pred == 0)))
        class_metrics = {}
        for c in (0, 1):
            tp_c = int(np.sum((y == c) & (pred == c)))
            fp_c = int(np.sum((y != c) & (pred == c)))
            fn_c = int(np.sum((y == c) & (pred != c)))
            pr = tp_c / (tp_c + fp_c) if tp_c + fp_c else 0.0
            re = tp_c / (tp_c + fn_c) if tp_c + fn_c else 0.0
            f1 = 2 * pr * re / (pr + re) if pr + re else 0.0
            class_metrics[str(c)] = {"precision": pr, "recall": re, "f1": f1,
                                     "support": int(np.sum(y == c))}
        bacc = sum(float(np.mean(pred[y == c] == c)) for c in (0, 1) if np.any(y == c)) / 2
        fm = [r for r in fold_metrics if r["model"] == model]
        runtimes = [float(x["runtime_seconds"]) for x in fm]
        rss_peak = [int(max(x["qnn_peak_rss_bytes"] or 0, x["rss_after_bytes"])) for x in fm]
        summaries[model] = {"records": len(rows), "accuracy": float(np.mean(y == pred)),
            "macro_f1": float(statistics.mean(v["f1"] for v in class_metrics.values())),
            "balanced_accuracy": float(bacc), "per_class": class_metrics,
            "confusion_matrix_labels_0_1": [[tn, fp], [fn, tp]],
            "fold_accuracy_mean": statistics.mean(x["accuracy"] for x in fm),
            "fold_accuracy_sd": statistics.stdev(x["accuracy"] for x in fm) if len(fm) > 1 else 0.0,
            "fold_macro_f1_mean": statistics.mean(x["macro_f1"] for x in fm),
            "fold_macro_f1_sd": statistics.stdev(x["macro_f1"] for x in fm) if len(fm) > 1 else 0.0,
            "fold_balanced_accuracy_mean": statistics.mean(x["balanced_accuracy"] for x in fm),
            "fold_balanced_accuracy_sd": statistics.stdev(x["balanced_accuracy"] for x in fm) if len(fm) > 1 else 0.0,
            "mean_brier_score": float(np.mean((p1 - y) ** 2)),
            "runtime_seconds_total": sum(runtimes), "runtime_seconds_mean_per_fold": statistics.mean(runtimes),
            "max_observed_process_rss_bytes": max(rss_peak)}
    return summaries


def run_campaign(root: Path, output: Path, config: dict[str, Any], resume: bool = False) -> dict[str, Any]:
    """Run bounded sampled 5-fold OOF; not a full-row fit or frozen-test run."""
    root, output = root.resolve(), output.resolve()
    required_start = memory_preflight(config)
    if output.exists() and any(output.iterdir()):
        checkpoint_dir = output / "fold_checkpoints"
        has_checkpoints = checkpoint_dir.is_dir() and any(
            checkpoint_dir.glob("fold_*.json"))
        if not resume:
            raise FileExistsError("Output exists; use a new directory or pass --resume to validate fold checkpoints")
        if not has_checkpoints:
            raise FileExistsError("Cannot resume: existing output has no completed fold or QNN epoch checkpoints; choose a new directory")
    elif output.exists() and not resume:
        # An empty directory is harmless; permit the first invocation to use it.
        pass
    output.mkdir(parents=True, exist_ok=True)
    stack = _load_scientific_stack()
    np, pd = stack["np"], stack["pd"]
    psutil = stack["psutil"]
    if stack["TARGET_MAP"] != LABEL_MAP:
        raise ValueError(f"Trainer binary target map {LABEL_MAP} differs from loader map {stack['TARGET_MAP']}")
    require_memory_floor(psutil, RAM_FLOOR_MIB, "After scientific imports")
    split = _reference_rows(root, np)
    cfg = dict(config)
    train_ids, valid_ids, sample_rows = [], [], []
    for fold in range(1, cfg["fold_count"] + 1):
        tr = stratified_origin_sample(split["strata"], fold, "train", cfg["per_origin_class_cap"],
                                      cfg["max_train_per_class_per_fold"], cfg["seed"])
        va = stratified_origin_sample(split["strata"], fold, "validation", cfg["per_origin_class_cap"],
                                      cfg["max_validation_per_class_per_fold"], cfg["seed"] + 13)
        if set(tr) & set(va):
            raise ValueError(f"Fold {fold} sampled overlapping train and validation IDs")
        train_ids.append(tr); valid_ids.append(va)
        for rid in tr:
            sample_rows.append({"record_id": rid, "fold": fold, "role": "train"})
        for rid in va:
            sample_rows.append({"record_id": rid, "fold": fold, "role": "validation"})
    unique_selected = sorted({int(rid) for ids in train_ids + valid_ids for rid in ids})
    estimate = required_start["estimated_peak_mib"]
    if psutil.virtual_memory().available // (1024 * 1024) < max(RAM_FLOOR_MIB, estimate - 700):
        raise MemoryError("Available RAM no longer supports the estimated sampled feature join")

    frame, loader_audit = stack["load_development_records"](root, record_ids=unique_selected)
    require_memory_floor(psutil, RAM_FLOOR_MIB, "During bounded development feature loading")
    fold_by_id = split["fold_by_id"]
    frame["reference_fold"] = [int(fold_by_id[int(rid)]) for rid in frame.record_id]
    if frame.reference_fold.eq(0).any() or frame.record_id.duplicated().any():
        raise ValueError("Loaded records do not align uniquely to the established five-fold split")
    origin_by_id = dict(zip(frame.record_id.astype(int), frame.origin_program.astype(str)))
    label_by_id = dict(zip(frame.record_id.astype(int), frame.y.astype(int)))
    sample_manifest_path = output / "sample_manifest.csv"
    versions = _software_versions()
    package_root = Path(__file__).resolve().parents[1]
    versions["trainer_source_sha256"] = file_sha256(Path(__file__).resolve())
    versions["qnn_model_source_sha256"] = file_sha256(package_root / "models" / "binary_vqc.py")
    versions["data_loader_source_sha256"] = file_sha256(package_root / "data" / "loader.py")
    versions["feature_encoder_source_sha256"] = file_sha256(package_root / "features" / "reduction.py")
    qnn_class = stack["BinaryVQC"]
    encoder_class = stack["ReducedFeatureEncoder"]
    all_predictions: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    fold_files = []
    process = psutil.Process()
    campaign_start = time.perf_counter()
    for fold in range(1, cfg["fold_count"] + 1):
        require_memory_floor(psutil, RAM_FLOOR_MIB, f"Fold {fold} before preprocessing")
        train = frame[frame.record_id.isin(train_ids[fold - 1])].copy().reset_index(drop=True)
        valid = frame[frame.record_id.isin(valid_ids[fold - 1])].copy().reset_index(drop=True)
        if set(train.reference_fold.unique()) & {fold} or set(valid.reference_fold.unique()) != {fold}:
            raise ValueError(f"Fold {fold} no longer matches the reference origin-grouped split")
        if set(train.y.astype(int)) != {0, 1} or set(valid.y.astype(int)) != {0, 1}:
            raise ValueError(f"Fold {fold} train/validation sample lacks both binary classes")
        groups_train = set(train.origin_program.astype(str)); groups_valid = set(valid.origin_program.astype(str))
        if groups_train & groups_valid:
            raise ValueError(f"Origin leakage in fold {fold}")
        encoder = encoder_class(n_features=cfg["qubits"], seed=cfg["seed"] + fold)
        encoder.fit(train[stack["FEATURES"]], train.y.to_numpy())
        x_train = encoder.transform(train[stack["FEATURES"]])
        x_valid = encoder.transform(valid[stack["FEATURES"]])
        y_train, y_valid = train.y.to_numpy(dtype=int), valid.y.to_numpy(dtype=int)
        for x in (x_train, x_valid):
            if not np.isfinite(x).all() or x.min() < -1e-9 or x.max() > 1 + 1e-9:
                raise ValueError("Fold-local encoded inputs must be finite and scaled to [0,1]")
        data_hash = hashlib.sha256()
        data_hash.update(np.asarray(x_train, dtype=np.float32).tobytes())
        data_hash.update(np.asarray(x_valid, dtype=np.float32).tobytes())
        data_hash.update(np.asarray(y_train, dtype=np.int8).tobytes())
        data_hash.update(np.asarray(y_valid, dtype=np.int8).tobytes())
        fingerprint = canonical_hash({"data_hash": data_hash.hexdigest(), "config": cfg,
            "fold": fold, "train_ids": sorted(map(int, train.record_id)),
            "validation_ids": sorted(map(int, valid.record_id)),
            "feature_schema": stack["FEATURES"], "features": encoder.selected_names,
            "source_sha256": loader_audit["source_sha256"],
            "qasm_cache_manifest_sha256": loader_audit["qasm_cache_manifest_sha256"],
            "reference_fold_archive_sha256": split["archive_sha256"],
            "target_map": LABEL_MAP,
            "software_versions": versions})

        fold_configuration = {**cfg, "target_map": LABEL_MAP, "feature_schema": stack["FEATURES"],
                              "software_versions": versions}
        checkpoint_dir = output / "fold_checkpoints"
        epoch_configuration = {**fold_configuration, "fold": fold,
                               "checkpoint_schema": EPOCH_CHECKPOINT_SCHEMA}

        def train_fold() -> dict[str, Any]:
            require_memory_floor(psutil, RAM_FLOOR_MIB, f"Fold {fold} before QNN fit")
            available_before_training = psutil.virtual_memory().available
            started = time.perf_counter()
            qnn = qnn_class(n_qubits=cfg["qubits"], depth=cfg["depth"], epochs=cfg["epochs"],
                learning_rate=cfg["learning_rate"], batch_size=cfg["batch_size"], seed=cfg["seed"] + fold,
                optimizer=cfg["optimizer"], encoding=cfg["encoding"])
            epoch_state = load_latest_qnn_epoch_checkpoint(
                checkpoint_dir, fold, fingerprint, epoch_configuration) if resume else None

            def persist_epoch(state: dict[str, Any]) -> None:
                save_qnn_epoch_checkpoint(checkpoint_dir, fold, fingerprint,
                                          epoch_configuration, state)

            qnn.fit(x_train, y_train, resume_state=epoch_state,
                    checkpoint_callback=persist_epoch)
            qnn_probability = qnn.predict_proba(x_valid)
            outputs = [("binary_vqc", qnn_probability),
                ("logistic_4f", stack["LogisticRegression"](C=1.0, class_weight="balanced",
                    solver="lbfgs", max_iter=500, random_state=cfg["seed"] + fold)),
                ("extra_trees_4f", stack["ExtraTreesClassifier"](n_estimators=cfg["classical_estimators"],
                    max_depth=16, min_samples_leaf=2, class_weight="balanced", n_jobs=1,
                    random_state=cfg["seed"] + fold))]
            predictions, results = [], []
            for model_name, model in outputs:
                require_memory_floor(psutil, RAM_FLOOR_MIB, f"Fold {fold} before {model_name}")
                model_started = time.perf_counter()
                rss_before = process.memory_info().rss
                if model_name == "binary_vqc":
                    probabilities = qnn_probability
                    probability_matrix = np.column_stack((1 - probabilities, probabilities))
                else:
                    model.fit(x_train, y_train)
                    probability_matrix = model.predict_proba(x_valid)
                pred, metrics = _metrics(y_valid, probability_matrix, stack)
                results.append({"fold": fold, "model": model_name, **metrics,
                    "runtime_seconds": (qnn.training_seconds + qnn.inference_seconds
                        if model_name == "binary_vqc" else time.perf_counter() - model_started),
                    "rss_before_bytes": int(rss_before),
                    "rss_after_bytes": int(process.memory_info().rss),
                    "qnn_peak_rss_bytes": int(qnn.peak_rss_bytes) if model_name == "binary_vqc" else None})
                for i, row in valid.iterrows():
                    predictions.append({"record_id": int(row.record_id), "origin_program": str(row.origin_program),
                        "fold": fold, "y_true": int(y_valid[i]), "model": model_name,
                        "y_pred": int(pred[i]), "prob_0": float(probability_matrix[i, 0]),
                        "prob_1": float(probability_matrix[i, 1])})
                require_memory_floor(psutil, RAM_FLOOR_MIB, f"Fold {fold} after {model_name}")
            elapsed = time.perf_counter() - started
            return {"predictions": predictions, "metrics": results,
                "feature_names": encoder.selected_names,
                "train_ids": sorted(map(int, train.record_id)),
                "validation_ids": sorted(map(int, valid.record_id)),
                "qnn_state": {"weights": np.asarray(qnn.weights).tolist(),
                    "readout_scale": float(qnn.readout_scale), "readout_bias": float(qnn.readout_bias),
                    "loss_history": [float(x) for x in qnn.loss_history]},
                "qnn_configuration": qnn.configuration(),
                "gradient_check": qnn.gradient_check_result,
                "train_records": len(train), "validation_records": len(valid),
                "train_origins": len(groups_train), "validation_origins": len(groups_valid),
                "elapsed_seconds": elapsed, "rss_bytes": process.memory_info().rss,
                "available_ram_before_bytes": available_before_training,
                "available_ram_after_bytes": psutil.virtual_memory().available,
                "estimated_peak_mib": estimate, "software_versions": versions}

        completed, was_resumed = load_or_train_fold(checkpoint_dir, fold, fingerprint,
            fold_configuration,
            train_fold, resume=resume)
        payload = completed["payload"]
        required_payload = {"predictions", "metrics", "qnn_state", "qnn_configuration",
                            "gradient_check", "feature_names", "train_ids", "validation_ids",
                            "software_versions", "train_records", "validation_records"}
        expected_models = {"binary_vqc", "logistic_4f", "extra_trees_4f"}
        if (not required_payload <= set(payload) or not payload.get("predictions") or
                {row.get("model") for row in payload["metrics"]} != expected_models or
                any(row.get("fold") != fold for row in payload["metrics"]) or
                {row.get("model") for row in payload["predictions"]} != expected_models or
                len(payload["predictions"]) != 3 * payload["validation_records"] or
                not {"weights", "readout_scale", "readout_bias", "loss_history"} <= set(payload["qnn_state"])):
            raise ValueError(f"Fold {fold} checkpoint is incomplete or incompatible with the expected model outputs")
        all_predictions.extend(payload["predictions"])
        fold_metrics.extend(payload["metrics"])
        fold_files.append({"fold": fold, "checkpoint": f"fold_checkpoints/fold_{fold:02d}.json",
            "resumed": was_resumed, "fingerprint": fingerprint})
        print(f"[fold {fold}/5] {'resumed' if was_resumed else 'completed'}; "
              f"train={payload['train_records']} validation={payload['validation_records']}; "
              f"RAM={psutil.virtual_memory().available // (1024 * 1024)} MiB", flush=True)
        del train, valid, x_train, x_valid, encoder

    require_memory_floor(psutil, RAM_FLOOR_MIB, "Before summary writing")
    sample_tmp = output / "sample_manifest.csv.tmp"
    with sample_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["record_id", "origin_program", "y", "fold", "role"])
        writer.writeheader()
        for row in sample_rows:
            row["origin_program"] = origin_by_id[row["record_id"]]
            row["y"] = label_by_id[row["record_id"]]
            writer.writerow(row)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(sample_tmp, sample_manifest_path)
    summaries = _summary_metrics(all_predictions, fold_metrics)
    prediction_path = output / "oof_predictions.csv"
    temp_predictions = output / "oof_predictions.csv.tmp"
    columns = ["record_id", "origin_program", "fold", "y_true", "model", "y_pred", "prob_0", "prob_1"]
    with temp_predictions.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(all_predictions)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp_predictions, prediction_path)
    report = {"schema": SCHEMA, "status": "completed_development_sampled_oof",
        "task": "QMutBench killed-vs-survived binary outcome; WOO=1, OPO=1, Survivor=0",
        "development_only": True, "frozen_test_accessed": False,
        "reference_fold_source": split["archive"], "reference_rows": split["rows"],
        "reference_fold_archive_sha256": split["archive_sha256"],
        "reference_origins": split["origins"], "reference_fold_counts": split["fold_counts"],
        "training_scope": "origin-stratified, capped development-only samples per training fold; not all 578,117 rows",
        "validation_scope": "origin-stratified, capped development-only sample from each held-out reference fold; not full-fold metrics",
        "sampled_records_used": int(len({int(r["record_id"]) for r in all_predictions})),
        "eligible_development_records": DEVELOPMENT_ROWS,
        "configuration": cfg, "software_versions": versions,
        "data_audit": loader_audit, "preflight": required_start,
        "fold_metrics": fold_metrics, "pooled_oof_metrics": summaries,
        "fold_checkpoints": fold_files, "prediction_file": prediction_path.name,
        "sample_manifest": sample_manifest_path.name,
        "elapsed_seconds": time.perf_counter() - campaign_start,
        "historical_comparator": {"name": "ExtraTrees mutation-aware 5-fold OOF",
            "accuracy": 0.79332903, "macro_f1": 0.7933135654357681,
            "balanced_accuracy": 0.79408748,
            "comparability": "historical score uses a distinct incumbent feature schema and all records; compare only as context, not a matched test"}}
    summary_path = output / "summary.json"
    tmp_summary = output / "summary.json.tmp"
    tmp_summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp_summary, summary_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/experiments/qmutbench_binary_qnn_development"))
    parser.add_argument("--resume", action="store_true", help="validate and reuse completed fold checkpoints")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--qubits", type=int, default=DEFAULT_CONFIG["qubits"])
    parser.add_argument("--depth", type=int, default=DEFAULT_CONFIG["depth"])
    parser.add_argument("--optimizer", choices=("adam", "gd"), default=DEFAULT_CONFIG["optimizer"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--max-train-per-class", type=int, default=DEFAULT_CONFIG["max_train_per_class_per_fold"])
    parser.add_argument("--max-validation-per-class", type=int, default=DEFAULT_CONFIG["max_validation_per_class_per_fold"])
    parser.add_argument("--per-origin-cap", type=int, default=DEFAULT_CONFIG["per_origin_class_cap"])
    parser.add_argument("--validate-config", action="store_true", help="validate configuration without ML imports or training")
    args = parser.parse_args(argv)
    config = {**DEFAULT_CONFIG, "seed": args.seed, "qubits": args.qubits, "depth": args.depth,
        "optimizer": args.optimizer, "learning_rate": args.learning_rate, "epochs": args.epochs,
        "batch_size": args.batch_size, "max_train_per_class_per_fold": args.max_train_per_class,
        "max_validation_per_class_per_fold": args.max_validation_per_class,
        "per_origin_class_cap": args.per_origin_cap}
    validate_config(config)
    if args.validate_config:
        print(json.dumps({"schema": SCHEMA, "configuration": config,
                          "target_map": LABEL_MAP, "folds": 5,
                          "status": "configuration_validated_no_data_loaded"}, indent=2))
        return 0
    root = Path.cwd()
    try:
        report = run_campaign(root, args.output, config, resume=args.resume)
    except MemoryError as exc:
        print(json.dumps({"status": "blocked_memory_guard", "error": str(exc),
                          "frozen_test_accessed": False}, indent=2))
        return 3
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "pooled_oof_metrics": report["pooled_oof_metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
