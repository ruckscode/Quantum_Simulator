"""Resume the RAM-stopped Bugs4Q-Robust/QSimBench 20-epoch QNN sweep.

The historical sweep checkpoints contain inference weights but no Adam or RNG
state. Completed epoch-20 folds are reused for scoring only; all other folds are
trained in a new, provenance-bound, resumable output. Historical artifacts are
left untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028"
CONFIGS = [
    {"name": "amplitude_hybrid_3q_depth1_lr004", "qubits": 3, "features": 8, "encoding": "amplitude", "depth": 1, "reupload": False, "learning_rate": .04},
    {"name": "amplitude_hybrid_3q_depth2_lr004", "qubits": 3, "features": 8, "encoding": "amplitude", "depth": 2, "reupload": False, "learning_rate": .04},
    {"name": "angle_reupload_4q_depth2_lr002", "qubits": 4, "features": 4, "encoding": "angle", "depth": 2, "reupload": True, "learning_rate": .02},
    {"name": "angle_reupload_4q_depth2_lr0035", "qubits": 4, "features": 4, "encoding": "angle", "depth": 2, "reupload": True, "learning_rate": .035},
    {"name": "angle_reupload_4q_depth2_lr007", "qubits": 4, "features": 4, "encoding": "angle", "depth": 2, "reupload": True, "learning_rate": .07},
    {"name": "angle_4q_depth2_lr004", "qubits": 4, "features": 4, "encoding": "angle", "depth": 2, "reupload": False, "learning_rate": .04},
]
TASKS = {
    "bugs4q_robust_execution": {
        "manifest": "bugs4q_robust_execution/normalized_manifest.csv",
        "processed": "data/processed/bugs4q_robust/harness_execution_outcome_v2/manifest.json",
        "split_reference": "bugs4q_robust_execution/resource_light_qnn/oof_predictions.csv",
        "group": "bug_id", "rows": 2220, "groups": 37,
    },
    "qsimbench_backend_fingerprint": {
        "manifest": "qsimbench_backend_fingerprint/normalized_manifest.csv",
        "processed": "data/processed/qsimbench/backend_fingerprint_v3/manifest.json",
        "split_reference": "qsimbench_backend_fingerprint/resource_light_qnn/oof_predictions.csv",
        "group": "bug_id", "rows": 300, "groups": 150,
    },
}
EPOCHS = 20
BATCH = 64
SEED = 28102026
FLOOR_MIB = 500


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> str:
    import io
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return hashlib.sha256(data).hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def source_and_folds(task: str, spec: dict, q) -> tuple[list[dict], np.ndarray, list[str], dict[int, tuple[np.ndarray, np.ndarray]], dict]:
    manifest = CAMPAIGN / spec["manifest"]
    processed_path = ROOT / spec["processed"]
    processed = json.loads(processed_path.read_text(encoding="utf-8"))
    if not processed.get("validation_pass") or processed.get("counts", {}).get("accepted") != spec["rows"]:
        raise ValueError(f"Processed source manifest is not validated for {task}")
    if sha256(ROOT / processed["input_path"]) != processed["input_sha256"]:
        raise ValueError(f"Validated input hash changed for {task}")
    rows = load_csv(manifest)
    if len(rows) != spec["rows"] or len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError(f"Unexpected row count or duplicate IDs for {task}")
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    groups = [str(r[spec["group"]]) for r in rows]
    if len(set(groups)) != spec["groups"] or set(y) != {0, 1}:
        raise ValueError(f"Unexpected source groups/classes for {task}")
    ref = load_csv(CAMPAIGN / spec["split_reference"])
    ref_assignment = {}
    for row in ref:
        rid = row["record_id"]
        value = (int(row["fold"]), str(row["group_id"]), int(row["y_true"]))
        if rid in ref_assignment and ref_assignment[rid] != value:
            raise ValueError(f"Existing baseline folds conflict for {rid}")
        ref_assignment[rid] = value
    if set(ref_assignment) != {r["record_id"] for r in rows}:
        raise ValueError(f"Existing grouped baseline OOF does not cover {task} source IDs")
    expected_folds = {}
    for fold, (tr, va) in enumerate(q.grouped_folds(y, groups), 1):
        for ix in va:
            row = rows[int(ix)]
            expected_folds[row["record_id"]] = fold
            old = ref_assignment[row["record_id"]]
            if old != (fold, groups[int(ix)], int(y[int(ix)])):
                raise ValueError(f"Saved baseline split/label/group differs from current source: {row['record_id']}")
        if set(groups[i] for i in tr) & set(groups[i] for i in va):
            raise ValueError(f"Group leakage in fold {fold}")
        if set(y[tr]) != {0, 1} or set(y[va]) != {0, 1}:
            raise ValueError(f"Fold {fold} lacks both classes for {task}")
    folds = {}
    for fold, (tr, va) in enumerate(q.grouped_folds(y, groups), 1):
        folds[fold] = (tr, va)
    source_hash = sha256(manifest)
    split_hash = hashlib.sha256(json.dumps(expected_folds, sort_keys=True).encode()).hexdigest()
    return rows, y, groups, folds, {
        "manifest_path": str(manifest.relative_to(ROOT)), "source_sha256": source_hash,
        "processed_manifest": str(processed_path.relative_to(ROOT)), "processed_manifest_sha256": sha256(processed_path),
        "processed_input_sha256": processed["input_sha256"], "baseline_split_reference": spec["split_reference"],
        "baseline_split_reference_sha256": sha256(CAMPAIGN / spec["split_reference"]),
        "group_fold_assignment_sha256": split_hash, "rows": len(rows), "groups": len(set(groups)),
    }


def model_features(raw: np.ndarray, config: dict) -> np.ndarray:
    x = raw[:, :config["features"]]
    if config["encoding"] == "amplitude":
        x = np.pad(x, ((0, 0), (0, max(0, 2 ** config["qubits"] - x.shape[1]))))[:, :2 ** config["qubits"]]
    return x


def task_run(task: str, *, resume: bool) -> dict:
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    import train_resource_light_qnn_tasks as q
    q.STEPS, q.BATCH, q.SEED = EPOCHS, BATCH, SEED
    if q.check_ram() < 700:
        raise MemoryError("QNN sweep launch requires 700 MiB available; existing 500 MiB training floor remains active")
    spec = TASKS[task]
    rows, y, groups, folds, source_receipt = source_and_folds(task, spec, q)
    manifest = CAMPAIGN / spec["manifest"]
    raw_x = np.asarray([q.features_from_text(row["code"], 8) for row in rows], dtype=np.float64)
    old_output = CAMPAIGN / task / "qnn_hyperparameter_sweep20"
    output = CAMPAIGN / task / "qnn_hyperparameter_sweep20_resumable_v2"
    code_hash = sha256(Path(__file__))
    base_config = {"task": task, "epochs": EPOCHS, "batch_size": BATCH, "seed": SEED,
                  "configs": CONFIGS, "feature_policy": "8 hashed-token counts of source code; fold-local minmax; model-width truncation/padding",
                  "group_split": "existing 5-fold GroupKFold verified against saved resource_light_qnn OOF",
                  "source_receipt": source_receipt, "runner_sha256": code_hash,
                  "frozen_test_accessed": False}
    progress_path = output / "progress.json"
    run_config = base_config
    config_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError(f"Output exists; use --resume after verifying its progress: {output}")
        if progress_path.exists():
            prior = json.loads(progress_path.read_text(encoding="utf-8"))
            prior_config = prior.get("config", {})
            comparable_prior = {k: v for k, v in prior_config.items() if k != "runner_sha256"}
            comparable_current = {k: v for k, v in base_config.items() if k != "runner_sha256"}
            if comparable_prior != comparable_current:
                raise RuntimeError("Existing sweep progress is incompatible with current source/config; refusing resume")
            # Preserve the original code receipt and config hash across a safe
            # implementation-only fix while retaining strict data/config checks.
            run_config = prior_config
            config_hash = prior["config_sha256"]
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(progress_path, {"status": "running", "config_sha256": config_hash,
                                "config": run_config, "completed_folds": [], "frozen_test_accessed": False})
    legacy_checks = []
    history_gaps = []
    all_predictions = []
    fold_results = []
    start_task = time.perf_counter()
    os.environ["LIGHT_QNN_RESUME"] = "1" if resume else "0"
    for fold, (train, valid) in folds.items():
        xt, xv, scaler = q.minmax_fit_apply(raw_x[train].tolist(), raw_x[valid].tolist())
        xt, xv = np.asarray(xt, dtype=np.float64), np.asarray(xv, dtype=np.float64)
        for config in CONFIGS:
            check = q.gradient_check(config)
            if not check["passed"]:
                raise ArithmeticError(f"Gradient check failed: task={task}, fold={fold}, config={config['name']}")
            train_x, valid_x = model_features(xt, config), model_features(xv, config)
            legacy_checkpoint = old_output / config["name"] / f"fold_{fold}" / "checkpoint.npz"
            legacy_receipt = None
            theta = head = bias = None
            if legacy_checkpoint.is_file():
                with np.load(legacy_checkpoint, allow_pickle=False) as saved:
                    saved_config = json.loads(str(saved["config"].item()))
                    epoch = int(saved["epoch"])
                    expected_config = {k: config[k] for k in config}
                    if saved_config != expected_config:
                        raise ValueError(f"Legacy checkpoint configuration mismatch: {legacy_checkpoint}")
                    if epoch == EPOCHS:
                        theta, head, bias = saved["theta"].copy(), saved["head"].copy(), float(saved["bias"])
                        legacy_receipt = {"kind": "legacy_inference_only_checkpoint_reused", "path": str(legacy_checkpoint.relative_to(ROOT)),
                                          "sha256": sha256(legacy_checkpoint), "epoch": epoch,
                                          "optimizer_rng_state_present": False,
                                          "source_sha256": source_receipt["source_sha256"],
                                          "fold_assignment_sha256": source_receipt["group_fold_assignment_sha256"]}
                    elif epoch > EPOCHS:
                        raise ValueError(f"Unexpected legacy epoch beyond requested budget: {legacy_checkpoint}")
                    else:
                        legacy_receipt = {"kind": "legacy_partial_checkpoint_preserved_but_not_resumed",
                                          "path": str(legacy_checkpoint.relative_to(ROOT)),
                                          "sha256": sha256(legacy_checkpoint), "epoch": epoch,
                                          "reason": "Legacy checkpoint has no Adam moments or RNG state; exact continuation is impossible."}
                if legacy_receipt:
                    legacy_checks.append({**legacy_receipt, "config": config["name"], "fold": fold})
            folder = output / config["name"] / f"fold_{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            provenance = {"source_sha256": source_receipt["source_sha256"], "split_sha256": source_receipt["group_fold_assignment_sha256"],
                          "task": task, "fold": fold, "seed": SEED + fold, "epochs": EPOCHS,
                          "config_sha256": config_hash}
            checkpoint_path, history_path = folder / "checkpoint.npz", folder / "training_history.json"
            checkpoint_kind = "resumable_adam_rng_checkpoint"
            if theta is None:
                if checkpoint_path.is_file():
                    with np.load(checkpoint_path, allow_pickle=False) as saved:
                        checkpoint_epoch = int(saved["epoch"])
                    existing_history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
                    recorded_epochs = {int(item["epoch"]) for item in existing_history if item.get("epoch") is not None}
                    missing_epochs = [epoch for epoch in range(1, checkpoint_epoch + 1) if epoch not in recorded_epochs]
                    if missing_epochs:
                        gap = {"schema": "qnn-training-history-gap-v1", "task": task, "model": config["name"],
                               "fold": fold, "missing_loss_epochs": missing_epochs,
                               "checkpoint_epoch": checkpoint_epoch,
                               "checkpoint_sha256": sha256(checkpoint_path),
                               "reason": "A checkpoint was durably saved, then the history replace hit a transient Windows sharing lock. Model/optimizer/RNG state remains resumable; no loss values were fabricated."}
                        gap_path = folder / "training_history_gap.json"
                        atomic_json(gap_path, gap)
                        history_gaps.append({**gap, "path": str(gap_path.relative_to(ROOT))})
                theta, head, bias, history, seconds = q.fit_model(
                    train_x, y[train].astype(float), config, SEED + fold, checkpoint_path, history_path,
                    provenance=provenance)
                checkpoint_hash = sha256(checkpoint_path)
            else:
                seconds = 0.0
                history = []
                checkpoint_kind = "legacy_inference_only_checkpoint_reused"
                checkpoint_hash = sha256(legacy_checkpoint)
                np.savez(folder / "inference_snapshot.npz", theta=theta, head=head, bias=np.asarray(bias),
                         config=np.asarray(json.dumps(config, sort_keys=True)), epoch=np.asarray(EPOCHS),
                         source_sha256=np.asarray(source_receipt["source_sha256"]),
                         legacy_checkpoint_sha256=np.asarray(checkpoint_hash),
                         fold_assignment_sha256=np.asarray(source_receipt["group_fold_assignment_sha256"]))
            probability = q.predict_prob(valid_x, theta, head, bias, config)
            prediction = (probability >= .5).astype(int)
            metric = q.metrics(y[valid], prediction)
            row = {"task": task, "model": config["name"], "fold": fold, "records": len(valid),
                   "groups": len({groups[int(i)] for i in valid}), "train_records": len(train),
                   "training_seconds": seconds, "available_ram_mib": q.check_ram(),
                   "gradient_check": check, "metrics": metric, "checkpoint_kind": checkpoint_kind,
                   "checkpoint_path": str((legacy_checkpoint if theta is not None and legacy_receipt and legacy_receipt["kind"] == "legacy_inference_only_checkpoint_reused" else checkpoint_path).relative_to(ROOT)),
                   "checkpoint_sha256": checkpoint_hash, "legacy_checkpoint_receipt": legacy_receipt}
            fold_results.append(row)
            scaler_path = folder / "preprocessing.npz"
            np.savez(scaler_path, minimum=np.asarray(scaler["min"]), maximum=np.asarray(scaler["max"]),
                     source_sha256=source_receipt["source_sha256"], fold=fold, seed=SEED + fold)
            for j, index in enumerate(valid):
                all_predictions.append({"record_id": rows[int(index)]["record_id"], "group_id": groups[int(index)],
                                        "fold": fold, "model": config["name"], "y_true": int(y[index]),
                                        "probability": float(probability[j]), "y_pred": int(prediction[j])})
        atomic_json(progress_path, {"status": "running", "config_sha256": config_hash, "config": run_config,
                                    "completed_folds": sorted({r["fold"] for r in fold_results}),
                                    "legacy_checkpoint_receipts": legacy_checks,
                                    "training_history_gaps": history_gaps,
                                    "fold_results": fold_results, "frozen_test_accessed": False})
        print(json.dumps({"task": task, "completed_fold": fold,
                          "available_ram_mib": q.check_ram()}), flush=True)
    predictions_path = output / "oof_predictions.csv"
    prediction_hash = atomic_csv(predictions_path, all_predictions,
                                 ["record_id", "group_id", "fold", "model", "y_true", "probability", "y_pred"])
    summaries = []
    for config in CONFIGS:
        rows_model = [r for r in all_predictions if r["model"] == config["name"]]
        metric = q.metrics([r["y_true"] for r in rows_model], [r["y_pred"] for r in rows_model])
        summaries.append({"model": config["name"], "configuration": config, "records": len(rows_model),
                          "folds": len(folds), **metric,
                          "training_seconds": sum(r["training_seconds"] for r in fold_results if r["model"] == config["name"])})
    result = {"status": "completed", "task": task, "rows": len(rows), "groups": len(set(groups)),
              "epochs": EPOCHS, "batch_size": BATCH, "seed": SEED, "optimizer": "Adam",
              "source_receipt": source_receipt, "config": run_config, "config_sha256": config_hash,
              "results": summaries, "fold_results": fold_results,
              "legacy_checkpoint_receipts": legacy_checks,
              "training_history_gaps": history_gaps,
              "oof_predictions_sha256": prediction_hash,
              "runtime_seconds": time.perf_counter() - start_task,
              "memory_floor_mib": FLOOR_MIB, "frozen_test_accessed": False,
              "evaluation_scope": "development GroupKFold OOF only; no frozen test"}
    atomic_json(output / "summary.json", result)
    atomic_json(output / "run_manifest.json", result)
    atomic_json(progress_path, {"status": "completed", "config_sha256": config_hash,
                                "completed_folds": list(folds), "summary_sha256": sha256(output / "summary.json"),
                                "oof_sha256": prediction_hash, "legacy_checkpoint_receipts": legacy_checks,
                                "frozen_test_accessed": False})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        result = task_run(args.task, resume=args.resume)
    except MemoryError as exc:
        spec = TASKS[args.task]
        output = CAMPAIGN / args.task / "qnn_hyperparameter_sweep20_resumable_v2"
        progress_path = output / "progress.json"
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            progress.update({"status": "partial_stopped_by_ram_guard", "stop_reason": str(exc),
                             "frozen_test_accessed": False})
            try:
                import psutil
                progress["available_ram_mib_at_stop"] = round(psutil.virtual_memory().available / 2**20, 1)
            except Exception:
                progress["available_ram_mib_at_stop"] = None
            atomic_json(progress_path, progress)
        print(json.dumps({"status": "partial_stopped_by_ram_guard", "task": args.task,
                          "reason": str(exc), "output": str(output.relative_to(ROOT))}, indent=2))
        raise SystemExit(2)
    print(json.dumps({"status": result["status"], "task": result["task"], "rows": result["rows"],
                      "groups": result["groups"], "runtime_seconds": result["runtime_seconds"],
                      "output": str((CAMPAIGN / args.task / "qnn_hyperparameter_sweep20_resumable_v2").relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
