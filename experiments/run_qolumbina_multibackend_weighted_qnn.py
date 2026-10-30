"""Development-only weighted-loss QNN follow-up for Qolumbina imbalance."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import train_resource_light_qnn_tasks as q
from train_qolumbina_multibackend_candidate_sweep import (
    EXPECTED_HASHES,
    SOURCE,
    TASK_SPEC,
    read_csv,
    sha256,
)

OUT = ROOT / "reports/experiments/qolumbina_multibackend_candidate_sweep_20261029/weighted_loss_followup"
MODEL = {"name": "angle_reupload_4q_depth2_lr0035_balanced_loss", "qubits": 4,
         "features": 4, "encoding": "angle", "depth": 2, "reupload": True,
         "learning_rate": 0.035}
EPOCHS, BATCH, SEED = 4, 64, 28102026


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def run() -> dict:
    manifest_path = SOURCE / TASK_SPEC["manifest"]
    split_path = ROOT / "data/processed/qolumbina/seeded_mutant_detection_multibackend/manifests/group_folds.csv"
    manifest_hash, split_hash = sha256(manifest_path), sha256(split_path)
    if (manifest_hash != EXPECTED_HASHES["manifest"] or
            split_hash != EXPECTED_HASHES["split"]):
        raise ValueError("Pinned source or grouped split changed before weighted experiment")
    rows, split = read_csv(manifest_path), read_csv(split_path)
    by_id = {row["record_id"]: row for row in rows}
    split_by_id = {row["record_id"]: row for row in split}
    if set(by_id) != set(split_by_id) or len(by_id) != len(rows):
        raise ValueError("Weighted experiment manifest and registered split do not align")
    labels = np.asarray([int(by_id[row["record_id"]]["label"]) for row in split], dtype=int)
    groups = [str(by_id[row["record_id"]]["bug_id"]) for row in split]
    folds = np.asarray([int(row["fold"]) for row in split], dtype=int)
    if any(len({int(folds[i]) for i, group in enumerate(groups) if group == key}) != 1
           for key in set(groups)):
        raise ValueError("Original Qolumbina programs cross registered folds")
    x = np.asarray([q.features_from_text(" ".join(str(by_id[row["record_id"]].get(key, ""))
                                                       for key in TASK_SPEC["text"]), 8)
                    for row in split], dtype=float)
    if not q.gradient_check(MODEL)["passed"]:
        raise ArithmeticError("Weighted QNN gradient check failed")

    OUT.mkdir(parents=True, exist_ok=True)
    predictions, fold_results = [], []
    q.STEPS, q.BATCH = EPOCHS, BATCH
    started = time.perf_counter()
    for fold in sorted(set(folds)):
        valid = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        y_train, y_valid = labels[train], labels[valid]
        if set(y_train) != {0, 1} or set(y_valid) != {0, 1}:
            raise ValueError(f"Fold {fold} lacks both classes")
        x_train, x_valid, scale = q.minmax_fit_apply(x[train].tolist(), x[valid].tolist())
        x_train = np.asarray(x_train, dtype=float)[:, :MODEL["features"]]
        x_valid = np.asarray(x_valid, dtype=float)[:, :MODEL["features"]]
        counts = np.bincount(y_train, minlength=2)
        sample_weights = len(y_train) / (2.0 * counts[y_train])
        folder = OUT / MODEL["name"] / f"fold_{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        theta, head, bias, history, seconds = q.fit_model(
            x_train, y_train.astype(float), MODEL, SEED + int(fold),
            folder / "checkpoint.npz", folder / "training_history.json",
            provenance={"manifest_sha256": manifest_hash, "split_sha256": split_hash,
                        "fold": int(fold), "class_weighting": "inverse-frequency equal class mass",
                        "model": MODEL, "seed": SEED + int(fold)},
            sample_weights=sample_weights)
        probability = q.predict_prob(x_valid, theta, head, bias, MODEL)
        predicted = (probability >= 0.5).astype(int)
        report = {"task": "Qolumbina multi-backend seeded-mutant detection",
                  "model": MODEL["name"], "fold": int(fold),
                  "train_rows": len(train), "validation_rows": len(valid),
                  "train_class_counts": {str(c): int(counts[c]) for c in (0, 1)},
                  "validation_class_counts": {str(c): int(np.sum(y_valid == c)) for c in (0, 1)},
                  "class_weight_rule": "n_train / (2 * n_class)",
                  "training_seconds": seconds, "available_ram_mib": q.check_ram(),
                  "gradient_check": q.gradient_check(MODEL),
                  "metrics": q.metrics(y_valid, predicted)}
        np.savez(folder / "preprocessing.npz", minimum=np.asarray(scale["min"]),
                 maximum=np.asarray(scale["max"]), fold=int(fold), seed=SEED + int(fold),
                 manifest_sha256=manifest_hash, split_sha256=split_hash,
                 feature_fields=json.dumps(TASK_SPEC["text"]), class_weight_rule=report["class_weight_rule"])
        atomic_json(folder / "fold_result.json", report)
        for j, ix in enumerate(valid):
            source = by_id[split[ix]["record_id"]]
            predictions.append({"record_id": split[ix]["record_id"], "bug_id": source["bug_id"],
                                "fold": int(fold), "y_true": int(y_valid[j]),
                                "probability": float(probability[j]), "y_pred": int(predicted[j])})
        fold_results.append(report)

    y = [row["y_true"] for row in predictions]
    predicted = [row["y_pred"] for row in predictions]
    summary = {"status": "completed_weighted_loss_development_oof", "task": TASK_SPEC["manifest"],
               "model": MODEL, "rows": len(predictions), "groups": len(set(groups)),
               "folds": len(set(folds)), "epochs": EPOCHS, "batch_size": BATCH,
               "optimizer": "Adam", "seed": SEED, "class_weight_rule": "n_train / (2 * n_class)",
               "metrics": q.metrics(y, predicted), "fold_results": fold_results,
               "source_manifest_sha256": manifest_hash, "split_sha256": split_hash,
               "runtime_seconds": time.perf_counter() - started,
               "minimum_ram_floor_mib": q.FLOOR_MIB, "frozen_test_accessed": False,
               "scope_note": "Development grouped OOF; class weights fitted from each training fold only."}
    pred_path = OUT / "oof_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader(); writer.writerows(predictions)
    atomic_json(OUT / "summary.json", summary)
    manifest = {"schema": "qolumbina-weighted-qnn-development-v1", "status": "VERIFIED",
                "summary_sha256": sha256(OUT / "summary.json"),
                "oof_sha256": sha256(pred_path), "source_manifest_sha256": manifest_hash,
                "split_sha256": split_hash, "checkpoint_hashes": {
                    str(fold): sha256(OUT / MODEL["name"] / f"fold_{fold}" / "checkpoint.npz")
                    for fold in sorted(set(folds))},
                "frozen_test_accessed": False}
    atomic_json(OUT / "run_manifest.json", manifest)
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "metrics": result["metrics"],
                      "runtime_seconds": result["runtime_seconds"],
                      "output": str(OUT)}, indent=2))
