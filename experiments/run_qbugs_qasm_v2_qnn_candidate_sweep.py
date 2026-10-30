"""QNN-only architecture candidates on the duplicate-aware QBugs QASM task."""
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

SOURCE = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/qbugs_qasm_v2"
MANIFEST = SOURCE / "normalized_manifest.csv"
FOLD = ROOT / "data/processed/qbugs_hornbro/correct_buggy_circuit_state_qasm_v2/manifests/group_folds.csv"
PILOT = ROOT / "reports/experiments/resumable_qnn_pilot_qbugs_qasm_v2_20261029/qnn_oof_predictions.csv"
OUT = ROOT / "reports/experiments/qbugs_qasm_v2_candidate_sweep_20261030"
EXPECTED = {"manifest": "809a4492d02246bdb6e182ac670e803b24ec9ae563fb133663384c7e2e336120",
            "split": "03ed9fa4234951abc6c1cc36e24fbde72f09c58ab7f0ae92f0cc2c14e95d63c8"}
CONFIGS = [
    {"name": "amplitude_3q_depth1_lr004_candidate", "qubits": 3, "features": 8,
     "encoding": "amplitude", "depth": 1, "reupload": False, "learning_rate": .04},
    {"name": "angle_reupload_4q_depth2_lr0035_candidate", "qubits": 4, "features": 4,
     "encoding": "angle", "depth": 2, "reupload": True, "learning_rate": .035},
]
EPOCHS, BATCH, SEED = 4, 64, 28102026


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run() -> dict:
    if q.check_ram() < 700:
        raise MemoryError("QBugs QASM candidate preflight requires 700 MiB available")
    hashes = {"manifest": digest(MANIFEST), "split": digest(FOLD), "pilot_oof": digest(PILOT)}
    if hashes["manifest"] != EXPECTED["manifest"] or hashes["split"] != EXPECTED["split"]:
        raise ValueError(f"QBugs QASM source/split fingerprint changed: {hashes}")
    rows, folds, pilot = read(MANIFEST), read(FOLD), read(PILOT)
    by_id = {row["record_id"]: row for row in rows}
    fold_by_id = {row["record_id"]: row for row in folds}
    pilot_by_id = {row["record_id"]: row for row in pilot}
    if (len(by_id) != len(rows) or set(by_id) != set(fold_by_id) or set(by_id) != set(pilot_by_id)):
        raise ValueError("QBugs QASM manifest, split, and pilot OOF IDs differ")
    group_folds: dict[str, set[int]] = {}
    for rid, row in by_id.items():
        if (int(row["label"]) != int(pilot_by_id[rid]["y_true"]) or
                int(fold_by_id[rid]["fold"]) != int(pilot_by_id[rid]["fold"])):
            raise ValueError(f"QBugs QASM target or fold mismatch at {rid}")
        group_folds.setdefault(str(row["bug_id"]), set()).add(int(fold_by_id[rid]["fold"]))
    if any(len(assigned) != 1 for assigned in group_folds.values()):
        raise ValueError("QBugs QASM source-pair group crosses folds")

    y = np.asarray([int(by_id[row["record_id"]]["label"]) for row in folds], dtype=int)
    fold_ids = np.asarray([int(row["fold"]) for row in folds], dtype=int)
    groups = [str(by_id[row["record_id"]]["bug_id"]) for row in folds]
    x = np.asarray([q.features_from_text(by_id[row["record_id"]]["code"], 8)
                    for row in folds], dtype=float)
    q.STEPS, q.BATCH = EPOCHS, BATCH
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model_outputs = []
    for config in CONFIGS:
        model_dir = OUT / config["name"]
        existing_summary = model_dir / "summary.json"
        existing_manifest = model_dir / "run_manifest.json"
        if existing_summary.is_file() and existing_manifest.is_file():
            saved = json.loads(existing_manifest.read_text(encoding="utf-8"))
            prediction_path = model_dir / "oof_predictions.csv"
            if (saved.get("source_hashes") == hashes and
                    saved.get("status") == "completed_grouped_qnn_candidate_oof" and
                    prediction_path.is_file() and digest(prediction_path) == saved.get("oof_sha256") and
                    all(digest(model_dir / f"fold_{fold}" / "checkpoint.npz") == checkpoint_hash
                        for fold, checkpoint_hash in saved.get("checkpoint_hashes", {}).items()) and
                    len(saved.get("checkpoint_hashes", {})) == len(set(int(r["fold"]) for r in folds))):
                model_outputs.append(saved)
                print(json.dumps({"model": config["name"], "status": "reused_hash_verified"}), flush=True)
                continue
        predictions, fold_results = [], []
        if not q.gradient_check(config)["passed"]:
            raise ArithmeticError(f"Gradient check failed: {config['name']}")
        for fold in sorted(set(fold_ids)):
            train, valid = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
            if set(y[train]) != {0, 1} or set(y[valid]) != {0, 1}:
                raise ValueError(f"QBugs QASM fold {fold} lacks both classes")
            x_train, x_valid, scaler = q.minmax_fit_apply(x[train].tolist(), x[valid].tolist())
            x_train = np.asarray(x_train, dtype=float)[:, :config["features"]]
            x_valid = np.asarray(x_valid, dtype=float)[:, :config["features"]]
            if config["encoding"] == "amplitude":
                dim = 2 ** config["qubits"]
                x_train = np.pad(x_train, ((0, 0), (0, max(0, dim-x_train.shape[1]))))[:, :dim]
                x_valid = np.pad(x_valid, ((0, 0), (0, max(0, dim-x_valid.shape[1]))))[:, :dim]
            folder = OUT / config["name"] / f"fold_{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            provenance = {"task": "qbugs_qasm_v2_source_state", "manifest_sha256": hashes["manifest"],
                          "split_sha256": hashes["split"], "fold": int(fold),
                          "configuration": config, "seed": SEED + int(fold),
                          "target": "paired correct vs buggy OpenQASM outcome"}
            theta, head, bias, history, seconds = q.fit_model(
                x_train, y[train].astype(float), config, SEED + int(fold),
                folder / "checkpoint.npz", folder / "training_history.json", provenance=provenance)
            probability = q.predict_prob(x_valid, theta, head, bias, config)
            predicted = (probability >= .5).astype(int)
            metrics = q.metrics(y[valid], predicted)
            result = {"fold": int(fold), "train_rows": len(train), "validation_rows": len(valid),
                      "train_groups": len({groups[i] for i in train}),
                      "validation_groups": len({groups[i] for i in valid}),
                      "training_seconds": seconds, "available_ram_mib": q.check_ram(),
                      "gradient_check": q.gradient_check(config), "metrics": metrics}
            np.savez(folder / "preprocessing.npz", minimum=np.asarray(scaler["min"]),
                     maximum=np.asarray(scaler["max"]), fold=int(fold), seed=SEED + int(fold),
                     manifest_sha256=hashes["manifest"], split_sha256=hashes["split"])
            atomic_json(folder / "fold_result.json", result)
            fold_results.append(result)
            for j, ix in enumerate(valid):
                row = folds[ix]
                predictions.append({"record_id": row["record_id"], "group_id": groups[ix],
                                    "fold": int(fold), "y_true": int(y[ix]),
                                    "probability": float(probability[j]), "y_pred": int(predicted[j])})
        pred_path = OUT / config["name"] / "oof_predictions.csv"
        with pred_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader(); writer.writerows(predictions)
        summary = {"status": "completed_grouped_qnn_candidate_oof", "task": "qbugs_qasm_v2_source_state",
                   "target_semantics": "paired correct versus buggy circuit outcome; not hardware diagnosis",
                   "rows": len(rows), "groups": len(group_folds), "folds": len(set(fold_ids)),
                   "label_counts": {str(c): int(np.sum(y == c)) for c in (0, 1)},
                   "configuration": config, "epochs": EPOCHS, "batch_size": BATCH,
                   "optimizer": "Adam", "seed": SEED,
                   "metrics": q.metrics([r["y_true"] for r in predictions], [r["y_pred"] for r in predictions]),
                   "fold_results": fold_results, "source_hashes": hashes,
                   "oof_sha256": digest(pred_path), "frozen_test_accessed": False,
                   "runtime_seconds": sum(r["training_seconds"] for r in fold_results)}
        atomic_json(OUT / config["name"] / "summary.json", summary)
        summary["summary_sha256"] = digest(OUT / config["name"] / "summary.json")
        summary["checkpoint_hashes"] = {str(fold): digest(OUT / config["name"] / f"fold_{fold}" / "checkpoint.npz")
                                        for fold in sorted(set(fold_ids))}
        atomic_json(OUT / config["name"] / "run_manifest.json", summary)
        model_outputs.append(summary)
        print(json.dumps({"model": config["name"], "metrics": summary["metrics"],
                          "runtime_seconds": summary["runtime_seconds"]}), flush=True)

    campaign = {"schema": "qbugs-qasm-v2-candidate-sweep-v1", "status": "VERIFIED",
                "task": "duplicate-aware paired correct/buggy QBugs QASM",
                "source_hashes": hashes, "rows": len(rows), "groups": len(group_folds),
                "folds": len(set(fold_ids)), "models": model_outputs,
                "elapsed_seconds": time.perf_counter() - started,
                "minimum_ram_floor_mib": q.FLOOR_MIB, "frozen_test_accessed": False,
                "classical_models_trained": False}
    atomic_json(OUT / "run_manifest.json", campaign)
    print(json.dumps({"status": "VERIFIED", "output": str(OUT),
                      "elapsed_seconds": campaign["elapsed_seconds"],
                      "available_ram_mib": q.check_ram()}, indent=2))
    return campaign


if __name__ == "__main__":
    run()
