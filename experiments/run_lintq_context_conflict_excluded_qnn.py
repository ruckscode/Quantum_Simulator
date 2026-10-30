"""Document and train a LintQ context variant excluding one contradictory input component."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import train_resource_light_qnn_tasks as q

SOURCE = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/lintq_context_v2"
OUT = ROOT / "reports/experiments/lintq_context_v2_conflict_excluded_20261029"
MODEL = {"name": "angle_reupload_4q_depth2_lr0035", "qubits": 4, "features": 4,
         "encoding": "angle", "depth": 2, "reupload": True, "learning_rate": 0.035}
EPOCHS, BATCH, SEED = 4, 64, 28102026


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run() -> dict:
    if q.check_ram() < 700:
        raise MemoryError("LintQ variant preflight requires 700 MiB available")
    manifest_path = SOURCE / "normalized_manifest.csv"
    split_path = SOURCE / "group_folds.csv"
    manifest, split = read_csv(manifest_path), read_csv(split_path)
    split_by_id = {row["record_id"]: row for row in split}
    if len(split_by_id) != len(split) or set(split_by_id) != {r["record_id"] for r in manifest}:
        raise ValueError("LintQ manifest and existing grouped split IDs differ")

    by_text: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        digest = hashlib.sha256(row["code"].encode("utf-8")).hexdigest()
        by_text[digest].append(row)
    conflicting = [component for component in by_text.values()
                   if len({int(row["label"]) for row in component}) > 1]
    if len(conflicting) != 1 or len(conflicting[0]) != 3:
        raise ValueError(f"Expected the audited single three-row conflicting component; found {[len(x) for x in conflicting]}")
    rejected = conflicting[0]
    rejected_ids = {row["record_id"] for row in rejected}
    accepted = [row for row in manifest if row["record_id"] not in rejected_ids]
    fold_rows = [{"record_id": row["record_id"], "group_id": row["model_group_id"],
                  "fold": int(split_by_id[row["record_id"]]["fold"])} for row in accepted]
    fold_by_id = {row["record_id"]: row["fold"] for row in fold_rows}
    for group in {row["group_id"] for row in fold_rows}:
        if len({row["fold"] for row in fold_rows if row["group_id"] == group}) != 1:
            raise ValueError(f"Group crosses folds after conflict-component exclusion: {group}")

    OUT.mkdir(parents=True, exist_ok=True)
    rejected_copy = [{**row, "rejection_reason": "byte_identical_model_input_has_conflicting_manual_labels"}
                     for row in rejected]
    fields = list(manifest[0])
    write_csv(OUT / "accepted_manifest.csv", accepted, fields)
    write_csv(OUT / "rejected_conflict_component.csv", rejected_copy,
              fields + ["rejection_reason"])
    write_csv(OUT / "group_folds.csv", fold_rows, ["record_id", "group_id", "fold"])

    labels = np.asarray([int(row["label"]) for row in accepted], dtype=int)
    groups = [row["model_group_id"] for row in accepted]
    fold_ids = np.asarray([fold_by_id[row["record_id"]] for row in accepted], dtype=int)
    features = np.asarray([q.features_from_text(row["code"], 8) for row in accepted], dtype=float)
    q.STEPS, q.BATCH = EPOCHS, BATCH
    predictions, fold_metrics = [], []
    started = time.perf_counter()
    for fold in sorted(set(fold_ids)):
        valid = np.flatnonzero(fold_ids == fold)
        train = np.flatnonzero(fold_ids != fold)
        if set(labels[train]) != {0, 1} or set(labels[valid]) != {0, 1}:
            raise ValueError(f"Fold {fold} lacks class coverage after conflict exclusion")
        x_train, x_valid, scaler = q.minmax_fit_apply(features[train].tolist(), features[valid].tolist())
        x_train = np.asarray(x_train, dtype=float)[:, :MODEL["features"]]
        x_valid = np.asarray(x_valid, dtype=float)[:, :MODEL["features"]]
        if not q.gradient_check(MODEL)["passed"]:
            raise ArithmeticError("LintQ QNN gradient check failed")
        folder = OUT / MODEL["name"] / f"fold_{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        provenance = {"variant": "exclude_full_conflicting_exact_context_component",
                      "source_sha256": sha256(manifest_path), "split_sha256": sha256(split_path),
                      "accepted_sha256": sha256(OUT / "accepted_manifest.csv"),
                      "rejected_ids": sorted(rejected_ids), "fold": int(fold),
                      "model": MODEL, "seed": SEED + int(fold)}
        theta, head, bias, history, seconds = q.fit_model(
            x_train, labels[train].astype(float), MODEL, SEED + int(fold),
            folder / "checkpoint.npz", folder / "training_history.json", provenance=provenance)
        probability = q.predict_prob(x_valid, theta, head, bias, MODEL)
        predicted = (probability >= .5).astype(int)
        fold_record = {"fold": int(fold), "train_rows": len(train), "validation_rows": len(valid),
                       "train_groups": len(set(groups[i] for i in train)),
                       "validation_groups": len(set(groups[i] for i in valid)),
                       "training_seconds": seconds, "available_ram_mib": q.check_ram(),
                       "gradient_check": q.gradient_check(MODEL),
                       "metrics": q.metrics(labels[valid], predicted)}
        np.savez(folder / "preprocessing.npz", minimum=np.asarray(scaler["min"]),
                 maximum=np.asarray(scaler["max"]), fold=int(fold), seed=SEED + int(fold),
                 manifest_sha256=sha256(manifest_path), split_sha256=sha256(split_path),
                 rejected_ids=json.dumps(sorted(rejected_ids)))
        atomic_json(folder / "fold_result.json", fold_record)
        for j, index in enumerate(valid):
            predictions.append({"record_id": accepted[index]["record_id"],
                                "group_id": accepted[index]["model_group_id"], "fold": int(fold),
                                "y_true": int(labels[index]), "probability": float(probability[j]),
                                "y_pred": int(predicted[j])})
        fold_metrics.append(fold_record)

    prediction_path = OUT / "oof_predictions.csv"
    write_csv(prediction_path, predictions, list(predictions[0]))
    summary = {"status": "completed_grouped_qnn_oof", "task": "LintQ warning TP vs FP with source context",
               "variant": "exclude the entire byte-identical code-context component with contradictory labels",
               "source_rows": len(manifest), "accepted_rows": len(accepted), "rejected_rows": len(rejected),
               "rejected_record_ids": sorted(rejected_ids),
               "rejection_rule": "All rows in an exact model-input component are excluded if labels conflict; no label is selected or fabricated.",
               "groups": len(set(groups)), "folds": len(set(fold_ids)),
               "label_counts": {str(c): int(np.sum(labels == c)) for c in (0, 1)},
               "model": MODEL, "epochs": EPOCHS, "batch_size": BATCH, "optimizer": "Adam",
               "seed": SEED, "feature": "8-bin deterministic BLAKE2 text counts; first four used",
               "scaling": "minmax fitted separately on each training fold",
               "metrics": q.metrics([r["y_true"] for r in predictions], [r["y_pred"] for r in predictions]),
               "fold_metrics": fold_metrics, "runtime_seconds": time.perf_counter() - started,
               "source_manifest_sha256": sha256(manifest_path), "original_split_sha256": sha256(split_path),
               "accepted_manifest_sha256": sha256(OUT / "accepted_manifest.csv"),
               "rejected_rows_sha256": sha256(OUT / "rejected_conflict_component.csv"),
               "split_sha256": sha256(OUT / "group_folds.csv"),
               "oof_sha256": sha256(prediction_path), "frozen_test_accessed": False,
               "development_only": True, "classical_models_trained": False}
    atomic_json(OUT / "summary.json", summary)
    summary["summary_sha256"] = sha256(OUT / "summary.json")
    summary["checkpoint_hashes"] = {str(fold): sha256(OUT / MODEL["name"] / f"fold_{fold}" / "checkpoint.npz")
                                    for fold in sorted(set(fold_ids))}
    atomic_json(OUT / "run_manifest.json", summary)
    print(json.dumps({"status": summary["status"], "accepted_rows": len(accepted),
                      "rejected_rows": len(rejected), "metrics": summary["metrics"],
                      "runtime_seconds": summary["runtime_seconds"], "output": str(OUT)}, indent=2))
    return summary


if __name__ == "__main__":
    run()
