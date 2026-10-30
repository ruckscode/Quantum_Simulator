"""Matched fold-local balanced-logistic controls for the resumed QNN sweep."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028"
TASKS = ("bugs4q_robust_execution", "qsimbench_backend_fingerprint")
FEATURE_WIDTHS = (4, 8)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> str:
    import io
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader(); w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def bootstrap_delta(qnn: list[dict], classical: list[dict], iterations: int = 1000) -> dict:
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    import train_resource_light_qnn_tasks as q
    q_by_group, c_by_group = defaultdict(list), defaultdict(list)
    for row in qnn: q_by_group[row["group_id"]].append(row)
    for row in classical: c_by_group[row["group_id"]].append(row)
    if set(q_by_group) != set(c_by_group): raise ValueError("QNN and classical OOF group sets differ")
    groups = sorted(q_by_group)
    rng = np.random.default_rng(28102026)
    deltas = []
    for _ in range(iterations):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        qr = [r for g in sampled for r in q_by_group[str(g)]]
        cr = [r for g in sampled for r in c_by_group[str(g)]]
        deltas.append(q.metrics([int(r["y_true"]) for r in qr], [int(r["y_pred"]) for r in qr])["macro_f1"] -
                      q.metrics([int(r["y_true"]) for r in cr], [int(r["y_pred"]) for r in cr])["macro_f1"])
    return {"groups_resampled": len(groups), "iterations": iterations,
            "macro_f1_difference_qnn_minus_classical": float(np.mean(deltas)),
            "percentile_95_ci": [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))]}


def evaluate(task: str) -> dict:
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    import train_resource_light_qnn_tasks as q
    from run_resumable_qnn_hyperparameter_sweep import TASKS as TASK_SPECS, source_and_folds, CONFIGS
    spec = TASK_SPECS[task]
    manifest = CAMPAIGN / spec["manifest"]
    processed_path = ROOT / spec["processed"]
    rows, y, groups, folds, source_receipt = source_and_folds(task, spec, q)
    output = CAMPAIGN / task / "qnn_hyperparameter_sweep20_resumable_v2"
    run_path = output / "run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    qnn_path = output / "oof_predictions.csv"
    qnn_oof = read_csv(qnn_path)
    if run.get("status") != "completed" or run.get("source_receipt") != source_receipt:
        raise ValueError(f"QNN run manifest/source mismatch for {task}")
    if sha256(qnn_path) != run.get("oof_predictions_sha256"):
        raise ValueError(f"QNN OOF hash mismatch for {task}")
    expected_ids = {r["record_id"] for r in rows}
    if len(qnn_oof) != len(expected_ids) * len(CONFIGS):
        raise ValueError(f"Unexpected QNN OOF row count for {task}")
    qnn_by_config = {cfg["name"]: [r for r in qnn_oof if r["model"] == cfg["name"]] for cfg in CONFIGS}
    qnn_by_id = {r["record_id"]: r for r in rows}
    raw_x = np.asarray([q.features_from_text(r["code"], 8) for r in rows], dtype=np.float64)
    start = time.perf_counter()
    ram_start = round(psutil.virtual_memory().available / 2**20, 1)
    predictions_by_width = {width: [] for width in FEATURE_WIDTHS}
    fold_metrics = []
    for fold, (train, valid) in folds.items():
        xt, xv, _scaler = q.minmax_fit_apply(raw_x[train].tolist(), raw_x[valid].tolist())
        xt, xv = np.asarray(xt), np.asarray(xv)
        for width in FEATURE_WIDTHS:
            model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=28102026 + fold)
            model.fit(xt[:, :width], y[train])
            p1 = model.predict_proba(xv[:, :width])[:, list(model.classes_).index(1)]
            pred = (p1 >= .5).astype(int)
            for j, index in enumerate(valid):
                row = rows[int(index)]
                predictions_by_width[width].append({"record_id": row["record_id"], "group_id": groups[int(index)],
                                                    "fold": fold, "y_true": int(y[int(index)]),
                                                    "probability": float(p1[j]), "y_pred": int(pred[j])})
            fold_metrics.append({"feature_width": width, "fold": fold, "records": len(valid),
                                 "train_records": len(train), "group_overlap": False,
                                 **q.metrics(y[valid], pred)})
    classical_metrics = {}
    classical_paths = {}
    for width, pred_rows in predictions_by_width.items():
        pred_rows.sort(key=lambda r: r["record_id"])
        path = output / f"matched_classical_oof_{width}f.csv"
        digest = write_csv(path, pred_rows, ["record_id", "group_id", "fold", "y_true", "probability", "y_pred"])
        classical_paths[str(width)] = {"path": str(path.relative_to(ROOT)), "sha256": digest}
        classical_metrics[str(width)] = {"model": f"balanced_logistic_{width}f",
            **q.metrics([r["y_true"] for r in pred_rows], [r["y_pred"] for r in pred_rows]),
            "features": width, "transform": "same fold-local minmax scaling and BLAKE2 hashed source-code features as QNN"}
    comparisons = []
    for config in CONFIGS:
        qrows = qnn_by_config[config["name"]]
        if {r["record_id"] for r in qrows} != expected_ids: raise ValueError(f"QNN IDs incomplete for {config['name']}")
        width = int(config["features"])
        crows = predictions_by_width[width]
        for qr in qrows:
            cr = next((r for r in crows if r["record_id"] == qr["record_id"]), None)
            if cr is None or (qr["fold"], qr["group_id"], qr["y_true"]) != (str(cr["fold"]), cr["group_id"], str(cr["y_true"])):
                raise ValueError(f"Matched QNN/classical OOF metadata mismatch: {config['name']} {qr['record_id']}")
        qmetric = q.metrics([int(r["y_true"]) for r in qrows], [int(r["y_pred"]) for r in qrows])
        cmetric = q.metrics([int(r["y_true"]) for r in crows], [int(r["y_pred"]) for r in crows])
        comparisons.append({"qnn_model": config["name"], "feature_width": width,
                            "qnn": qmetric, "classical": cmetric,
                            "paired_group_bootstrap_macro_f1": bootstrap_delta(qrows, crows)})
    report = {"schema": "matched-qnn-classical-hyperparameter-sweep-v1", "status": "COMPLETED",
              "task": task, "rows": len(rows), "groups": len(set(groups)), "folds": len(folds),
              "source_sha256": sha256(manifest), "processed_manifest_sha256": sha256(processed_path),
              "qnn_run_manifest_sha256": sha256(run_path), "qnn_oof_sha256": sha256(qnn_path),
              "classical_oof": classical_paths, "classical_metrics_by_feature_width": classical_metrics,
              "comparisons": comparisons, "fold_classical_metrics": fold_metrics,
              "runtime_seconds": time.perf_counter() - start,
              "resources": {"available_ram_start_mib": ram_start,
                            "available_ram_end_mib": round(psutil.virtual_memory().available / 2**20, 1),
                            "max_process_rss_mib": round(psutil.Process().memory_info().rss / 2**20, 1)},
              "training_scope": "Development grouped OOF; no frozen test access",
              "frozen_test_accessed": False}
    path = output / "matched_classical_comparison.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    args = parser.parse_args()
    reports = [evaluate(t) for t in (TASKS if args.task == "all" else (args.task,))]
    print(json.dumps({"status": "COMPLETED", "tasks": [{"task": r["task"], "comparisons": len(r["comparisons"]),
          "output": str((CAMPAIGN / r["task"] / "qnn_hyperparameter_sweep20_resumable_v2/matched_classical_comparison.json").relative_to(ROOT))}
          for r in reports]}, indent=2))


if __name__ == "__main__":
    main()
