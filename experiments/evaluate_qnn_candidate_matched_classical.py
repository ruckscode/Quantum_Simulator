"""Evaluate fixed classical controls on the exact inputs and folds of a QNN sweep.

This is a development-only comparison. It reuses the QNN sweep's hashed text
features, per-fold min-max scaling, saved group folds, and any source-restricted
training rows. It never opens frozen-test artifacts.
"""
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
import train_remaining_standalone_qml_tasks as qml_sweep
import train_resource_light_qnn_tasks as qml

SOURCE = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028"
SWEEP = ROOT / "reports/experiments/qml_standalone_accuracy_optimization_20261028/remaining_task_sweeps"
OUT = SWEEP / "matched_classical_controls"
SEED = 20261028
TASK_DIRS = {
    "Bugs4Q core": "bugs4q_core",
    "Bugs4Q-NA": "bugs4q_na",
    "Bugs4Q pooled": "bugs4q_pooled",
    "Bugs4Q core-only transfer": "bugs4q_core_only_transfer",
    "Bugs4Q-NA-only transfer": "bugs4q_na_only_transfer",
    "LintQ": "lintq",
    "QBugs HornBro subset": "qbugs_hornbro_subset",
    "Qolumbina ideal": "qolumbina_ideal",
    "Quantum platform comment task": "quantum_platform_comment_task",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    labels = sorted(set(y.tolist()) | set(pred.tolist()))
    per_class = {}
    f1s, recalls = [], []
    for label in labels:
        tp = int(np.sum((y == label) & (pred == label)))
        fp = int(np.sum((y != label) & (pred == label)))
        fn = int(np.sum((y == label) & (pred != label)))
        support = int(np.sum(y == label))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(label)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1s.append(f1)
        recalls.append(recall)
    confusion = [[int(np.sum((y == a) & (pred == b))) for b in labels] for a in labels]
    return {"accuracy": float(np.mean(y == pred)), "macro_f1": float(np.mean(f1s)),
            "balanced_accuracy": float(np.mean(recalls)), "per_class": per_class,
            "confusion_matrix": confusion, "confusion_matrix_labels": labels}


def main() -> None:
    import sklearn
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    available = qml.check_ram()
    if available < 600:
        raise MemoryError(f"Matched classical control preflight requires 600 MiB; found {available:.1f} MiB")
    models = {
        "majority_dummy": lambda seed: DummyClassifier(strategy="most_frequent"),
        "logistic_balanced": lambda seed: LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed),
        "decision_tree_balanced": lambda seed: DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, class_weight="balanced", random_state=seed),
        "random_forest_balanced": lambda seed: RandomForestClassifier(n_estimators=100, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=1),
        "extra_trees_balanced": lambda seed: ExtraTreesClassifier(n_estimators=100, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=1),
    }
    all_results = []
    started = time.perf_counter()
    for task_name, output_name in TASK_DIRS.items():
        if qml.check_ram() < 550:
            raise MemoryError(f"Stopping before task {task_name}: available RAM below 550 MiB")
        spec = qml_sweep.TASKS[task_name]
        manifest_path = SOURCE / spec["manifest"]
        manifest = read_csv(manifest_path)
        qnn_summary = json.loads((SWEEP / output_name / "summary.json").read_text(encoding="utf-8"))
        reference_predictions = read_csv(SWEEP / output_name / "oof_predictions.csv")
        reference_by_id = {row["record_id"]: row for row in reference_predictions
                           if row["model"] == qnn_summary["results"][0]["model"]}
        if len(reference_by_id) != len(manifest):
            raise ValueError(f"Saved QNN fold map does not cover {task_name} exactly")
        ids = [row["record_id"] for row in manifest]
        y = np.asarray([int(row["label"]) for row in manifest], dtype=np.int64)
        groups = np.asarray([str(row["bug_id"]) for row in manifest], dtype=object)
        fold_by_row = np.asarray([int(reference_by_id[rid]["fold"]) for rid in ids], dtype=np.int64)
        for group in set(groups):
            if len(set(fold_by_row[groups == group])) != 1:
                raise AssertionError(f"Group leakage in saved fold map for {task_name}: {group}")
        features = np.asarray([qml.features_from_text(qml_sweep.text_for(row, spec), 8)
                               for row in manifest], dtype=np.float64)
        fold_rows = []
        task_predictions = []
        for feature_count in (4, 8):
            for model_name, factory in models.items():
                for fold in sorted(set(fold_by_row.tolist())):
                    valid = np.flatnonzero(fold_by_row == fold)
                    train = np.flatnonzero(fold_by_row != fold)
                    if spec.get("train_source"):
                        train = np.asarray([i for i in train if manifest[i].get("source") == spec["train_source"]], dtype=int)
                    if len(set(y[train].tolist())) < 2 or len(set(y[valid].tolist())) < 2:
                        raise ValueError(f"Missing class coverage: {task_name}, fold {fold}, {model_name}")
                    train_x, valid_x, scaler = qml.minmax_fit_apply(features[train].tolist(), features[valid].tolist())
                    train_x = np.asarray(train_x, dtype=np.float64)[:, :feature_count]
                    valid_x = np.asarray(valid_x, dtype=np.float64)[:, :feature_count]
                    model = factory(SEED + int(fold))
                    model.fit(train_x, y[train])
                    pred = model.predict(valid_x).astype(np.int64)
                    fold_metric = score(y[valid], pred)
                    fold_rows.append({"task": task_name, "model": model_name, "feature_count": feature_count,
                                      "fold": int(fold), "train_rows": len(train), "validation_rows": len(valid),
                                      "train_groups": int(len(set(groups[train]))),
                                      "validation_groups": int(len(set(groups[valid]))), **fold_metric})
                    for j, i in enumerate(valid):
                        task_predictions.append({"record_id": ids[i], "group_id": str(groups[i]), "fold": int(fold),
                            "task": task_name, "model": model_name, "feature_count": feature_count,
                            "y_true": int(y[i]), "y_pred": int(pred[j])})
        leaderboard = []
        for feature_count in (4, 8):
            for model_name in models:
                rows = [row for row in task_predictions if row["model"] == model_name and row["feature_count"] == feature_count]
                metrics = score(np.asarray([row["y_true"] for row in rows]), np.asarray([row["y_pred"] for row in rows]))
                leaderboard.append({"task": task_name, "model": model_name, "feature_count": feature_count,
                                    "records": len(rows), "folds": 5, **metrics})
        task_out = OUT / output_name
        task_out.mkdir(parents=True, exist_ok=True)
        with (task_out / "fold_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fold_rows[0]))
            writer.writeheader(); writer.writerows(fold_rows)
        with (task_out / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(task_predictions[0]))
            writer.writeheader(); writer.writerows(task_predictions)
        summary = {"status": "completed_matched_classical_grouped_oof", "task": task_name,
                   "target": qnn_summary.get("task", task_name), "records": len(manifest),
                   "groups": int(len(set(groups))), "folds": 5,
                   "feature_policy": "same deterministic 8-bin BLAKE2b token counts as QNN; fold-local min-max; compare 4-feature and 8-feature controls",
                   "split_source": "saved development QNN grouped out-of-fold fold assignment",
                   "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   "scikit_learn": sklearn.__version__, "models": leaderboard,
                   "fold_metrics_path": "fold_metrics.csv", "oof_predictions_path": "oof_predictions.csv",
                   "tuning": "fixed published-in-this-run configurations; no hyperparameter selection on validation folds",
                   "frozen_test_accessed": False}
        (task_out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        all_results.extend(leaderboard)
        print(json.dumps({"task": task_name, "baseline_best_4f": max((r for r in leaderboard if r["feature_count"] == 4), key=lambda r:r["macro_f1"])["model"],
                          "baseline_best_8f": max((r for r in leaderboard if r["feature_count"] == 8), key=lambda r:r["macro_f1"])["model"],
                          "rows": len(manifest), "groups": len(set(groups))}), flush=True)
    overall = {"status": "completed_matched_classical_grouped_sweeps", "tasks": len(TASK_DIRS),
               "results": all_results, "runtime_seconds": time.perf_counter() - started,
               "minimum_available_ram_mib": 500, "frozen_test_accessed": False,
               "comparison_note": "Compare QNN 4-feature models to 4-feature controls and amplitude 8-feature models to 8-feature controls. Never compare these hash-feature results to prior SVD/metadata controls as if inputs matched."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "campaign_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps({"status": overall["status"], "tasks": len(TASK_DIRS), "candidate_rows": len(all_results),
                      "runtime_seconds": overall["runtime_seconds"], "available_ram_mib": qml.check_ram()}, indent=2))


if __name__ == "__main__":
    main()
