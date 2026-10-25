"""Temporary, explicitly non-sklearn-ensemble evaluation for Windows policy-blocked hosts.

The existing ``ml_training.py`` remains the intended evaluation. This runner uses
its feature definitions, labels, folds, and metrics, while implementing the
Random Forest as bootstrap aggregation of approved sklearn decision trees.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml_evaluation import LABELS, OBSERVABLE_ONLY, WITH_PROGRAM_COMPARISON, load_dataset


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ml_results" / "fallback_ml_evaluation"
CALIBRATION = {"calibration_readout_error_max", "calibration_t1_min", "calibration_t2_min"}
FEATURE_SETS = {
    "OBSERVABLE_ONLY": list(OBSERVABLE_ONLY),
    "WITH_PROGRAM_COMPARISON": list(WITH_PROGRAM_COMPARISON),
    "OBSERVABLE_PLUS_IDEAL_CIRCUIT": list(OBSERVABLE_ONLY) + [
        "ideal_operation_count", "ideal_unique_gate_count", "ideal_qubit_count",
    ],
    "OBSERVABLE_ONLY_NO_CALIBRATION": [f for f in OBSERVABLE_ONLY if f not in CALIBRATION],
    "OBSERVABLE_ONLY_DEDUP_CALIBRATION": [f for f in OBSERVABLE_ONLY if f != "calibration_t2_min"],
}


def _add_ideal_circuit_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add structural features from the unchanged reference circuit only."""
    enriched = []
    for row in rows:
        report_path = ROOT / row["report_path"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        ideal = report.get("circuit", {}).get("ideal", []) or []
        gate_names = []
        for operation in ideal:
            if isinstance(operation, dict):
                name = operation.get("name", operation.get("op", operation.get("gate", "")))
            elif isinstance(operation, (list, tuple)) and operation:
                name = operation[0]
            else:
                continue
            gate_names.append(str(name).lower())
        enriched.append({
            **row,
            "ideal_operation_count": float(len(ideal)),
            "ideal_unique_gate_count": float(len(set(gate_names))),
            "ideal_qubit_count": float(report.get("configuration", {}).get("num_qubits", np.nan)),
        })
    return enriched


class BootstrapRandomForest(ClassifierMixin, BaseEstimator):
    """Custom random forest using sklearn CART trees, bootstrap samples and sqrt subspaces."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 8, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BootstrapRandomForest":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        counts = Counter(y.tolist())
        # Match sklearn RandomForestClassifier(class_weight="balanced"): class
        # weights are computed from the full training fold, then bootstrap counts
        # determine each tree's effective sample weights.
        class_weights = {
            label: len(y) / (len(self.classes_) * counts[label]) for label in self.classes_
        }
        row_weights = np.asarray([class_weights[label] for label in y], dtype=float)
        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []
        n_rows = len(y)
        for _ in range(self.n_estimators):
            sample = rng.randint(0, n_rows, size=n_rows)
            tree_seed = int(rng.randint(0, np.iinfo(np.int32).max))
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features="sqrt",
                random_state=tree_seed,
            )
            tree.fit(X[sample], y[sample], sample_weight=row_weights[sample])
            self.estimators_.append(tree)
        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.estimators_], axis=0
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        votes = np.zeros((len(X), len(self.classes_)), dtype=np.int32)
        class_indices = {label: i for i, label in enumerate(self.classes_)}
        for tree in self.estimators_:
            predicted = tree.predict(X)
            for label, index in class_indices.items():
                votes[:, index] += predicted == label
        return self.classes_[np.argmax(votes, axis=1)]


def _models() -> dict[str, Any]:
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        ),
        "Random Forest (custom bootstrap forest)": make_pipeline(
            SimpleImputer(strategy="median"), BootstrapRandomForest()
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _permutation_importances(estimator: Any, X: np.ndarray, y: np.ndarray, features: list[str], fold: int) -> list[dict[str, Any]]:
    baseline = f1_score(y, estimator.predict(X), labels=LABELS, average="macro", zero_division=0)
    rng = np.random.RandomState(42)
    results = []
    for column, feature in enumerate(features):
        scores = []
        for _ in range(5):
            permuted = X.copy()
            permuted[:, column] = rng.permutation(permuted[:, column])
            score = f1_score(y, estimator.predict(permuted), labels=LABELS, average="macro", zero_division=0)
            scores.append(baseline - score)
        results.append({
            "feature_set": "",
            "fold": fold,
            "importance_type": "permutation_macro_f1",
            "feature": feature,
            "importance_mean": float(np.mean(scores)),
            "importance_std": float(np.std(scores, ddof=0)),
        })
    return results


def main() -> int:
    # This diagnostic runner used to treat nominal family IDs as independent
    # and could fit the rejected historical dataset outside the strict gate.
    # Reuse the read-only provenance preflight before loading features/fitting.
    from ml_training import run_preflight

    preflight = run_preflight()
    if not preflight["ready"]:
        print("Fallback evaluation blocked by dataset preflight:")
        print(json.dumps(preflight["blockers"], indent=2))
        return 2

    rows = _add_ideal_circuit_features(load_dataset())
    if len(rows) != 270:
        raise ValueError(f"Expected the existing 270-record dataset, found {len(rows)}")
    if len({row["circuit_family_id"] for row in rows}) != 30:
        raise ValueError("Expected 30 independent circuit-family groups")
    if Counter(row["ground_truth_label"] for row in rows) != Counter({label: 90 for label in LABELS}):
        raise ValueError("Expected exactly 90 records in each class")

    y = np.asarray([row["ground_truth_label"] for row in rows])
    groups = np.asarray([row["circuit_family_id"] for row in rows])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(np.zeros((len(rows), 1)), y, groups))
    for fold_i, (train, valid) in enumerate(folds, 1):
        train_groups, valid_groups = set(groups[train]), set(groups[valid])
        if train_groups & valid_groups:
            raise AssertionError(f"Circuit family leaked across fold {fold_i}")
        if set(y[train]) != set(LABELS) or set(y[valid]) != set(LABELS):
            raise AssertionError(f"Fold {fold_i} does not contain all classes on both sides")

    OUT.mkdir(parents=True, exist_ok=True)
    cm_dir = OUT / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    markdown_rows = []

    for set_name, features in FEATURE_SETS.items():
        X = np.asarray([
            [float(row.get(feature)) if row.get(feature) is not None else np.nan for feature in features]
            for row in rows
        ], dtype=float)
        for model_name, estimator in _models().items():
            metrics: dict[str, list[float]] = {k: [] for k in ("accuracy", "macro_f1", "weighted_f1")}
            total_cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
            class_reports: dict[str, Any] = {}
            class_metric_values = {label: {metric: [] for metric in ("precision", "recall", "f1-score")}
                                   for label in LABELS}

            for fold_i, (train, valid) in enumerate(folds, 1):
                estimator.fit(X[train], y[train])
                prediction = estimator.predict(X[valid])
                metrics["accuracy"].append(float(accuracy_score(y[valid], prediction)))
                metrics["macro_f1"].append(float(f1_score(y[valid], prediction, labels=LABELS, average="macro", zero_division=0)))
                metrics["weighted_f1"].append(float(f1_score(y[valid], prediction, labels=LABELS, average="weighted", zero_division=0)))
                cm = confusion_matrix(y[valid], prediction, labels=LABELS)
                total_cm += cm
                report = classification_report(
                    y[valid], prediction, labels=LABELS, target_names=LABELS,
                    output_dict=True, zero_division=0,
                )
                class_reports[str(fold_i)] = report
                for label in LABELS:
                    for metric in class_metric_values[label]:
                        class_metric_values[label][metric].append(float(report[label][metric]))
                        per_class_rows.append({
                            "feature_set": set_name, "model": model_name, "fold": fold_i,
                            "class": label, "metric": metric, "value": float(report[label][metric]),
                        })
                fold_rows.append({
                    "feature_set": set_name, "model": model_name, "fold": fold_i,
                    "train_groups": len(set(groups[train])), "validation_groups": len(set(groups[valid])),
                    "accuracy": metrics["accuracy"][-1], "macro_f1": metrics["macro_f1"][-1],
                    "weighted_f1": metrics["weighted_f1"][-1],
                    "classification_report_json": json.dumps(report, sort_keys=True),
                })

                if model_name.startswith("Random Forest"):
                    forest = estimator[-1]
                    for feature, value in zip(features, forest.feature_importances_):
                        importance_rows.append({
                            "feature_set": set_name, "fold": fold_i, "importance_type": "impurity",
                            "feature": feature, "importance_mean": float(value), "importance_std": 0.0,
                        })
                    permutation_rows = _permutation_importances(estimator, X[valid], y[valid], features, fold_i)
                    for item in permutation_rows:
                        item["feature_set"] = set_name
                    importance_rows.extend(permutation_rows)

            safe = f"{set_name}__{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
            np.savetxt(cm_dir / f"{safe}.csv", total_cm, delimiter=",", fmt="%d", header=",".join(LABELS), comments="")
            (cm_dir / f"{safe}_classification_reports.json").write_text(
                json.dumps(class_reports, indent=2), encoding="utf-8"
            )
            summary = {"feature_set": set_name, "model": model_name}
            for metric, values in metrics.items():
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = float(np.std(values, ddof=1))
            summary_rows.append(summary)
            for label in LABELS:
                for metric, values in class_metric_values[label].items():
                    per_class_rows.append({
                        "feature_set": set_name, "model": model_name, "fold": "mean",
                        "class": label, "metric": metric, "value": float(np.mean(values)),
                    })
            markdown_rows.append(summary)
            print(f"completed {set_name} / {model_name}", flush=True)

    # Fold-wise majority baseline matches the original evaluation report.
    baseline_rows = []
    for fold_i, (train, valid) in enumerate(folds, 1):
        counts = Counter(y[train])
        majority = max(LABELS, key=lambda label: counts[label])
        pred = np.repeat(majority, len(valid))
        baseline_rows.append({
            "fold": fold_i,
            "accuracy": float(accuracy_score(y[valid], pred)),
            "macro_f1": float(f1_score(y[valid], pred, labels=LABELS, average="macro", zero_division=0)),
        })

    summary_rows.sort(key=lambda row: row["macro_f1_mean"], reverse=True)
    _write_csv(OUT / "model_comparison.csv", summary_rows)
    _write_csv(OUT / "fold_results.csv", fold_rows)
    _write_csv(OUT / "per_class_metrics.csv", per_class_rows)
    _write_csv(OUT / "feature_importance.csv", importance_rows)
    _write_csv(OUT / "majority_baseline.csv", baseline_rows)

    metadata = {
        "evaluation_path": "fallback_ml_evaluation",
        "dataset_records": len(rows),
        "dataset_directory": "reports/experiments/full_dataset",
        "labels": list(LABELS),
        "group_column": "circuit_family_id",
        "splitter": "sklearn.model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)",
        "folds": [
            {"fold": i, "train_groups": len(set(groups[train])), "validation_groups": len(set(groups[valid]))}
            for i, (train, valid) in enumerate(folds, 1)
        ],
        "models": {
            "Logistic Regression": "sklearn LogisticRegression with the existing median-impute/standardize pipeline",
            "Decision Tree": "sklearn DecisionTreeClassifier with the existing median-impute pipeline",
            "Random Forest (custom bootstrap forest)": "custom bootstrap aggregation of sklearn DecisionTreeClassifier trees; sqrt feature subsampling, max_depth=8, class-balanced weights; not sklearn RandomForestClassifier",
        },
        "additional_features": {
            "ideal_operation_count": "count of operations in circuit.ideal; excludes observed/injected circuit and all labels/metadata",
            "ideal_unique_gate_count": "number of distinct operation names in circuit.ideal",
            "ideal_qubit_count": "configuration.num_qubits for the reference circuit; structural circuit width",
        },
        "sklearn_ensemble_imported": False,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        "# Fallback ML evaluation",
        "",
        "Temporary results generated by `ml_fallback_training.py` because Windows Application Control blocks scikit-learn's HistGradientBoosting native extension.",
        "The existing sklearn evaluation in `ml_training.py` is unchanged and remains the intended pipeline.",
        "",
        "Dataset: 270 existing reports (90 per class), 30 circuit-family groups. No dataset or labels were regenerated or edited.",
        "Folds: 5-fold `StratifiedGroupKFold`, shuffled with random state 42; holdout unit is `circuit_family_id`.",
        "",
        "Logistic Regression and Decision Tree use the installed sklearn estimators that successfully fit in this environment. The Random Forest below is a custom forest of sklearn decision trees; it is not sklearn's RandomForestClassifier.",
        "",
        "| Feature set | Model | Accuracy | Macro F1 | Weighted F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['feature_set']} | {row['model']} | {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | "
            f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | {row['weighted_f1_mean']:.4f} ± {row['weighted_f1_std']:.4f} |"
        )
    lines.extend([
        "",
        "Per-class precision, recall, and F1 are in `per_class_metrics.csv` (fold results and fold means). Aggregate confusion matrices and fold classification reports are in `confusion_matrices/`. Random Forest impurity and permutation macro-F1 importances are in `feature_importance.csv`.",
        "",
        "`OBSERVABLE_PLUS_IDEAL_CIRCUIT` adds only reference-circuit structure (ideal operation count, unique ideal gate count, and qubit count). These are available in the existing reports and are independent of the injected observed circuit and label fields.",
        "",
        "These fallback results are temporary and should not be treated as sklearn Random Forest results or as a replacement for the existing sklearn evaluation.",
    ])
    (OUT / "fallback_ml_evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"fallback_ml_evaluation reports written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
