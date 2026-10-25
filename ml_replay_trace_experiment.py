"""Grouped evaluation of replay-derived trace features; writes a separate artifact."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml_evaluation import LABELS, load_dataset
from replay_trace_features import DATASET_NAME as REPLAY_DATASET_NAME
from replay_trace_features import FEATURES as REPLAY_FEATURES
from replay_trace_features import TRACE_ORIGIN


ROOT = Path(__file__).resolve().parent
REPLAY_FEATURES_CSV = ROOT / "reports" / "experiments" / "replay_derived_trace_features" / "replay_trace_features.csv"
OUTPUT_DIR = ROOT / "ml_results" / "replay_trace_feature_experiment"
BASELINE_FEATURES = (
    "total_variation_distance",
    "js_divergence",
    "hellinger_distance",
    "observed_entropy_bits",
    "observed_max_probability",
)
FEATURE_SETS = {
    "OBSERVABLE_ONLY": BASELINE_FEATURES,
    "OBSERVABLE_PLUS_REPLAY_TRACE": BASELINE_FEATURES + REPLAY_FEATURES,
}
FORBIDDEN_MODEL_FEATURES = {
    "ground_truth_label", "run_id", "circuit_family_id", "repetition_index", "random_seed",
    "diagnosis", "diagnosis.category", "diagnosis.program_evidence_candidates",
    "diagnosis.hardware_evidence_candidates", "calibration_readout_error_max",
    "calibration_t1_min", "calibration_t2_min", "gate_mismatch_fraction", "gate_error",
    "readout_error", "t1", "t2", "gate_failed", "trace_step", "step", "position",
    "program_fault_id", "hardware_condition_id", "hardware_affected_component",
}
MATERIAL_DELTA = 0.01  # descriptive one-percentage-point threshold; no tuning is performed


def models():
    """Mirror the existing ml_training.models() definitions exactly."""
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
        ),
    }


def _load_joined_rows() -> list[dict[str, Any]]:
    observable_rows = load_dataset()
    if len(observable_rows) != 270:
        raise ValueError(f"Expected 270 baseline report rows, found {len(observable_rows)}")
    audit_path = REPLAY_FEATURES_CSV.with_name("audit_report.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("audit_pass") is not True:
        raise ValueError("Replay feature audit did not pass")
    expected_csv_hash = audit.get("replay_feature_csv_sha256")
    actual_csv_hash = hashlib.sha256(REPLAY_FEATURES_CSV.read_bytes()).hexdigest()
    if not expected_csv_hash or actual_csv_hash != expected_csv_hash:
        raise ValueError("Replay feature CSV does not match its provenance audit")

    with REPLAY_FEATURES_CSV.open(newline="", encoding="utf-8") as stream:
        replay_rows = list(csv.DictReader(stream))
    replay_by_id = {row["run_id"]: row for row in replay_rows}
    if len(replay_rows) != 270 or len(replay_by_id) != 270:
        raise ValueError("Replay feature CSV must contain 270 unique run IDs")

    joined = []
    for row in observable_rows:  # preserves the existing report-loader order for identical folds
        replay = replay_by_id.get(row["run_id"])
        if replay is None:
            raise ValueError(f"Missing replay feature record for run {row['run_id']}")
        if replay["ground_truth_label"] != row["ground_truth_label"]:
            raise ValueError(f"Target mismatch for run {row['run_id']}")
        if replay["circuit_family_id"] != row["circuit_family_id"]:
            raise ValueError(f"Group mismatch for run {row['run_id']}")
        if int(replay["repetition_index"]) != int(row["repetition_index"]):
            raise ValueError(f"Repetition mismatch for run {row['run_id']}")
        joined.append({**row, **{name: float(replay[name]) for name in REPLAY_FEATURES}})
    return joined


def _correlation_audit(rows: list[dict[str, Any]], feature_names: tuple[str, ...]) -> dict[str, Any]:
    matrix = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    corr = np.corrcoef(matrix, rowvar=False)
    pairs = []
    suspicious = []
    for left_i, left in enumerate(feature_names):
        for right_i in range(left_i + 1, len(feature_names)):
            right = feature_names[right_i]
            value = float(corr[left_i, right_i])
            pair = {"feature_a": left, "feature_b": right, "pearson_r": value}
            pairs.append(pair)
            if abs(value) >= 0.95:
                suspicious.append(pair)
    constants = [feature for i, feature in enumerate(feature_names) if np.std(matrix[:, i]) <= 1e-12]
    return {
        "method": "Pearson correlation over the 270 joined records; computed without labels as inputs",
        "high_correlation_cutoff_absolute_r": 0.95,
        "near_duplicate_cutoff_absolute_r": 0.995,
        "constant_features": constants,
        "correlations": pairs,
        "high_correlation_pairs": suspicious,
        "near_duplicate_pairs": [pair for pair in pairs if abs(pair["pearson_r"]) >= 0.995],
    }


def _eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    grand_mean = float(np.mean(values))
    between = sum(
        int(np.sum(labels == label)) * (float(np.mean(values[labels == label])) - grand_mean) ** 2
        for label in LABELS
    )
    total = float(np.sum((values - grand_mean) ** 2))
    return between / total if total > 0 else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    rows = _load_joined_rows()
    source_identity = []
    for row in rows:
        report_path = Path(row["report_path"])
        if not report_path.is_absolute():
            report_path = ROOT / report_path
        source_identity.append({
            "run_id": row["run_id"],
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "declared_source_sha256": row.get("_source_sha256"),
        })
    source_identity.sort(key=lambda row: row["run_id"])
    source_data_fingerprint = hashlib.sha256(
        json.dumps(source_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    trace_features_sha256 = hashlib.sha256(REPLAY_FEATURES_CSV.read_bytes()).hexdigest()
    trace_audit_path = REPLAY_FEATURES_CSV.with_name("audit_report.json")
    trace_audit_sha256 = hashlib.sha256(trace_audit_path.read_bytes()).hexdigest()
    trace_implementation_sha256 = json.loads(trace_audit_path.read_text(encoding="utf-8")).get(
        "implementation_sha256", {}
    )
    labels = np.asarray([row["ground_truth_label"] for row in rows])
    groups = np.asarray([row["circuit_family_id"] for row in rows])
    if set(np.unique(labels)) != set(LABELS) or len(set(groups)) != 30:
        raise ValueError("Expected all three labels and 30 circuit families")

    feature_audit = {
        name: {
            "features": list(features),
            "forbidden_feature_intersection": sorted(set(features) & FORBIDDEN_MODEL_FEATURES),
            "feature_count": len(features),
        }
        for name, features in FEATURE_SETS.items()
    }
    if any(item["forbidden_feature_intersection"] for item in feature_audit.values()):
        raise ValueError("Feature configuration contains a forbidden metadata/diagnosis/calibration field")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(np.zeros((len(rows), 1)), labels, groups))
    for train, valid in folds:
        if set(groups[train]) & set(groups[valid]):
            raise AssertionError("Circuit family leaked across folds")
        if set(labels[train]) != set(LABELS) or set(labels[valid]) != set(LABELS):
            raise AssertionError("A fold is missing one or more classes")

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    pooled_matrices: dict[tuple[str, str], np.ndarray] = {}
    pooled_predictions: dict[tuple[str, str], list[tuple[str, str]]] = {}

    for set_name, features in FEATURE_SETS.items():
        X = np.asarray([[float(row[feature]) for feature in features] for row in rows], dtype=float)
        if not np.all(np.isfinite(X)):
            raise ValueError(f"Non-finite input found in {set_name}")
        for model_name, estimator in models().items():
            accuracies, macro_f1s = [], []
            per_class_fold_recall = {label: [] for label in LABELS}
            total_matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
            predictions: list[tuple[str, str]] = []
            for fold_index, (train, valid) in enumerate(folds, 1):
                estimator.fit(X[train], labels[train])
                predicted = estimator.predict(X[valid])
                actual = labels[valid]
                matrix = confusion_matrix(actual, predicted, labels=LABELS)
                total_matrix += matrix
                accuracies.append(float(accuracy_score(actual, predicted)))
                macro_f1s.append(float(f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)))
                fold_recall = recall_score(actual, predicted, labels=LABELS, average=None, zero_division=0)
                for label, score in zip(LABELS, fold_recall):
                    per_class_fold_recall[label].append(float(score))
                predictions.extend((str(y), str(yhat)) for y, yhat in zip(actual, predicted))
                fold_rows.append({
                    "feature_set": set_name,
                    "model": model_name,
                    "fold": fold_index,
                    "train_family_count": len(set(groups[train])),
                    "validation_family_count": len(set(groups[valid])),
                    "accuracy": accuracies[-1],
                    "macro_f1": macro_f1s[-1],
                    **{f"recall_{label}": per_class_fold_recall[label][-1] for label in LABELS},
                })

            key = (set_name, model_name)
            pooled_matrices[key] = total_matrix
            pooled_predictions[key] = predictions
            summary_rows.append({
                "feature_set": set_name,
                "model": model_name,
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies, ddof=1)),
                "macro_f1_mean": float(np.mean(macro_f1s)),
                "macro_f1_std": float(np.std(macro_f1s, ddof=1)),
            })
            for class_index, label in enumerate(LABELS):
                pooled_recall = float(total_matrix[class_index, class_index] / total_matrix[class_index].sum())
                recall_rows.append({
                    "feature_set": set_name,
                    "model": model_name,
                    "class": label,
                    "pooled_oof_recall": pooled_recall,
                    "fold_recall_mean": float(np.mean(per_class_fold_recall[label])),
                    "fold_recall_std": float(np.std(per_class_fold_recall[label], ddof=1)),
                    "support": int(total_matrix[class_index].sum()),
                })

    # Same training-fold majority-label rule as ml_training.py.
    majority_fold_rows = []
    majority_matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for fold_index, (train, valid) in enumerate(folds, 1):
        counts = {label: int(np.sum(labels[train] == label)) for label in LABELS}
        majority_label = max(LABELS, key=lambda label: counts[label])
        prediction = np.repeat(majority_label, len(valid))
        actual = labels[valid]
        matrix = confusion_matrix(actual, prediction, labels=LABELS)
        majority_matrix += matrix
        majority_fold_rows.append({
            "fold": fold_index,
            "majority_label": majority_label,
            "accuracy": float(accuracy_score(actual, prediction)),
            "macro_f1": float(f1_score(actual, prediction, labels=LABELS, average="macro", zero_division=0)),
        })
    confusion_rows.append({"feature_set": "MAJORITY_CLASS_BASELINE", "model": "Majority Class", "matrix": majority_matrix.tolist()})

    for (set_name, model_name), matrix in pooled_matrices.items():
        confusion_rows.append({"feature_set": set_name, "model": model_name, "matrix": matrix.tolist()})

    summary_by_pair = {(row["feature_set"], row["model"]): row for row in summary_rows}
    comparisons = []
    model_names = list(models())
    for model_name in model_names:
        baseline = summary_by_pair[("OBSERVABLE_ONLY", model_name)]
        trace = summary_by_pair[("OBSERVABLE_PLUS_REPLAY_TRACE", model_name)]
        delta_accuracy = trace["accuracy_mean"] - baseline["accuracy_mean"]
        delta_macro_f1 = trace["macro_f1_mean"] - baseline["macro_f1_mean"]
        if delta_macro_f1 >= MATERIAL_DELTA and delta_accuracy > -MATERIAL_DELTA:
            interpretation = "improves"
        elif delta_macro_f1 <= -MATERIAL_DELTA or delta_accuracy <= -MATERIAL_DELTA:
            interpretation = "hurts"
        else:
            interpretation = "no material change"
        comparisons.append({
            "model": model_name,
            "baseline_accuracy_mean": baseline["accuracy_mean"],
            "trace_accuracy_mean": trace["accuracy_mean"],
            "delta_accuracy": delta_accuracy,
            "baseline_macro_f1_mean": baseline["macro_f1_mean"],
            "trace_macro_f1_mean": trace["macro_f1_mean"],
            "delta_macro_f1": delta_macro_f1,
            "interpretation": interpretation,
            "materiality_cutoff_absolute_delta": MATERIAL_DELTA,
        })

    class_etas = {}
    for feature in FEATURE_SETS["OBSERVABLE_PLUS_REPLAY_TRACE"]:
        values = np.asarray([float(row[feature]) for row in rows])
        class_etas[feature] = _eta_squared(values, labels)
    correlation_audit = _correlation_audit(rows, FEATURE_SETS["OBSERVABLE_PLUS_REPLAY_TRACE"])
    leakage_audit = {
        "target_field": "ground_truth_label",
        "target_used_only_for_y": True,
        "model_feature_names": {key: list(value) for key, value in FEATURE_SETS.items()},
        "forbidden_feature_names": sorted(FORBIDDEN_MODEL_FEATURES),
        "metadata_used_only_for_join_or_grouping": ["run_id", "circuit_family_id", "repetition_index", "random_seed"],
        "diagnosis_output_used": False,
        "calibration_fields_used": False,
        "gate_mismatch_used": False,
        "trace_position_or_localization_fields_used": False,
        "replay_origin": TRACE_ORIGIN,
        "replay_feature_provenance": "Probability deviations are generated by replaying each stored ideal/observed circuit with stored calibration; no labels or diagnosis fields enter feature computation.",
        "class_association_eta_squared_descriptive_only": class_etas,
        "interpretation": "Strong association is a leakage warning to inspect, not proof of direct label leakage. Replay trace values can reflect injected program or hardware effects by design.",
    }

    per_class_comparison = []
    recall_map = {(row["feature_set"], row["model"], row["class"]): row for row in recall_rows}
    for model_name in model_names:
        for label in LABELS:
            baseline = recall_map[("OBSERVABLE_ONLY", model_name, label)]
            trace = recall_map[("OBSERVABLE_PLUS_REPLAY_TRACE", model_name, label)]
            per_class_comparison.append({
                "model": model_name,
                "class": label,
                "observable_only_pooled_oof_recall": baseline["pooled_oof_recall"],
                "observable_plus_trace_pooled_oof_recall": trace["pooled_oof_recall"],
                "delta_pooled_oof_recall": trace["pooled_oof_recall"] - baseline["pooled_oof_recall"],
            })

    report = {
        "experiment_name": "REPLAY_TRACE_FEATURE_COMPARISON",
        "replay_dataset": REPLAY_DATASET_NAME,
        "trace_origin": TRACE_ORIGIN,
        "n_records": len(rows),
        "class_counts": {label: int(np.sum(labels == label)) for label in LABELS},
        "group_count": len(set(groups)),
        "input_provenance": {
            "baseline_report_set_sha256": source_data_fingerprint,
            "replay_feature_csv_sha256": trace_features_sha256,
            "replay_feature_audit_sha256": trace_audit_sha256,
            "trace_implementation_sha256": trace_implementation_sha256,
            "baseline_report_sha256_coverage": len(source_identity),
            "embedded_source_hash_coverage": sum(row.get("_source_sha256") is not None for row in rows),
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
        "methodology": {
            "splitter": "StratifiedGroupKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "group_column": "circuit_family_id",
            "preprocessing": "SimpleImputer(strategy='median'); StandardScaler only for Logistic Regression",
            "hyperparameter_tuning": False,
            "models": list(models()),
            "metrics": ["accuracy", "macro_f1"],
            "mean_std": "mean and sample standard deviation (ddof=1) across the same five folds",
            "feature_sets": {key: list(value) for key, value in FEATURE_SETS.items()},
        },
        "summary": summary_rows,
        "comparisons": comparisons,
        "material_change_rule": f"Macro-F1 improvement of at least {MATERIAL_DELTA:.2f} with accuracy loss smaller than {MATERIAL_DELTA:.2f}; symmetric loss rule for hurt; otherwise no material change.",
        "majority_baseline": {
            "folds": majority_fold_rows,
            "accuracy_mean": float(np.mean([row["accuracy"] for row in majority_fold_rows])),
            "accuracy_std": float(np.std([row["accuracy"] for row in majority_fold_rows], ddof=1)),
            "macro_f1_mean": float(np.mean([row["macro_f1"] for row in majority_fold_rows])),
            "macro_f1_std": float(np.std([row["macro_f1"] for row in majority_fold_rows], ddof=1)),
            "confusion_matrix": majority_matrix.tolist(),
        },
        "fold_results": fold_rows,
        "confusion_matrices": confusion_rows,
        "per_class_recall": recall_rows,
        "per_class_recall_comparison": per_class_comparison,
        "feature_correlation_audit": correlation_audit,
        "leakage_audit": leakage_audit,
    }

    (output_dir / "experiment_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(output_dir / "fold_results.csv", fold_rows)
    _write_csv(output_dir / "model_summary.csv", summary_rows)
    _write_csv(output_dir / "per_class_recall.csv", recall_rows)
    _write_csv(output_dir / "per_class_recall_comparison.csv", per_class_comparison)
    _write_csv(output_dir / "confusion_matrices.csv", confusion_rows)
    (output_dir / "feature_correlation_audit.json").write_text(json.dumps(correlation_audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "leakage_audit.json").write_text(json.dumps(leakage_audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    try:
        displayed_output_dir = output_dir.relative_to(ROOT).as_posix()
    except ValueError:
        displayed_output_dir = str(output_dir)
    print(json.dumps({
        "output_dir": displayed_output_dir,
        "summary": summary_rows,
        "comparisons": comparisons,
        "majority_baseline": report["majority_baseline"],
        "per_class_recall_comparison": per_class_comparison,
        "high_correlation_pairs": correlation_audit["high_correlation_pairs"],
        "class_association_eta_squared": class_etas,
    }, indent=2))
    return report


if __name__ == "__main__":
    run_experiment()

