import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "qmutbench_ml" / "qmutbench_results.csv"
OUT = ROOT / "data" / "qmutbench_ml" / "final"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "trace_max_probability_difference",
    "trace_mean_probability_difference",
    "trace_median_probability_difference",
    "trace_deviation_concentration",
    "trace_high_deviation_fraction",
]
NORMAL = "NORMAL"
FAULT = "PROGRAM_FAULT"


def metrics(actual, predicted):
    # Positive class is NORMAL.
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))

    precision_normal = tp / (tp + fp) if tp + fp else 0.0
    recall_normal = tp / (tp + fn) if tp + fn else 0.0
    f1_normal = (
        2 * precision_normal * recall_normal /
        (precision_normal + recall_normal)
        if precision_normal + recall_normal else 0.0
    )

    precision_fault = tn / (tn + fn) if tn + fn else 0.0
    recall_fault = tn / (tn + fp) if tn + fp else 0.0
    f1_fault = (
        2 * precision_fault * recall_fault /
        (precision_fault + recall_fault)
        if precision_fault + recall_fault else 0.0
    )

    accuracy = (tp + tn) / len(actual) if len(actual) else 0.0
    balanced = (recall_normal + recall_fault) / 2
    macro_f1 = (f1_normal + f1_fault) / 2

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "normal_precision": precision_normal,
        "normal_recall": recall_normal,
        "normal_f1": f1_normal,
        "program_fault_precision": precision_fault,
        "program_fault_recall": recall_fault,
        "program_fault_f1": f1_fault,
        "confusion_matrix_actual_rows_predicted_columns": [[tn, fp], [fn, tp]],
        "confusion_matrix_label_order": [FAULT, NORMAL],
    }


def sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic(X, y, iterations=3000, learning_rate=0.05, l2=0.01):
    # y=1 means NORMAL; y=0 means PROGRAM_FAULT.
    n, p = X.shape
    positives = max(int(y.sum()), 1)
    negatives = max(n - positives, 1)

    # Class-balanced sample weights.
    weights = np.where(y == 1, n / (2 * positives), n / (2 * negatives))
    beta = np.zeros(p + 1, dtype=float)
    design = np.column_stack([np.ones(n), X])

    for _ in range(iterations):
        probabilities = sigmoid(design @ beta)
        gradient = (design.T @ (weights * (probabilities - y))) / weights.sum()
        gradient[1:] += l2 * beta[1:]
        beta -= learning_rate * gradient

    return beta


def main():
    df = pd.read_csv(DATA)
    df = df[df["label"].isin([NORMAL, FAULT])].copy()

    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + ["original_circuit_id", "label"])

    if df["original_circuit_id"].nunique() < 5:
        raise ValueError("Not enough distinct circuit groups for a useful split.")

    X = df[FEATURES].to_numpy(dtype=float)
    y = (df["label"].to_numpy() == NORMAL).astype(int)
    groups = df["original_circuit_id"].astype(str).to_numpy()

    # Split entire original-circuit groups, never individual rows.
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(42)
    rng.shuffle(unique_groups)
    test_group_count = max(1, round(0.20 * len(unique_groups)))
    test_groups = set(unique_groups[:test_group_count])
    test_mask = np.array([g in test_groups for g in groups])
    train_mask = ~test_mask

    X_train_raw, X_test_raw = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "Training split contains only one class. Change the group split seed."
        )

    # Standardize using training data only.
    mean = X_train_raw.mean(axis=0)
    scale = X_train_raw.std(axis=0)
    scale[scale == 0] = 1.0
    X_train = (X_train_raw - mean) / scale
    X_test = (X_test_raw - mean) / scale

    # Majority-class baseline.
    majority_class = int(np.bincount(y_train).argmax())
    baseline_pred = np.full(len(y_test), majority_class, dtype=int)

    # NumPy-only logistic regression.
    beta = train_logistic(X_train, y_train)
    probabilities = sigmoid(np.column_stack([np.ones(len(X_test)), X_test]) @ beta)
    predictions = (probabilities >= 0.5).astype(int)

    baseline_metrics = metrics(y_test, baseline_pred)
    model_metrics = metrics(y_test, predictions)

    comparison = pd.DataFrame([
        {"model": "Majority-class baseline", **baseline_metrics},
        {"model": "NumPy logistic regression", **model_metrics},
    ])
    comparison.to_csv(OUT / "model_comparison.csv", index=False)

    report = {
        "dataset": str(DATA),
        "records_used": int(len(df)),
        "features": FEATURES,
        "class_counts": {
            str(k): int(v) for k, v in df["label"].value_counts().items()
        },
        "train_records": int(train_mask.sum()),
        "test_records": int(test_mask.sum()),
        "train_circuit_groups": int(len(set(groups[train_mask]))),
        "test_circuit_groups": int(len(set(groups[test_mask]))),
        "overlapping_circuit_groups": int(
            len(set(groups[train_mask]) & set(groups[test_mask]))
        ),
        "baseline": baseline_metrics,
        "logistic_regression": model_metrics,
        "note": (
            "This is a NumPy-only logistic-regression baseline, not a "
            "scikit-learn implementation. Labels represent the supplied "
            "QMutBench-derived program-fault labels, not verified hardware faults."
        ),
    }
    (OUT / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # Save parameters for reproducibility and future inference.
    np.savez(
        OUT / "numpy_logistic_model.npz",
        coefficients=beta,
        feature_mean=mean,
        feature_scale=scale,
        feature_names=np.array(FEATURES),
        positive_class=np.array(NORMAL),
    )

    pd.DataFrame({
        "original_circuit_id": groups[test_mask],
        "actual_label": np.where(y_test == 1, NORMAL, FAULT),
        "predicted_label": np.where(predictions == 1, NORMAL, FAULT),
        "normal_probability": probabilities,
    }).to_csv(OUT / "test_predictions.csv", index=False)

    print("\nDATASET:", len(df), "records")
    print("TRAIN:", int(train_mask.sum()), "records;",
          len(set(groups[train_mask])), "circuit groups")
    print("TEST:", int(test_mask.sum()), "records;",
          len(set(groups[test_mask])), "circuit groups")
    print("GROUP OVERLAP:", report["overlapping_circuit_groups"])
    print("\nBASELINE:")
    print(json.dumps(baseline_metrics, indent=2))
    print("\nNUMPY LOGISTIC REGRESSION:")
    print(json.dumps(model_metrics, indent=2))
    print("\nSAVED OUTPUTS:", OUT)
    print(" - model_comparison.csv")
    print(" - evaluation_report.json")
    print(" - test_predictions.csv")
    print(" - numpy_logistic_model.npz")


if __name__ == "__main__":
    main()
