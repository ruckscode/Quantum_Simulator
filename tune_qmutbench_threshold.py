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
NORMAL, FAULT = "NORMAL", "PROGRAM_FAULT"


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic(X, y, iterations=3000, lr=0.05, l2=0.01):
    n, p = X.shape
    pos, neg = max(y.sum(), 1), max(n - y.sum(), 1)
    weights = np.where(y == 1, n / (2 * pos), n / (2 * neg))
    design = np.column_stack([np.ones(n), X])
    beta = np.zeros(p + 1)

    for _ in range(iterations):
        error = sigmoid(design @ beta) - y
        grad = design.T @ (weights * error) / weights.sum()
        grad[1:] += l2 * beta[1:]
        beta -= lr * grad

    return beta


def score(y, pred):
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))

    precision_n = tp / (tp + fp) if tp + fp else 0
    recall_n = tp / (tp + fn) if tp + fn else 0
    precision_f = tn / (tn + fn) if tn + fn else 0
    recall_f = tn / (tn + fp) if tn + fp else 0

    f1_n = 2 * precision_n * recall_n / (precision_n + recall_n) if precision_n + recall_n else 0
    f1_f = 2 * precision_f * recall_f / (precision_f + recall_f) if precision_f + recall_f else 0

    return {
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (recall_n + recall_f) / 2,
        "macro_f1": (f1_n + f1_f) / 2,
        "normal_precision": precision_n,
        "normal_recall": recall_n,
        "program_fault_precision": precision_f,
        "program_fault_recall": recall_f,
        "confusion_matrix_actual_rows_predicted_columns": [[tn, fp], [fn, tp]],
    }


df = pd.read_csv(DATA)
df = df[df["label"].isin([NORMAL, FAULT])].copy()
df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES + ["original_circuit_id", "label"])

X = df[FEATURES].to_numpy(float)
y = (df["label"].to_numpy() == NORMAL).astype(int)
groups = df["original_circuit_id"].astype(str).to_numpy()

# 60% training groups, 20% validation groups, 20% test groups.
unique_groups = np.unique(groups)
rng = np.random.default_rng(42)
rng.shuffle(unique_groups)
n_train = int(0.60 * len(unique_groups))
n_val = int(0.20 * len(unique_groups))
train_groups = set(unique_groups[:n_train])
val_groups = set(unique_groups[n_train:n_train + n_val])
test_groups = set(unique_groups[n_train + n_val:])

train = np.array([g in train_groups for g in groups])
val = np.array([g in val_groups for g in groups])
test = np.array([g in test_groups for g in groups])

if any(len(np.unique(y[m])) < 2 for m in (train, val, test)):
    raise ValueError("A split contains only one class; group distribution needs review.")

mean = X[train].mean(axis=0)
scale = X[train].std(axis=0)
scale[scale == 0] = 1
Xs = (X - mean) / scale

beta = fit_logistic(Xs[train], y[train])
val_prob = sigmoid(np.column_stack([np.ones(val.sum()), Xs[val]]) @ beta)
val_y = y[val]

# Select threshold on validation data only, maximizing macro-F1.
thresholds = np.arange(0.05, 0.951, 0.01)
candidates = []
for threshold in thresholds:
    result = score(val_y, (val_prob >= threshold).astype(int))
    candidates.append((result["macro_f1"], result["balanced_accuracy"], threshold, result))

_, _, best_threshold, validation_metrics = max(candidates, key=lambda x: (x[0], x[1]))

test_prob = sigmoid(np.column_stack([np.ones(test.sum()), Xs[test]]) @ beta)
test_pred = (test_prob >= best_threshold).astype(int)
test_metrics = score(y[test], test_pred)

report = {
    "records_used": int(len(df)),
    "feature_names": FEATURES,
    "split_group_counts": {
        "train": len(train_groups),
        "validation": len(val_groups),
        "test": len(test_groups),
    },
    "split_record_counts": {
        "train": int(train.sum()),
        "validation": int(val.sum()),
        "test": int(test.sum()),
    },
    "group_overlap": {
        "train_validation": len(train_groups & val_groups),
        "train_test": len(train_groups & test_groups),
        "validation_test": len(val_groups & test_groups),
    },
    "selected_threshold": float(best_threshold),
    "validation_metrics_at_selected_threshold": validation_metrics,
    "held_out_test_metrics": test_metrics,
    "note": "Threshold selected on validation groups; final metrics reported on separate test groups. This evaluates QMutBench-derived program-fault labels, not verified hardware faults."
}

(OUT / "threshold_tuning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
pd.DataFrame([{
    "threshold": float(t),
    **score(val_y, (val_prob >= t).astype(int))
} for t in thresholds]).to_csv(OUT / "threshold_validation_curve.csv", index=False)

np.savez(
    OUT / "numpy_logistic_model_tuned.npz",
    coefficients=beta,
    feature_mean=mean,
    feature_scale=scale,
    feature_names=np.array(FEATURES),
    threshold=best_threshold,
    positive_class=np.array(NORMAL),
)

pd.DataFrame({
    "original_circuit_id": groups[test],
    "actual_label": np.where(y[test] == 1, NORMAL, FAULT),
    "predicted_label": np.where(test_pred == 1, NORMAL, FAULT),
    "normal_probability": test_prob,
}).to_csv(OUT / "tuned_test_predictions.csv", index=False)

print(json.dumps(report, indent=2))
print("\nSaved threshold_tuning_report.json, threshold_validation_curve.csv,")
print("numpy_logistic_model_tuned.npz and tuned_test_predictions.csv to", OUT)
