import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "qmutbench_ml" / "qmutbench_results.csv"
OUT = ROOT / "data" / "qmutbench_ml" / "final"
OUT.mkdir(parents=True, exist_ok=True)

features = [
    "trace_max_probability_difference",
    "trace_mean_probability_difference",
    "trace_median_probability_difference",
    "trace_deviation_concentration",
    "trace_high_deviation_fraction",
]
labels = ["NORMAL", "PROGRAM_FAULT"]

df = pd.read_csv(DATA)
df = df[df["label"].isin(labels)].copy()
df[features] = df[features].apply(pd.to_numeric, errors="coerce")
df[features] = df[features].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features + ["original_circuit_id", "label"])

X = df[features]
y = df["label"]
groups = df["original_circuit_id"]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

models = {
    "Dummy baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=6,
        min_samples_leaf=5,
        random_state=42
    ),
}

summary = []
details = {}
best_name, best_score, best_model = None, -1, None

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    scores = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(
            y_test, pred, labels=labels, average="macro", zero_division=0
        )),
        "normal_precision": float(precision_score(
            y_test, pred, pos_label="NORMAL", zero_division=0
        )),
        "normal_recall": float(recall_score(
            y_test, pred, pos_label="NORMAL", zero_division=0
        )),
        "program_fault_f1": float(f1_score(
            y_test, pred, pos_label="PROGRAM_FAULT", zero_division=0
        )),
    }
    summary.append({"model": name, **scores})
    details[name] = {
        **scores,
        "confusion_matrix_rows_actual_columns_predicted": confusion_matrix(
            y_test, pred, labels=labels
        ).tolist(),
        "classification_report": classification_report(
            y_test, pred, labels=labels, zero_division=0, output_dict=True
        ),
    }

    print(f"\n{'=' * 55}\n{name}")
    print(json.dumps(scores, indent=2))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print("Label order:", labels)
    print(np.array(details[name]["confusion_matrix_rows_actual_columns_predicted"]))
    print(classification_report(y_test, pred, labels=labels, zero_division=0))

    if name != "Dummy baseline" and scores["macro_f1"] > best_score:
        best_name, best_score, best_model = name, scores["macro_f1"], model

results = {
    "dataset": str(DATA),
    "records_used": int(len(df)),
    "features": features,
    "label_counts": {k: int(v) for k, v in y.value_counts().items()},
    "train_records": int(len(train_idx)),
    "test_records": int(len(test_idx)),
    "train_circuit_groups": int(groups.iloc[train_idx].nunique()),
    "test_circuit_groups": int(groups.iloc[test_idx].nunique()),
    "overlapping_circuit_groups": int(
        len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))
    ),
    "models": details,
    "best_model_by_macro_f1_excluding_baseline": best_name,
}

pd.DataFrame(summary).to_csv(OUT / "model_comparison.csv", index=False)
(OUT / "evaluation_report.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)

if best_model is not None:
    joblib.dump(best_model, OUT / "best_program_fault_model.joblib")

# Keep test predictions for error analysis.
if best_model is not None:
    pred = best_model.predict(X_test)
    pd.DataFrame({
        "original_circuit_id": groups.iloc[test_idx].to_numpy(),
        "actual_label": y_test.to_numpy(),
        "predicted_label": pred,
    }).to_csv(OUT / "test_predictions.csv", index=False)

print("\nFINAL OUTPUT DIRECTORY:", OUT)
print("Best model by macro-F1:", best_name)
print("Overlap between train/test circuit groups:", results["overlapping_circuit_groups"])
print("Saved model comparison, evaluation report and test predictions.")
