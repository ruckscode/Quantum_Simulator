"""Leakage-aware, circuit-family-held-out evaluation for controlled simulator data.

This is a methodological stress test, not official benchmark or IBM accuracy.
It reuses the transparent scenario generator and compares feature ablations using
leave-one-circuit-family-out evaluation. All folds hold out entire circuit families.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from simulator_generated_ml_baseline import LABELS, generate
except ModuleNotFoundError:
    from experiments.simulator_generated_ml_baseline import LABELS, generate

STRUCTURAL = ["qubits", "depth", "gate_count", "shots"]
SUMMARY = ["tvd_from_reference", "observed_entropy", "reference_entropy", "distinct_outcomes"]


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    probability = sorted(c for c in df.columns if c.startswith("p_"))
    return {
        "all_features": STRUCTURAL + SUMMARY + probability,
        "without_tvd": STRUCTURAL + [c for c in SUMMARY if c != "tvd_from_reference"] + probability,
        "distribution_only": probability,
        "summary_without_tvd": STRUCTURAL + [c for c in SUMMARY if c != "tvd_from_reference"],
    }


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    groups = df["family_id"]
    y = df["label"]
    logo = LeaveOneGroupOut()
    rows = []
    reports = {}
    for feature_name, features in feature_sets(df).items():
        X = df[features].fillna(0).astype(float)
        for model_name in ("Logistic Regression", "Random Forest"):
            predictions = pd.Series(index=df.index, dtype="object")
            fold_rows = []
            for train_idx, test_idx in logo.split(X, y, groups):
                if set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]):
                    raise RuntimeError("Circuit-family leakage detected in a fold")
                if set(y.iloc[train_idx]) != set(LABELS):
                    raise RuntimeError("Training fold does not contain every class")
                if model_name == "Logistic Regression":
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
                    )
                else:
                    model = RandomForestClassifier(
                        n_estimators=250, class_weight="balanced",
                        min_samples_leaf=2, random_state=42,
                    )
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                pred = model.predict(X.iloc[test_idx])
                predictions.iloc[test_idx] = pred
                fold_rows.append({
                    "held_out_family": str(groups.iloc[test_idx].iloc[0]),
                    "accuracy": accuracy_score(y.iloc[test_idx], pred),
                    "macro_f1": f1_score(y.iloc[test_idx], pred, labels=LABELS, average="macro", zero_division=0),
                })
            if predictions.isna().any():
                raise RuntimeError("Some rows did not receive an out-of-fold prediction")
            pred_list = predictions.tolist()
            report = classification_report(y, pred_list, labels=LABELS, output_dict=True, zero_division=0)
            summary = {
                "feature_set": feature_name,
                "model": model_name,
                "accuracy": accuracy_score(y, pred_list),
                "macro_precision": precision_score(y, pred_list, labels=LABELS, average="macro", zero_division=0),
                "macro_recall": recall_score(y, pred_list, labels=LABELS, average="macro", zero_division=0),
                "macro_f1": f1_score(y, pred_list, labels=LABELS, average="macro", zero_division=0),
                "rows": len(df),
                "circuit_families": int(groups.nunique()),
                "evaluation": "LeaveOneGroupOut; each test fold holds out one complete circuit family",
            }
            rows.append(summary)
            reports[f"{feature_name}::{model_name}"] = {
                **summary,
                "per_class": report,
                "confusion_matrix_labels": LABELS,
                "confusion_matrix": confusion_matrix(y, pred_list, labels=LABELS).tolist(),
                "fold_metrics": fold_rows,
                "features": features,
            }
    return pd.DataFrame(rows), reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/robust_ml_evaluation")
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repetitions < 2 or args.shots < 16:
        parser.error("repetitions must be >=2 and shots >=16")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = generate(args.repetitions, args.shots, args.seed)
    metrics, detailed = evaluate(data)
    data.to_csv(out / "controlled_dataset.csv", index=False)
    metrics.to_csv(out / "group_cv_metrics.csv", index=False)
    (out / "group_cv_details.json").write_text(json.dumps(detailed, indent=2), encoding="utf-8")
    ax = metrics.pivot(index="feature_set", columns="model", values="macro_f1").plot(kind="bar", ylim=(0, 1), rot=20)
    ax.set_ylabel("Out-of-fold macro F1")
    ax.set_title("Held-out circuit-family evaluation (controlled simulator data)")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "feature_ablation_macro_f1.png", dpi=160)
    plt.close()
    print(metrics.to_string(index=False))
    print("\nSaved robust evaluation artifacts to", out)
    print("WARNING: controlled simulator data only; this is not official benchmark or physical IBM accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
