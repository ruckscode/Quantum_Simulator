import csv
import numpy as np

from ml_evaluation import LABELS, OBSERVABLE_ONLY, load_dataset
from ml_fallback_training import _models, _add_ideal_circuit_features
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = "ml_results/reduced_calibration_experiment"
import os
os.makedirs(OUT, exist_ok=True)

rows = load_dataset()
if len(rows) != 270 or len({r["circuit_family_id"] for r in rows}) != 30:
    raise ValueError("Dataset differs from the expected 270 records / 30 groups.")

rows = _add_ideal_circuit_features(rows)
y = np.array([r["ground_truth_label"] for r in rows])
groups = np.array([r["circuit_family_id"] for r in rows])

observable = [
    "total_variation_distance",
    "js_divergence",
    "hellinger_distance",
    "observed_entropy_bits",
    "observed_max_probability",
]
readout = observable + ["calibration_readout_error_max"]
readout_t1 = readout + ["calibration_t1_min"]
baseline = list(OBSERVABLE_ONLY)

feature_sets = {
    "BASELINE_ALL_CALIBRATION": baseline,
    "OBSERVABLES_PLUS_READOUT": readout,
    "OBSERVABLES_READOUT_T1": readout_t1,
    "OBSERVABLES_ONLY": observable,
}

splitter = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)
folds = list(splitter.split(np.zeros(len(y)), y, groups))
results = []

for set_name, features in feature_sets.items():
    X = np.array([
        [float(r[f]) if r.get(f) is not None else np.nan for f in features]
        for r in rows
    ], dtype=float)

    for model_name, estimator in _models().items():
        # Evaluate only LR and DT here; the custom forest is slower and
        # did not outperform LR on the baseline feature set.
        if model_name.startswith("Random Forest"):
            continue

        accuracies, macro_f1s = [], []
        for train, test in folds:
            if set(groups[train]) & set(groups[test]):
                raise AssertionError("Circuit-family leakage detected")

            # Imputation and scaling are fitted only on each training fold.
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                *([StandardScaler()] if model_name == "Logistic Regression" else []),
                estimator
            )
            model.fit(X[train], y[train])
            pred = model.predict(X[test])
            accuracies.append(accuracy_score(y[test], pred))
            macro_f1s.append(
                f1_score(y[test], pred, labels=LABELS,
                         average="macro", zero_division=0)
            )

        results.append({
            "feature_set": set_name,
            "model": model_name,
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies, ddof=1),
            "macro_f1_mean": np.mean(macro_f1s),
            "macro_f1_std": np.std(macro_f1s, ddof=1),
        })
        print(f"Completed {set_name} / {model_name}", flush=True)

path = os.path.join(OUT, "comparison.csv")
with open(path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(sorted(
        results, key=lambda r: r["macro_f1_mean"], reverse=True
    ))

print("\n=== REDUCED-CALIBRATION RESULTS ===")
for r in sorted(results, key=lambda r: r["macro_f1_mean"], reverse=True):
    print(
        f'{r["feature_set"]:30} {r["model"]:22} '
        f'Accuracy={r["accuracy_mean"]:.3f} +/- {r["accuracy_std"]:.3f} '
        f'Macro-F1={r["macro_f1_mean"]:.3f} +/- {r["macro_f1_std"]:.3f}'
    )
print(f"\nSaved to {path}")
