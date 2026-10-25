from pathlib import Path
import pandas as pd

root = Path("data/qmutbench_ml")
df = pd.read_csv(root / "qmutbench_results.csv")

features = [
    "trace_max_probability_difference",
    "trace_mean_probability_difference",
    "trace_median_probability_difference",
    "trace_deviation_concentration",
    "trace_high_deviation_fraction",
]

print("DATA QUALITY CHECK")
print("=" * 72)
print("Rows:", len(df))
print("\nRecord types:")
print(df["record_type"].value_counts(dropna=False).to_string())
print("\nLabels by record type:")
print(pd.crosstab(df["record_type"], df["label"], dropna=False).to_string())

for col in features:
    if col not in df.columns:
        print(f"\nMISSING FEATURE COLUMN: {col}")
        continue

    print(f"\n{col}")
    print(
        df.groupby("record_type", dropna=False)[col]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(6)
        .to_string()
    )

print("\nFailure rows:", len(pd.read_csv(root / "qmutbench_failures.csv"))
      if (root / "qmutbench_failures.csv").exists() else "failure file not found")

print("\nInterpretation: original records labelled NORMAL are generated reference examples,")
print("not independently verified ground-truth examples of all normal program behaviour.")
