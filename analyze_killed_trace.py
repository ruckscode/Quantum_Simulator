from pathlib import Path
import pandas as pd
import json

root = Path(__file__).resolve().parent
source = root / "data" / "qmutbench_ml" / "qmutbench_results.csv"
output = root / "data" / "qmutbench_ml" / "final" / "killed_trace_analysis.json"

df = pd.read_csv(source)
features = [
    "trace_max_probability_difference",
    "trace_mean_probability_difference",
    "trace_median_probability_difference",
    "trace_deviation_concentration",
    "trace_high_deviation_fraction",
]

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

mutants = df[df["record_type"].eq("mutant")].copy()
mutants["all_zero_trace"] = mutants[features].fillna(0).eq(0).all(axis=1)
mutants["max_difference_positive"] = mutants["trace_max_probability_difference"] > 0

report = {
    "mutant_count": int(len(mutants)),
    "killed_categories": {},
    "killed_by_zero_trace": {},
    "killed_by_positive_max_difference": {},
    "trace_statistics_by_killed": {},
}

for field, target in [
    ("Killed", "killed_categories"),
    ("all_zero_trace", "killed_by_zero_trace"),
    ("max_difference_positive", "killed_by_positive_max_difference"),
]:
    if field == "Killed":
        report[target] = {
            str(k): int(v)
            for k, v in mutants[field].fillna("MISSING").value_counts().items()
        }
    else:
        report[target] = {
            str(k): {str(label): int(count) for label, count in values.items()}
            for k, values in pd.crosstab(
                mutants[field], mutants["Killed"].fillna("MISSING")
            ).iterrows()
        }

for label, group in mutants.groupby("Killed", dropna=False):
    report["trace_statistics_by_killed"][str(label)] = {
        "rows": int(len(group)),
        "zero_trace_rows": int(group["all_zero_trace"].sum()),
        "nonzero_max_difference_rows": int(group["max_difference_positive"].sum()),
        "mean_max_difference": float(group["trace_max_probability_difference"].mean()),
        "median_max_difference": float(group["trace_max_probability_difference"].median()),
    }

output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print(f"\nSaved: {output}")
