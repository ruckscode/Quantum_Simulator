from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "qmutbench_ml" / "qmutbench_results.csv"
FAILURES = ROOT / "data" / "qmutbench_ml" / "qmutbench_failures.csv"
OUT = ROOT / "data" / "qmutbench_ml" / "final" / "dataset_audit.json"

FEATURES = [
    "trace_max_probability_difference",
    "trace_mean_probability_difference",
    "trace_median_probability_difference",
    "trace_deviation_concentration",
    "trace_high_deviation_fraction",
]

if not DATA.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA}")

df = pd.read_csv(DATA)
required = ["label", "record_type", "original_circuit_id"] + FEATURES
missing_columns = [c for c in required if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

features = df[FEATURES].apply(pd.to_numeric, errors="coerce")
zero_rows = features.fillna(0).eq(0).all(axis=1)

report = {
    "dataset": str(DATA),
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "label_counts": {
        str(k): int(v) for k, v in df["label"].value_counts(dropna=False).items()
    },
    "record_type_counts": {
        str(k): int(v) for k, v in df["record_type"].value_counts(dropna=False).items()
    },
    "unique_original_circuit_groups": int(df["original_circuit_id"].nunique()),
    "missing_values_by_column": {
        str(k): int(v)
        for k, v in df.isna().sum().items()
        if int(v) > 0
    },
    "all_zero_feature_rows": int(zero_rows.sum()),
    "all_zero_feature_rate": float(zero_rows.mean()),
    "feature_summary_by_record_type": {},
    "feature_summary_by_label": {},
    "label_record_type_crosstab": {},
    "group_label_mixing": {},
    "failure_file_exists": FAILURES.exists(),
}

for grouping, destination in [
    ("record_type", "feature_summary_by_record_type"),
    ("label", "feature_summary_by_label"),
]:
    for name, indexes in df.groupby(grouping, dropna=False).groups.items():
        subset = features.loc[indexes]
        report[destination][str(name)] = {
            "rows": int(len(subset)),
            "all_zero_feature_rows": int(zero_rows.loc[indexes].sum()),
            "all_zero_feature_rate": float(zero_rows.loc[indexes].mean()),
            "feature_means": {
                col: None if pd.isna(subset[col].mean()) else float(subset[col].mean())
                for col in FEATURES
            },
        }

cross = pd.crosstab(
    df["record_type"].fillna("MISSING").astype(str),
    df["label"].fillna("MISSING").astype(str),
)
report["label_record_type_crosstab"] = {
    str(index): {str(col): int(value) for col, value in row.items()}
    for index, row in cross.iterrows()
}

group_counts = df.groupby("original_circuit_id")["label"].nunique()
report["group_label_mixing"] = {
    "groups_with_multiple_labels": int((group_counts > 1).sum()),
    "total_groups": int(len(group_counts)),
}

if FAILURES.exists():
    failures = pd.read_csv(FAILURES)
    report["failure_rows"] = int(len(failures))
    report["failure_columns"] = list(failures.columns)
else:
    report["failure_rows"] = None

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(json.dumps({
    "rows": report["rows"],
    "labels": report["label_counts"],
    "record_types": report["record_type_counts"],
    "all_zero_feature_rows": report["all_zero_feature_rows"],
    "all_zero_feature_rate": round(report["all_zero_feature_rate"], 4),
    "groups_with_multiple_labels": report["group_label_mixing"]["groups_with_multiple_labels"],
    "failure_rows": report["failure_rows"],
    "report_saved": str(OUT),
}, indent=2))
