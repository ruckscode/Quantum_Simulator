from pathlib import Path
import json
import pandas as pd

root = Path(__file__).resolve().parent
path = root / "data" / "qmutbench_ml" / "qmutbench_results.csv"
out = root / "data" / "qmutbench_ml" / "final" / "zero_feature_diagnostic.json"

features = [
    "trace_max_probability_difference",
    "trace_mean_probability_difference",
    "trace_median_probability_difference",
    "trace_deviation_concentration",
    "trace_high_deviation_fraction",
]

df = pd.read_csv(path)
x = df[features].apply(pd.to_numeric, errors="coerce")
zero = x.fillna(0).eq(0).all(axis=1)

columns = [
    c for c in [
        "label", "record_type", "original_circuit_id",
        "qmutbench_row_index", "algorithm", "operation",
        "gate", "Killed", "mutant_qasm_path", "original_qasm_path",
    ] if c in df.columns
]

selected = df.loc[zero, columns].copy()
summary = {
    "total_zero_feature_rows": int(zero.sum()),
    "zero_rows_by_label": {
        str(k): int(v) for k, v in df.loc[zero, "label"].value_counts(dropna=False).items()
    },
    "zero_rows_by_record_type": {
        str(k): int(v) for k, v in df.loc[zero, "record_type"].value_counts(dropna=False).items()
    },
    "zero_rows_by_label_and_type": {
        str(label): {str(kind): int(count) for kind, count in kinds.items()}
        for label, kinds in pd.crosstab(
            df.loc[zero, "label"], df.loc[zero, "record_type"]
        ).iterrows()
    },
    "examples": selected.head(20).astype(object).where(pd.notna(selected.head(20)), None).to_dict("records"),
}

out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))
print("\nExample zero-feature records:")
print(selected.head(12).to_string(index=False))
print(f"\nSaved: {out}")
