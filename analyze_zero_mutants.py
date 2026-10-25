from pathlib import Path
import pandas as pd
import json

root = Path(__file__).resolve().parent
data_path = root / "data" / "qmutbench_ml" / "qmutbench_results.csv"
out_path = root / "data" / "qmutbench_ml" / "final" / "zero_mutant_analysis.json"

df = pd.read_csv(data_path)

features = [
    "trace_max_probability_difference",
    "trace_mean_probability_difference",
    "trace_median_probability_difference",
    "trace_deviation_concentration",
    "trace_high_deviation_fraction",
]

x = df[features].apply(pd.to_numeric, errors="coerce")
zero = x.fillna(0).eq(0).all(axis=1)

mutants = df.loc[zero & df["record_type"].eq("mutant")].copy()

summary = {
    "zero_feature_mutants": int(len(mutants)),
    "killed_distribution": {
        str(k): int(v)
        for k, v in mutants["Killed"].fillna("MISSING").value_counts().items()
    },
    "operation_distribution": {
        str(k): int(v)
        for k, v in mutants["operation"].fillna("MISSING").value_counts().items()
    },
    "gate_distribution": {
        str(k): int(v)
        for k, v in mutants["gate"].fillna("MISSING").value_counts().items()
    },
    "algorithm_distribution": {
        str(k): int(v)
        for k, v in mutants["algorithm"].fillna("MISSING").value_counts().items()
    },
    "examples": mutants[
        [
            "qmutbench_row_index", "original_circuit_id",
            "algorithm", "operation", "gate", "Killed",
            "original_qasm_path", "mutant_qasm_path"
        ]
    ].head(30).astype(object).where(
        pd.notna(mutants[
            [
                "qmutbench_row_index", "original_circuit_id",
                "algorithm", "operation", "gate", "Killed",
                "original_qasm_path", "mutant_qasm_path"
            ]
        ].head(30)), None
    ).to_dict("records"),
}

out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))
print(f"\nSaved: {out_path}")
