"""Read-only, source-pinned audit of QMutBench outcome metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.qmutbench_outcome_ml import (
    CATEGORICAL, DEFAULT_DATA, EXPECTED_BYTES, EXPECTED_ROWS, EXPECTED_SHA256,
    FEATURES, GROUP_COLUMN, REQUIRED, TARGET_COLUMN, TARGET_MAP, DTYPES, sha256_file,
)


def audit_dataset(path: Path = DEFAULT_DATA, *, report_dir: Path | None = None) -> dict:
    if path.stat().st_size != EXPECTED_BYTES:
        raise ValueError("Official CSV byte size does not match the recorded source manifest")
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError("Official CSV SHA-256 does not match the recorded source manifest")
    frame = pd.read_csv(path, dtype=DTYPES, low_memory=False)
    missing_cols = sorted(REQUIRED - set(frame.columns))
    extra_cols = sorted(set(frame.columns) - REQUIRED)
    if missing_cols or extra_cols:
        raise ValueError(f"Schema mismatch: missing={missing_cols}; extra={extra_cols}")
    missingness = {str(k): int(v) for k, v in frame.isna().sum().items()}
    labels = frame[TARGET_COLUMN].astype(str).value_counts(dropna=False)
    unknown = sorted(set(labels.index) - set(TARGET_MAP))
    if unknown:
        raise ValueError(f"Unexpected raw labels: {unknown}")

    feature_key = FEATURES
    feature_group = frame.groupby(feature_key, observed=True, sort=False, dropna=False)
    feature_summary = feature_group.agg(
        rows=(TARGET_COLUMN, "size"),
        target_classes=(TARGET_COLUMN, "nunique"),
        origins=(GROUP_COLUMN, "nunique"),
    )
    conflicting = feature_summary[feature_summary["target_classes"] > 1]
    repeated = feature_summary[feature_summary["rows"] > 1]
    origin_counts = frame[GROUP_COLUMN].value_counts()
    raw_duplicates_without_row_index_or_parent = int(frame.duplicated(
        subset=[col for col in frame.columns if col not in {"Unnamed: 0", GROUP_COLUMN}],
    ).sum())
    exact_full_duplicates = int(frame.duplicated().sum())
    target_counts = {str(k): int(v) for k, v in labels.items()}
    detected = target_counts.get("WOO", 0) + target_counts.get("OPO", 0)
    resolved_path = path.resolve()
    try:
        display_path = resolved_path.relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        display_path = f"external-source:{path.name}"
    result = {
        "schema": "qmutbench-outcome-data-audit-v1",
        "source": {
            "path": display_path, "source_repository": "https://github.com/EnautMendi/QMutBench",
            "source_commit": "01f5e7700c051fe715fc368b307deaa7e6869ba8",
            "sha256": actual_hash, "bytes": int(path.stat().st_size),
            "expected_rows": EXPECTED_ROWS, "actual_rows": int(len(frame)),
            "parsed_rows_match_expected": len(frame) == EXPECTED_ROWS,
            "license": "GPL-3.0 (upstream repository license)",
        },
        "schema_columns": list(frame.columns),
        "missing_values_by_column": missingness,
        "unique_counts": {
            "origin_program": int(frame[GROUP_COLUMN].nunique()),
            "algorithm": int(frame["algorithm"].nunique()),
            "algorithm_group": int(frame["algorithm_group"].nunique()),
            "gate": int(frame["gate"].nunique()),
            "mutation_operator": int(frame["operation"].nunique()),
        },
        "qubit_range": [int(frame["qubits"].min()), int(frame["qubits"].max())],
        "raw_Killed_label_counts": target_counts,
        "binary_target_counts": {"Survivor": target_counts.get("Survivor", 0), "Detected_WOO_or_OPO": int(detected)},
        "duplicates": {
            "exact_all_columns": exact_full_duplicates,
            "same_metadata_outcome_across_rows_ignoring_row_index_and_origin": raw_duplicates_without_row_index_or_parent,
            "repeated_feature_profiles": int(len(repeated)),
            "rows_in_repeated_feature_profiles": int(repeated["rows"].sum()),
            "feature_profiles_crossing_origin_programs": int((feature_summary["origins"] > 1).sum()),
            "feature_profiles_with_conflicting_outcomes": int(len(conflicting)),
            "rows_in_conflicting_feature_profiles": int(conflicting["rows"].sum()),
            "note": "Exact input-profile recurrence is measured because origin IDs are excluded from model features; all mutants from one Origin_program are kept in one split.",
        },
        "origin_group_sizes": {
            "min": int(origin_counts.min()), "median": float(origin_counts.median()),
            "max": int(origin_counts.max()),
        },
        "model_task": {
            "target": "binary prediction of recorded Killed outcome: WOO/OPO detected vs Survivor",
            "features": FEATURES,
            "excluded": sorted(REQUIRED - set(FEATURES) - {TARGET_COLUMN}),
            "identifiers_and_index_excluded": ["Unnamed: 0", GROUP_COLUMN],
            "categorical_features": CATEGORICAL,
            "numeric_features": [col for col in FEATURES if col not in CATEGORICAL],
            "post_outcome_features_excluded": [TARGET_COLUMN],
            "simulator_derived_features_used": False,
            "mutation_operator_is_input_metadata": True,
            "intended_claim_scope": "historical benchmark oracle outcome prediction only; not diagnosis or hardware robustness",
        },
        "memory": {"pandas_deep_bytes": int(frame.memory_usage(index=True, deep=True).sum())},
    }
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(frame)}")
    if any(missingness.values()):
        raise ValueError(f"Missing data requires an explicit policy: {missingness}")
    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "data_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(audit_dataset(report_dir=Path("reports/experiments/qmutbench_outcome_ml")), indent=2))
