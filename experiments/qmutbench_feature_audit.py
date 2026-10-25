"""Fold-training-only association and schema audit for QMutBench predictors."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from experiments.qmutbench_master_optimization import _sha_ids
from experiments.qmutbench_outcome_ml import (DEFAULT_DATA, EXPECTED_SHA256, FEATURES,
    GROUP_COLUMN, TARGET_COLUMN, TARGET_MAP, group_holdout, load_dataset, prepare_xy,
    sha256_file)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/qmutbench_ml/master_optimization_v1"
OUTPUT = ROOT / "reports/experiments/qmutbench_feature_audit_20261025.json"
SAMPLE_ROWS_PER_FOLD = 50000
SEED = 20261025

DEFINITIONS = {
    "algorithm": "Algorithm family associated with the original circuit.",
    "operation": "Mutation operator: Add, Remove, or Replace.",
    "gate": "Quantum gate named by the mutation metadata.",
    "dominant_state": "QMutBench source category indicating whether the original circuit has a dominant output state.",
    "qubits": "Number of qubits in the original circuit metadata.",
    "gates": "Original-circuit gate count in the dataset metadata.",
    "Position_int": "Integer mutation-position value parsed into the local CSV. It is available at prediction time, but its scale exceeds gate count on some records; a direct ratio to gates is not assumed valid.",
    "position": "QMutBench source field documented as Position (Circuit %), stored in ten percentage bands.",
    "algorithm_group": "QMutBench source grouping of algorithm names; likely redundant with algorithm and not part of the selected schema.",
}


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a.astype(str), b.astype(str))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    correction = ((k - 1) * (r - 1)) / max(1, n - 1)
    phi2_corr = max(0.0, phi2 - correction)
    r_corr = r - ((r - 1) ** 2) / max(1, n - 1)
    k_corr = k - ((k - 1) ** 2) / max(1, n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


def run(data: Path = DEFAULT_DATA, output: Path = OUTPUT) -> dict:
    if sha256_file(data) != EXPECTED_SHA256:
        raise ValueError("Dataset checksum differs from validated QMutBench source")
    manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    frame = load_dataset(data)
    x, y, groups = prepare_xy(frame)
    dev_idx, _frozen_idx = group_holdout(y, groups, seed=SEED)
    dev = frame.iloc[dev_idx].reset_index(drop=True)
    del frame, x, y, groups
    features = list(FEATURES) + ["position", "algorithm_group"]
    fold_associations = {name: [] for name in features}
    fold_samples = []
    for fold_id, fold in enumerate(manifest["config"]["folds"], 1):
        train_mask = dev[GROUP_COLUMN].astype(str).isin(fold["fit_groups"]).to_numpy()
        train = dev.loc[train_mask, [GROUP_COLUMN, *features, TARGET_COLUMN]].copy()
        sample = train.sample(n=min(SAMPLE_ROWS_PER_FOLD, len(train)), random_state=SEED + fold_id)
        target = sample[TARGET_COLUMN].astype(str).map(TARGET_MAP).astype(str)
        for feature in features:
            values = sample[feature]
            if feature in {"qubits", "gates", "Position_int"}:
                values = pd.qcut(values, q=10, duplicates="drop").astype(str)
            fold_associations[feature].append(_cramers_v(values, target))
        fold_samples.append({"fold": fold_id, "training_rows": int(len(train)),
            "sample_rows": int(len(sample)), "training_groups": int(train[GROUP_COLUMN].nunique()),
            "fit_group_sha256": _sha_ids(fold["fit_groups"])})
    schema = {}
    for feature in features:
        values = dev[feature]
        schema[feature] = {"definition": DEFINITIONS[feature], "dtype": str(values.dtype),
            "development_cardinality": int(values.nunique(dropna=True)),
            "missing_rows": int(values.isna().sum()),
            "available_at_prediction_time": feature not in {GROUP_COLUMN, TARGET_COLUMN, "Unnamed: 0"},
            "selected_in_current_model": feature in FEATURES,
            "fold_training_target_association_cramers_v": [float(x) for x in fold_associations[feature]],
            "mean_fold_training_target_association_cramers_v": float(np.mean(fold_associations[feature]))}
    excluded = {
        "Origin_program": {"role": "group key and diagnostic only", "predictor_allowed": False,
            "reason": "Origin grouping is held out to prevent sibling-record leakage; identifier could allow memorization."},
        "Unnamed: 0": {"role": "row identifier", "predictor_allowed": False,
            "reason": "Not a semantic feature and can encode data ordering."},
        "Killed": {"role": "target", "predictor_allowed": False,
            "reason": "WOO and OPO map to detected=1; Survivor maps to 0."},
    }
    result = {"schema": "qmutbench-feature-audit-v1", "dataset_sha256": EXPECTED_SHA256,
        "development_rows": int(len(dev)), "development_groups": int(dev[GROUP_COLUMN].nunique()),
        "historical_frozen_test_accessed": False,
        "association_method": "bias-corrected Cramer's V against binary target, computed independently on a fixed random sample of training-side rows within each saved grouped fold; numeric features discretized into training-sample deciles",
        "association_sample_rows_per_fold": SAMPLE_ROWS_PER_FOLD,
        "folds": fold_samples, "features": schema, "excluded_fields": excluded,
        "interpretation": "Associations are descriptive, univariate training-partition statistics. They do not establish causality or independent predictive gain; model value is judged by grouped held-fold experiments.",
        "source_documentation": "data/qmutbench_full/README.md describes position as circuit percent and lists operator, gate, algorithm, and qubit metadata."}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    audit = run()
    print(json.dumps({"rows": audit["development_rows"], "groups": audit["development_groups"],
        "features": {k: {"cardinality": v["development_cardinality"],
                          "cramers_v_mean": v["mean_fold_training_target_association_cramers_v"]}
                     for k, v in audit["features"].items()}, "output": str(OUTPUT)}, indent=2))
