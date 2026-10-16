from __future__ import annotations

import pandas as pd

from experiments.robust_ml_evaluation import feature_sets
from experiments.benchmark_label_audit import audit_csv


def test_feature_ablation_has_variant_without_tvd():
    frame = pd.DataFrame({
        "qubits": [1], "depth": [2], "gate_count": [1], "shots": [128],
        "tvd_from_reference": [0.1], "observed_entropy": [0.5],
        "reference_entropy": [0.5], "distinct_outcomes": [2],
        "p_0": [0.5], "p_1": [0.5],
    })
    sets = feature_sets(frame)
    assert "tvd_from_reference" in sets["all_features"]
    assert "tvd_from_reference" not in sets["without_tvd"]
    assert sets["distribution_only"] == ["p_0", "p_1"]


def test_benchmark_label_audit_reports_metadata_without_relabeling(tmp_path):
    csv_path = tmp_path / "qmutbench_sample.csv"
    pd.DataFrame({
        "MutationOperator": ["Add", "Replace", "Remove"],
        "Killed": ["WOO", "OPO", "Survivor"],
        "MutationID": [1, 2, 3],
    }).to_csv(csv_path, index=False)
    report = audit_csv(csv_path, chunksize=2)
    assert report["rows"] == 3
    assert report["diagnosis_ground_truth_column_present"] is False
    assert "Killed" in report["candidate_label_columns"]
    assert "do not map" in report["candidate_label_columns"]["Killed"]["interpretation"]
