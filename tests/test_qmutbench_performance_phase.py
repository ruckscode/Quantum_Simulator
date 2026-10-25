import numpy as np
import pandas as pd

from experiments.qmutbench_outcome_ml import FEATURES
from experiments.qmutbench_performance_phase import add_features, _nested_thresholds


def test_feature_interactions_are_deterministic_and_use_only_registered_inputs():
    source = pd.DataFrame({
        "algorithm": ["qft", "qft"], "operation": ["Add", "Replace"],
        "gate": ["h", "h"], "dominant_state": ["yes", "no"],
        "qubits": [4, 4], "gates": [30, 30], "Position_int": [2, 2],
        "position": ["0-10%", "91-100%"],
        "Origin_program": ["origin-a", "origin-b"], "Killed": ["WOO", "Survivor"],
    })
    first = add_features(source, "algorithm_operation")
    again = add_features(source, "algorithm_operation")
    assert first.equals(again)
    assert list(first.columns) == [*FEATURES, "algorithm_operation"]
    assert first.algorithm_operation.tolist() == ["qft::Add", "qft::Replace"]
    assert "Origin_program" not in first and "Killed" not in first
    category = add_features(source, "position_bin")
    fraction = add_features(source, "position_fraction")
    assert category.position.tolist() == ["0-10%", "91-100%"]
    assert fraction.position_fraction.tolist() == [0.05, 0.955]


def test_nested_threshold_keeps_validation_fold_out_of_threshold_fit():
    rows = []
    for fold, base in [(1, .3), (2, .5), (3, .7)]:
        for index in range(20):
            y = index % 2
            rows.append({"y": y, "probability": base + (.2 if y else -.2),
                         "origin_program": f"origin-{fold}", "fold": fold})
    result = _nested_thresholds(pd.DataFrame(rows))
    assert [row["held_fold"] for row in result] == [1, 2, 3]
    assert all(0 <= row["threshold_fit_on_other_folds"] <= 1 for row in result)
    assert all(row["threshold_selection_scope"] == "other two development folds only" for row in result)
    assert all(np.isfinite(row["held_macro_f1"]) for row in result)
