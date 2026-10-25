import joblib
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import qmutbench_master_optimization as master


def test_registered_features_exclude_outcome_and_metadata():
    for columns in master.FEATURE_SETS.values():
        assert master.TARGET_COLUMN not in columns
        assert master.GROUP_COLUMN not in columns
        assert "Unnamed: 0" not in columns


def test_group_membership_rejects_overlap_and_frozen_test():
    master._validate_membership({"a"}, {"b"}, {"test"})
    with pytest.raises(AssertionError, match="leakage"):
        master._validate_membership({"a"}, {"a"}, set())
    with pytest.raises(AssertionError, match="Frozen test"):
        master._validate_membership({"test"}, {"b"}, {"test"})


def test_nested_threshold_requires_fold_labels_and_is_finite():
    oof = pd.DataFrame({
        "y_true": [0, 1, 0, 1, 0, 1],
        "p_detected": [.1, .8, .4, .6, .2, .9],
        "origin_program": ["a", "a", "b", "b", "c", "c"],
        "fold": [1, 1, 2, 2, 3, 3],
    })
    report = master._nested_threshold_report(oof, seed=1)
    assert len(report["folds"]) == 3
    assert all(0 <= f["nested_threshold_metrics"]["macro_f1"]["threshold_fit_on_other_folds"] <= 1
               for f in report["folds"])
    with pytest.raises(ValueError, match="fold"):
        master._nested_threshold_report(oof.drop(columns="fold"), seed=1)


def test_predict_rejects_target_and_writes_feature_only_predictions(tmp_path):
    features = {name: [1, 2] for name in master.FEATURES}
    frame = pd.DataFrame(features)
    model_path = tmp_path / "model.joblib"
    joblib.dump(DummyClassifier(strategy="prior").fit(frame, [0, 1]), model_path)
    source, target = tmp_path / "input.csv", tmp_path / "predictions.csv"
    frame.to_csv(source, index=False)
    master.predict(source, target, model_path=model_path)
    result = pd.read_csv(target)
    assert len(result) == 2
    assert result.probability_detected.between(0, 1).all()
    frame[master.TARGET_COLUMN] = ["Survivor", "WOO"]
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="must not contain"):
        master.predict(source, target, model_path=model_path)
