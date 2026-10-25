import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from experiments.qmutbench_outcome_ml import (
    FEATURES,
    _fold_cv,
    group_holdout,
    make_pipeline,
    predict_metadata,
    prepare_xy,
)
from experiments.render_qmutbench_outcome_ml_figures import comparison_svg, confusion_svg


def _frame():
    rows = []
    for group in range(10):
        for outcome, operator in (("Survivor", "Add"), ("WOO", "Replace")):
            rows.append({
                "Unnamed: 0": len(rows),
                "Origin_program": f"origin_{group}",
                "algorithm": "heldout_only_algorithm" if group == 9 else f"algorithm_{group % 3}",
                "qubits": 4 + (group % 2),
                "gates": 10 + group,
                "operation": operator,
                "Position_int": 1 + group,
                "Killed": outcome,
                "gate": "cx" if operator == "Add" else "h",
                "algorithm_group": f"family_{group % 3}",
                "dominant_state": "yes",
                "position": "0-10%",
            })
    return pd.DataFrame(rows)


def test_registered_features_exclude_target_parent_and_row_identifier():
    frame = _frame()
    x, y, groups = prepare_xy(frame)
    assert list(x.columns) == FEATURES
    assert "Killed" not in x
    assert "Origin_program" not in x
    assert "Unnamed: 0" not in x
    assert set(np.unique(y)) == {0, 1}
    assert len(groups) == len(frame)


def test_group_holdout_is_deterministic_and_keeps_origins_disjoint():
    _, y, groups = prepare_xy(_frame())
    first = group_holdout(y, groups, seed=17, folds=5)
    second = group_holdout(y, groups, seed=17, folds=5)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert not (set(groups[first[0]]) & set(groups[first[1]]))


def test_preprocessing_fits_on_training_only_and_unknown_categories_load():
    frame = _frame()
    x, y, groups = prepare_xy(frame)
    train = groups != "origin_9"
    test = ~train
    model = make_pipeline("logistic_regression", seed=23)
    model.fit(x.loc[train], y[train])
    categories = model.named_steps["preprocess"].named_transformers_["categorical"] \
        .named_steps["onehot"].categories_[0]
    assert "heldout_only_algorithm" not in categories
    predictions = model.predict(x.loc[test])
    assert np.array_equal(predictions, model.predict(x.loc[test]))


def test_group_cv_checkpoint_resumes_matching_completed_folds(tmp_path):
    x, y, groups = prepare_xy(_frame())
    train_idx = np.arange(len(y))
    first = _fold_cv(x, y, groups, train_idx, seed=31, folds=3,
                     cache_dir=tmp_path / "cv", cache_key="frozen-key")
    cache_files = list((tmp_path / "cv").glob("*.json"))
    assert len(cache_files) == 12
    second = _fold_cv(x, y, groups, train_idx, seed=31, folds=3,
                      cache_dir=tmp_path / "cv", cache_key="frozen-key")
    assert first == second


def test_report_figures_are_valid_self_contained_svg():
    scores = {name: {"mean_macro_f1": value} for name, value in (
        ("majority", 0.5), ("logistic_regression", 0.6),
        ("decision_tree", 0.7), ("random_forest", 0.8),
    )}
    result = {
        "selection": {"selected_model": "random_forest", "cv": scores},
        "final_test": {
            "confusion_matrix_labels_0_1": [[8, 2], [3, 7]],
            "accuracy": 0.75, "macro_f1": 0.75,
        },
    }
    assert ET.fromstring(comparison_svg(result)).tag.endswith("svg")
    assert ET.fromstring(confusion_svg(result)).tag.endswith("svg")


def test_saved_model_predicts_metadata_without_label_and_rejects_target(tmp_path):
    frame = _frame()
    x, y, _ = prepare_xy(frame)
    model = make_pipeline("logistic_regression", seed=41)
    model.fit(x.iloc[:16], y[:16])
    model_path = tmp_path / "model.joblib"
    import joblib
    joblib.dump(model, model_path)
    prediction_input = tmp_path / "new.csv"
    frame[FEATURES].iloc[16:18].to_csv(prediction_input, index=False)
    prediction_output = tmp_path / "predictions.csv"
    predict_metadata(prediction_input, model_path, prediction_output)
    assert len(pd.read_csv(prediction_output)) == 2
    frame.iloc[16:18].to_csv(prediction_input, index=False)
    try:
        predict_metadata(prediction_input, model_path, prediction_output)
    except ValueError as exc:
        assert "must not contain the Killed target" in str(exc)
    else:
        raise AssertionError("prediction accepted target-bearing input")
