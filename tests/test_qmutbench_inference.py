import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from qmutbench_ml import QMutBenchOutcomeModel


@pytest.fixture
def bundle(tmp_path):
    features = ["kind", "size"]
    x = pd.DataFrame({"kind": ["a", "b", "a", "b"], "size": [1., 2., 3., 4.]})
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["kind"]),
        ("num", "passthrough", ["size"]),
    ])
    model = Pipeline([("preprocess", prep), ("model", DummyClassifier(strategy="prior"))])
    model.fit(x, [0, 1, 0, 1])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "feature_schema": features,
        "target_map": {"OPO": 1, "Survivor": 0, "WOO": 1},
        "inference_threshold": 0.485,
        "model_sha256": digest,
        "model_version": "test-v1",
    }
    meta = tmp_path / "best_model_metadata.json"
    meta.write_text(json.dumps(metadata), encoding="utf-8")
    return path, meta


def test_single_and_batch_prediction_use_saved_threshold(bundle):
    predictor = QMutBenchOutcomeModel(*bundle)
    single = predictor.predict({"kind": "a", "size": 8})
    batch = predictor.predict([{"kind": "a", "size": 8}, {"kind": "b", "size": 9}])
    assert single["applied_threshold"] == 0.485
    assert single["predicted_class"] == 1
    assert single["predicted_label"] == "Detected_WOO_or_OPO"
    assert single["model_version"] == "test-v1"
    assert "uncalibrated" in single["probability_semantics"]
    assert len(batch) == 2
    assert all(0 <= row["probability_detected"] <= 1 for row in batch)


@pytest.mark.parametrize("record, message", [
    ({"kind": "a"}, "missing"),
    ({"kind": "a", "size": 1, "extra": 2}, "unexpected"),
    ({"kind": "x", "size": 1}, "Unsupported"),
    ({"kind": "a", "size": float("nan")}, "non-finite"),
    ({"kind": "a", "size": "bad"}, "numeric"),
    ({"kind": "a", "size": 1, "Killed": "WOO"}, "target"),
])
def test_rejects_invalid_input(bundle, record, message):
    with pytest.raises(ValueError, match=message):
        QMutBenchOutcomeModel(*bundle).predict(record)


def test_checksum_and_threshold_validation(bundle):
    path, meta = bundle
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        QMutBenchOutcomeModel(path, meta)


def test_source_position_category_derivation_and_validation(tmp_path):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = pd.DataFrame({"size": [1., 2., 3., 4.], "position_fraction": [.05, .25, .75, .955]})
    estimator = make_pipeline(StandardScaler(), DummyClassifier(strategy="prior"))
    estimator.fit(x, [0, 0, 1, 1])
    model_path = tmp_path / "position_model.joblib"
    joblib.dump(estimator, model_path)
    metadata = {
        "feature_schema": ["size", "position_fraction"],
        "input_feature_schema": ["size", "position"],
        "feature_derivations": {"position_fraction": {
            "source_column": "position",
            "mapping": {"0-10%": .05, "21-30%": .255, "71-80%": .755, "91-100%": .955},
        }},
        "target_map": {"OPO": 1, "Survivor": 0, "WOO": 1},
        "inference_threshold": .5,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    meta_path = tmp_path / "position_model_metadata.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    predictor = QMutBenchOutcomeModel(model_path, meta_path)
    result = predictor.predict({"size": 5, "position": "91-100%"})
    assert result["model_version"] == "qmutbench-outcome-v1"
    with pytest.raises(ValueError, match="Unsupported.*position"):
        predictor.predict({"size": 5, "position": "unknown"})
