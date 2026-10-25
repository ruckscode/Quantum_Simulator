import copy
import json

import numpy as np
import pytest

from quantum_simulator.ml_diagnosis import CLASSES, FEATURE_NAMES
from quantum_simulator.observable_model_adapter import (
    ARTIFACT_TYPE,
    DEFAULT_ARTIFACT_PATH,
    PortableLogisticRegressionAdapter,
)
from quantum_simulator.ml_diagnosis import ModelBackendUnavailable


def _synthetic_artifact():
    return {
        "artifact_type": ARTIFACT_TYPE,
        "training_status": "FINAL DEPLOYABLE MODEL",
        "model_type": "sklearn.linear_model.LogisticRegression",
        "feature_names": list(FEATURE_NAMES),
        "class_names": list(CLASSES),
        "preprocessing": {
            "median_imputer_statistics": [0.0] * len(FEATURE_NAMES),
            "standard_scaler_mean": [0.0] * len(FEATURE_NAMES),
            "standard_scaler_scale": [1.0] * len(FEATURE_NAMES),
        },
        "coefficients": [[0.0] * len(FEATURE_NAMES) for _ in CLASSES],
        "intercept": [2.0, 1.0, 0.0],
        "probability_calibration": "none",
        "probabilities_calibrated": False,
        "confidence_policy": {
            "cutoff": 0.60,
            "meaning": "operational confidence rule; not a guarantee of 60% empirical correctness",
            "probability_source": "uncalibrated Logistic Regression predict_proba output",
        },
        "metadata": {
            "dataset_source": "synthetic test fixture",
            "evaluation_methodology": "synthetic adapter contract test",
            "training_families": ["F1_GHZ_CHAIN", "F2_TOFFOLI_COMPUTE"],
            "excluded_families": ["F3_SWAP_ROUTING", "F4_PARITY_TREE", "F5_PHASE_LAYER"],
        },
    }


def _write_artifact(tmp_path, artifact):
    path = tmp_path / "synthetic_model.json"
    path.write_text(json.dumps(artifact, allow_nan=True), encoding="utf-8")
    return path


def test_valid_synthetic_artifact_loads_and_predicts_in_declared_class_order(tmp_path):
    adapter = PortableLogisticRegressionAdapter(_write_artifact(tmp_path, _synthetic_artifact()))
    features = {name: 0.0 for name in FEATURE_NAMES}

    prediction = adapter.predict(features)

    assert tuple(prediction.class_probabilities) == CLASSES
    assert prediction.predicted_class == "NORMAL"
    assert prediction.class_probabilities["NORMAL"] > prediction.class_probabilities["PROGRAM_FAULT"]
    assert np.isclose(sum(prediction.class_probabilities.values()), 1.0)


def test_missing_artifact_raises_documented_exception(tmp_path):
    with pytest.raises(ModelBackendUnavailable, match="model artifact is unavailable"):
        PortableLogisticRegressionAdapter(tmp_path / "missing.json")


def test_default_artifact_is_valid_when_present_or_fails_closed_when_absent():
    assert DEFAULT_ARTIFACT_PATH.name == "observable_logistic_regression.json"
    if not DEFAULT_ARTIFACT_PATH.exists():
        with pytest.raises(ModelBackendUnavailable, match="model artifact is unavailable"):
            PortableLogisticRegressionAdapter(DEFAULT_ARTIFACT_PATH)
        return

    adapter = PortableLogisticRegressionAdapter(DEFAULT_ARTIFACT_PATH)
    assert adapter.probabilities_calibrated is False
    assert adapter.deployment_validated is False
    assert adapter.metadata["training_families"] == ["F1_GHZ_CHAIN", "F2_TOFFOLI_COMPUTE"]
    prediction = adapter.predict({name: 0.0 for name in FEATURE_NAMES})
    assert set(prediction.class_probabilities) == set(CLASSES)
    assert np.isclose(sum(prediction.class_probabilities.values()), 1.0)


def test_malformed_json_and_missing_required_schema_are_rejected(tmp_path):
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not read model artifact"):
        PortableLogisticRegressionAdapter(malformed)

    missing_type = _synthetic_artifact()
    del missing_type["artifact_type"]
    with pytest.raises(ValueError, match="Unsupported observable model artifact type"):
        PortableLogisticRegressionAdapter(_write_artifact(tmp_path, missing_type))


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("feature_names", list(reversed(FEATURE_NAMES)), "feature names/order"),
        ("class_names", list(reversed(CLASSES)), "class names/order"),
        ("coefficients", [[0.0] * len(FEATURE_NAMES)] * 2, "coefficient shapes"),
        ("intercept", [0.0, 1.0], "coefficient shapes"),
    ],
)
def test_incompatible_order_and_parameter_shapes_are_rejected(tmp_path, field, value, message):
    artifact = _synthetic_artifact()
    artifact[field] = value
    with pytest.raises(ValueError, match=message):
        PortableLogisticRegressionAdapter(_write_artifact(tmp_path, artifact))


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda item: item["coefficients"][0].__setitem__(0, float("nan")), "non-finite parameters"),
        (lambda item: item["preprocessing"]["standard_scaler_scale"].__setitem__(0, 0.0), "invalid scaler scales"),
        (lambda item: item["metadata"].__setitem__("dataset_source", ""), "provenance metadata"),
        (lambda item: item["metadata"].pop("evaluation_methodology"), "provenance metadata"),
    ],
)
def test_invalid_numeric_values_and_provenance_are_rejected(tmp_path, mutation, message):
    artifact = copy.deepcopy(_synthetic_artifact())
    mutation(artifact)

    with pytest.raises(ValueError, match=message):
        PortableLogisticRegressionAdapter(_write_artifact(tmp_path, artifact))
