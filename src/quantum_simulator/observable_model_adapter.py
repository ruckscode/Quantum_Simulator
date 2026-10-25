"""Portable JSON adapter for an exported observable-only sklearn LR model.

The JSON contains coefficients and fitted preprocessing statistics exported
from the existing sklearn pipeline. It is deliberately not a substitute or a
fallback model: an artifact must first be produced by the dedicated exporter.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .ml_diagnosis import (
    CLASSES,
    FEATURE_NAMES,
    ClassifierPrediction,
    ModelBackendUnavailable,
)

ARTIFACT_TYPE = "quantum_simulator.observable_logistic_regression.v1"
DEVELOPMENT_CANDIDATE_STATUS = "DEVELOPMENT CANDIDATE; NOT VALIDATED FOR DEPLOYMENT"
DEFAULT_ARTIFACT_PATH = Path(__file__).parent / "models" / "observable_logistic_regression.json"


class PortableLogisticRegressionAdapter:
    """Load exported sklearn LR parameters and implement its inference path."""

    feature_names = FEATURE_NAMES

    def __init__(self, artifact_path: str | Path = DEFAULT_ARTIFACT_PATH):
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.is_file():
            raise ModelBackendUnavailable(
                f"Genuine observable-only model artifact is unavailable: {self.artifact_path}"
            )
        try:
            artifact = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read model artifact {self.artifact_path}: {exc}") from exc
        if artifact.get("artifact_type") != ARTIFACT_TYPE:
            raise ValueError("Unsupported observable model artifact type")
        if tuple(artifact.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Artifact feature names/order do not match the approved five-feature schema")
        if tuple(artifact.get("class_names", ())) != CLASSES:
            raise ValueError("Artifact class names/order do not match the supported diagnosis classes")
        if artifact.get("model_type") != "sklearn.linear_model.LogisticRegression":
            raise ValueError("Artifact model type is not the selected Logistic Regression model")
        training_status = artifact.get("training_status")
        if training_status not in {"FINAL DEPLOYABLE MODEL", DEVELOPMENT_CANDIDATE_STATUS}:
            raise ValueError("Artifact training status is unsupported")
        if artifact.get("probability_calibration") != "none" or artifact.get("probabilities_calibrated") is not False:
            raise ValueError("Phase 5 artifact must declare uncalibrated Logistic Regression probabilities")
        policy = artifact.get("confidence_policy", {})
        if policy.get("cutoff") != 0.60 or policy.get("meaning") != "operational confidence rule; not a guarantee of 60% empirical correctness":
            raise ValueError("Artifact confidence policy must preserve the fixed operational 0.60 cutoff")
        if policy.get("probability_source") != "uncalibrated Logistic Regression predict_proba output":
            raise ValueError("Artifact confidence policy must identify uncalibrated probability outputs")

        preprocessing = artifact.get("preprocessing", {})
        self.impute_values = np.asarray(preprocessing.get("median_imputer_statistics", []), dtype=float)
        self.scaler_mean = np.asarray(preprocessing.get("standard_scaler_mean", []), dtype=float)
        self.scaler_scale = np.asarray(preprocessing.get("standard_scaler_scale", []), dtype=float)
        self.coefficients = np.asarray(artifact.get("coefficients", []), dtype=float)
        self.intercept = np.asarray(artifact.get("intercept", []), dtype=float)
        expected_vector = (len(FEATURE_NAMES),)
        expected_matrix = (len(CLASSES), len(FEATURE_NAMES))
        if any(vector.shape != expected_vector for vector in (self.impute_values, self.scaler_mean, self.scaler_scale)):
            raise ValueError("Artifact preprocessing vectors have an invalid shape")
        if self.coefficients.shape != expected_matrix or self.intercept.shape != (len(CLASSES),):
            raise ValueError("Artifact coefficient shapes do not match the feature and class schemas")
        if not all(np.isfinite(array).all() for array in (
            self.impute_values, self.scaler_mean, self.scaler_scale,
            self.coefficients, self.intercept,
        )) or np.any(self.scaler_scale <= 0):
            raise ValueError("Artifact contains non-finite parameters or invalid scaler scales")
        self.metadata = artifact.get("metadata", {})
        self.training_status = training_status
        self.deployment_validated = (
            training_status == "FINAL DEPLOYABLE MODEL"
            and self.metadata.get("deployment_validation") == "PASSED"
        )
        if not self.metadata.get("dataset_source") or not self.metadata.get("evaluation_methodology"):
            raise ValueError("Artifact is missing dataset/evaluation provenance metadata")
        if tuple(self.metadata.get("training_families", ())) != ("F1_GHZ_CHAIN", "F2_TOFFOLI_COMPUTE"):
            raise ValueError("Artifact provenance must show training on F1/F2 only")
        if tuple(self.metadata.get("excluded_families", ())) != (
            "F3_SWAP_ROUTING", "F4_PARITY_TREE", "F5_PHASE_LAYER"
        ):
            raise ValueError("Artifact provenance must exclude F3/F4/F5 from fitting")
        self.probabilities_calibrated = False
        self.confidence_cutoff = 0.60

    def predict(self, features: Mapping[str, float]) -> ClassifierPrediction:
        if tuple(features) != FEATURE_NAMES:
            raise ValueError("Inference features must use the exact approved names and order")
        row = np.asarray([features[name] for name in FEATURE_NAMES], dtype=float)
        if row.shape != (len(FEATURE_NAMES),) or np.isinf(row).any():
            raise ValueError("Inference features must be numeric and cannot contain infinity")
        imputed = np.where(np.isnan(row), self.impute_values, row)
        standardized = (imputed - self.scaler_mean) / self.scaler_scale
        logits = self.coefficients @ standardized + self.intercept
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / exp_logits.sum()
        class_probabilities = {label: float(prob) for label, prob in zip(CLASSES, probabilities)}
        predicted = max(class_probabilities, key=class_probabilities.get)
        return ClassifierPrediction(predicted, class_probabilities)


__all__ = ["ARTIFACT_TYPE", "DEVELOPMENT_CANDIDATE_STATUS", "DEFAULT_ARTIFACT_PATH", "PortableLogisticRegressionAdapter"]
