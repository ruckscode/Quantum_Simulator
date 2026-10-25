"""Pluggable observable-only ML classification and diagnosis integration.

Feature extraction, classifier inference, and evidence-based localization are
kept separate. A production sklearn adapter can be supplied when its genuine
validated estimator artifact is available; this module never trains a model.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Mapping, Protocol, Sequence

import numpy as np

from .diagnosis_engine import (
    DiagnosisCategory,
    DiagnosisResult,
    _canonicalize_circuit,
    diagnose_execution,
    has_relevant_calibration_fault_evidence,
)
from .fault_localizer import _unsupported_operations
from .statistics import compare_distributions

FEATURE_NAMES = (
    "total_variation_distance", "js_divergence", "hellinger_distance",
    "observed_entropy_bits", "observed_max_probability",
)
CLASSES = ("NORMAL", "PROGRAM_FAULT", "HARDWARE_ANOMALY")
# Fixed operational abstention rule over the classifier's output probabilities.
# Phase 5 does not calibrate these probabilities; 0.60 is not a guarantee of
# 60% empirical correctness. Fault labels additionally require localization.
MIN_CONFIDENCE = 0.60


@dataclass(frozen=True)
class ClassifierPrediction:
    predicted_class: str
    class_probabilities: Mapping[str, float]


class ObservableClassifier(Protocol):
    """Inference adapter contract. Implementations receive approved named features."""

    feature_names: Sequence[str]

    def predict(self, features: Mapping[str, float]) -> ClassifierPrediction: ...


class ModelBackendUnavailable(RuntimeError):
    """Raised when no genuine trained model adapter has been configured."""


class UnavailableObservableClassifier:
    feature_names = FEATURE_NAMES

    def predict(self, features: Mapping[str, float]) -> ClassifierPrediction:
        raise ModelBackendUnavailable(
            "No trained observable-only model artifact/backend is available. "
            "Supply an ObservableClassifier adapter; no model is trained during inference."
        )


@dataclass
class MLDiagnosisResult:
    predicted_class: str
    class_probabilities: dict[str, float]
    ml_features: dict[str, float]
    localization_result: DiagnosisResult | None
    final_diagnosis: DiagnosisCategory
    confidence: float
    summary: str


def extract_observable_features(
    expected: Mapping[str, float] | Sequence[float],
    observed: Mapping[str, float] | Sequence[float],
) -> dict[str, float]:
    """Extract precisely the five approved independent observable features."""
    comparison = compare_distributions(expected, observed)
    probabilities = tuple(comparison.observed_distribution.values())
    entropy = -sum(p * log2(p) for p in probabilities if p > 0)
    return {
        "total_variation_distance": comparison.total_variation_distance,
        "js_divergence": comparison.js_divergence,
        "hellinger_distance": comparison.hellinger_distance,
        "observed_entropy_bits": float(entropy),
        "observed_max_probability": float(max(probabilities)),
    }


def _validated_prediction(classifier: ObservableClassifier, features: dict[str, float]):
    try:
        feature_names = tuple(classifier.feature_names)
    except (AttributeError, TypeError) as exc:
        raise ValueError("Classifier must declare the approved feature names and order") from exc
    if feature_names != FEATURE_NAMES:
        raise ValueError("Classifier feature names/order do not match approved observable-only schema")
    prediction = classifier.predict(features)
    if not isinstance(prediction, ClassifierPrediction):
        raise TypeError("classifier.predict must return ClassifierPrediction")
    if not isinstance(prediction.class_probabilities, Mapping):
        raise TypeError("Classifier class_probabilities must be a mapping")
    probabilities = dict(prediction.class_probabilities)
    if set(probabilities) != set(CLASSES):
        raise ValueError(f"Classifier probabilities must cover exactly {CLASSES}")
    if prediction.predicted_class not in CLASSES:
        raise ValueError(f"Unsupported classifier class: {prediction.predicted_class!r}")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value) or value < 0 or value > 1
        for value in probabilities.values()
    ):
        raise ValueError("Classifier probabilities must be finite values in [0, 1]")
    if not np.isclose(sum(probabilities.values()), 1.0, atol=1e-6):
        raise ValueError("Classifier probabilities must sum to 1")
    if prediction.predicted_class != max(probabilities, key=probabilities.get):
        raise ValueError("predicted_class must be the highest-probability class")
    return probabilities


def diagnose_with_ml(
    expected_distribution, observed_distribution, *, expected_circuit=None,
    observed_circuit=None, hardware_model=None, threshold: float = 0.05,
    shots: int | None = None, precomputed_trace=None,
    classifier: ObservableClassifier | None = None,
    min_confidence: float = MIN_CONFIDENCE,
) -> MLDiagnosisResult:
    """Predict broad diagnosis class, then use existing evidence to localize."""
    if (
        isinstance(min_confidence, (bool, np.bool_))
        or not isinstance(min_confidence, (int, float, np.integer, np.floating))
        or not np.isfinite(min_confidence)
        or not 0 <= min_confidence <= 1
    ):
        raise ValueError("min_confidence must be between 0 and 1")
    features = extract_observable_features(expected_distribution, observed_distribution)
    backend = classifier if classifier is not None else UnavailableObservableClassifier()
    probabilities = _validated_prediction(backend, features)
    predicted = max(probabilities, key=probabilities.get)
    confidence = probabilities[predicted]

    if confidence < min_confidence:
        return MLDiagnosisResult(
            predicted, probabilities, features, None,
            DiagnosisCategory.INSUFFICIENT_EVIDENCE, confidence,
            f"ML confidence {confidence:.3f} is below the {min_confidence:.3f} abstention threshold.",
        )
    if predicted == "NORMAL":
        comparison = compare_distributions(
            expected_distribution,
            observed_distribution,
            threshold=threshold,
            shots=shots,
        )
        expected_normalized = _canonicalize_circuit(expected_circuit)
        observed_normalized = _canonicalize_circuit(observed_circuit)
        unsupported_operations = _unsupported_operations(
            expected_normalized, observed_normalized,
        )
        direct_circuit_mismatch = (
            expected_normalized is not None
            and observed_normalized is not None
            and expected_normalized != observed_normalized
        )
        circuit_for_calibration = observed_circuit or expected_circuit
        calibration_fault = has_relevant_calibration_fault_evidence(
            hardware_model, circuit_for_calibration,
        )
        # Trace differences alone are not independent fault evidence: distinct
        # outcomes from separate superposition executions can be legitimate.
        if (
            not comparison.anomaly_detected
            and not direct_circuit_mismatch
            and not calibration_fault
            and not unsupported_operations
        ):
            return MLDiagnosisResult(
                predicted, probabilities, features, None,
                DiagnosisCategory.NO_ANOMALY, confidence,
                "ML classified NORMAL; output, circuit, and calibration checks found no contradictory evidence.",
            )
    # Reuse rule-based diagnosis and its independent localizers/evidence. Its
    # trace-only deviation candidates remain non-program evidence by design.
    localization = diagnose_execution(
        expected_distribution, observed_distribution,
        expected_circuit=expected_circuit, observed_circuit=observed_circuit,
        hardware_model=hardware_model, threshold=threshold, shots=shots,
        precomputed_trace=precomputed_trace,
    )
    if predicted == "NORMAL":
        # A model prediction cannot erase direct fault evidence. Keep the
        # evidence-based diagnosis (including mixed-fault ambiguity) whenever
        # the rule engine finds anything beyond a genuinely normal execution.
        if localization.category is not DiagnosisCategory.NO_ANOMALY:
            return MLDiagnosisResult(
                predicted, probabilities, features, localization,
                localization.category, confidence,
                f"ML classified NORMAL, but evidence-based diagnosis reports: {localization.summary}",
            )
        return MLDiagnosisResult(
            predicted, probabilities, features, None, DiagnosisCategory.NO_ANOMALY,
            confidence, "ML classified NORMAL; evidence checks found no direct fault or output anomaly.",
        )

    supported = (
        bool(localization.program_behavioral_candidates)
        if predicted == "PROGRAM_FAULT"
        else bool(localization.hardware_evidence_candidates)
    )
    # A syntactic source diff can localize a candidate without establishing
    # behavioral impact. Do not let either the classifier or that diff turn
    # an observationally unchanged execution into a program-fault diagnosis.
    if predicted == "PROGRAM_FAULT" and localization.anomaly_detected is False:
        supported = False
    final = (
        DiagnosisCategory.PROGRAM_FAULT if predicted == "PROGRAM_FAULT"
        else DiagnosisCategory.HARDWARE_ANOMALY
    ) if supported else DiagnosisCategory.INSUFFICIENT_EVIDENCE
    if supported:
        summary = localization.summary
    elif predicted == "PROGRAM_FAULT" and localization.anomaly_detected is False:
        summary = (
            "A source-level mismatch was localized, but observed outputs show no behavioral "
            "anomaly; abstaining on program-fault attribution."
        )
    elif predicted == "PROGRAM_FAULT" and localization.program_evidence_candidates:
        summary = (
            "A source mismatch was localized, but counterfactual evidence does not link it "
            "to the measured outputs; abstaining on program-fault attribution."
        )
    else:
        summary = f"ML classified {predicted}, but existing localization evidence does not support a definitive cause."
    return MLDiagnosisResult(predicted, probabilities, features, localization, final, confidence, summary)


__all__ = [
    "FEATURE_NAMES", "CLASSES", "MIN_CONFIDENCE", "ClassifierPrediction",
    "ObservableClassifier", "UnavailableObservableClassifier", "ModelBackendUnavailable",
    "MLDiagnosisResult", "extract_observable_features", "diagnose_with_ml",
]
