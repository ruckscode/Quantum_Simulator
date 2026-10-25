"""Reproducible Phase 5 observable LR training and portable JSON export.

This module consumes already prepared Phase 5 training rows; it never scans
datasets, simulates cases, calibrates probabilities, or reads locked families.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quantum_simulator.ml_diagnosis import CLASSES, FEATURE_NAMES
from quantum_simulator.observable_model_adapter import ARTIFACT_TYPE

from .phase5_benchmark_infrastructure import FAMILIES, PARTITIONS

TRAINING_FAMILIES = PARTITIONS["development"]
RANDOM_STATE = 42
TRAINING_STATUS = "DEVELOPMENT CANDIDATE; NOT VALIDATED FOR DEPLOYMENT"


def _validate_training_rows(rows: Sequence[Mapping]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("Phase 5 training rows are required")
    families = {row.get("family_id") for row in rows}
    if not families <= TRAINING_FAMILIES:
        raise ValueError("training rows may contain only F1/F2 development families")
    if families != set(TRAINING_FAMILIES):
        raise ValueError("training rows must include both F1 and F2 development families")
    features, labels = [], []
    for row in rows:
        if set(row) != {"family_id", "features", "label"}:
            raise ValueError("training rows must contain only family_id, features, and label")
        values = row["features"]
        if not isinstance(values, Mapping) or tuple(values) != FEATURE_NAMES:
            raise ValueError("training feature names/order must exactly match the frozen schema")
        vector = np.asarray([values[name] for name in FEATURE_NAMES], dtype=float)
        if np.isinf(vector).any() or not np.isfinite(vector).all():
            raise ValueError("Phase 5 development features must be finite")
        label = row["label"]
        if label not in CLASSES:
            raise ValueError("training labels must be one of the three supported single-cause classes")
        features.append(vector)
        labels.append(label)
    if set(labels) != set(CLASSES):
        raise ValueError("F1/F2 training rows must contain all supported classes")
    return np.asarray(features, dtype=float), np.asarray(labels)


def train_phase5_artifact(rows: Sequence[Mapping], output_path: str | Path) -> dict:
    """Fit the frozen LR pipeline on F1/F2 and create an uncalibrated artifact.

    Existing output is never overwritten. F3 and locked-family records are
    rejected before fitting; this function has no dataset-loading behavior.
    """
    X, y = _validate_training_rows(rows)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing model artifact: {destination}")
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
    )
    pipeline.fit(X, y)
    imputer, scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1], pipeline.steps[2][1]
    class_rows = [list(classifier.classes_).index(name) for name in CLASSES]
    artifact = {
        "artifact_type": ARTIFACT_TYPE,
        "training_status": TRAINING_STATUS,
        "model_type": "sklearn.linear_model.LogisticRegression",
        "feature_names": list(FEATURE_NAMES),
        "class_names": list(CLASSES),
        "preprocessing": {
            "median_imputer_statistics": imputer.statistics_.tolist(),
            "standard_scaler_mean": scaler.mean_.tolist(),
            "standard_scaler_scale": scaler.scale_.tolist(),
        },
        "coefficients": classifier.coef_[class_rows].tolist(),
        "intercept": classifier.intercept_[class_rows].tolist(),
        "probability_calibration": "none",
        "probabilities_calibrated": False,
        "confidence_policy": {
            "cutoff": 0.60,
            "meaning": "operational confidence rule; not a guarantee of 60% empirical correctness",
            "probability_source": "uncalibrated Logistic Regression predict_proba output",
        },
        "metadata": {
            "dataset_source": "Phase 5 prepared training rows",
            "evaluation_methodology": "development-family-only fit; no probability calibration",
            "deployment_validation": "NOT_PERFORMED",
            "training_families": sorted(TRAINING_FAMILIES),
            "excluded_families": sorted(set(FAMILIES) - set(TRAINING_FAMILIES)),
            "row_count": len(rows),
            "random_state": RANDOM_STATE,
            "class_order_source": "quantum_simulator.ml_diagnosis.CLASSES",
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temp_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(artifact, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link creation is atomic and fails if another artifact appeared.
        os.link(temp_path, destination)
        temp_path.unlink()
    except BaseException:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return artifact


__all__ = ["TRAINING_FAMILIES", "train_phase5_artifact"]
