"""Validated inference for the saved QMutBench mutation-outcome model.

Probabilities are model scores, not calibrated confidence estimates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


class QMutBenchOutcomeModel:
    """Load and run a versioned model bundle using its saved feature contract."""

    def __init__(self, model_path: str | Path, metadata_path: str | Path | None = None):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path) if metadata_path else self.model_path.with_name("best_model_metadata.json")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Model metadata not found: {self.metadata_path}")
        try:
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read model metadata: {self.metadata_path}") from exc
        self.features = tuple(self.metadata.get("feature_schema", ()))
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("Metadata must define a unique, non-empty feature_schema")
        self.input_features = tuple(self.metadata.get("input_feature_schema", self.features))
        if not self.input_features or len(set(self.input_features)) != len(self.input_features):
            raise ValueError("Metadata must define a unique, non-empty input_feature_schema")
        self.feature_derivations = self.metadata.get("feature_derivations", {})
        prohibited = {"Killed", "Origin_program", "Unnamed: 0", "record_id", "row_index"}
        if prohibited.intersection(self.features) or prohibited.intersection(self.input_features):
            raise ValueError("Feature schema contains a prohibited outcome, group, or row identifier")
        for target, derivation in self.feature_derivations.items():
            if target not in self.features or derivation.get("source_column") not in self.input_features:
                raise ValueError(f"Invalid metadata derivation for feature {target!r}")
        expected_hash = self.metadata.get("model_sha256")
        if not expected_hash:
            raise ValueError("Metadata is missing model_sha256")
        actual_hash = self._sha256(self.model_path)
        if actual_hash != expected_hash:
            raise ValueError("Model checksum does not match metadata; artifact bundle may be inconsistent")
        self.threshold = self.metadata.get("inference_threshold")
        if not isinstance(self.threshold, (int, float)) or not 0 <= self.threshold <= 1:
            raise ValueError("Metadata inference_threshold must be between 0 and 1")
        target_map = self.metadata.get("target_map", {})
        if target_map != {"OPO": 1, "Survivor": 0, "WOO": 1}:
            raise ValueError("Metadata target_map does not match the documented QMutBench encoding")
        try:
            self.model = joblib.load(self.model_path)
        except Exception as exc:
            raise ValueError(f"Could not load model artifact {self.model_path}: {exc}") from exc
        if not callable(getattr(self.model, "predict_proba", None)):
            raise ValueError("Model artifact must provide predict_proba")
        if hasattr(self.model, "feature_names_in_") and tuple(self.model.feature_names_in_) != self.features:
            raise ValueError("Model fitted feature order disagrees with metadata feature_schema")
        self._categorical_values = self._learned_categories()
        self.model_version = str(self.metadata.get("model_version", self.metadata.get("schema", "qmutbench-outcome-v1")))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _learned_categories(self) -> dict[str, set[str]]:
        """Read accepted categories from fitted OneHotEncoder objects when available."""
        result: dict[str, set[str]] = {}
        try:
            transformers = self.model.named_steps["preprocess"].transformers_
        except (AttributeError, KeyError, TypeError):
            return result
        for name, transformer, columns in transformers:
            if name == "remainder" or transformer == "drop":
                continue
            try:
                encoder = transformer.named_steps["onehot"]
            except (AttributeError, KeyError):
                encoder = transformer if hasattr(transformer, "categories_") else None
            if encoder is None:
                continue
            for column, values in zip(columns, encoder.categories_):
                result[str(column)] = {str(value) for value in values}
        return result

    def _features(self, records: pd.DataFrame | Mapping[str, Any] | list[Mapping[str, Any]]) -> tuple[pd.DataFrame, bool]:
        is_single = isinstance(records, Mapping)
        if is_single:
            frame = pd.DataFrame([dict(records)])
        elif isinstance(records, pd.DataFrame):
            frame = records.copy()
        elif isinstance(records, list) and all(isinstance(item, Mapping) for item in records):
            frame = pd.DataFrame(records)
        else:
            raise TypeError("Input must be a record mapping, a list of mappings, or a pandas DataFrame")
        if frame.empty:
            raise ValueError("At least one input record is required")
        if "Killed" in frame.columns:
            raise ValueError("Input must not contain target column 'Killed'")
        missing = sorted(set(self.input_features) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(self.input_features))
        if missing or extra:
            raise ValueError(f"Input schema mismatch; missing={missing}; unexpected={extra}")
        frame = frame.loc[:, list(self.input_features)].copy()
        for target, derivation in self.feature_derivations.items():
            source = derivation["source_column"]
            mapping = {str(key): float(value) for key, value in derivation["mapping"].items()}
            if frame[source].isna().any():
                raise ValueError(f"Derived feature source {source!r} contains missing values")
            raw = frame[source].astype(str)
            invalid = sorted(set(raw) - set(mapping))
            if invalid:
                raise ValueError(f"Unsupported value(s) for {source!r}: {invalid[:5]}")
            frame[target] = raw.map(mapping)
            if source not in self.features:
                frame = frame.drop(columns=[source])
        if set(frame.columns) != set(self.features):
            missing_derived = sorted(set(self.features) - set(frame.columns))
            unexpected_derived = sorted(set(frame.columns) - set(self.features))
            raise ValueError(f"Derived feature schema mismatch; missing={missing_derived}; unexpected={unexpected_derived}")
        frame = frame.loc[:, list(self.features)]
        categorical = set(self._categorical_values)
        numeric = set(self.features) - categorical
        for column in numeric:
            try:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature {column!r} must contain numeric values") from exc
            values = frame[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Feature {column!r} contains missing or non-finite numeric values")
        for column, accepted in self._categorical_values.items():
            if frame[column].isna().any():
                raise ValueError(f"Categorical feature {column!r} contains missing values")
            invalid = sorted(set(frame[column].astype(str)) - accepted)
            if invalid:
                raise ValueError(f"Unsupported value(s) for {column!r}: {invalid[:5]}")
            frame[column] = frame[column].astype(str)
        if frame.isna().any().any():
            columns = frame.columns[frame.isna().any()].tolist()
            raise ValueError(f"Input contains missing values in: {columns}")
        return frame, is_single

    def predict(self, records: pd.DataFrame | Mapping[str, Any] | list[Mapping[str, Any]], *, threshold: float | None = None):
        """Return one dict for a mapping, otherwise a list of prediction dicts."""
        frame, is_single = self._features(records)
        cut = self.threshold if threshold is None else threshold
        if not isinstance(cut, (int, float)) or not np.isfinite(cut) or not 0 <= cut <= 1:
            raise ValueError("threshold must be a finite number between 0 and 1")
        try:
            classes = list(self.model.classes_)
            positive_index = classes.index(1)
        except (AttributeError, ValueError) as exc:
            raise ValueError("Model classes must include positive class 1") from exc
        probabilities = np.asarray(self.model.predict_proba(frame), dtype=float)[:, positive_index]
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Model returned invalid probabilities outside [0, 1]")
        labels = probabilities >= cut
        rows = [{"predicted_class": int(label),
                 "predicted_label": "Detected_WOO_or_OPO" if label else "Survivor",
                 "probability_detected": float(probability),
                 "probability_semantics": "uncalibrated model probability",
                 "applied_threshold": float(cut), "model_version": self.model_version}
                for label, probability in zip(labels, probabilities)]
        return rows[0] if is_single else rows
