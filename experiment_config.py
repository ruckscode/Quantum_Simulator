"""Configuration model and loader for reproducible local experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


REQUIRED_FIELDS = {"experiment_name", "num_qubits", "shots", "circuit", "random_seed"}
ALLOWED_FIELDS = REQUIRED_FIELDS | {
    "validation_label",
    "observed_circuit",
    "hardware_parameters",
    "noise_parameters",
    "anomaly_threshold",
    "backend_name",
    "observed_counts",
    "description",
}


@dataclass
class ExperimentConfig:
    experiment_name: str
    num_qubits: int
    shots: int
    circuit: list[Any]
    random_seed: int
    validation_label: str = "SYNTHETIC_VALIDATION"
    observed_circuit: list[Any] | None = None
    hardware_parameters: dict[str, Any] | None = None
    noise_parameters: dict[str, Any] | None = None
    anomaly_threshold: float = 0.05
    backend_name: str | None = None
    observed_counts: dict[str, int] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise ValueError("experiment_name must be a non-empty string")
        if not isinstance(self.validation_label, str) or self.validation_label != "SYNTHETIC_VALIDATION":
            raise ValueError("local experiment configurations must be labeled SYNTHETIC_VALIDATION")
        if not isinstance(self.num_qubits, int) or isinstance(self.num_qubits, bool) or self.num_qubits <= 0:
            raise ValueError("num_qubits must be a positive integer")
        if not isinstance(self.shots, int) or isinstance(self.shots, bool) or self.shots <= 0:
            raise ValueError("shots must be a positive integer")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool) or self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        if not isinstance(self.circuit, list):
            raise ValueError("circuit must be a list of universal circuit operations")
        if self.observed_circuit is not None and not isinstance(self.observed_circuit, list):
            raise ValueError("observed_circuit must be a list when provided")
        if not isinstance(self.anomaly_threshold, (int, float)) or isinstance(self.anomaly_threshold, bool):
            raise ValueError("anomaly_threshold must be numeric")
        if not 0.0 <= float(self.anomaly_threshold):
            raise ValueError("anomaly_threshold must be non-negative")
        self.anomaly_threshold = float(self.anomaly_threshold)
        if self.backend_name is not None and not isinstance(self.backend_name, str):
            raise ValueError("backend_name must be a string or null")
        if self.hardware_parameters is not None and not isinstance(self.hardware_parameters, dict):
            raise ValueError("hardware_parameters must be an object or null")
        if self.noise_parameters is not None and not isinstance(self.noise_parameters, dict):
            raise ValueError("noise_parameters must be an object or null")
        if self.observed_counts is not None:
            if not isinstance(self.observed_counts, dict) or not self.observed_counts:
                raise ValueError("observed_counts must be a non-empty object when provided")
            if any(not isinstance(key, str) for key in self.observed_counts):
                raise ValueError("observed_counts keys must be strings")
            if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in self.observed_counts.values()):
                raise ValueError("observed_counts values must be non-negative integers")
            if sum(self.observed_counts.values()) != self.shots:
                raise ValueError("observed_counts must sum to shots")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        if not isinstance(value, Mapping):
            raise ValueError("experiment configuration must be a JSON object")
        missing = sorted(REQUIRED_FIELDS - set(value))
        if missing:
            raise ValueError(f"missing required experiment configuration field(s): {', '.join(missing)}")
        unknown = sorted(set(value) - ALLOWED_FIELDS)
        if unknown:
            raise ValueError(f"unknown experiment configuration field(s): {', '.join(unknown)}")
        return cls(**dict(value))

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not load experiment configuration {source}: {exc}") from exc
        return cls.from_mapping(value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.from_json(path)


__all__ = ["ExperimentConfig", "load_experiment_config"]
