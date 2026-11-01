"""Statistical anomaly detection for finite-shot quantum measurements.

This module uses Total Variation Distance as the primary comparison metric and
adds a heuristic finite-shot guard for probability-only inputs. The guard is
not a p-value, confidence interval, or universal statistical guarantee. When
raw independent count samples are available, callers can use
``count_statistics.compare_count_samples`` for a conditional permutation test.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Mapping, Sequence

import numpy as np


@dataclass
class DistributionComparisonResult:
    expected_distribution: dict[str, float]
    observed_distribution: dict[str, float]
    total_variation_distance: float
    hellinger_distance: float
    js_divergence: float
    threshold: float
    finite_shot_guard: float
    shots: int
    anomaly_detected: bool
    reason: str
    message: str


def _normalize_nonnegative_values(values, name: str) -> list[float]:
    normalized = []
    for key, value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"{name} values must be numeric; got {type(value).__name__} for key {key!r}")
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError(f"{name} values must be finite; got {numeric} for key {key!r}")
        if numeric < 0:
            raise ValueError(f"{name} values must be non-negative; got {numeric} for key {key!r}")
        normalized.append(numeric)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    scale = max(normalized)
    if scale <= 0:
        raise ValueError(f"{name} must contain a positive total mass")
    scaled = [value / scale for value in normalized]
    total = sum(scaled)
    return [value / total for value in scaled]


def _validate_nonnegative_mapping(mapping: Mapping[str, float], name: str) -> dict[str, float]:
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{name} must be a mapping of outcome -> numeric value")
    keys = [str(key) for key in mapping]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} keys must remain unique when converted to outcome strings")
    values = _normalize_nonnegative_values(zip(keys, mapping.values()), name)
    return dict(zip(keys, values))


def _require_same_keys(expected: Mapping[str, float], observed: Mapping[str, float]) -> None:
    if set(expected.keys()) != set(observed.keys()):
        raise ValueError("Expected and observed distributions must have identical outcome keys")


def _as_probability_dict(raw: Mapping[str, float] | Sequence[float], label: str) -> dict[str, float]:
    if isinstance(raw, Mapping):
        return _validate_nonnegative_mapping(raw, label)
    if isinstance(raw, (list, tuple, np.ndarray)):
        if isinstance(raw, np.ndarray) and raw.ndim != 1:
            raise ValueError(f"{label} sequence must be one-dimensional")
        values = _normalize_nonnegative_values(enumerate(raw), label)
        return {str(i): value for i, value in enumerate(values)}
    raise TypeError(f"{label} must be a mapping or sequence of numeric values")


def _total_variation_distance(expected: dict[str, float], observed: dict[str, float]) -> float:
    return 0.5 * sum(abs(expected[key] - observed[key]) for key in expected)


def _hellinger_distance(expected: dict[str, float], observed: dict[str, float]) -> float:
    return sqrt(0.5 * sum((sqrt(expected[key]) - sqrt(observed[key])) ** 2 for key in expected))


def _js_divergence(expected: dict[str, float], observed: dict[str, float]) -> float:
    keys = sorted(set(expected) | set(observed))
    midpoint = {key: 0.5 * (expected.get(key, 0.0) + observed.get(key, 0.0)) for key in keys}

    def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
        total = 0.0
        for key in keys:
            p_i = p.get(key, 0.0)
            q_i = q.get(key, 0.0)
            if p_i > 0 and q_i > 0:
                total += p_i * np.log(p_i / q_i)
        return float(total)

    return 0.5 * (kl_divergence(expected, midpoint) + kl_divergence(observed, midpoint))


def _finite_shot_guard(shots: int | None, threshold: float) -> float:
    if shots is None or shots <= 0:
        return 0.0
    return max(float(threshold), 1.0 / sqrt(shots))


def compare_distributions(
    expected: Mapping[str, float] | Sequence[float],
    observed: Mapping[str, float] | Sequence[float],
    *,
    threshold: float = 0.05,
    shots: int | None = None,
) -> DistributionComparisonResult:
    """Compare two finite-sample distributions.

    We use Total Variation Distance as the primary metric because it is a clear,
    interpretable distance in probability space and is natural for finite-shot
    quantum measurements. For probability-only inputs, this helper applies the
    heuristic guard ``max(threshold, 1/sqrt(shots))``. This is not a significance
    test or calibrated confidence bound. For independent raw count samples, use
    ``count_statistics.compare_count_samples`` instead.
    """

    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise TypeError("threshold must be a numeric value")
    threshold = float(threshold)
    if not isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")

    if shots is not None:
        if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)):
            raise TypeError("shots must be an integer or None")
        if shots <= 0:
            raise ValueError("shots must be positive")

    expected_prob = _as_probability_dict(expected, "expected")
    observed_prob = _as_probability_dict(observed, "observed")
    _require_same_keys(expected_prob, observed_prob)

    keys = sorted(expected_prob)
    expected_arr = {key: expected_prob[key] for key in keys}
    observed_arr = {key: observed_prob[key] for key in keys}

    tvd = _total_variation_distance(expected_arr, observed_arr)
    hellinger = _hellinger_distance(expected_arr, observed_arr)
    jsd = _js_divergence(expected_arr, observed_arr)
    finite_guard = _finite_shot_guard(shots, threshold)
    effective_threshold = max(threshold, finite_guard)
    anomaly_detected = tvd > effective_threshold

    if anomaly_detected:
        reason = "Distance exceeds configured threshold and finite-shot tolerance"
        message = (
            f"Observed distribution differs from expected distribution by TVD={tvd:.6f}, "
            f"which exceeds threshold={effective_threshold:.6f}."
        )
    else:
        reason = "No statistically meaningful anomaly detected"
        message = (
            f"Observed distribution differs by TVD={tvd:.6f}, which is within the configured "
            f"threshold={effective_threshold:.6f}."
        )

    return DistributionComparisonResult(
        expected_distribution=expected_arr,
        observed_distribution=observed_arr,
        total_variation_distance=tvd,
        hellinger_distance=hellinger,
        js_divergence=jsd,
        threshold=threshold,
        finite_shot_guard=finite_guard,
        shots=shots if shots is not None else int(round(1.0 / max(tvd, 1e-12))),
        anomaly_detected=anomaly_detected,
        reason=reason,
        message=message,
    )


def compare_counts(
    expected_counts: Mapping[str, int] | Sequence[int],
    observed_counts: Mapping[str, int] | Sequence[int],
    *,
    threshold: float = 0.05,
    shots: int | None = None,
) -> DistributionComparisonResult:
    """Compare two count dictionaries or sequences by normalizing them to distributions."""

    if shots is None:
        shots = int(sum(float(v) for v in _as_probability_dict(expected_counts, "expected_counts").values()) * 0) if False else None

    expected_prob = _as_probability_dict(expected_counts, "expected_counts")
    observed_prob = _as_probability_dict(observed_counts, "observed_counts")
    _require_same_keys(expected_prob, observed_prob)

    if shots is None:
        expected_total = sum(float(v) for v in expected_counts.values()) if isinstance(expected_counts, Mapping) else sum(float(v) for v in expected_counts)
        observed_total = sum(float(v) for v in observed_counts.values()) if isinstance(observed_counts, Mapping) else sum(float(v) for v in observed_counts)
        shots = max(int(round(max(expected_total, observed_total))), 1)

    if shots <= 0:
        raise ValueError("shots must be positive")

    # Explicit negative-count validation before normalization.
    if isinstance(expected_counts, Mapping):
        if any(float(v) < 0 for v in expected_counts.values()):
            raise ValueError("Expected counts must be non-negative")
    elif isinstance(expected_counts, (list, tuple, np.ndarray)):
        if any(float(v) < 0 for v in expected_counts):
            raise ValueError("Expected counts must be non-negative")

    if isinstance(observed_counts, Mapping):
        if any(float(v) < 0 for v in observed_counts.values()):
            raise ValueError("Observed counts must be non-negative")
    elif isinstance(observed_counts, (list, tuple, np.ndarray)):
        if any(float(v) < 0 for v in observed_counts):
            raise ValueError("Observed counts must be non-negative")

    return compare_distributions(expected_prob, observed_prob, threshold=threshold, shots=shots)


__all__ = [
    "DistributionComparisonResult",
    "compare_distributions",
    "compare_counts",
]
