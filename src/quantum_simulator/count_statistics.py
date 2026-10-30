"""Count-aware statistical tests for independently sampled outcomes.

Kept separate from the legacy probability-only helper so count-aware callers
can use a calibrated sampling model without changing its frozen behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from .statistics import DistributionComparisonResult, compare_distributions


@dataclass
class CountSampleComparisonResult(DistributionComparisonResult):
    p_value: float = 1.0
    method: str = "conditional_tvd_permutation"


def compare_count_samples(
    expected_counts: Mapping[str, int] | Sequence[int],
    observed_counts: Mapping[str, int] | Sequence[int],
    *,
    threshold: float = 0.05,
    alpha: float = 0.05,
    permutations: int = 999,
    permutation_seed: int = 0,
) -> DistributionComparisonResult:
    """Compare independent categorical count samples by conditional TVD test.

    Under equality of outcome probabilities, pooled categorical observations
    are exchangeable between the samples. The test conditions on pooled counts
    and both sample sizes; `threshold` is a separate practical-effect gate.
    The permutation seed is a fixed, predeclared input for reproducibility.
    """
    for name, value in (("alpha", alpha), ("threshold", threshold)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"{name} must be numeric")
        if not isfinite(float(value)) or float(value) < 0 or (name == "alpha" and float(value) > 1):
            raise ValueError(f"{name} is outside its valid range")
    if isinstance(permutations, (bool, np.bool_)) or not isinstance(permutations, (int, np.integer)) or permutations <= 0:
        raise ValueError("permutations must be a positive integer")
    if isinstance(permutation_seed, (bool, np.bool_)) or not isinstance(permutation_seed, (int, np.integer)):
        raise TypeError("permutation_seed must be an integer")

    def vector(counts, label):
        if isinstance(counts, Mapping):
            keys = [str(key) for key in counts]
            if len(set(keys)) != len(keys):
                raise ValueError(f"{label} keys must remain unique when converted to outcome strings")
            raw, labels = list(counts.values()), keys
        elif isinstance(counts, (list, tuple, np.ndarray)) and not (
            isinstance(counts, np.ndarray) and counts.ndim != 1
        ):
            raw = list(counts)
            labels = [str(index) for index in range(len(raw))]
        else:
            raise TypeError(f"{label} must be a mapping or one-dimensional sequence")
        values = []
        for value in raw:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{label} must contain non-negative integer counts")
            values.append(int(value))
        if not values or sum(values) == 0:
            raise ValueError(f"{label} must have positive total count")
        return dict(zip(labels, values))

    first = vector(expected_counts, "expected_counts")
    second = vector(observed_counts, "observed_counts")
    keys = sorted(set(first) | set(second))
    a = np.asarray([first.get(key, 0) for key in keys], dtype=np.int64)
    b = np.asarray([second.get(key, 0) for key in keys], dtype=np.int64)
    n_first, n_second = int(a.sum()), int(b.sum())
    tvd = 0.5 * np.abs(a / n_first - b / n_second).sum()

    pooled = np.repeat(np.arange(len(keys)), a + b)
    rng = np.random.default_rng(int(permutation_seed))
    extreme = 0
    for _ in range(int(permutations)):
        shuffled = rng.permutation(pooled)
        first_permuted = np.bincount(shuffled[:n_first], minlength=len(keys))
        second_permuted = np.bincount(shuffled[n_first:], minlength=len(keys))
        permuted_tvd = 0.5 * np.abs(first_permuted / n_first - second_permuted / n_second).sum()
        extreme += permuted_tvd >= tvd - 1e-12
    p_value = (extreme + 1) / (int(permutations) + 1)

    base = compare_distributions(a, b, threshold=float(threshold), shots=max(n_first, n_second))
    detected = bool(p_value <= float(alpha) and tvd > float(threshold))
    reason = (
        "Permutation test rejects equal distributions and observed TVD exceeds effect threshold"
        if detected else
        "No statistically significant distribution change above the effect threshold"
    )
    message = (
        f"Observed TVD={tvd:.6f}, permutation p={p_value:.4f}, "
        f"effect threshold={float(threshold):.6f}."
    )
    return CountSampleComparisonResult(
        expected_distribution=base.expected_distribution,
        observed_distribution=base.observed_distribution,
        total_variation_distance=float(tvd),
        hellinger_distance=base.hellinger_distance,
        js_divergence=base.js_divergence,
        threshold=float(threshold),
        finite_shot_guard=0.0,
        shots=max(n_first, n_second),
        anomaly_detected=detected,
        reason=reason,
        message=message,
        p_value=float(p_value),
    )


def compare_counts_to_distribution(
    expected_distribution: Mapping[str, float],
    observed_counts: Mapping[str, int],
    *,
    threshold: float = 0.05,
    alpha: float = 0.05,
    simulations: int = 1999,
    seed: int = 0,
) -> CountSampleComparisonResult:
    """Test one observed count sample against an exact expected distribution.

    A seeded multinomial parametric bootstrap estimates the null distribution
    of TVD. Statistical significance and the practical effect threshold are
    separate gates, avoiding the conservative 1/sqrt(shots) rule when the
    reference probabilities are exact rather than another finite-shot sample.
    """
    if not isinstance(expected_distribution, Mapping) or not expected_distribution:
        raise TypeError("expected_distribution must be a non-empty probability mapping")
    if not isinstance(observed_counts, Mapping) or not observed_counts:
        raise TypeError("observed_counts must be a non-empty count mapping")
    for name, value in (("alpha", alpha), ("threshold", threshold)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"{name} must be numeric")
        if not isfinite(float(value)) or float(value) < 0 or (name == "alpha" and float(value) > 1):
            raise ValueError(f"{name} is outside its valid range")
    if isinstance(simulations, (bool, np.bool_)) or not isinstance(simulations, (int, np.integer)) or simulations <= 0:
        raise ValueError("simulations must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")

    def normalized_probabilities(values: Mapping[str, float]) -> dict[str, float]:
        result: dict[str, float] = {}
        for raw_key, raw_value in values.items():
            key = str(raw_key)
            if key in result:
                raise ValueError("outcome keys must be unique after string conversion")
            if isinstance(raw_value, (bool, np.bool_)) or not isinstance(raw_value, (int, float, np.integer, np.floating)):
                raise TypeError("expected probabilities must be numeric")
            value = float(raw_value)
            if not isfinite(value) or value < 0:
                raise ValueError("expected probabilities must be finite and non-negative")
            result[key] = value
        total = sum(result.values())
        if total <= 0:
            raise ValueError("expected probabilities must have positive total mass")
        return {key: value / total for key, value in result.items()}

    probabilities = normalized_probabilities(expected_distribution)
    counts: dict[str, int] = {}
    for raw_key, raw_value in observed_counts.items():
        key = str(raw_key)
        if key in counts:
            raise ValueError("outcome keys must be unique after string conversion")
        if isinstance(raw_value, (bool, np.bool_)) or not isinstance(raw_value, (int, np.integer)) or raw_value < 0:
            raise ValueError("observed_counts must contain non-negative integer values")
        counts[key] = int(raw_value)
    shots = sum(counts.values())
    if shots <= 0:
        raise ValueError("observed_counts must contain positive total shots")

    keys = sorted(set(probabilities) | set(counts))
    p = np.asarray([probabilities.get(key, 0.0) for key in keys], dtype=float)
    observed = np.asarray([counts.get(key, 0) for key in keys], dtype=np.int64)
    observed_probability = observed / shots
    tvd = float(0.5 * np.abs(observed_probability - p).sum())
    rng = np.random.default_rng(int(seed))
    extreme = 0
    for _ in range(int(simulations)):
        simulated = rng.multinomial(shots, p)
        simulated_tvd = 0.5 * np.abs(simulated / shots - p).sum()
        extreme += simulated_tvd >= tvd - 1e-12
    p_value = (extreme + 1) / (int(simulations) + 1)
    detected = bool(p_value <= float(alpha) and tvd > float(threshold))
    observed_mapping = {key: float(value) for key, value in zip(keys, observed_probability)}
    expected_mapping = {key: float(value) for key, value in zip(keys, p)}
    base = compare_distributions(expected_mapping, observed_mapping, threshold=float(threshold))
    base.threshold = float(threshold)
    base.finite_shot_guard = 0.0
    base.shots = shots
    base.anomaly_detected = detected
    base.reason = (
        "Multinomial goodness-of-fit rejects the exact reference and TVD exceeds the practical-effect threshold"
        if detected else
        "No significant distribution change above the practical-effect threshold"
    )
    base.message = (
        f"Observed TVD={tvd:.6f}, multinomial-bootstrap p={p_value:.4f}, "
        f"effect threshold={float(threshold):.6f}, shots={shots}."
    )
    return CountSampleComparisonResult(
        **{name: getattr(base, name) for name in DistributionComparisonResult.__dataclass_fields__},
        p_value=float(p_value),
        method="multinomial_parametric_bootstrap",
    )


__all__ = ["CountSampleComparisonResult", "compare_count_samples", "compare_counts_to_distribution"]
