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


__all__ = ["CountSampleComparisonResult", "compare_count_samples"]
