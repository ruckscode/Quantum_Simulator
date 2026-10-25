"""Development-only statistical candidates for circuit diagnosis.

These helpers are not wired into the simulator's default diagnosis path.  The
two-sample permutation test below returns a statistical p-value, not a fault
probability.  Its null interpretation requires independent, exchangeable shot
outcomes from the two executions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def permutation_tvd_p_value(
    reference_counts: Sequence[int],
    submitted_counts: Sequence[int],
    *,
    repetitions: int,
    seed: int,
) -> float:
    """Return a seeded Monte Carlo permutation p-value for two count vectors.

    The pooled per-shot category labels are permuted between the two samples,
    conditioning on pooled category counts and the observed sample sizes.  The
    add-one correction prevents a reported zero Monte Carlo p-value.  This is
    an experimental detector component; it neither attributes nor localizes
    a cause and makes no calibrated-probability claim.
    """

    first = _validated_counts(reference_counts, "reference_counts")
    second = _validated_counts(submitted_counts, "submitted_counts")
    if first.shape != second.shape:
        raise ValueError("count vectors must have the same number of categories")
    if repetitions <= 0 or isinstance(repetitions, bool):
        raise ValueError("repetitions must be a positive integer")
    if not isinstance(repetitions, int):
        raise TypeError("repetitions must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")

    n_first, n_second = int(first.sum()), int(second.sum())
    if n_first == 0 or n_second == 0:
        raise ValueError("each count vector must contain at least one shot")

    pooled = np.repeat(np.arange(first.size), first + second)
    observed = _tvd(first / n_first, second / n_second)
    rng = np.random.default_rng(seed)
    at_least_observed = 0
    for _ in range(repetitions):
        permuted = rng.permutation(pooled)
        perm_first = np.bincount(permuted[:n_first], minlength=first.size)
        perm_second = np.bincount(permuted[n_first:], minlength=first.size)
        permuted_tvd = _tvd(perm_first / n_first, perm_second / n_second)
        if permuted_tvd >= observed - 1e-12:
            at_least_observed += 1
    return (at_least_observed + 1) / (repetitions + 1)


def _validated_counts(values: Sequence[int], name: str) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a one-dimensional integer count vector") from error
    if raw.ndim != 1 or raw.size < 2:
        raise ValueError(f"{name} must contain at least two categories")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise TypeError(f"{name} must contain nonnegative integers")
    if np.any(raw < 0):
        raise ValueError(f"{name} must contain nonnegative integers")
    return raw.astype(np.int64, copy=False)


def _tvd(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.abs(first - second).sum())
