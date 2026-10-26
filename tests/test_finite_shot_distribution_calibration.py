"""Bounded independent calibration of finite-shot distribution comparisons."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from quantum_simulator.diagnosis_engine import diagnose_circuit


FIXTURE = Path(__file__).parent / "fixtures" / "finite_shot_distribution_calibration.json"


def _tvd(first, second):
    return 0.5 * sum(abs(float(a) - float(b)) for a, b in zip(first, second))


def _permutation_p_value(first_counts, second_counts, repetitions, seed):
    """Monte Carlo two-sample TVD permutation test for categorical counts.

    Under the null, the pooled outcome labels are exchangeable between the two
    independently sampled groups. Permuting labels therefore conditions on
    the pooled category counts and accounts for the observed outcome support.
    """
    first = np.asarray(first_counts, dtype=np.int64)
    second = np.asarray(second_counts, dtype=np.int64)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("count vectors must be one-dimensional and have equal length")
    if np.any(first < 0) or np.any(second < 0):
        raise ValueError("count vectors must be non-negative")
    n_first, n_second = int(first.sum()), int(second.sum())
    if n_first <= 0 or n_second <= 0:
        raise ValueError("each count vector must have a positive total")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    pooled = np.repeat(np.arange(len(first)), first + second)
    observed = _tvd(first / n_first, second / n_second)
    rng = np.random.default_rng(seed)
    at_least_observed = 0
    for _ in range(repetitions):
        permuted = rng.permutation(pooled)
        perm_first = np.bincount(permuted[:n_first], minlength=len(first))
        perm_second = np.bincount(permuted[n_first:], minlength=len(first))
        if _tvd(perm_first / n_first, perm_second / n_second) >= observed - 1e-12:
            at_least_observed += 1
    return (at_least_observed + 1) / (repetitions + 1)


def _counts(distribution, support, shots):
    return np.asarray([round(distribution.get(key, 0.0) * shots) for key in support], dtype=np.int64)


def evaluate_calibration():
    spec = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rows = []
    for case in spec["cases"]:
        for shots in spec["shot_counts"]:
            for seed in spec["seeds"]:
                sample_seed = seed + spec["shot_seed_offsets"][str(shots)]
                # Labels and scenario metadata are deliberately not passed to diagnosis.
                result = diagnose_circuit(
                    case["submitted"], reference_circuit=case["reference"],
                    shots=shots, seed=sample_seed, threshold=0.05,
                )
                stats = result.statistics_result
                # Use the actual outcome labels returned by the simulator. A
                # hard-coded four-bit support can silently produce all-zero
                # vectors when a backend uses a different label representation.
                support = sorted(set(stats.expected_distribution) | set(stats.observed_distribution))
                first = _counts(stats.expected_distribution, support, shots)
                second = _counts(stats.observed_distribution, support, shots)
                p_value = _permutation_p_value(
                    first, second, spec["permutation_repetitions"],
                    seed=sample_seed + sum(case["id"].encode("utf-8")),
                )
                rows.append({
                    "case_id": case["id"], "scenario": case["scenario"],
                    "shots": shots, "seed": sample_seed,
                    "expected_anomaly": case["expected_anomaly"],
                    "tvd": stats.total_variation_distance,
                    "guard": stats.finite_shot_guard,
                    "current_detected": stats.anomaly_detected,
                    "permutation_p": p_value,
                    "permutation_detected": p_value <= spec["permutation_alpha"],
                })
    return spec, rows


def test_independent_calibration_fixture_and_permutation_comparison():
    spec, rows = evaluate_calibration()
    assert spec["predeclared_before_evaluation"] is True
    assert len(rows) == 20
    assert {row["scenario"] for row in rows} == {
        "identical_control", "ambiguous_observable_equivalence",
        "observable_fault_subtle", "observable_fault_moderate", "observable_fault_large",
    }
    for case in spec["cases"]:
        if "expected_ideal_tvd" in case:
            theta = case["submitted"][0][2]
            ideal_tvd = abs(math.sin(theta / 2) ** 2 - 0.5)
            assert round(ideal_tvd, 4) == case["expected_ideal_tvd"]
    for method in ("current_detected", "permutation_detected"):
        positives = [row for row in rows if row["expected_anomaly"]]
        negatives = [row for row in rows if not row["expected_anomaly"]]
        tp = sum(row[method] for row in positives)
        fp = sum(row[method] for row in negatives)
        print(
            f"{method}: sensitivity={tp}/{len(positives)} "
            f"({tp / len(positives):.3f}), FPR={fp}/{len(negatives)} "
            f"({fp / len(negatives):.3f})"
        )
    for scenario in (
        "identical_control", "ambiguous_observable_equivalence",
        "observable_fault_subtle", "observable_fault_moderate", "observable_fault_large",
    ):
        subset = [row for row in rows if row["scenario"] == scenario]
        print(scenario, [(row["shots"], row["seed"], round(row["tvd"], 4),
                         row["current_detected"], round(row["permutation_p"], 4))
                        for row in subset])


@pytest.mark.parametrize(
    "first, second",
    [
        ([0, 0], [1, 0]),
        ([1, 0], [0, 0]),
        ([0, 0], [0, 0]),
    ],
)
def test_permutation_test_rejects_empty_count_vectors(first, second):
    with pytest.raises(ValueError, match="positive total"):
        _permutation_p_value(first, second, 99, seed=17)


def test_permutation_test_rejects_negative_counts():
    with pytest.raises(ValueError, match="non-negative"):
        _permutation_p_value([2, -1], [1, 1], 99, seed=17)


def test_permutation_test_is_dimension_aware_and_seeded():
    first = [40, 40, 40, 40]
    same = [40, 40, 40, 40]
    shifted = [70, 30, 40, 20]
    p_same = _permutation_p_value(first, same, 499, seed=17)
    p_shifted = _permutation_p_value(first, shifted, 499, seed=17)
    assert p_same == _permutation_p_value(first, same, 499, seed=17)
    assert p_shifted < p_same


def test_proposed_calibration_plan_is_preregistered_and_bounded():
    spec = json.loads(FIXTURE.read_text(encoding="utf-8"))
    plan = spec["proposed_next_calibration_plan"]
    assert plan["status"] == "preregistered_not_executed"
    assert plan["trials_per_stratum_per_shot_per_split"] == 120
    assert plan["false_positive_targets"]["maximum_false_positives_at_n"] == 1
    assert "999 Monte Carlo resamples" in plan["candidate_test"]
    assert plan["minimum_detected_at_n_120"] == {
        "0.50": 70, "0.80": 104, "0.90": 114, "0.95": 119,
    }
    assert plan["workload"]["shot_circuit_executions"] == (
        2 * 5 * 120 * 2 * sum(plan["shot_counts"])
    )
    assert plan["workload"]["diagnosis_configurations"] == (
        2 * 5 * 120 * len(plan["shot_counts"])
    )
    pilot = plan["workload"]["runtime_pilot"]
    assert pilot["diagnosis_configurations"] == pilot["scenarios"] * len(pilot["shot_counts"])
    assert pilot["shot_circuit_executions"] == (
        pilot["scenarios"] * 2 * sum(pilot["shot_counts"])
    )
    assert (
        plan["design"]["template_generation_seed_roots"]["calibration"]
        != plan["design"]["template_generation_seed_roots"]["untouched_validation"]
    )
    assert (
        plan["design"]["sampling_seed_roots"]["calibration"]
        != plan["design"]["sampling_seed_roots"]["untouched_validation"]
    )
