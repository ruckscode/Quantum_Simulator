import pytest
import numpy as np

from quantum_simulator.statistics import DistributionComparisonResult, compare_counts, compare_distributions


def test_identical_distributions_are_not_anomalous():
    result = compare_distributions({"00": 0.5, "11": 0.5}, {"00": 0.5, "11": 0.5})
    assert result.anomaly_detected is False
    assert result.total_variation_distance == 0.0


def test_small_statistical_variation_is_not_anomalous():
    expected = {"0": 500, "1": 500}
    observed = {"0": 510, "1": 490}
    result = compare_counts(expected, observed, threshold=0.05)
    assert result.anomaly_detected is False


def test_clearly_different_distributions_are_anomalous():
    expected = {"0": 500, "1": 500}
    observed = {"0": 900, "1": 100}
    result = compare_counts(expected, observed, threshold=0.05)
    assert result.anomaly_detected is True


def test_bell_state_distribution():
    expected = {"00": 0.5, "11": 0.5}
    observed = {"00": 0.47, "11": 0.53}
    result = compare_distributions(expected, observed, threshold=0.1)
    assert result.anomaly_detected is False


def test_different_shot_counts():
    expected = {"0": 100, "1": 100}
    observed = {"0": 120, "1": 80}
    result = compare_counts(expected, observed, shots=200)
    assert result.shots == 200
    assert hasattr(result, "total_variation_distance")


def test_configurable_threshold_behavior():
    expected = {"0": 500, "1": 500}
    observed = {"0": 550, "1": 450}
    weak = compare_counts(expected, observed, threshold=0.01)
    strong = compare_counts(expected, observed, threshold=0.2)
    assert weak.anomaly_detected is True
    assert strong.anomaly_detected is False


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        compare_counts({"0": -1, "1": 2}, {"0": 1, "1": 1})
    with pytest.raises(ValueError):
        compare_counts({"0": 10}, {"0": 5, "1": 5})
    with pytest.raises(ValueError):
        compare_counts({"0": 10, "1": 10}, {"0": 5, "1": 5}, shots=0)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), -1.0])
def test_nonfinite_or_negative_mapping_and_sequence_values_are_rejected(invalid):
    with pytest.raises(ValueError):
        compare_distributions({"0": 1.0, "1": 0.0}, {"0": invalid, "1": 1.0})
    with pytest.raises(ValueError):
        compare_distributions([1.0, 0.0], [invalid, 1.0])


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, True])
def test_invalid_thresholds_are_rejected(invalid):
    with pytest.raises((TypeError, ValueError)):
        compare_distributions([1, 0], [1, 0], threshold=invalid)


def test_distribution_normalization_is_stable_for_large_finite_values():
    result = compare_distributions([1e308, 1e308], [1e308, 1e308])
    assert result.anomaly_detected is False
    assert result.total_variation_distance == 0.0


def test_colliding_stringified_outcome_keys_and_multidimensional_arrays_fail_closed():
    with pytest.raises(ValueError, match="keys must remain unique"):
        compare_distributions({0: 1.0, "0": 1.0}, {"0": 1.0})
    with pytest.raises(ValueError, match="one-dimensional"):
        compare_distributions(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]))
