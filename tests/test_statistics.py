import pytest

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


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_distribution_mapping_rejects_non_finite_values(bad_value):
    with pytest.raises(ValueError):
        compare_distributions({"0": bad_value, "1": 1.0}, {"0": 0.5, "1": 0.5})


@pytest.mark.parametrize("bad_value", [-0.1, float("nan"), float("inf"), float("-inf")])
def test_distribution_sequences_reject_negative_or_non_finite_values(bad_value):
    with pytest.raises(ValueError):
        compare_distributions([bad_value, 1.0], [0.5, 0.5])


@pytest.mark.parametrize("bad_threshold", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_threshold_is_rejected(bad_threshold):
    with pytest.raises(ValueError):
        compare_distributions([0.5, 0.5], [0.5, 0.5], threshold=bad_threshold)
