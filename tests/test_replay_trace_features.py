import math

import pytest

from replay_trace_features import _extract_features


def test_replay_features_are_deterministic_and_match_documented_statistics():
    trace = [
        {"probability_difference": 0.01},
        {"probability_difference": 0.05},
        {"probability_difference": 0.8},
    ]

    first = _extract_features(trace)
    second = _extract_features(trace)

    assert first == second
    assert first == {
        "trace_max_probability_difference": 0.8,
        "trace_mean_probability_difference": pytest.approx(0.86 / 3),
        "trace_median_probability_difference": 0.05,
        "trace_deviation_concentration": pytest.approx(0.8 / 0.86),
        "trace_high_deviation_fraction": pytest.approx(1 / 3),
    }


def test_probability_difference_clamps_only_floating_point_roundoff():
    result = _extract_features([{"probability_difference": 1.0 + 5e-13}])

    assert result["trace_max_probability_difference"] == 1.0


@pytest.mark.parametrize(
    "trace, message",
    [
        ([], "at least one operation row"),
        ([{}], "numeric probability_difference"),
        ([{"probability_difference": "bad"}], "numeric probability_difference"),
        ([{"probability_difference": True}], "numeric probability_difference"),
        ([{"probability_difference": math.nan}], "finite and between 0 and 1"),
        ([{"probability_difference": math.inf}], "finite and between 0 and 1"),
        ([{"probability_difference": -0.1}], "finite and between 0 and 1"),
        ([{"probability_difference": 1.1}], "finite and between 0 and 1"),
    ],
)
def test_malformed_or_out_of_range_trace_fails_transparently(trace, message):
    with pytest.raises(ValueError, match=message):
        _extract_features(trace)
