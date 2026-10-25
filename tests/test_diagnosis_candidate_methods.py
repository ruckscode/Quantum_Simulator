"""Small contract tests for the development-only TVD permutation candidate."""

import pytest

from experiments.diagnosis_candidate_methods import permutation_tvd_p_value


def test_permutation_candidate_is_seed_reproducible_and_not_a_fault_probability():
    reference = [8, 2, 0, 0]
    submitted = [7, 3, 0, 0]
    first = permutation_tvd_p_value(reference, submitted, repetitions=31, seed=41)
    second = permutation_tvd_p_value(reference, submitted, repetitions=31, seed=41)

    assert first == second
    assert 1 / 32 <= first <= 1


def test_identical_count_vectors_return_non_significant_p_value():
    p_value = permutation_tvd_p_value([4, 6, 2], [4, 6, 2], repetitions=31, seed=7)
    assert p_value == 1.0


@pytest.mark.parametrize(
    "first,second,message",
    [
        ([1, -1], [1, 1], "nonnegative"),
        ([1, 2], [1, 2, 3], "same number of categories"),
        ([0, 0], [1, 1], "at least one shot"),
        ([1], [1], "at least two categories"),
    ],
)
def test_invalid_count_inputs_fail_clearly(first, second, message):
    with pytest.raises((TypeError, ValueError), match=message):
        permutation_tvd_p_value(first, second, repetitions=9, seed=3)


def test_invalid_permutation_controls_fail_clearly():
    with pytest.raises(ValueError, match="positive integer"):
        permutation_tvd_p_value([1, 1], [1, 1], repetitions=0, seed=3)
    with pytest.raises(TypeError, match="seed must be an integer"):
        permutation_tvd_p_value([1, 1], [1, 1], repetitions=9, seed=True)
