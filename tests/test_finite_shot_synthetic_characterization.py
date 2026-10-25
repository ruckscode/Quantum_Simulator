import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.finite_shot_synthetic_characterization import (
    MAX_PAIRS, distribution_for_delta, interval, one_sided_upper, pair_outcome, planned_pair_count, tvd,
)


def test_pair_cap_is_exact_and_not_exceeded():
    assert planned_pair_count() == 12000
    assert planned_pair_count() <= MAX_PAIRS


def test_alternative_distribution_has_registered_tvd():
    assert distribution_for_delta(0.1) == (0.4, 0.35, 0.15, 0.1)
    assert tvd((0.5, 0.25, 0.15, 0.1), distribution_for_delta(0.1)) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        distribution_for_delta(0.075)


def test_exact_intervals_boundary_and_reference_values():
    lo, hi = interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.3084971078187608, abs=1e-12)
    lo, hi = interval(10, 10)
    assert lo == pytest.approx(0.6915028921812392, abs=1e-12) and hi == 1.0
    assert one_sided_upper(0, 2000) == pytest.approx(0.0014967448951883062, abs=1e-12)
    assert one_sided_upper(250, 500) == pytest.approx(0.538, abs=0.002)


def test_seeded_pair_is_reproducible():
    p = (0.5, 0.25, 0.15, 0.1)
    a = pair_outcome(256, "null_equal", 8, p, p)
    b = pair_outcome(256, "null_equal", 8, p, p)
    assert a is b
