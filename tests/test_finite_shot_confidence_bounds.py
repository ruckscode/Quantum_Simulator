"""Focused edge tests for the calibration binomial confidence bounds."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.finite_shot_calibration_plan import _cp_lower


def test_lower_bound_zero_successes_is_zero():
    assert _cp_lower(0, 120) == 0.0


def test_lower_bound_all_successes():
    assert _cp_lower(120, 120) == pytest.approx(0.9753445989443437, abs=1e-12)


def test_lower_bound_intermediate_counts():
    assert _cp_lower(70, 120) == pytest.approx(0.504201076877167, abs=1e-12)
    assert _cp_lower(104, 120) == pytest.approx(0.8045561538017076, abs=1e-12)
    assert _cp_lower(114, 120) == pytest.approx(0.903703682623062, abs=1e-12)
    assert _cp_lower(119, 120) == pytest.approx(0.9610791531398315, abs=1e-12)

def test_lower_bound_boundaries_and_monotonicity():
    assert _cp_lower(0, 0) is None
    assert _cp_lower(0, 1) == 0.0
    assert _cp_lower(1, 1) == pytest.approx(0.05, abs=1e-12)
    assert _cp_lower(1, 2) < _cp_lower(2, 2)
    assert _cp_lower(120, 120) <= 1.0
