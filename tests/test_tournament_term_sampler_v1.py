"""Coefficient-sampling invariants for a bounded-work tournament pilot."""

import math

import pytest

from tournament_term_sampler_v1 import sample_coefficients


def test_small_group_is_exact():
    coefficients = {"win": 1.0, "lose": -1.0}
    assert sample_coefficients(coefficients, terms=16, seed=7) == coefficients


def test_large_group_preserves_each_side_mass():
    coefficients = {**{f"p{i}": 0.1 for i in range(20)}, **{f"n{i}": -0.2 for i in range(10)}}
    sampled = sample_coefficients(coefficients, terms=16, seed=23)
    assert len(sampled) <= 16
    assert sum(v for v in sampled.values() if v > 0) == pytest.approx(2.0)
    assert sum(v for v in sampled.values() if v < 0) == pytest.approx(-2.0)


def test_each_coefficient_is_unbiased_empirically():
    coefficients = {**{f"p{i}": (i + 1) / 55 for i in range(10)},
                    **{f"n{i}": -(i + 1) / 55 for i in range(10)}}
    totals = {key: 0.0 for key in coefficients}
    trials = 3000
    for seed in range(trials):
        for key, value in sample_coefficients(coefficients, terms=8, seed=seed).items():
            totals[key] += value
    for key, value in coefficients.items():
        assert math.isclose(totals[key] / trials, value, abs_tol=0.025)


def test_invalid_budget_rejected():
    with pytest.raises(ValueError):
        sample_coefficients({"p": 1, "n": -1}, terms=3, seed=0)
