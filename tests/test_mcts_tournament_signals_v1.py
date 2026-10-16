import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mcts_tournament_signals_v1 import _groups


def trace(rows, cost, cc=1):
    return {"rows": tuple(rows), "cost": cost, "cc": cc, "moves": cost - cc}


def coeff(group):
    return group["coefficients"]


def sample():
    # The two routes diverge at root and reconverge at a resource-identical
    # state. Each route is a complete verified-solution analogue.
    best = trace([("root", 1), ("left", 3), ("join", 5)], 10, cc=1)
    near = trace([("root", 2), ("right", 4), ("join", 6)], 12, cc=2)
    far = trace([("root", 7), ("far", 8), ("join", 9)], 16, cc=3)
    return [best, near, far]


PARAMS = dict(seed=7, k=2, near_margin=3, gap_scale=2, gap_cap=10)


def test_one_control_plus_eight_real_variants():
    assert _groups(sample(), "A", **PARAMS) == []
    for arm in "BCDEFGHI":
        groups = _groups(sample(), arm, **PARAMS)
        assert groups and groups[0]["coefficients"]
    with pytest.raises(ValueError, match="unknown arm"):
        _groups(sample(), "J", **PARAMS)


def test_first_fork_and_shared_context_are_distinct():
    first = coeff(_groups(sample(), "C", **PARAMS)[0])
    assert set(state for state, _ in first) == {"root"}
    shared = coeff(_groups(sample(), "E", **PARAMS)[0])
    assert {state for state, _ in shared} == {"root", "join"}
    assert sum(shared.values()) == pytest.approx(0)


def test_full_suffix_keeps_only_uncancelled_terms():
    winner = trace([("root", 1), ("shared", 3), ("left", 4)], 10)
    loser = trace([("root", 1), ("shared", 3), ("right", 5)], 12)
    signal = coeff(_groups([winner, loser], "D", **PARAMS)[0])
    assert ("root", 1) not in signal
    assert ("shared", 3) not in signal
    assert signal[("left", 4)] == pytest.approx(1)
    assert signal[("right", 5)] == pytest.approx(-1)


def test_near_best_positives_and_separate_high_cc_objective():
    groups = _groups(sample(), "F", **PARAMS)
    assert groups[0]["positive_paths"] == 2
    assert groups[0]["negative_paths"] == 1
    groups = _groups(sample(), "G", **PARAMS)
    assert [g["objective"] for g in groups] == [
        "lambda_cost",
        "lexicographic_cc_extra",
    ]


def test_gap_weights_total_one_and_favor_larger_cost_gap():
    uniform = coeff(_groups(sample(), "E", **PARAMS)[0])
    weighted = coeff(_groups(sample(), "I", **PARAMS)[0])
    assert uniform[("root", 2)] == pytest.approx(-0.5)
    assert uniform[("root", 7)] == pytest.approx(-0.5)
    assert weighted[("root", 2)] == pytest.approx(-0.25)
    assert weighted[("root", 7)] == pytest.approx(-0.75)
    assert sum(weighted.values()) == pytest.approx(0)


def test_no_strictly_worse_verified_solution_means_no_preference():
    best = sample()[0]
    tied = {**best, "rows": (("root", 10), ("other", 11))}
    assert _groups([best, tied], "E", **PARAMS) == []
