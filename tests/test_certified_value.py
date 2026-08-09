"""The floor is only worth having if it is admissible: it must never overshoot.

A value floor that exceeds the true cost is worse than no floor at all -- it
would clamp the network away from the correct answer and prune the optimal
branch. So the test that matters is the one against knots whose `u` is known.
"""

from __future__ import annotations

import pytest

from rf_knots.certified_value import (
    certified_floor,
    certified_floor_report,
    clamp_cost_to_go,
    clamp_value,
)

pytest.importorskip("spherogram", reason="certified bounds need the 'bounds' extra")

# word, strands, true u
KNOWN = [
    ((1, 1, 1), 2, 1),                     # 3_1
    ((-1, -1, -1), 2, 1),                  # 3_1 mirror
    ((1, -2, 1, -2), 3, 1),                # 4_1: all three bounds are silent
    ((1, 1, 1, 1, 1), 2, 2),               # 5_1
    ((1, 1, 1, 2, -1, 2), 3, 1),           # 5_2
    ((1,) * 7, 2, 3),                      # 7_1
    ((1, 2, 1, 2, 1, 2, 1, 2), 3, 3),      # T(3,4) = 8_19
]


@pytest.mark.parametrize(("word", "strands", "truth"), KNOWN)
def test_the_floor_never_exceeds_the_true_unknotting_number(word, strands, truth):
    """Admissibility. If this fails the floor prunes optimal branches."""
    floor = certified_floor(word, strands)
    assert floor.crossing_changes <= truth, (
        f"{word}: floor {floor.crossing_changes} exceeds u = {truth} "
        f"via {floor.method}"
    )


@pytest.mark.parametrize(("word", "strands", "truth"), KNOWN)
def test_the_cost_floor_never_exceeds_the_true_cost(word, strands, truth):
    """Same statement in objective units, at both ends of the A:B range."""
    for ratio in (1000.0, 10.0, 1.0, 0.1):
        floor = certified_floor(word, strands, ratio=ratio)
        assert floor.cost <= ratio * truth + 1e-9


def test_the_floor_is_zero_exactly_when_it_knows_nothing():
    """`4_1` has sigma = 0, tau = 0 and cyclic H_1, so nothing fires."""
    silent = certified_floor((1, -2, 1, -2), 3)
    assert silent.crossing_changes == 0
    assert not silent.informative
    loud = certified_floor((1, 1, 1, 1, 1), 2)
    assert loud.crossing_changes == 2
    assert loud.informative


def test_the_solved_state_has_a_zero_floor():
    floor = certified_floor((), 1)
    assert floor.crossing_changes == 0 and floor.cost == 0.0
    assert floor.method == "solved"


def test_clamping_cost_raises_and_never_lowers():
    floor = certified_floor((1, 1, 1, 1, 1), 2, ratio=10.0)  # cost floor 20
    assert clamp_cost_to_go(5.0, floor) == floor.cost
    assert clamp_cost_to_go(999.0, floor) == 999.0


def test_clamping_value_is_a_ceiling_not_a_floor():
    """Value is negated cost, so a cost floor must *lower* an optimistic value."""
    floor = certified_floor((1, 1, 1, 1, 1), 2, ratio=1.0)  # 2 crossing changes
    # An optimistic network says "nearly solved"; the theorem says it cannot be.
    assert clamp_value(0.9, floor, cap=4.0) == pytest.approx(-0.5)
    # A pessimistic prediction is left alone.
    assert clamp_value(-0.95, floor, cap=4.0) == pytest.approx(-0.95)
    # A zero cap disables clamping rather than dividing by zero.
    assert clamp_value(0.9, floor, cap=0.0) == pytest.approx(0.9)


def test_the_report_says_how_often_the_bound_is_informative():
    """The pre-check: a bound-based candidate is only as good as its coverage."""
    report = certified_floor_report([(w, n) for w, n, _ in KNOWN])
    assert report["instances"] == len(KNOWN)
    # 4_1 and 5_2 are silent; the rest fire.
    assert 0.0 < report["informative_fraction"] < 1.0
    assert report["max_crossing_change_floor"] == 3
    assert set(report["by_method"]) <= {
        "signature", "ozsvath-szabo-tau", "montesinos-cyclic",
    }


def test_the_bound_is_cached_so_a_search_can_afford_it():
    """Type-preserving moves leave it unchanged, so a search hits this constantly."""
    from rf_knots.certified_value import _bound

    _bound.cache_clear()
    certified_floor((1, 2, 1, 2, 1, 2, 1, 2), 3)
    certified_floor((1, 2, 1, 2, 1, 2, 1, 2), 3)
    assert _bound.cache_info().hits >= 1
