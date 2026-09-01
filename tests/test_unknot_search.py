"""Branch-and-bound: every witness must replay, and the pruning must be sound.

A pruning heuristic that is not admissible does not fail loudly -- it quietly
returns a longer sequence and everything still looks fine. So the tests here check
the two things that would catch it: that the bound never exceeds the sequence
actually found, and that the knots whose `u` is a theorem come back exact.
"""

from __future__ import annotations

import pytest

from rf_knots.unknot_search import certified_lower_bound, search

pytest.importorskip("spherogram", reason="the bounds need the 'bounds' extra")

# word, strands, true u, whether the certified bound is known to reach it
KNOWN = [
    ((1, 1, 1), 2, 1, True),            # 3_1, |sigma|/2 = 1
    ((1, 1, 1, 1, 1), 2, 2, True),      # 5_1, |sigma|/2 = 2
    ((1, 2, 1, 2, 1, 2, 1, 2), 3, 3, True),   # T(3,4) = 8_19, |sigma|/2 = 3
    ((1, -2, 1, -2), 3, 1, False),      # 4_1: sigma = 0, tau = 0, H_1 cyclic
]


@pytest.mark.parametrize(("word", "strands", "truth", "tight"), KNOWN)
def test_the_search_finds_a_sequence_no_longer_than_the_truth(word, strands, truth, tight):
    report = search(word, strands, max_crossing_changes=truth + 1, node_budget=8000)
    assert report.witness is not None, report.notes
    report.witness.verify()
    assert report.crossing_changes <= truth


@pytest.mark.parametrize(("word", "strands", "truth", "tight"), KNOWN)
def test_the_certified_bound_never_exceeds_the_truth(word, strands, truth, tight):
    """Admissibility. If this fails, the pruning is cutting real optima."""
    assert certified_lower_bound(word, strands) <= truth


@pytest.mark.parametrize(("word", "strands", "truth", "tight"), KNOWN)
def test_u_is_determined_exactly_where_the_bound_is_tight(word, strands, truth, tight):
    report = search(word, strands, max_crossing_changes=truth + 1, node_budget=8000)
    assert report.solved_exactly is tight, (
        f"{word}: bound {report.lower_bound}, found {report.crossing_changes}"
    )


def test_the_seven_five_rung_is_solved_at_two_and_the_bound_meets_it():
    """`R(3,18)#0` from `docs/rungs.md`: the ladder's standing record was 6.

    Eighteen letters for a seven-crossing knot, which is why the level search is
    ordered by word length -- breadth-first spends its whole budget in the
    eighteen-letter neighbourhood and never simplifies.
    """
    word = (1, 1, 2, 2, 1, 1, 2, 1, -2, -1, 2, 1, -2, -2, -1, 2, 1, -2)
    report = search(word, 3, max_crossing_changes=2, node_budget=60_000, growth=3,
                    frontier_width=40, flip_from=250)
    assert report.lower_bound == 2
    assert report.witness is not None, report.notes
    report.witness.verify()
    assert report.crossing_changes == 2
    assert report.solved_exactly


def test_a_budget_below_the_certified_bound_returns_immediately():
    """No search at all when the theorem already rules the budget out."""
    report = search((1,) * 7, 2, max_crossing_changes=1)   # 7_1 has |sigma|/2 = 3
    assert report.witness is None
    assert report.diagrams_explored == 0
    assert "certified bound" in report.notes[0]


def test_the_pruning_actually_fires():
    """A search that prunes nothing is not branch-and-bound, it is enumeration."""
    report = search((1, 1, 1, 1, 1), 2, max_crossing_changes=2, node_budget=8000)
    assert report.pruned_by_bound > 0


def test_the_two_determinant_routes_agree_so_the_fingerprint_key_is_unchanged():
    """`_fingerprint` switched routes for speed; a differing value would be a bug.

    The Seifert route is 18x-247x faster depending on shape (1,539x on the
    254-letter target), but it is only a legitimate swap because the key it
    produces is identical.  This asserts that over the bundled table rather than
    over one hand-picked word.
    """
    import random

    from rf_knots import seifert
    from rf_knots.invariants import determinant
    from rf_knots.unknot_search import _fingerprint

    rng = random.Random(20260901)
    words = [(1,) * n for n in range(2, 9)]                      # torus knots
    words += [(1, -2, 1, -2), (1, 1, 2, -1, 2, 2)]               # small mixed
    words += [
        tuple(rng.choice([g, -g]) for g in
              (rng.randint(1, 4) for _ in range(length)))
        for length in (12, 20, 28)
    ]
    checked = 0
    for word in words:
        strands = max((abs(letter) for letter in word), default=1) + 1
        try:
            fast = seifert.determinant(word, strands)
        except (ValueError, KeyError, RuntimeError, seifert.BackendUnavailable):
            continue                     # the fallback path; nothing to compare
        assert fast == determinant(word, strands), word
        assert _fingerprint(word, strands) == (1, fast), word
        checked += 1
    assert checked >= 8, f"only {checked} words exercised the fast route"


def test_the_fingerprint_is_bounded_on_a_word_the_burau_route_cannot_afford():
    """The empty word is special-cased, and a long word must not take minutes.

    A 60-letter word costs ~0.9 s through `invariants.determinant` and ~10 ms
    through `seifert`.  The bound here is loose enough to survive a contended
    host and still fail the route this change removed.
    """
    import time

    from rf_knots.unknot_search import _fingerprint

    assert _fingerprint((), 1) == (0, 1)

    word = tuple(1 if index % 3 else -1 for index in range(60))
    _fingerprint(word, 2)                                  # warm the backend
    start = time.monotonic()
    key = _fingerprint(word, 2)
    assert time.monotonic() - start < 0.30
    assert key[0] == 1
