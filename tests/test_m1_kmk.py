"""The `K # m(K)` family, checked where it could quietly be wrong.

Four things this construction claims, and each is checked against something that
does not share its implementation:

* the braid word really is the connected sum -- `connected_summands` splits it
  back into the two factors, which is `spherogram` and not this module;
* the `2u(K)` upper bound is a **program**, not subadditivity: it replays to the
  empty one-braid through `UnknotWitness` or the seed refuses to exist;
* the certified floor is 2 for every member, which is the premise §M1 rests on
  and the reason the family was nominated at all;
* the §4.3 screen fires on the target.  The interesting half is the *negative*:
  the shipped `Quarantine.screen_name` cannot catch `7_1 # m(7_1)`, so a test
  that only checked `screen` would pass while the guard it names did nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))

import m1_kmk  # noqa: E402

from rf_knots import reference  # noqa: E402
from rf_knots.actions import ActionSpec  # noqa: E402
from rf_knots.interleaved_generator import load_quarantine  # noqa: E402
from rf_knots.invariants import connected_summands, invariants  # noqa: E402
from rf_knots.lower_bounds import computed_claims, strongest  # noqa: E402


def test_the_composite_word_closes_to_a_knot_and_splits_back_into_its_factors():
    word, strands = m1_kmk.mirror_connected_sum((1,) * 5, 2)
    assert strands == 3
    assert reference.num_components(word, strands) == 1
    pieces = connected_summands(word, strands)
    assert pieces is not None and len(pieces) == 2
    #: One factor is positive and one negative -- that is what "mirror" means at
    #: the level of the word, and it is checked here rather than assumed.
    signs = sorted(min(p) > 0 for p in pieces)
    assert signs == [False, True]
    assert sorted(len(p) for p in pieces) == [5, 5]


@pytest.mark.parametrize("m", [1, 2, 3])
def test_the_upper_bound_is_a_replayed_program_and_not_subadditivity(m):
    spec = ActionSpec(max_len=2 * (2 * m + 1), max_strands=4)
    seed = m1_kmk.kmk_torus_seed(m, spec)
    #: `verify_program` replays through `UnknotWitness` and raises unless the
    #: replay ends at the empty one-braid having spent exactly `u_upper`.
    assert seed.verify_program(spec) == 2 * m
    assert seed.replay_complete


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5])
def test_the_certified_floor_is_two_for_every_member(m):
    """The premise, as a regression test: `L` does not track `2u(K)`."""
    word, strands = m1_kmk.mirror_connected_sum((1,) * (2 * m + 1), 2)
    assert strongest(computed_claims(word, strands)) == 2


def test_the_only_bound_that_fires_is_montesinos_because_the_composite_is_slice():
    word, strands = m1_kmk.mirror_connected_sum((1,) * 5, 2)
    values = {claim.method: claim.value for claim in computed_claims(word, strands)}
    assert values["signature"] == 0
    assert values["ozsvath-szabo-tau"] == 0
    assert values["montesinos-cyclic"] == 2


def test_the_interval_is_tight_exactly_when_u_of_k_is_one():
    tight = m1_kmk.premise_row("T(2,3)", (1, 1, 1), 2, 1)
    loose = m1_kmk.premise_row("T(2,5)", (1,) * 5, 2, 2)
    assert tight.width == 0 and not tight.loose
    assert loose.width == 2 and loose.loose


def test_the_shipped_name_screen_cannot_catch_the_target_but_the_alias_screen_can():
    word, strands = m1_kmk.mirror_connected_sum((1,) * 7, 2)
    inv = invariants(word, strands)
    quarantine = load_quarantine()
    #: The identification is ambiguous (§1.2's trap) and the mirror is unmarked,
    #: so the exact-set test in the shipped screen is false.  This is the defect,
    #: pinned so that a later fix to `Quarantine` shows up as a failing test
    #: rather than as silence.
    assert quarantine.screen_name(inv.name, inv.summands) is None
    assert m1_kmk.summand_alias_hit(inv.summands)
    assert not m1_kmk.summand_alias_hit(("3_1", "3_1"))
    assert not m1_kmk.summand_alias_hit(("7_1",))


def test_the_screen_rejects_the_target_and_admits_a_clean_member():
    quarantine = load_quarantine()
    target, n = m1_kmk.mirror_connected_sum((1,) * 7, 2)
    clean, k = m1_kmk.mirror_connected_sum((1,) * 9, 2)
    assert m1_kmk.screen(quarantine, target, n).rejected
    assert not m1_kmk.screen(quarantine, clean, k).rejected


def test_the_summandwise_shortcut_is_measured_and_needs_no_growth():
    word, strands = m1_kmk.mirror_connected_sum((1,) * 5, 2)
    shortcut = m1_kmk.summandwise(word, strands)
    assert not shortcut.prime and shortcut.determined
    assert shortcut.total_upper == 4
    assert shortcut.needs_growth is False
    #: `beats` is about the *instance's* optimum, so it is None until one exists
    #: and False when the optimum is exactly the shortcut.
    assert shortcut.beats(None) is None
    assert shortcut.beats(4) is False
    assert shortcut.beats(3) is True
