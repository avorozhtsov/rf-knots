"""The reference implementation is the oracle for everything else, so it gets
tested against mathematics rather than against other code."""

from __future__ import annotations

from rf_knots.actions import PASS, ActionSpec
from rf_knots.reference import (
    apply,
    bfs_unknot,
    equal_in_braid_group,
    format_word,
    free_reduce,
    legal_actions,
    num_components,
    permutation,
    successors,
    writhe,
)


def test_free_reduce() -> None:
    assert free_reduce((1, -1)) == ()
    assert free_reduce((1, 2, -2, -1)) == ()
    assert free_reduce((1, 2, -2, 3)) == (1, 3)
    assert free_reduce((1, 1, -1)) == (1,)


def test_artin_is_a_representation_of_the_braid_relations() -> None:
    # sigma_1 sigma_2 sigma_1 = sigma_2 sigma_1 sigma_2  in B_3
    assert equal_in_braid_group((1, 2, 1), (2, 1, 2), 3)
    # and the all-negative form, which is its inverse
    assert equal_in_braid_group((-1, -2, -1), (-2, -1, -2), 3)
    # far commutation in B_4
    assert equal_in_braid_group((1, 3), (3, 1), 4)
    assert equal_in_braid_group((1, -3), (-3, 1), 4)
    # free reduction
    assert equal_in_braid_group((1, -1), (), 3)
    assert equal_in_braid_group((2, 1, -1, -2), (), 3)


def test_artin_separates_distinct_elements() -> None:
    assert not equal_in_braid_group((1,), (2,), 3)
    assert not equal_in_braid_group((1, 2), (2, 1), 3)  # B_3 is non-abelian
    assert not equal_in_braid_group((1,), (), 3)
    assert not equal_in_braid_group((1, 1), (1,), 3)
    # adjacent generators do NOT commute
    assert not equal_in_braid_group((1, 2, 1), (1, 1, 2), 3)


def test_mixed_sign_braid_relation_is_derivable() -> None:
    # (-b, a, b) = (a, b, -a) with a=1, b=2 -- the relation the move set reaches
    # only via insert + braid + reduce. It must still be a true identity.
    assert equal_in_braid_group((-2, 1, 2), (1, 2, -1), 3)


def test_permutation_and_components() -> None:
    assert permutation((), 3) == [0, 1, 2]
    assert permutation((1,), 3) == [1, 0, 2]
    assert num_components((), 3) == 3  # closure of the identity braid is a 3-unlink
    assert num_components((1, 2), 3) == 1  # 3-cycle -> a knot (this one is the unknot)
    assert num_components((1, 1, 1), 2) == 1  # trefoil as a 2-braid
    assert num_components((1, 1), 2) == 2  # Hopf link
    assert num_components((1,), 3) == 2  # sigma_1 in B_3: unknot plus a stray strand


def test_writhe() -> None:
    assert writhe((1, 1, 1)) == 3
    assert writhe((1, -1)) == 0
    assert writhe(()) == 0


def test_bfs_finds_the_empty_braid() -> None:
    spec = ActionSpec(max_len=10, max_strands=4)
    # sigma_1 on 2 strands is the unknot: one destabilisation away.
    path = bfs_unknot(spec, (1,), 2, max_depth=3)
    assert path is not None and len(path) == 1

    # sigma_1 sigma_2 in B_3 is the unknot: two destabilisations.
    path = bfs_unknot(spec, (1, 2), 3, max_depth=4)
    assert path is not None and len(path) == 2

    # One free reduction is needed first, and no shorter route exists: two
    # destabilisations are forced, and they only remove two of the four letters.
    path = bfs_unknot(spec, (1, 2, 2, -2), 3, max_depth=5)
    assert path is not None and len(path) == 3


def test_bfs_returns_none_past_the_cutoff() -> None:
    spec = ActionSpec(max_len=10, max_strands=4)
    assert bfs_unknot(spec, (1,), 2, max_depth=0) is None


def test_successors_agrees_with_legal_actions_and_apply() -> None:
    """The fast enumeration used by search must match the exhaustive scan."""
    spec = ActionSpec(max_len=10, max_strands=4)
    cases = [
        ((), 1),
        ((1,), 2),
        ((1, 2), 3),
        ((1, -1, 2), 3),
        ((1, 2, 1, -3), 4),
        ((2, 1, 1, 2, -2), 3),
        ((1,) * 10, 2),  # full word: no insertions or stabilisations fit
    ]
    for word, n in cases:
        for allow_crossing in (False, True):
            fast = sorted(successors(spec, word, n, allow_crossing))
            slow = sorted(
                (action, *apply(spec, word, n, action))
                for action in legal_actions(spec, word, n, allow_crossing)
                if spec.decode(action)[0] != PASS
            )
            assert fast == slow, f"{format_word(word, n)} (crossing={allow_crossing})"
