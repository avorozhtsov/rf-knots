"""Invariants are checked against theorems, not against previous runs of this code.

Every expected value below is either a published invariant of a named knot or a
property every knot must have. A regression test that pins whatever the code
happened to print would have passed for the fitted Seifert matrix described in
`rf_knots.invariants`, which was wrong.
"""

from __future__ import annotations

import pytest

from rf_knots.generator import torus_braid
from rf_knots.invariants import (
    alexander_polynomial,
    determinant,
    invariants,
    jones_polynomial,
    kauffman_bracket,
    signature,
)


def mirror(word):
    return tuple(-x for x in word)


def test_unknot_is_trivial():
    assert alexander_polynomial((), 1) == {0: 1}
    assert jones_polynomial((), 1) == {0: 1}
    assert kauffman_bracket((), 1) == {0: 1}


def test_alexander_of_torus_knots():
    # Delta(T(2,q)) = 1 - t + t^2 - ... + t^(q-1).
    for q in (3, 5, 7, 9):
        word, strands = torus_braid(2, q)
        expected = {e: (1 if e % 2 == 0 else -1) for e in range(q)}
        assert alexander_polynomial(tuple(word), strands) == expected


def test_alexander_of_figure_eight():
    assert alexander_polynomial((1, -2, 1, -2), 3) == {0: 1, 1: -3, 2: 1}


def test_jones_of_named_knots():
    # Published values, for the closure of the braid word given.
    cases = {
        (1, 1, 1): (2, {1: 1, 3: 1, 4: -1}),                      # trefoil, positive
        (1, -2, 1, -2): (3, {-2: 1, -1: -1, 0: 1, 1: -1, 2: 1}),  # figure-eight, 4_1
        tuple([1, 2] * 4): (3, {3: 1, 5: 1, 8: -1}),              # T(3,4) = 8_19
        tuple([1, 2] * 5): (3, {4: 1, 6: 1, 10: -1}),             # T(3,5) = 10_124
    }
    for word, (strands, expected) in cases.items():
        assert jones_polynomial(word, strands) == expected


def test_jones_of_the_mirror_is_the_substitution_t_to_one_over_t():
    for word, strands in [((1, 1, 1), 2), ((1, -2, 1, -2), 3), (tuple([1, 2] * 4), 3)]:
        direct = jones_polynomial(word, strands)
        assert jones_polynomial(mirror(word), strands) == {-e: c for e, c in direct.items()}


def test_alexander_is_symmetric_and_evaluates_to_one():
    """Two facts true of every knot: Delta(t) = t^deg Delta(1/t), and Delta(1) = +-1."""
    for word, strands in [((1, 1, 1), 2), ((1, -2, 1, -2), 3), (tuple([1, 2] * 5), 3),
                          ((1, 1, 2, 2, 1, 1, 2, 1, -2, -1, 2, 1, -2, -2, -1, 2, 1, -2), 3)]:
        delta = alexander_polynomial(word, strands)
        top = max(delta)
        assert delta == {top - e: c for e, c in delta.items()}
        assert abs(sum(delta.values())) == 1


def test_determinant_is_odd_for_a_knot():
    for word, strands in [((1, 1, 1), 2), ((1, -2, 1, -2), 3), (tuple([1, 2] * 4), 3)]:
        assert determinant(word, strands) % 2 == 1


def test_determinant_of_named_knots():
    assert determinant((1, 1, 1), 2) == 3           # trefoil
    assert determinant((1, -2, 1, -2), 3) == 5      # figure-eight
    assert determinant(tuple([1, 2] * 4), 3) == 3   # 8_19
    assert determinant(tuple([1, 2] * 5), 3) == 1   # 10_124


def test_span_of_jones_bounds_the_crossing_number():
    """span V <= c, with equality exactly for alternating diagrams (Kauffman-
    Murasugi-Thistlethwaite). The figure-eight is alternating and attains it."""
    jones = jones_polynomial((1, -2, 1, -2), 3)
    assert max(jones) - min(jones) == 4


@pytest.mark.parametrize(
    "p,q,expected",
    [(2, 3, -2), (2, 5, -4), (2, 7, -6), (2, 9, -8), (3, 4, -6), (3, 5, -8)],
)
def test_signature_of_torus_knots(p, q, expected):
    word, strands = torus_braid(p, q)
    value = signature(tuple(word), strands)
    if value is None:
        pytest.skip("spherogram is not installed")
    assert value == expected


def test_identifies_the_rung_that_looked_unlabelled():
    """`R(3,18)#0` is eighteen letters and is the seven-crossing knot 7_5.

    This is the case the whole module exists for: the ladder recorded it as an
    unlabelled challenge knot with u unknown, and its unknotting number has been
    a theorem since the knot tables.
    """
    word = (1, 1, 2, 2, 1, 1, 2, 1, -2, -1, 2, 1, -2, -2, -1, 2, 1, -2)
    result = invariants(word, 3)
    assert result.determinant == 17
    assert result.crossings == 18
    if result.name is None:
        pytest.skip("knot table has not been built")
    assert result.name == "7_5"
    assert result.identified_crossings == 7
    assert result.unknotting == 2


def test_genus_bounds_bracket_the_truth_for_torus_knots():
    """g(T(p,q)) = (p-1)(q-1)/2, and the Bennequin surface of the standard braid
    realises it -- so for these the two bounds meet."""
    for p, q in [(2, 3), (2, 5), (3, 4), (3, 5)]:
        word, strands = torus_braid(p, q)
        result = invariants(tuple(word), strands, identify_knot=False)
        genus = (p - 1) * (q - 1) // 2
        assert result.genus_lower == genus
        assert result.genus_upper == genus
