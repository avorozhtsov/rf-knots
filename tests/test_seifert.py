"""The certified lower bounds, checked against tables rather than against belief.

Two of these tests are differential and large: the determinant against every knot
in the bundled table, and the exact signature against the float-eigenvalue path
already in `invariants`. They are what caught the bug that made this module
return an odd signature -- and a knot's signature is always even, so the classical
five-knot check below would have caught it too. Both are kept: the small one says
*what* is wrong, the large one says *how often*.
"""

from __future__ import annotations

import json
from importlib.resources import files

import pytest

from rf_knots import seifert
from rf_knots.invariants import signature as signature_via_eigenvalues

pytest.importorskip("spherogram", reason="certified bounds need the 'bounds' extra")

# name, braid word, strands, sigma, det, tau
CLASSICAL = [
    ("3_1 right", (1, 1, 1), 2, -2, 3, 1),
    ("3_1 left", (-1, -1, -1), 2, 2, 3, -1),
    ("4_1", (1, -2, 1, -2), 3, 0, 5, 0),
    ("5_1", (1, 1, 1, 1, 1), 2, -4, 5, 2),
    ("5_2", (1, 1, 1, 2, -1, 2), 3, -2, 7, 1),
    ("7_1", (1,) * 7, 2, -6, 7, 3),
    ("T(3,4) = 8_19", (1, 2, 1, 2, 1, 2, 1, 2), 3, -6, 3, 3),
]


@pytest.mark.parametrize(("name", "word", "strands", "sigma", "det", "tau"), CLASSICAL)
def test_the_classical_invariants_are_reproduced(name, word, strands, sigma, det, tau):
    assert seifert.signature(word, strands) == sigma
    assert seifert.determinant(word, strands) == det
    assert seifert.tau(word, strands) == tau


@pytest.mark.parametrize(("name", "word", "strands", "sigma", "det", "tau"), CLASSICAL)
def test_the_bounds_never_exceed_the_true_unknotting_number(
    name, word, strands, sigma, det, tau
):
    """Every bound here is a theorem, so it must not exceed the tabulated `u`."""
    truth = {"3_1 right": 1, "3_1 left": 1, "4_1": 1, "5_1": 2, "5_2": 1,
             "7_1": 3, "T(3,4) = 8_19": 3}[name]
    assert abs(sigma) // 2 <= truth
    assert abs(tau) <= truth


def test_a_knot_signature_is_always_even():
    """The property that catches a broken congruence.

    Even-ness holds for **knots**, not for links -- a 3-component closure can
    legitimately have signature -1, which is how this test first failed. Filtering
    on the component count is the fix; weakening the assertion would have thrown
    away the property that catches the bug.
    """
    import numpy as np

    from rf_knots.reference import num_components

    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(400):
        strands = int(rng.integers(2, 6))
        length = int(rng.integers(2, 12))
        word = tuple(
            int(rng.choice((-1, 1))) * int(rng.integers(1, strands))
            for _ in range(length)
        )
        if num_components(word, strands) != 1:
            continue
        try:
            value = seifert.signature(word, strands)
        except (ValueError, seifert.BackendUnavailable):
            continue
        checked += 1
        assert value % 2 == 0, f"odd signature {value} for {word} on {strands} strands"
    assert checked > 50


def test_the_exact_signature_agrees_with_the_eigenvalue_one():
    """`invariants.signature` uses a float tolerance; this one does not."""
    import numpy as np

    rng = np.random.default_rng(1)
    compared = 0
    for _ in range(120):
        strands = int(rng.integers(2, 6))
        length = int(rng.integers(2, 12))
        word = tuple(
            int(rng.choice((-1, 1))) * int(rng.integers(1, strands))
            for _ in range(length)
        )
        try:
            exact = seifert.signature(word, strands)
        except (ValueError, seifert.BackendUnavailable):
            continue
        approximate = signature_via_eigenvalues(word, strands)
        if approximate is None:
            continue
        compared += 1
        assert exact == approximate, f"{word} on {strands}: {exact} vs {approximate}"
    assert compared > 50


def test_the_determinant_matches_every_knot_in_the_bundled_table():
    table = json.loads(
        files("rf_knots").joinpath("data/knot_table.json").read_text()
    )["knots"]
    checked = wrong = 0
    for row in table.values():
        try:
            got = seifert.determinant(tuple(row["braid"]), row["strands"])
        except (ValueError, seifert.BackendUnavailable):
            continue
        checked += 1
        if got != row["determinant"]:
            wrong += 1
    assert checked > 2500
    assert wrong == 0


def test_the_double_cover_of_an_unknotting_number_one_knot_is_cyclic():
    """The Montesinos condition, on knots whose `u` is 1 by the tables."""
    for word, strands in [((1, 1, 1), 2), ((1, -2, 1, -2), 3), ((1, 1, 1, 2, -1, 2), 3)]:
        assert seifert.double_cover_is_cyclic(word, strands)
        assert len(seifert.branched_cover_homology(word, strands)) == 1


def test_the_empty_braid_is_refused_rather_than_answered():
    with pytest.raises(ValueError, match="unknot"):
        seifert.signature((), 1)


def test_exact_integer_determinant_agrees_with_a_naive_expansion():
    """The Bareiss elimination, against cofactor expansion on small matrices."""
    import numpy as np

    def naive(matrix):
        size = len(matrix)
        if size == 1:
            return matrix[0][0]
        total = 0
        for column in range(size):
            minor = [row[:column] + row[column + 1:] for row in matrix[1:]]
            total += (-1) ** column * matrix[0][column] * naive(minor)
        return total

    rng = np.random.default_rng(2)
    for _ in range(40):
        size = int(rng.integers(1, 6))
        matrix = [[int(rng.integers(-4, 5)) for _ in range(size)] for _ in range(size)]
        assert seifert.integer_determinant(matrix) == naive(matrix)


def test_coprime_orders_are_one_cyclic_factor_not_two():
    """A wrong length here is an unsound *lower* bound, which prunes witnesses.

    `Z/3 + Z/29` is cyclic of order 87.  Reporting it as a rank-2 group emits a
    false Montesinos `u >= 2`, and `unknot_search` prunes on that bound -- so the
    error direction discards correct answers rather than merely wasting time.
    Measured blast radius before the fix: 25 of 9,708 nodes carrying a replayed
    `u <= 1` were given a provably false `u >= 2`.
    """
    from rf_knots.seifert import _divisibility_chain

    assert _divisibility_chain([3, 29]) == [87]
    assert _divisibility_chain([2, 3, 5, 7]) == [210]
    assert _divisibility_chain([9]) == [9]
    assert _divisibility_chain([]) == []


def test_the_chain_is_a_chain_and_preserves_the_group_order():
    """`d_1 | d_2 | ... | d_k`, no units, same order -- the name's whole promise.

    The previous implementation returned `[5, 5, 5] -> [5, 1, 25]`, which has the
    right product and is not a divisibility chain, and `[3, 29] -> [1, 87]`, whose
    unit inflated the rank.
    """
    import math
    import random

    from rf_knots.seifert import _divisibility_chain

    assert _divisibility_chain([5, 5, 5]) == [5, 5, 5]
    assert _divisibility_chain([2, 3, 4]) == [2, 12]

    rng = random.Random(20260901)
    for _ in range(60):
        orders = [rng.randint(2, 60) for _ in range(rng.randint(1, 5))]
        chain = _divisibility_chain(list(orders))
        assert math.prod(chain) == math.prod(orders), orders
        assert all(d > 1 for d in chain), (orders, chain)
        for smaller, larger in zip(chain, chain[1:], strict=False):
            assert larger % smaller == 0, (orders, chain)


# `test_the_chain_is_a_chain_and_preserves_the_group_order` above checks three
# properties -- divisibility, no units, same order -- and those three do not pin
# the answer. For the multiset `[9, 4]` the chain `[6, 6]` satisfies all of them
# and is still wrong: `Z/9 + Z/4` is the cyclic `Z/36`, so the truth is `[36]`.
# The wrong answer is longer, which is the same direction the merged defect
# failed in -- an inflated rank, hence a false Montesinos `u >= 2`. So these add
# the cases that separate the right chain from the merely well-formed ones.
COPRIME_CASES = [
    ([5, 7], [35]),
    ([9, 4], [36]),
    ([2, 4], [2, 4]),
    ([3, 3], [3, 3]),
    ([6, 10, 15], [30, 30]),
    ([12, 18, 30], [6, 6, 180]),
]


@pytest.mark.parametrize(("orders", "expected"), COPRIME_CASES)
def test_the_chain_is_the_one_smith_form_and_not_merely_well_formed(orders, expected):
    assert seifert._divisibility_chain(orders) == expected


def test_the_chain_agrees_with_an_independent_gcd_lcm_construction():
    """Differential check against a second algorithm that shares no code.

    `seifert` builds the invariant factors from sorted prime exponents. The
    classical alternative works on the diagonal directly: `diag(a, b)` is
    equivalent to `diag(gcd(a, b), lcm(a, b))` over `Z`, so sweeping that
    replacement across adjacent pairs until nothing moves also lands on the Smith
    form. Two routes to a decomposition that is unique -- so any disagreement is
    a defect in one of them, and no amount of property-checking substitutes for
    it, since the properties admit wrong answers.
    """
    import math
    import random
    from math import gcd

    def by_gcd_lcm(values):
        chain = sorted(abs(value) for value in values if value not in (0, 1))
        changed = True
        while changed:
            changed = False
            for index in range(len(chain) - 1):
                first, second = chain[index], chain[index + 1]
                divisor = gcd(first, second)
                if divisor != first:
                    chain[index] = divisor
                    chain[index + 1] = first // divisor * second
                    changed = True
        return [value for value in chain if value != 1]

    rng = random.Random(20260902)
    primes = [2, 3, 5, 7, 11, 13, 101, 9973]
    for _ in range(400):
        orders = [
            rng.choice(primes) ** rng.randint(1, 3)
            if rng.random() < 0.5
            else rng.randint(2, 5000)
            for _ in range(rng.randint(1, 5))
        ]
        chain = seifert._divisibility_chain(list(orders))
        assert chain == by_gcd_lcm(orders), orders
        assert math.prod(chain) == math.prod(orders), orders
