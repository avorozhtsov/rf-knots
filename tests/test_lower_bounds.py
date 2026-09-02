from dataclasses import replace

import pytest

from rf_knots.invariants import invariants
from rf_knots.lower_bounds import (
    claims_for,
    computed_claims,
    strongest,
    tabulated_claims,
)


def test_trefoil_has_three_independent_tabulated_lower_bounds():
    claims = tabulated_claims("3_1")
    assert {claim.method for claim in claims} == {
        "rasmussen-s",
        "ozsvath-szabo-tau",
        "nakanishi-index",
    }
    assert {claim.value for claim in claims} == {1}


def test_rasmussen_and_tau_strengthen_the_signature_when_available():
    # 8_4 has signature-derived lower bound 1 as well; this checks aggregation
    # and provenance without assuming one table field is the source of another.
    claims = tabulated_claims("8_4")
    assert strongest(claims) == 1
    assert all(claim.source.startswith("KnotInfo snapshot") for claim in claims)


def test_computed_and_tabulated_claims_combine():
    inv = replace(invariants((1, 1, 1), 2), signature=-2)
    claims = claims_for(inv)
    assert strongest(claims) == 1
    assert "signature" in {claim.method for claim in claims}


def test_a_cyclic_double_cover_does_not_certify_an_unknotting_number_of_two():
    """11n_184, whose `u` is 1 by a replayed CC=1 edge to the unknot.

    `tests/test_seifert.py` pins the invariant factors; this pins the *consumer*,
    which is where the bound actually became unsound. `H_1(Sigma_2) = Z/87` is
    cyclic, so Montesinos says nothing here -- but `computed_claims` only ever
    sees `len(homology)`, so any future change that inflates the rank re-emits
    `u >= 2` on a knot that unknots in one crossing change, and `unknot_search`
    prunes on exactly this maximum.
    """
    pytest.importorskip("spherogram", reason="computed bounds need the 'bounds' extra")
    word, strands = (-2, -2, 1, 3, 2, 2, 3, 1, -2, 1, 3), 4
    claims = computed_claims(word, strands)
    assert claims, "the computed bounds are unavailable, so this proves nothing"
    assert "montesinos-cyclic" not in {claim.method for claim in claims}
    assert strongest(claims) == 1
