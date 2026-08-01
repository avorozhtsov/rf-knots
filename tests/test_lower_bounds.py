from dataclasses import replace

from rf_knots.invariants import invariants
from rf_knots.lower_bounds import claims_for, strongest, tabulated_claims


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
