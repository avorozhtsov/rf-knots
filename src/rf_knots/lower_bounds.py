"""Certified unknotting lower bounds with explicit computational provenance."""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files

from rf_knots.evidence import LowerBoundClaim
from rf_knots.invariants import Invariants

MURASUGI = "Murasugi signature bound: abs(sigma(K))/2 <= u(K)"


@cache
def _table() -> dict:
    path = files("rf_knots").joinpath("data/lower_bounds.json")
    return json.loads(path.read_text())


def tabulated_claims(name: str) -> tuple[LowerBoundClaim, ...]:
    """KnotInfo s, tau and Nakanishi lower bounds for a canonical knot name."""
    table = _table()
    row = table["values"].get(name)
    if row is None:
        return ()
    source = f"KnotInfo snapshot {table['source']['retrieved_at']}"
    citation = table["source"]["url"]
    claims = []
    if "rasmussen_s" in row:
        value = abs(int(row["rasmussen_s"])) // 2
        claims.append(LowerBoundClaim(value, "rasmussen-s", source, citation,
                                      {"s": int(row["rasmussen_s"])}))
    if "ozsvath_szabo_tau" in row:
        value = abs(int(row["ozsvath_szabo_tau"]))
        claims.append(LowerBoundClaim(value, "ozsvath-szabo-tau", source, citation,
                                      {"tau": int(row["ozsvath_szabo_tau"])}))
    if "nakanishi_lower" in row:
        claims.append(LowerBoundClaim(int(row["nakanishi_lower"]), "nakanishi-index",
                                      source, citation,
                                      {"raw": row.get("nakanishi_raw", "")}))
    return tuple(claims)


def computed_claims(word, strands: int) -> tuple[LowerBoundClaim, ...]:
    """Bounds computed from the braid word itself, with no table lookup.

    This is the half that matters for knots nobody has named: the ladder's random
    rungs, and anything an agent proposes. Each is a theorem, so each is a claim
    about `u` rather than an estimate of it.

    Returns an empty tuple when the optional backend is missing, which is a real
    degradation and not a silent one -- `rf_knots selfcheck` reports whether the
    computed bounds are available at all.
    """
    from rf_knots import seifert

    claims: list[LowerBoundClaim] = []
    try:
        sigma = seifert.signature(word, strands)
    except (ValueError, seifert.BackendUnavailable):
        return ()
    claims.append(
        LowerBoundClaim(abs(sigma) // 2, "signature", "rf_knots.seifert",
                        details={"signature": sigma, "theorem": seifert.MURASUGI})
    )
    # A non-cyclic H_1 of the double branched cover rules out u = 1 outright. It
    # is the only bound here that fires when the signature and tau are both zero,
    # which is exactly the case the other two cannot separate.
    try:
        homology = seifert.branched_cover_homology(word, strands)
    except (ValueError, seifert.BackendUnavailable):
        homology = None
    if homology is not None and len(homology) > 1:
        claims.append(
            LowerBoundClaim(2, "montesinos-cyclic", "rf_knots.seifert",
                            details={"H_1(Sigma_2)": homology,
                                     "theorem": seifert.MONTESINOS})
        )
    try:
        value = seifert.tau(word, strands)
    except (ValueError, KeyError, RuntimeError, seifert.BackendUnavailable):
        value = None
    if value is not None:
        claims.append(
            LowerBoundClaim(abs(value), "ozsvath-szabo-tau", "rf_knots.seifert",
                            details={"tau": value, "theorem": seifert.OZSVATH_SZABO})
        )
    return tuple(claims)


def claims_for(inv: Invariants, computed: bool = True) -> tuple[LowerBoundClaim, ...]:
    """All currently available independent lower-bound claims for a fingerprint.

    `computed=False` restores the table-only behaviour, which is what the
    zero-human-knowledge audit in [research/12 §4] wants when it needs to know how
    much of a result came from a human-curated table.
    """
    claims: list[LowerBoundClaim] = []
    if inv.signature is not None:
        claims.append(
            LowerBoundClaim(abs(inv.signature) // 2, "signature", "rf_knots.invariants",
                            details={"signature": inv.signature, "theorem": MURASUGI})
        )
    if computed and not claims:
        claims.extend(computed_claims(inv.word, inv.strands))
    elif computed:
        # The signature is already present from `invariants`; add only the bounds
        # it does not carry, so a knot is never counted twice for one theorem.
        claims.extend(
            claim for claim in computed_claims(inv.word, inv.strands)
            if claim.method != "signature"
        )
    if inv.name is not None:
        claims.extend(tabulated_claims(inv.name))
    return tuple(claims)


def strongest(claims: tuple[LowerBoundClaim, ...]) -> int | None:
    return max((claim.value for claim in claims), default=None)
