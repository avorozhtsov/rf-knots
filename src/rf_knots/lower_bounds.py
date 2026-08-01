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


def claims_for(inv: Invariants) -> tuple[LowerBoundClaim, ...]:
    """All currently available independent lower-bound claims for a fingerprint."""
    claims: list[LowerBoundClaim] = []
    if inv.signature is not None:
        claims.append(
            LowerBoundClaim(abs(inv.signature) // 2, "signature", "rf_knots.invariants",
                            details={"signature": inv.signature, "theorem": MURASUGI})
        )
    if inv.name is not None:
        claims.extend(tabulated_claims(inv.name))
    return tuple(claims)


def strongest(claims: tuple[LowerBoundClaim, ...]) -> int | None:
    return max((claim.value for claim in claims), default=None)
