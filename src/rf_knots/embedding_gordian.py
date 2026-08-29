"""Read-only supervision contract for catalogue Gordian-distance bounds."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SCHEMA = "rf-knots-gordian-lower-bounds-v0"
_FACTOR = re.compile(r"K(10|[0-9])([0-9]+)(m?)")


@dataclass(frozen=True)
class GordianEndpoint:
    source_id: str
    canonical_knot_id: str
    representation_ids: tuple[int, ...]
    mirror: bool
    split: str


@dataclass(frozen=True)
class GordianPair:
    left: GordianEndpoint
    right: GordianEndpoint
    lower: float
    upper: float
    exact: bool
    provenance: str
    split: str


def parse_source_id(source_id: str) -> tuple[tuple[str, bool], ...]:
    """Parse source names such as K31m and K31K41 without guessing."""
    if source_id == "K01":
        return (("0_1", False),)
    position = 0
    factors = []
    while position < len(source_id):
        match = _FACTOR.match(source_id, position)
        if match is None:
            raise ValueError(f"unsupported Gordian source id {source_id!r}")
        crossing, index, mirror = match.groups()
        factors.append((f"{int(crossing)}_{int(index)}", bool(mirror)))
        position = match.end()
    return tuple(factors)


def canonical_knot_id(source_id: str) -> tuple[str, bool] | None:
    """Map a prime source endpoint to a sidecar knot id and mirror transform.

    Mixed-mirror connected sums need an explicit identified representation and
    are deliberately left unmapped. A non-mirrored composite can use an exact
    posting already present in UnknotDB.
    """
    factors = parse_source_id(source_id)
    if len(factors) == 1:
        name, mirror = factors[0]
        return f"knot:{name}", mirror
    if any(mirror for _, mirror in factors):
        return None
    return "knot:" + "#".join(name for name, _ in factors), False


def load_gordian_pairs(path: str | Path, split: str) -> tuple[GordianPair, ...]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"invalid split {split!r}")
    connection = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    metadata = dict(connection.execute("SELECT key,value FROM meta"))
    if metadata.get("schema") != SUPPORTED_SCHEMA:
        connection.close()
        raise ValueError(f"unsupported Gordian schema {metadata.get('schema')!r}")
    postings: dict[str, list[int]] = {}
    for source_id, representation_id in connection.execute(
        "SELECT source_id,representation_id FROM representation_postings "
        "WHERE split=? ORDER BY source_id,rank,representation_id",
        (split,),
    ):
        postings.setdefault(str(source_id), []).append(int(representation_id))
    endpoints = {
        str(source_id): GordianEndpoint(
            str(source_id),
            str(knot_id),
            tuple(postings.get(str(source_id), ())),
            bool(mirror),
            str(endpoint_split),
        )
        for source_id, knot_id, mirror, endpoint_split in connection.execute(
            "SELECT source_id,canonical_knot_id,mirror_transform,split "
            "FROM source_knots WHERE split=? AND mapped=1",
            (split,),
        )
    }
    rows = []
    for left, right, lower, upper, exact, provenance in connection.execute(
        "SELECT left_source_id,right_source_id,distance_lower,distance_upper,"
        "is_exact,provenance FROM distance_pairs WHERE split=? ORDER BY pair_id",
        (split,),
    ):
        left_endpoint = endpoints.get(str(left))
        right_endpoint = endpoints.get(str(right))
        if left_endpoint is None or right_endpoint is None:
            continue
        rows.append(
            GordianPair(
                left_endpoint,
                right_endpoint,
                float(lower),
                float(upper),
                bool(exact),
                str(provenance),
                split,
            )
        )
    connection.close()
    return tuple(rows)


def protected_knot_ids(path: str | Path) -> frozenset[str]:
    """Knot identities reserved wholly for validation/test."""
    connection = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT DISTINCT canonical_knot_id FROM source_knots "
        "WHERE split IN ('validation','test') AND mapped=1"
    )
    result = frozenset(str(row[0]) for row in rows)
    connection.close()
    return result
