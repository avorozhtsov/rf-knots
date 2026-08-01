"""Build `src/rf_knots/data/knot_table.json`, the table `knot_table.identify` reads.

Run it with spherogram available; the result is committed, so nothing at runtime
needs spherogram:

    uv run --with snappy --with-editable . python scripts/build_knot_table.py

`spherogram` supplies one thing only -- a braid word for each tabulated knot.
Every invariant in the table is then computed from that word by
`rf_knots.invariants`, so the table and the thing being identified are measured
with the same instrument. A published table of polynomials would instead make a
convention mismatch look like a failed identification.

Coverage is the Rolfsen table (up to 10 crossings) plus Hoste-Thistlethwaite 11
and 12 crossings. Enumeration walks the index until a name stops resolving
rather than hard-coding how many knots there are at each crossing number, which
is one fewer number to get wrong.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rf_knots.invariants import jones_polynomial, to_pairs  # noqa: E402

KNOTINFO = {
    "name": "KnotInfo",
    "url": "https://knotinfo.org/",
    "database_url": "https://knotinfo.org/knotinfo_data_complete.xls",
    "retrieved_at": "2026-08-01",
    "sha256": "bd454dcb6bcd5effe205b27ca9de172bb21cf87ce190e15f870e1b07a714ccbe",
    "scope": ("identifier correspondence only; braid representatives come from Spherogram "
              "and invariants are computed by rf-knots"),
}

# Published unknotting numbers deliberately live in
# `src/rf_knots/data/unknotting_numbers.json` rather than in this table. They are
# the only numbers in the package that were read off someone else's table rather
# than derived here, and keeping them separate means adding one costs an edit
# instead of a half-hour rebuild of three thousand fingerprints.


def rolfsen_names():
    counts = {3: 1, 4: 1, 5: 2, 6: 3, 7: 7, 8: 21, 9: 49, 10: 165}
    for crossings, total in counts.items():
        for index in range(1, total + 1):
            yield f"{crossings}_{index}", crossings


def census_names(spherogram):
    """`K11a1`, `K11n1`, `K12a1`, `K12n1`, ... until the name stops resolving."""
    for crossings in (11, 12):
        for kind in ("a", "n"):
            index = 1
            while True:
                name = f"K{crossings}{kind}{index}"
                try:
                    spherogram.Link(name)
                except Exception:
                    break
                yield name, crossings
                index += 1


def knotinfo_name(spherogram_name: str) -> str:
    """Convert Spherogram's `K12n570` spelling to KnotInfo's `12n_570`."""
    match = re.fullmatch(r"K(\d+)([an])(\d+)", spherogram_name)
    if match is None:
        return spherogram_name
    crossings, kind, index = match.groups()
    return f"{crossings}{kind}_{index}"


# The Kauffman bracket is evaluated in TL_n, whose dimension is Catalan(n): 429
# at 7 strands, 1430 at 8, 4862 at 9, 208012 at 12. The hundred knots whose
# spherogram braid needs nine strands cost more than the other 2870 put
# together, so the cap stops at eight: 96% of the table for a fifth of the time.
# Every knot skipped is named in the table's metadata, because a knot missing
# from the table is otherwise indistinguishable from a knot that was never
# tabulated at all.
MAX_STRANDS = 8


def main() -> int:
    import spherogram

    out = {
        "schema_version": 2,
        "canonical_names": "KnotInfo",
        "catalogue_provenance": KNOTINFO,
        "identifier_sources": {},
        "knots": {},
        "max_crossings": 12,
        "skipped": [],
    }
    started = time.time()
    failures = []
    names = list(rolfsen_names()) + list(census_names(spherogram))
    print(f"{len(names)} knots to fingerprint", flush=True)

    for position, (spherogram_name, crossings) in enumerate(names, start=1):
        name = knotinfo_name(spherogram_name)
        out["identifier_sources"][name] = {"KnotInfo": name, "Spherogram": spherogram_name}
        try:
            word = tuple(int(x) for x in spherogram.Link(spherogram_name).braid_word())
            strands = max((abs(x) for x in word), default=0) + 1
            if strands > MAX_STRANDS:
                out["skipped"].append({"name": name, "strands": strands})
                continue
            # The Alexander polynomial is deliberately not stored. It costs far
            # more than the Jones polynomial on a wide braid -- the reduced Burau
            # entries reach the degree of the whole word, so evaluating them is
            # arithmetic on thousand-bit integers -- and it buys nothing here,
            # because `det(K) = |Delta(-1)| = |V(-1)|`. The determinant below is
            # read straight off the Jones polynomial, and a query fingerprint
            # computed the other way round agrees with it by that identity.
            jones = jones_polynomial(word, strands)
            out["knots"][name] = {
                "crossings": crossings,
                "braid": list(word),
                "strands": strands,
                "jones": [list(p) for p in to_pairs(jones)],
                # `(-1) ** e` is a *float* for negative e, and Jones exponents
                # are routinely negative -- which silently made every such
                # determinant 17.0 rather than 17, and no key ever matched.
                "determinant": abs(sum(c * (-1) ** (e % 2) for e, c in jones.items())),
            }
        except Exception as exc:                          # noqa: BLE001
            failures.append((name, f"{type(exc).__name__}: {exc}"))
        if position % 250 == 0:
            print(f"  {position}/{len(names)}  {time.time() - started:.0f}s", flush=True)

    # Collisions are not errors -- mutants share fingerprints -- but they are the
    # thing that makes an identification ambiguous, so they get counted out loud.
    seen: dict[tuple, list[str]] = {}
    for name, entry in out["knots"].items():
        key = (entry["determinant"], json.dumps(entry["jones"]))
        seen.setdefault(key, []).append(name)
    collisions = {k: v for k, v in seen.items() if len(v) > 1}

    target = Path(__file__).resolve().parents[1] / "src" / "rf_knots" / "data"
    target.mkdir(parents=True, exist_ok=True)
    (target / "knot_table.json").write_text(json.dumps(out, separators=(",", ":"), sort_keys=True))

    print(f"wrote {len(out['knots'])} knots in {time.time() - started:.0f}s")
    print(f"fingerprint collisions: {len(collisions)} groups covering "
          f"{sum(len(v) for v in collisions.values())} knots")
    for group in list(collisions.values())[:10]:
        print("   ", " = ".join(group))
    if failures:
        print(f"{len(failures)} failed:")
        for name, why in failures[:10]:
            print("   ", name, why)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
