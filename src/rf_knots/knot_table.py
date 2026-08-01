"""Naming a knot by its invariants, against a bundled table.

The table is `data/knot_table.json`: one entry per tabulated knot up to twelve
crossings -- Rolfsen through ten, then Hoste-Thistlethwaite -- each carrying a
braid word and the fingerprint computed *from that word by this repository's own
code*. That matters more than it sounds. A
transcribed table of published polynomials would be a place to make a silent
mistake, and worse, a place where a convention mismatch -- a mirror, a variable
substitution -- would look like a failed identification rather than a bug. Every
number in the table came out of `rf_knots.invariants`, so a match is a match.

Identification is by `(determinant, jones)`. The Jones polynomial is sensitive to
mirrors, so a knot that matches under `V(t)` is the table's knot and one that
matches under `V(1/t)` is its mirror image, which is reported rather than hidden.
The Alexander polynomial is not part of the key: `det = |Delta(-1)| = |V(-1)|`
already ties the two together, and computing Alexander for a nine-strand braid
costs more than everything else in the table combined. Where the table holds
a published unknotting number it is attached too, with its source recorded in the
file -- those are the only numbers here that this repository did not derive.

A knot past twelve crossings will not be found, and neither will one whose
tabulated braid is wider than the build's strand cap -- those are listed in the
table's `skipped` field. Both are limits of the table, not claims about the knot,
and `identify` says so in `notes` rather than leaving a bare `None` to be misread
as "not a knot".
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

from rf_knots.invariants import Invariants

DATA = Path(__file__).with_name("data")
TABLE_PATH = DATA / "knot_table.json"
UNKNOTTING_PATH = DATA / "unknotting_numbers.json"


@functools.lru_cache(maxsize=1)
def unknotting_numbers() -> dict:
    """Published unknotting numbers, and where they were read from."""
    if not UNKNOTTING_PATH.exists():
        return {"source": "", "values": {}}
    return json.loads(UNKNOTTING_PATH.read_text())


# A knot table starts at the trefoil, because the unknot is not a table entry --
# but the ladder starts at the unknot, and reporting its first two rungs as
# "unidentified" would be false in the most basic way available. It is added
# here rather than to the generated file: nothing about it needs computing.
UNKNOT = {"crossings": 0, "braid": [], "strands": 1, "jones": [[0, 1]], "determinant": 1}


@functools.lru_cache(maxsize=1)
def load_table() -> dict:
    """The bundled table, keyed by fingerprint for both chiralities."""
    if not TABLE_PATH.exists():
        return {"by_fingerprint": {}, "knots": {"0_1": UNKNOT}, "max_crossings": 0}
    raw = json.loads(TABLE_PATH.read_text())
    raw["knots"].setdefault("0_1", UNKNOT)
    index: dict[str, list[tuple[str, bool]]] = {}
    for name, entry in raw["knots"].items():
        # Every candidate is kept, not just the first. 384 fingerprints in this
        # table are shared by more than one knot -- `5_1` and `10_132` have the
        # same Jones polynomial, and so do 857 others -- so a dictionary that
        # kept one name per key would answer those queries with an arbitrary
        # member of the group and look completely confident doing it.
        direct = _key(entry["determinant"], entry["jones"])
        mirrored = _key(entry["determinant"], [[-e, c] for e, c in entry["jones"]])
        index.setdefault(direct, {}).setdefault(name, False)
        # A knot whose Jones polynomial is symmetric -- the unknot, the
        # figure-eight, anything amphichiral -- has the same key both ways round,
        # and registering it twice would report it as ambiguous with itself.
        index.setdefault(mirrored, {}).setdefault(name, mirrored != direct)
    return {"by_fingerprint": {k: sorted(v.items()) for k, v in index.items()},
            "knots": raw["knots"], "max_crossings": raw.get("max_crossings", 0)}


def _key(determinant: int, jones) -> str:
    return f"{determinant}|" + ";".join(f"{int(e)}:{int(c)}" for e, c in sorted(jones))


@functools.lru_cache(maxsize=4096)
def _table_alexander(name: str) -> tuple[tuple[int, int], ...]:
    """The Alexander polynomial of a table knot, computed on demand.

    It is not stored in the table because computing it for every wide braid costs
    more than everything else put together. It is only ever needed to separate
    knots that already share a Jones polynomial, which is a few hundred of them,
    so it is computed here and cached instead.
    """
    from rf_knots.invariants import alexander_polynomial, to_pairs

    entry = load_table()["knots"][name]
    return to_pairs(alexander_polynomial(tuple(entry["braid"]), entry["strands"]))


def identify(inv: Invariants) -> Invariants:
    """Return `inv` with the knot's name filled in, where the table knows it."""
    table = load_table()
    candidates = table["by_fingerprint"].get(_key(inv.determinant, inv.jones))
    if not candidates:
        limit = table["max_crossings"]
        note = (f"not in the bundled table (tabulated knots up to {limit} crossings); "
                "this bounds the table, not the knot")
        return _replace(inv, notes=inv.notes + (note,))

    notes = inv.notes
    if len(candidates) > 1:
        # The Alexander polynomial is canonicalised and so blind to mirroring,
        # which is exactly right here: it is being used to separate genuinely
        # different knots, not the two chiralities of one.
        narrowed = [c for c in candidates if _table_alexander(c[0]) == inv.alexander]
        shared = ", ".join(sorted({name for name, _ in candidates}))
        if len(narrowed) == 1:
            notes = notes + (f"Jones polynomial shared with {shared}; "
                             "separated by the Alexander polynomial",)
        elif narrowed:
            still = ", ".join(sorted({name for name, _ in narrowed}))
            notes = notes + (f"ambiguous: {still} agree on both the Jones and the "
                             "Alexander polynomial, so this names a set, not a knot",)
        else:
            return _replace(inv, notes=notes + (
                f"fingerprint matched {shared} on the Jones polynomial, but none of "
                "them on the Alexander polynomial -- not identified",))
        candidates = narrowed

    # Order by crossing number so an ambiguous answer names the simplest knot
    # first: "5_1 or 10_132" is the useful way round, and reporting only one of
    # them would be a guess wearing the clothes of an identification.
    candidates = sorted(candidates, key=lambda c: (table["knots"][c[0]]["crossings"], c[0]))
    name, mirrored = candidates[0]
    entry = table["knots"][name]
    published = unknotting_numbers()["values"].get(name)
    unique = len(candidates) == 1
    if unique and published is None:
        notes = notes + (f"{name} identified, but its unknotting number has not been "
                         "looked up in data/unknotting_numbers.json",)
    return _replace(
        inv,
        name=name if unique else " or ".join(n for n, _ in candidates),
        identified_crossings=entry["crossings"] if unique else None,
        mirror=mirrored,
        unknotting=published if unique else None,
        notes=notes,
    )


def _replace(inv: Invariants, **changes) -> Invariants:
    import dataclasses

    return dataclasses.replace(inv, **changes)


def lookup(name: str) -> dict | None:
    """The table's entry for a named knot, or `None`."""
    return load_table()["knots"].get(name)
