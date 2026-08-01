"""Compute the invariants of every knot the ladder climbs, and write them down.

    uv run --with snappy python scripts/rung_invariants.py

Produces `docs/rungs.json` (machine-readable) and `docs/rungs.md` (the table).
`snappy` is optional and supplies only the signature; without it every other
column is still exact, and the committed files were produced with it.

The rung list is read out of `pgx-mcts-bench/src/pgx_mcts_bench/ladder.py` when
that repository is beside this one, and otherwise from the copy already in
`docs/rungs.json`. Parsing the real thing rather than keeping a second list is
deliberate: the ladder's docstring says outright that the count "has changed
twice and the prose did not", and a hand-copied ladder would repeat exactly that
failure.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rf_knots.config import BraidConfig  # noqa: E402
from rf_knots.generator import GradedGenerator  # noqa: E402
from rf_knots.invariants import format_polynomial, invariants  # noqa: E402

LADDER = ROOT.parent / "pgx-mcts-bench" / "src" / "pgx_mcts_bench" / "ladder.py"

# The generator settings the ladder runs with, from its BraidGameConfig. If these
# drift the rung *names* stay the same while the words behind them change, which
# is the one kind of drift that would silently invalidate this whole table -- so
# the words are written into `rungs.json` too, and can be diffed.
POOL = dict(
    max_crossings=22,
    positive_braids=3,
    positive_seed=0,
    random_crossings=(10, 12, 14, 16, 18, 20, 22, 24, 26),
    random_per_grade=1,
    random_seed=0,
)


def read_stages() -> tuple[list[tuple[str, int]], str]:
    """`[(source, scramble), ...]` in ladder order, and where it came from."""
    if LADDER.exists():
        tree = ast.parse(LADDER.read_text())
        for node in tree.body:
            targets = getattr(node, "targets", []) or (
                [node.target] if hasattr(node, "target") else [])
            if any(getattr(t, "id", None) == "STAGES" for t in targets):
                stages = ast.literal_eval(node.value)
                return [(str(a), int(b)) for a, b in stages], str(LADDER)
    previous = ROOT / "docs" / "rungs.json"
    if previous.exists():
        data = json.loads(previous.read_text())
        return [(r["source"], r["scramble"]) for r in data["rungs"]], "docs/rungs.json (cached)"
    raise SystemExit("no ladder.py beside this repo and no cached rungs.json")


def main() -> int:
    stages, origin = read_stages()
    generator = GradedGenerator(BraidConfig(max_len=48, max_strands=5), **POOL)
    sources = {s.name: s for s in generator.sources}
    missing = sorted({name for name, _ in stages} - set(sources))
    if missing:
        raise SystemExit(f"the generator does not produce these rungs: {missing}")

    print(f"{len(stages)} rungs from {origin}", flush=True)
    computed: dict[str, dict] = {}
    for name in dict.fromkeys(name for name, _ in stages):
        source = sources[name]
        inv = invariants(source.word, source.strands)
        computed[name] = {
            "name": name,
            "braid": list(source.word),
            "strands": source.strands,
            "word_length": len(source.word),
            "generator_crossing_number": source.crossing_number,
            "generator_unknotting_number": source.unknotting_number,
            "writhe": inv.writhe,
            "alexander": [list(p) for p in inv.alexander],
            "alexander_pretty": format_polynomial(inv.alexander_polynomial),
            "determinant": inv.determinant,
            "jones": [list(p) for p in inv.jones],
            "jones_pretty": format_polynomial(inv.jones_polynomial),
            "signature": inv.signature,
            "genus_lower": inv.genus_lower,
            "genus_upper": inv.genus_upper,
            "unknotting_lower_bound": inv.unknotting_lower,
            "identified_as": inv.name,
            "identified_crossings": inv.identified_crossings,
            "identified_mirror": inv.mirror,
            "unknotting_number": inv.unknotting,
            "unknotting_upper_bound": inv.unknotting_upper,
            "unknotting_exact": inv.unknotting_known,
            "summands": list(inv.summands),
            "notes": list(inv.notes),
        }
        label = inv.name or (" # ".join(inv.summands) if inv.summands else "unidentified")
        exact = inv.unknotting_known
        print(f"  {name:12} -> {label:24} det={inv.determinant:<5} "
              f"sigma={str(inv.signature):<3} u={exact if exact is not None else '?'}", flush=True)

    payload = {
        "source": origin,
        "pool": {k: list(v) if isinstance(v, tuple) else v for k, v in POOL.items()},
        "rungs": [
            {"index": i, "source": name, "scramble": scramble}
            for i, (name, scramble) in enumerate(stages)
        ],
        "knots": computed,
    }
    (ROOT / "docs" / "rungs.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    (ROOT / "docs" / "rungs-invariants.md").write_text(render(payload))
    print(f"wrote docs/rungs.json and docs/rungs-invariants.md: "
          f"{len(stages)} rungs over {len(computed)} distinct knots")
    return 0


def render(payload: dict) -> str:
    """The generated half of the documentation. Prose lives in `docs/rungs.md`."""
    knots, rungs = payload["knots"], payload["rungs"]
    lines = [
        "# Rung invariants",
        "",
        "Generated by `scripts/rung_invariants.py` -- do not edit by hand. The",
        "narrative, and why the rungs are in this order, is in [rungs.md](rungs.md).",
        "",
        f"{len(rungs)} rungs over {len(knots)} distinct knots. `braid length` is the",
        "generator's word; `c(K)` is the crossing number of the knot it actually",
        "closes to, where the knot could be identified. Where the two differ, the",
        "ladder has been grading itself on the wrong number.",
        "",
        "## The ladder, in order",
        "",
        "| # | source | scramble | knot | c(K) | u |",
        "|---|---|---|---|---|---|",
    ]
    for rung in rungs:
        entry = knots[rung["source"]]
        name = entry["identified_as"] or (" # ".join(entry["summands"]) or "--")
        if entry["identified_as"] and entry["identified_mirror"]:
            name += " (mirror)"
        crossings = entry["identified_crossings"]
        unknotting = entry["unknotting_exact"]
        if unknotting is None and entry["generator_unknotting_number"] >= 0:
            unknotting = f"{entry['generator_unknotting_number']} (theorem)"
        elif unknotting is None and entry["unknotting_upper_bound"] is not None:
            unknotting = f"<= {entry['unknotting_upper_bound']}"
        lines.append(
            f"| {rung['index']} | `{rung['source']}` | +{rung['scramble']} | {name} | "
            f"{crossings if crossings is not None else '--'} | "
            f"{unknotting if unknotting is not None else '?'} |"
        )

    lines += [
        "",
        "## The knots",
        "",
        "| knot | strands | braid length | c(K) | writhe | det | sigma | genus "
        "| u lower | u |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, entry in knots.items():
        genus = (f"{entry['genus_lower']}" if entry["genus_lower"] == entry["genus_upper"]
                 else f"{entry['genus_lower']}-{entry['genus_upper']}")
        identified = (entry["identified_as"]
                      or " # ".join(entry["summands"]) or "--")
        lower = entry["unknotting_lower_bound"]
        crossings = entry["identified_crossings"]
        sigma = entry["signature"]
        exact = entry["unknotting_exact"]
        lines.append(
            f"| `{name}` = {identified} | {entry['strands']} | {entry['word_length']} | "
            f"{crossings if crossings else '--'} | "
            f"{entry['writhe']} | {entry['determinant']} | "
            f"{sigma if sigma is not None else '--'} | {genus} | "
            f"{lower if lower is not None else '--'} | "
            f"{exact if exact is not None else '?'} |"
        )

    lines += ["", "## Polynomials", ""]
    for name, entry in knots.items():
        lines += [
            f"### `{name}`" + (f" = {entry['identified_as']}" if entry["identified_as"] else ""),
            "",
            f"* braid: `{entry['braid']}` on {entry['strands']} strands",
            f"* Alexander: `{entry['alexander_pretty']}`",
            f"* Jones: `{entry['jones_pretty']}`",
        ]
        for note in entry["notes"]:
            lines.append(f"* note: {note}")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
