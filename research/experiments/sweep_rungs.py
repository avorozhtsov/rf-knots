r"""Run the certified branch-and-bound over every rung with a standing record.

Compares three numbers per knot:

* **bound** -- the strongest certified lower bound (`max` of `|sigma|/2`, `|tau|`,
  and the Montesinos `u >= 2` obstruction). A theorem.
* **record** -- the fewest crossing changes any trained agent has used, from the
  ratchet in `pgx-mcts-bench/artifacts/bounds.jsonl`. An upper bound with no
  replayable witness.
* **found** -- what the search returns here, with a witness that replays.

`found == bound` means `u` is **determined**, not bounded. `found < record` means
the ratchet moves. Every improvement is written out as a verified claim.

The search budget is capped at the standing record: there is no point looking for
a sequence longer than one somebody already has. Where the cap binds the run is
reported as `capped`, so an absent improvement is never confused with a proof that
none exists -- the search is a beam and never claims completeness.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rf_knots import verified_bounds as vb
from rf_knots.invariants import invariants
from rf_knots.unknot_search import certified_lower_bound, search

AGENT = "branch-and-bound (rf_knots.unknot_search)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rungs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-budget", type=int, default=40_000)
    parser.add_argument("--frontier", type=int, default=24)
    parser.add_argument("--flip-from", type=int, default=150)
    parser.add_argument("--growth", type=int, default=3)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "run.log"
    claims = args.output / "bounds-verified.jsonl"

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        with log_path.open("a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    rungs = json.loads(args.rungs.read_text())
    log(f"{len(rungs)} rungs, node budget {args.node_budget}")
    rows = []
    for rung in rungs:
        word, strands, record = tuple(rung["word"]), rung["strands"], rung["record"]
        if not word:
            continue
        inv = invariants(word, strands)
        bound = certified_lower_bound(word, strands)
        started = time.perf_counter()
        report = search(
            word, strands,
            max_crossing_changes=record,
            node_budget=args.node_budget,
            growth=args.growth,
            frontier_width=args.frontier,
            flip_from=args.flip_from,
        )
        took = time.perf_counter() - started
        found = report.crossing_changes
        row = {
            "knot": inv.name or "unnamed",
            "word": list(word), "strands": strands,
            "length": len(word), "record": record, "bound": bound,
            "found": found, "exact": bool(report.solved_exactly),
            "improved": found is not None and found < record,
            "capped": found is None,
            "seconds": took,
        }
        if report.witness is not None:
            report.witness.verify()
            vb.claim(claims, vb.from_witness(
                report.witness, agent=AGENT, lower_bound=bound,
                note=f"ratchet record was {record}"))
            row["moves"] = report.witness.moves
        rows.append(row)
        flag = ("EXACT" if row["exact"] else "improved") if row["improved"] else (
            "exact" if row["exact"] else "--")
        log(f"  {(inv.name or 'unnamed'):10s} n={strands} len={len(word):2d} "
            f"bound={bound} record={record:2d} found={str(found):>4s} {flag:8s} "
            f"{took:5.0f}s")
        (args.output / "rows.json").write_text(json.dumps(rows, indent=2))

    (args.output / "report.md").write_text(render(rows))
    improved = sum(r["improved"] for r in rows)
    exact = sum(r["exact"] for r in rows)
    log(f"done: {improved}/{len(rows)} improved, {exact}/{len(rows)} determined exactly")


def render(rows) -> str:
    lines = [
        "# Certified branch-and-bound over the ladder's rungs",
        "",
        "`bound` is the strongest certified lower bound and is a theorem.",
        "`record` is the ratchet's standing upper bound, from a trained agent, with",
        "no replayable witness. `found` is this search, with one. **exact** means",
        "`found == bound`, so `u` is determined rather than bounded.",
        "",
        "| knot | n | len | bound | record | found | verdict |",
        "|---|--:|--:|--:|--:|--:|---|",
    ]
    for row in sorted(rows, key=lambda r: (-int(r["improved"]), r["knot"])):
        verdict = []
        if row["improved"]:
            verdict.append(f"**improved by {row['record'] - row['found']}**")
        if row["exact"]:
            verdict.append("**u determined**")
        if row["capped"]:
            verdict.append("no sequence within budget")
        lines.append(
            f"| `{row['knot']}` | {row['strands']} | {row['length']} | {row['bound']} | "
            f"{row['record']} | {row['found'] if row['found'] is not None else '--'} | "
            f"{', '.join(verdict) or 'matches record'} |"
        )
    improved = sum(r["improved"] for r in rows)
    exact = sum(r["exact"] for r in rows)
    lines += [
        "",
        f"{len(rows)} rungs: **{improved} improved**, **{exact} with `u` determined**.",
        "",
        "The search is a beam capped at the standing record, so a rung with no",
        "improvement is not evidence that none exists.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
