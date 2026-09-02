r"""Can the crossing-change search grow the braid, and does growing help?

Note 22 §2.3 and §3.5 record the defect: `unknot_search.search` built its
`ActionSpec` at the source's own strand count, so `STABILIZE_POS`/`STABILIZE_NEG`
-- which take `n -> n+1` and are legal only while `n < spec.max_strands` -- were
illegal from the first node.  The search has never been able to widen a braid.
§M1b measured what that costs: the published witness for `7_1 # m(7_1)` creates
crossings sixteen times, and the 20-letter entry word admits only two of its five
crossing changes without inflating.  So M0.3's `witness_rate@(cap=5) = 0.00` was
measured by a search structurally incapable of the move the target requires.

This measures the fix.  Four arms, everything but the growth dials held
identical -- same instances, same node budget, same fingerprint, same certified
pruning, same terminal test, same beam widths:

    growth-off        growth=2  strand_growth=0                 today's search
    length-matched    growth=3  strand_growth=0                 the extra letter, no extra strand
    growth-on-p0      growth=2  strand_growth=1  width_penalty=0  growth, ratchet uncontrolled
    growth-on-p1      growth=2  strand_growth=1  width_penalty=1  growth, ratchet charged lightly
    growth-on-p2      growth=2  strand_growth=1  width_penalty=2  growth, ratchet charged

`length-matched` exists because opening the strand budget also opens the length
cap by one (a stabilisation costs a letter as well as a strand).  Without it, a
`growth-off` vs `growth-on` gap could be the extra letter rather than the extra
strand, and the comparison would be worthless.

Stages:

    --stage project    time a handful of instances per arm and project the sweep
    --stage sweep      all 57 landmark-search instances, four arms
    --stage target     `7_1 # m(7_1)` from the 20-letter B5 word, cap 5 and cap 6

Partial results are flushed after every instance, because there is no `timeout`
on this machine and a killed run must still leave its measurements behind.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from dataclasses import asdict
from pathlib import Path

from rf_knots.unknot_search import search

RF = Path(__file__).resolve().parents[2]
LANDMARK = RF / "artifacts/landmark-search/report.json"
OUT = RF / "artifacts/search-growth"

# The 20-letter B5 representative of 7_1 # m(7_1), from
# unknotdb/tools/audit_brittenham_b5.py (Brittenham-Hermiller).
B5_WORD = (1, -4, 2, 3, 3, 3, 2, 3, 2, 2, 4, -3, -3, -3, -3, -1, -3, -2, -3, -3)
B5_STRANDS = 5

ARMS = {
    "growth-off": dict(growth=2, strand_growth=0, width_penalty=0),
    "length-matched": dict(growth=3, strand_growth=0, width_penalty=0),
    "growth-on-p0": dict(growth=2, strand_growth=1, width_penalty=0),
    "growth-on-p1": dict(growth=2, strand_growth=1, width_penalty=1),
    "growth-on-p2": dict(growth=2, strand_growth=1, width_penalty=2),
}

# The target gets its own arms.  §M1b measured a certified **117-move floor just
# to inflate** the 20-letter chart far enough for changes 3-5 to be visible, so
# `strand_growth=1` with two letters of length headroom -- the sweep's setting --
# is not a serious attempt at it.  The generous arms open both dials wide, and the
# length-only control is there so that any difference is attributable to width
# rather than to the extra letters that come with it.
TARGET_ARMS = {
    "growth-off": dict(growth=2, strand_growth=0, width_penalty=0),
    "growth-off-long": dict(growth=6, strand_growth=0, width_penalty=0),
    "growth-on": dict(growth=2, strand_growth=1, width_penalty=2),
    "growth-on-generous-p2": dict(growth=6, strand_growth=3, width_penalty=2),
    "growth-on-generous-p0": dict(growth=6, strand_growth=3, width_penalty=0),
}

START = time.monotonic()
_LAST = START


def log(*parts) -> None:
    print(f"[{time.monotonic() - START:8.1f}s]", *parts, flush=True)


def heartbeat(message: str) -> None:
    global _LAST
    now = time.monotonic()
    if now - _LAST >= 15.0:
        _LAST = now
        log(message)


class Timeout(Exception):
    pass


def _raise(_signum, _frame):
    raise Timeout


def run_one(word, strands, cap, arm: dict, *, node_budget: int, seconds: float) -> dict:
    """One `search` call under a wall-clock cap, with the counters extracted.

    The cap is a `SIGALRM`; the search has no internal clock and a hard instance
    on the widened action space will not return on its own.
    """
    started = time.monotonic()
    signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    timed_out = False
    error = None
    report = None
    try:
        report = search(
            tuple(word), int(strands),
            max_crossing_changes=int(cap),
            node_budget=node_budget,
            **arm,
        )
    except Timeout:
        timed_out = True
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    elapsed = time.monotonic() - started

    row = {
        "seconds": elapsed,
        "timed_out": timed_out,
        "error": error,
        "solved": False,
    }
    if report is None:
        return row
    witness = report.witness
    replayed = None
    if witness is not None:
        try:
            witness.verify()
            replayed = True
        except Exception as exc:  # noqa: BLE001
            replayed = f"{type(exc).__name__}: {exc}"
    row.update(
        solved=witness is not None,
        replay_verified=replayed,
        crossing_changes=report.crossing_changes,
        certified_lower_bound=report.lower_bound,
        exact=report.exact,
        moves=witness.moves if witness is not None else None,
        witness_stabilizations=report.witness_stabilizations,
        diagrams_explored=report.diagrams_explored,
        knots_expanded=report.knots_expanded,
        pruned_by_bound=report.pruned_by_bound,
        expansions=report.expansions,
        successors_generated=report.successors_generated,
        successors_admitted=report.successors_admitted,
        branching_factor=report.branching_factor,
        admitted_branching_factor=report.admitted_branching_factor,
        stabilizations_generated=report.stabilizations_generated,
        destabilizations_generated=report.destabilizations_generated,
        max_strands_expanded=report.max_strands_expanded,
        expansions_above_source_width=report.expansions_above_source_width,
        width_drift=report.width_drift,
        flip_sites=report.flip_sites,
        flip_sites_above_source_width=report.flip_sites_above_source_width,
        growth=report.growth,
        strand_growth=report.strand_growth,
        length_growth=report.length_growth,
        width_penalty=report.width_penalty,
        strand_cap=report.strand_cap,
        notes=list(report.notes),
    )
    if witness is not None:
        row["witness_moves_flat"] = [
            step.action.kind for step in witness.steps
        ]
    return row


def instances() -> list[dict]:
    payload = json.loads(LANDMARK.read_text())
    rows = []
    for item in payload["instances"]:
        rows.append({
            "instance_id": item["instance_id"],
            "family": item["family"],
            "word": list(item["word"]),
            "strands": int(item["strands"]),
            "cap": int(item["cap"]),
            "known_u": int(item["known_u"]),
            "bnb_solved_in_landmark": bool(
                item["branch_and_bound_control"].get("solved")
            ),
        })
    return rows


def atomic(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    tmp.replace(path)


def stage_project(args) -> None:
    rows = instances()
    hard = [r for r in rows if not r["bnb_solved_in_landmark"]]
    easy = [r for r in rows if r["bnb_solved_in_landmark"]]
    sample = hard[: args.sample] + easy[: args.sample]
    log(f"projection on {len(sample)} instances x {len(ARMS)} arms")
    out = []
    for row in sample:
        for name, arm in ARMS.items():
            res = run_one(row["word"], row["strands"], row["cap"], arm,
                          node_budget=args.node_budget, seconds=args.seconds)
            out.append({"instance_id": row["instance_id"], "arm": name, **res})
            log(f"  {row['instance_id']:28s} {name:16s} "
                f"{res['seconds']:7.1f}s solved={res['solved']} "
                f"exp={res.get('expansions')} bf={res.get('branching_factor', 0):.1f} "
                f"drift={res.get('width_drift', 0):.3f}")
            atomic(OUT / "projection.json", out)
    by_arm: dict[str, float] = {}
    for item in out:
        by_arm[item["arm"]] = by_arm.get(item["arm"], 0.0) + item["seconds"]
    n = len(sample)
    total = sum(v / n * len(rows) for v in by_arm.values())
    log("mean seconds/instance by arm: "
        + ", ".join(f"{k}={v / n:.1f}" for k, v in by_arm.items()))
    log(f"PROJECTED full sweep ({len(rows)} instances x {len(ARMS)} arms): "
        f"{total / 60:.1f} min")


def stage_sweep(args) -> None:
    rows = instances()
    log(f"sweep: {len(rows)} instances x {len(ARMS)} arms, "
        f"node_budget={args.node_budget}, cap {args.seconds:.0f}s/instance")
    out: list[dict] = []
    for index, row in enumerate(rows):
        for name, arm in ARMS.items():
            res = run_one(row["word"], row["strands"], row["cap"], arm,
                          node_budget=args.node_budget, seconds=args.seconds)
            out.append({
                "instance_id": row["instance_id"], "family": row["family"],
                "cap": row["cap"], "strands": row["strands"],
                "letters": len(row["word"]),
                "bnb_solved_in_landmark": row["bnb_solved_in_landmark"],
                "arm": name, **res,
            })
            atomic(OUT / "sweep.json", out)
        done = index + 1
        rate = (time.monotonic() - START) / done
        log(f"[{done}/{len(rows)}] {row['instance_id']:28s} "
            + " ".join(
                f"{r['arm'].split('-')[-1]}:{'S' if r['solved'] else '.'}"
                f"{r.get('expansions', 0)}"
                for r in out[-len(ARMS):]
            )
            + f"  eta {(len(rows) - done) * rate / 60:.1f} min")
    log("sweep done")


def stage_target(args) -> None:
    log("target: 7_1 # m(7_1), 20 letters on B5, caps 5 and 6")
    out: list[dict] = []
    for cap in (5, 6):
        for name, arm in TARGET_ARMS.items():
            log(f"  cap {cap} arm {name} starting "
                f"(cap {args.target_seconds:.0f}s, budget {args.target_nodes})")
            res = run_one(B5_WORD, B5_STRANDS, cap, arm,
                          node_budget=args.target_nodes,
                          seconds=args.target_seconds)
            out.append({"cap": cap, "arm": name,
                        "node_budget": args.target_nodes, **res})
            atomic(OUT / "target.json", out)
            atomic(OUT / f"target-nodebudget-{args.target_nodes}.json", out)
            log(f"  cap {cap} arm {name:16s} {res['seconds']:7.1f}s "
                f"solved={res['solved']} timed_out={res['timed_out']} "
                f"exp={res.get('expansions')} bf={res.get('branching_factor', 0):.1f} "
                f"drift={res.get('width_drift', 0):.3f} "
                f"stab_in_witness={res.get('witness_stabilizations')}")
    log("target done")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["project", "sweep", "target"])
    parser.add_argument("--node-budget", type=int, default=12_000)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--sample", type=int, default=3)
    parser.add_argument("--target-nodes", type=int, default=12_000)
    parser.add_argument("--target-seconds", type=float, default=600.0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"project": stage_project, "sweep": stage_sweep, "target": stage_target}[
        args.stage
    ](args)


if __name__ == "__main__":
    main()
