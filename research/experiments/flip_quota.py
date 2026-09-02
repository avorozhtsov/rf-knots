r"""Does applying a crossing change to a *widened* diagram help at all?

Note 22 §3.6's "one cheap experiment first, before any head".  `strand_growth`
works -- the search can widen, drift is held at 12.1% -- and **no witness
anywhere uses `STABILIZE`**, 139 replay-complete sweep witnesses and 5 on the
target.  §3.6 located the blocking mechanism at one line of `unknot_search`:

    ordered_by_length = sorted(reached, key=lambda item: len(item[0][0]))[:flip_from]

Crossing-change sites are the `flip_from` **shortest** diagrams, and a stabilised
diagram is always longer, so under the policy that actually holds the ratchet a
widened diagram never reaches the stage where a change is applied:
`flip_sites_above_source_width` was **0 of 6,066** on the hard subset and **0 of
7,500** on the target.  That is §Status consequence 3's length-monotone bias
surviving one stage later, as a length-monotone *selection*.

`flip_growth_quota` reserves a fraction of those slots for diagrams above the
source width, filling the rest by length as before; `0.0` is the default and
changes nothing.  This sweeps it, and the three outcomes §3.6 names are all
publishable:

    1. a widened diagram gets flipped and something new is solved
    2. widened diagrams get flipped and nothing improves
    3. the quota cannot be filled -- a frontier problem, not a selection problem

Arms.  The quota only has anything to reserve if widened diagrams reach the
frontier at all, and the two frontier policies differ sharply on that: at
`width_penalty=2` drift is 12.1% (and **0.0%** on the hard subset), at
`width_penalty=0` it is 57.3%.  So the quota is swept on both, because a null
result at `p2` alone could not be told apart from outcome 3.

    p2-q0    growth=2 strand_growth=1 width_penalty=2 quota=0     control
    p2-q10   ...                                      quota=0.10
    p2-q25   ...                                      quota=0.25
    p2-q50   ...                                      quota=0.50
    p0-q0    growth=2 strand_growth=1 width_penalty=0 quota=0     wide-frontier control
    p0-q25   ...                                      quota=0.25

`p2-q0` reproduces `search-growth`'s `growth-on-p2` and `p0-q0` its
`growth-on-p0`, bit for bit; both are re-run rather than quoted so the control
and the arms are measured in the same conditions.

**Fewer expansions is not a win if the answer got worse.**  §3.6's trap was
exactly that: `growth-on-p0` spent 29,325 fewer expansions and returned a
3-change sequence where the control returned a certified-optimal 2.  Every stage
here records `crossing_changes` and `exact` per instance so the paired table can
say so.

Stages:

    --stage project    a handful of instances per arm, and project the sweep
    --stage sweep      all 57 landmark-search instances, six arms
    --stage target     `7_1 # m(7_1)` at cap 5, with cap 6 reported separately

Partial results are flushed after every instance.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from search_growth import (  # noqa: E402  -- same directory, shared machinery
    B5_STRANDS,
    B5_WORD,
    atomic,
    instances,
    run_one,
)

RF = Path(__file__).resolve().parents[2]
OUT = RF / "artifacts/flip-quota"

BASE_P2 = dict(growth=2, strand_growth=1, width_penalty=2)
BASE_P0 = dict(growth=2, strand_growth=1, width_penalty=0)

ARMS = {
    "p2-q0": dict(BASE_P2, flip_growth_quota=0.0),
    "p2-q10": dict(BASE_P2, flip_growth_quota=0.10),
    "p2-q25": dict(BASE_P2, flip_growth_quota=0.25),
    "p2-q50": dict(BASE_P2, flip_growth_quota=0.50),
    "p0-q0": dict(BASE_P0, flip_growth_quota=0.0),
    "p0-q25": dict(BASE_P0, flip_growth_quota=0.25),
    "p0-q50": dict(BASE_P0, flip_growth_quota=0.50),
}

# The target gets the generous dials.  §M1b measured a certified 117-move floor
# just to inflate the 20-letter chart far enough for changes 3-5 to be visible,
# so `strand_growth=1` is not a serious attempt at it; and at `width_penalty=2`
# the target's drift was 0.000, so only the `p0` arms have anything to reserve.
GEN_P2 = dict(growth=6, strand_growth=3, width_penalty=2)
GEN_P0 = dict(growth=6, strand_growth=3, width_penalty=0)
TARGET_ARMS = {
    "generous-p2-q0": dict(GEN_P2, flip_growth_quota=0.0),
    "generous-p2-q25": dict(GEN_P2, flip_growth_quota=0.25),
    "generous-p0-q0": dict(GEN_P0, flip_growth_quota=0.0),
    "generous-p0-q25": dict(GEN_P0, flip_growth_quota=0.25),
    "generous-p0-q50": dict(GEN_P0, flip_growth_quota=0.50),
}

START = time.monotonic()


def log(*parts) -> None:
    print(f"[{time.monotonic() - START:8.1f}s]", *parts, flush=True)


def brief(res: dict) -> str:
    return (f"{res['seconds']:6.1f}s solved={str(res['solved']):5s} "
            f"cc={res.get('crossing_changes')} exact={res.get('exact')} "
            f"exp={res.get('expansions')} "
            f"wideflip={res.get('flip_sites_above_source_width')}"
            f"/{res.get('flip_sites')} "
            f"quota={res.get('flip_quota_filled')}/{res.get('flip_quota_slots')} "
            f"stab={res.get('witness_stabilizations')}")


def enrich(res: dict, report_fields: dict) -> dict:
    res.update(report_fields)
    return res


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
            log(f"  {row['instance_id']:28s} {name:8s} {brief(res)}")
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
                f"{r['arm']}:{'S' if r['solved'] else '.'}"
                f"{r.get('crossing_changes') if r['solved'] else ''}"
                f"/w{r.get('flip_sites_above_source_width', 0)}"
                for r in out[-len(ARMS):]
            )
            + f"  eta {(len(rows) - done) * rate / 60:.1f} min")
    log("sweep done")


def stage_target(args) -> None:
    log("target: 7_1 # m(7_1), 20 letters on B5, cap 5 (and cap 6, not a result)")
    out: list[dict] = []
    for cap in (5, 6):
        for name, arm in TARGET_ARMS.items():
            log(f"  cap {cap} arm {name} starting "
                f"(wall cap {args.target_seconds:.0f}s, budget {args.target_nodes})")
            res = run_one(B5_WORD, B5_STRANDS, cap, arm,
                          node_budget=args.target_nodes,
                          seconds=args.target_seconds)
            out.append({"cap": cap, "arm": name,
                        "node_budget": args.target_nodes, **res})
            atomic(OUT / "target.json", out)
            log(f"  cap {cap} arm {name:18s} {brief(res)} "
                f"drift={res.get('width_drift', 0):.3f} "
                f"timed_out={res['timed_out']}")
    log("target done")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["project", "sweep", "target"])
    parser.add_argument("--node-budget", type=int, default=12_000)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--sample", type=int, default=3)
    parser.add_argument("--target-nodes", type=int, default=1_000_000)
    parser.add_argument("--target-seconds", type=float, default=900.0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"project": stage_project, "sweep": stage_sweep, "target": stage_target}[
        args.stage
    ](args)


if __name__ == "__main__":
    main()
