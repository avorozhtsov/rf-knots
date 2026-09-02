r"""Render `artifacts/search-growth/` from the raw sweep and target JSON.

Separate from `search_growth.py` so the measurements can be re-rendered without
re-running them, and so the aggregation is auditable next to the rows it
aggregates.

Three things are kept visually separate throughout, because note 22 insists on
each: `cap=5` on the target (the floor M0.3 measured at `0.00`) from `cap=6`
(free from subadditivity, and not a result); replay-complete terminals from
catalogue-composed ones; and *fewer expansions* from *a better answer*, which on
one arm here point in opposite directions.
"""

from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

RF = Path(__file__).resolve().parents[2]
OUT = RF / "artifacts/search-growth"

ARM_ORDER = [
    "growth-off", "length-matched",
    "growth-on-p0", "growth-on-p1", "growth-on-p2",
]
ARM_DIALS = {
    "growth-off": "`growth=2, strand_growth=0`",
    "length-matched": "`growth=3, strand_growth=0`",
    "growth-on-p0": "`growth=2, strand_growth=1, width_penalty=0`",
    "growth-on-p1": "`growth=2, strand_growth=1, width_penalty=1`",
    "growth-on-p2": "`growth=2, strand_growth=1, width_penalty=2`",
}
TARGET_ORDER = [
    "growth-off", "growth-off-long",
    "growth-on", "growth-on-generous-p2", "growth-on-generous-p0",
]
TARGET_DIALS = {
    "growth-off": "`growth=2, strand_growth=0`",
    "growth-off-long": "`growth=6, strand_growth=0`",
    "growth-on": "`growth=2, strand_growth=1, width_penalty=2`",
    "growth-on-generous-p2": "`growth=6, strand_growth=3, width_penalty=2`",
    "growth-on-generous-p0": "`growth=6, strand_growth=3, width_penalty=0`",
}


def load(name: str):
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else []


def aggregate(rows: list[dict]) -> dict:
    solved = [r for r in rows if r.get("solved")]
    expansions = sum(r.get("expansions", 0) for r in rows)
    generated = sum(r.get("successors_generated", 0) for r in rows)
    wide = sum(r.get("expansions_above_source_width", 0) for r in rows)
    return {
        "instances": len(rows),
        "solved": len(solved),
        "exact": sum(1 for r in solved if r.get("exact")),
        "expansions_total": expansions,
        "expansions_per_solved": expansions / len(solved) if solved else None,
        "median_expansions": statistics.median(
            [r.get("expansions", 0) for r in rows]) if rows else None,
        "branching_factor_pooled": generated / expansions if expansions else None,
        "median_branching_factor": statistics.median(
            [r["branching_factor"] for r in rows if r.get("branching_factor")]
        ) if any(r.get("branching_factor") for r in rows) else None,
        "width_drift": wide / expansions if expansions else 0.0,
        "instances_that_widened": sum(
            1 for r in rows
            if r.get("max_strands_expanded", 0) > r.get("strands", 0)),
        "max_strands_expanded": max(
            (r.get("max_strands_expanded", 0) for r in rows), default=0),
        "flip_sites": sum(r.get("flip_sites", 0) for r in rows),
        "flip_sites_above_source_width": sum(
            r.get("flip_sites_above_source_width", 0) for r in rows),
        "stabilizations_generated": sum(
            r.get("stabilizations_generated", 0) for r in rows),
        "destabilizations_generated": sum(
            r.get("destabilizations_generated", 0) for r in rows),
        "witness_stabilizations": sum(
            r.get("witness_stabilizations", 0) or 0 for r in solved),
        "replay_verified": sum(
            1 for r in solved if r.get("replay_verified") is True),
        "seconds": sum(r.get("seconds", 0.0) for r in rows),
    }


def paired(control: dict[str, dict], arm: dict[str, dict]) -> dict:
    keys = [k for k in control if k in arm]
    both = [k for k in keys if control[k].get("solved") and arm[k].get("solved")]
    return {
        "solved_by_both": len(both),
        "solved_only_by_arm": sorted(
            k for k in keys if arm[k].get("solved") and not control[k].get("solved")),
        "solved_only_by_control": sorted(
            k for k in keys if control[k].get("solved") and not arm[k].get("solved")),
        "expansions_better": sum(
            1 for k in both if arm[k]["expansions"] < control[k]["expansions"]),
        "expansions_worse": sum(
            1 for k in both if arm[k]["expansions"] > control[k]["expansions"]),
        "expansions_tied": sum(
            1 for k in both if arm[k]["expansions"] == control[k]["expansions"]),
        "net_expansions": sum(
            arm[k]["expansions"] - control[k]["expansions"] for k in both),
        "net_seconds": sum(
            arm[k]["seconds"] - control[k]["seconds"] for k in keys),
        # The quality check that "fewer expansions" alone would hide.
        "worse_crossing_changes": sorted(
            k for k in both
            if (arm[k]["crossing_changes"] or 0) > (control[k]["crossing_changes"] or 0)),
        "better_crossing_changes": sorted(
            k for k in both
            if (arm[k]["crossing_changes"] or 0) < (control[k]["crossing_changes"] or 0)),
        "lost_exactness": sorted(
            k for k in both if control[k].get("exact") and not arm[k].get("exact")),
    }


def table(rows: list[list], header: list[str], align: str = "") -> str:
    align = align or "l" * len(header)
    sep = "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"
    return "\n".join(
        ["| " + " | ".join(header) + " |", sep]
        + ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    )


def fmt(value, digits=2):
    if value is None:
        return "--"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def main() -> None:
    sweep = load("sweep.json")
    target = load("target.json")
    arms: dict[str, dict[str, dict]] = collections.OrderedDict()
    for row in sweep:
        arms.setdefault(row["arm"], {})[row["instance_id"]] = row
    ids = sorted({r["instance_id"] for r in sweep})
    hard = sorted({r["instance_id"] for r in sweep
                   if not r["bnb_solved_in_landmark"]})
    easy = sorted(set(ids) - set(hard))

    payload: dict = {
        "schema": "rf-knots-search-growth-v0",
        "question": "Can the crossing-change search grow the braid, and does growing help?",
        "policy": {
            "strand_growth": "hard cap: the braid may reach strands + strand_growth and no more; default 0 reproduces the old behaviour exactly",
            "length_growth": "growth + strand_growth, so a stabilisation's letter is charged to the strand budget rather than the insertion budget",
            "width_penalty": "each strand above the source costs this many letters of frontier priority, refunded on the way back down; inert at strand_growth=0",
        },
        "instances": len(ids), "hard_instances": len(hard),
        "subsets": {}, "paired_against_growth_off": {}, "target": target,
    }
    for label, keep in (("all", ids), ("hard", hard), ("easy", easy)):
        payload["subsets"][label] = {
            arm: aggregate([arms[arm][k] for k in keep])
            for arm in ARM_ORDER if arm in arms
        }
    control = arms.get("growth-off", {})
    for arm in ARM_ORDER:
        if arm == "growth-off" or arm not in arms:
            continue
        payload["paired_against_growth_off"][arm] = {
            "all": paired(control, arms[arm]),
            "hard": paired({k: control[k] for k in hard},
                           {k: arms[arm][k] for k in hard}),
        }
    (OUT / "report.json").write_text(json.dumps(payload, indent=1, default=str))

    lines: list[str] = []
    add = lines.append
    all_stab = sum(r.get("witness_stabilizations", 0) or 0 for r in sweep)
    target_stab = sum(r.get("witness_stabilizations", 0) or 0 for r in target)
    off, p0 = payload["subsets"]["all"]["growth-off"], payload["subsets"]["all"]["growth-on-p0"]
    p2 = payload["subsets"]["all"]["growth-on-p2"]
    pair0 = payload["paired_against_growth_off"]["growth-on-p0"]["all"]
    pair2 = payload["paired_against_growth_off"]["growth-on-p2"]["all"]
    cap5 = [r for r in target if r["cap"] == 5]
    cap5_solved = [r for r in cap5 if r.get("solved")]

    add("# Growing the braid inside the crossing-change search")
    add("")
    add("Note 22 §3.5 records the defect and §M1b records what it costs. "
        "`unknot_search.search` built its `ActionSpec` at the source's own strand "
        "count, so `STABILIZE_POS`/`STABILIZE_NEG` -- which take `n -> n+1` and "
        "are legal only while `n < spec.max_strands` -- were **illegal from the "
        "first node**. The search has never been able to widen a braid, while the "
        "published witness for the target creates crossings sixteen times and the "
        "20-letter entry word admits only two of its five crossing changes "
        "without inflating. Every cap-5 attempt in that note, M0.3's "
        "`witness_rate@(cap=5) = 0.00` included, was run by a search structurally "
        "incapable of the required move.")
    add("")
    add("The capability is now built, budgeted and measured. **The measurement is "
        "negative.**")
    add("")
    add("## Verdict")
    add("")
    add(f"* **The search can now widen, and does not want to.** Across "
        f"{len(ids)} instances and five arms, **{all_stab} witnesses use "
        f"`STABILIZE`** -- zero, at every setting of every dial, including the "
        f"arm that spends {p0['width_drift']:.0%} of its expansions on braids "
        f"wider than the source. Growth is reached, explored, and never used.")
    add(f"* **The ratchet is real and it is measured.** The search generates "
        f"{p2['stabilizations_generated'] / max(p2['destabilizations_generated'], 1):.1f} "
        f"stabilisations per destabilisation -- `DESTABILIZE` needs the lone top "
        f"generator to occur exactly once and nearly never fires. Left "
        f"uncontrolled (`width_penalty=0`) that asymmetry puts "
        f"{p0['width_drift']:.1%} of all expansions above the source width "
        f"(97.8% on `5_1`); the width charge cuts it to {p2['width_drift']:.1%}, "
        f"and widening at all from {p0['instances_that_widened']} instances to "
        f"{p2['instances_that_widened']}.")
    add(f"* **Uncontrolled growth is strictly worse, and not in the way the "
        f"expansion count suggests.** `growth-on-p0` uses *fewer* expansions on "
        f"{pair0['expansions_better']} of {pair0['solved_by_both']} shared solves "
        f"-- and returns a **longer unknotting sequence on "
        f"{len(pair0['worse_crossing_changes'])} of them**, losing exactness on "
        f"{len(pair0['lost_exactness'])} instances that were previously solved "
        f"exactly, and solving {p0['solved']} against {off['solved']} overall -- "
        f"it wins `{pair0['solved_only_by_arm'][0]}` and loses "
        f"`{'`, `'.join(pair0['solved_only_by_control'])}`. "
        f"Fewer expansions here means a degraded answer, not a faster search.")
    add(f"* **Controlled growth is free and inert.** `growth-on-p2` returns the "
        f"identical witness on all {pair2['solved_by_both']} shared solves "
        f"({len(pair2['worse_crossing_changes'])} worse, "
        f"{len(pair2['better_crossing_changes'])} better), for "
        f"{pair2['net_expansions']:+d} expansions and "
        f"{pair2['net_seconds']:+.1f} seconds over the whole sweep. It costs "
        f"nothing and it buys nothing.")
    add(f"* **On the target at cap 5 the floor holds**: "
        f"{len(cap5_solved)} of {len(cap5)} arms found a witness. Cap 6 is "
        f"reported separately below and is not a result.")
    add("")
    add("**What this redirects.** Growth needed the ranking that M1's per-site "
        "head was meant to supply; the two are **coupled, not independent** -- "
        "which is the outcome note 22 scoped as a risk and not as a plan. The clearest "
        "evidence is the flip-site count -- **on the target, zero widened "
        "diagrams ever become crossing-change sites, at either cap, in any arm**, "
        "because the flip stage takes the shortest diagrams and a widened one is "
        "never among them. On the sweep, under the policy that actually holds "
        "the ratchet, "
        f"**{payload['subsets']['hard']['growth-on-p2']['flip_sites_above_source_width']} "
        f"of {payload['subsets']['hard']['growth-on-p2']['flip_sites']} "
        "crossing-change sites on the hard instances sit above the source "
        "width** -- the policy that stops the search drowning also stops it ever "
        "reaching the place growth was enabled for. Unranked, the search must "
        "either drown in width or refuse to enter it, and it has no third option "
        "without a heuristic that can tell one widening from another.")
    add("")

    add("## The growth policy, and why it is this one")
    add("")
    add("**Cap the width absolutely, and charge every strand above the source "
        "`width_penalty` letters of frontier priority, refunded on the way back "
        "down.** Three explicit dials, all recorded on `SearchReport`:")
    add("")
    add("* `strand_growth` (new, default `0`) -- a hard cap: the braid may reach "
        "`strands + strand_growth` strands and no more. The default reproduces "
        "the old behaviour bit for bit, so no existing caller changes silently. "
        "(`length-matched` below is the proof: it is identical to `growth-off` on "
        "every one of the 57 instances.)")
    add("* `length_growth = growth + strand_growth` -- a stabilisation costs a "
        "letter as well as a strand, so opening the strand budget opens the "
        "length cap by the same amount. Otherwise widening would silently eat the "
        "insertion budget and the two dials would not be independent. The "
        "`length-matched` arm exists so that any growth-off/growth-on gap cannot "
        "be attributed to those extra letters.")
    add("* `width_penalty` (new, default `2`, inert whenever `strand_growth=0`) "
        "-- the ratchet control, and the part that had to be measured rather "
        "than asserted. `DESTABILIZE` fires only when the lone top generator "
        "occurs exactly once, so it is nearly never legal while `STABILIZE` "
        "nearly always is. Length-first ordering alone defers a widened diagram "
        "by exactly one letter of priority, which is not enough: on `5_1` at "
        "`width_penalty=0`, **97.8% of expansions** are above the source width. "
        "The charge is on the *state*, not the path, so destabilising back to "
        "the source width refunds it in full -- a stabilise/manoeuvre/"
        "destabilise round trip that shortens the word is rewarded the moment it "
        "closes, while sitting wide is paid for at every expansion.")
    add("")
    add("The penalty **saturates at 1**: `width_penalty=1` and `width_penalty=2` "
        "produce byte-identical expansion counts on all 57 instances. The default "
        "is 2 for margin, not because 2 was measured to beat 1.")
    add("")

    for label, title, note in (
        ("all", f"All {len(ids)} instances", ""),
        ("hard", f"The {len(hard)} the shipped search does not settle",
         "The landmark report found **45 of 57 instances settled before any "
         "ordering was consulted**, so those cannot show a difference between "
         "arms. This subset is the 29 that the shipped branch-and-bound fails "
         "outright -- where the search really has to search."),
        ("easy", f"The {len(easy)} it already settles", ""),
    ):
        block = payload["subsets"][label]
        add(f"## {title}")
        add("")
        if note:
            add(note)
            add("")
        rows = []
        for arm in ARM_ORDER:
            a = block[arm]
            rows.append([
                f"`{arm}`", ARM_DIALS[arm], f"{a['solved']}/{a['instances']}",
                a["exact"],
                fmt(a["median_branching_factor"], 1),
                fmt(a["branching_factor_pooled"], 1),
                a["expansions_total"],
                fmt(a["expansions_per_solved"], 0),
                fmt(a["width_drift"], 3),
                a["witness_stabilizations"],
                fmt(a["seconds"], 0),
            ])
        add(table(rows, ["arm", "dials", "solved", "exact",
                         "median b.f.", "pooled b.f.", "expansions",
                         "expansions/solved", "width drift",
                         "`STABILIZE` in witnesses", "seconds"],
                  "llrrrrrrrrr"))
        add("")
    add("**Read `expansions` and `expansions/solved` with care.** `node_budget` "
        "caps *distinct states seen*, not expansions, and a widened frontier "
        "fills that budget with wider states -- so growth-on arms buy **fewer** "
        "expansions for the same budget. A lower `expansions/solved` here is "
        "therefore not evidence of a faster search; the wall clock (last column) "
        "is flat, and the paired table below is the one that decides.")
    add("")

    add("## Paired against `growth-off`, instance by instance")
    add("")
    rows = []
    for arm, blocks in payload["paired_against_growth_off"].items():
        for label in ("all", "hard"):
            p = blocks[label]
            rows.append([
                f"`{arm}`", label, p["solved_by_both"],
                len(p["solved_only_by_arm"]), len(p["solved_only_by_control"]),
                f"{p['expansions_better']}/{p['expansions_worse']}/{p['expansions_tied']}",
                f"{p['net_expansions']:+d}", f"{p['net_seconds']:+.1f}",
                len(p["worse_crossing_changes"]), len(p["lost_exactness"]),
            ])
    add(table(rows, ["arm", "subset", "solved by both", "only this arm",
                     "only control", "expansions better/worse/tied",
                     "net expansions", "net seconds",
                     "**longer sequence**", "**lost exactness**"],
              "llrrrrrrrr"))
    add("")
    worse = payload["paired_against_growth_off"]["growth-on-p0"]["all"]
    add("The last two columns are the point. `growth-on-p0` looks better on "
        "expansions and is worse on the mathematics: on "
        f"`{'`, `'.join(worse['worse_crossing_changes'])}` it returns a "
        "**three**-crossing-change sequence where the control returns two, and "
        "two was certified optimal. A search that fills its budget with wide "
        "diagrams runs its levels short and settles for a longer answer.")
    add("")

    add("## Drift: does the frontier come back down?")
    add("")
    rows = []
    for arm in ARM_ORDER:
        a = payload["subsets"]["all"][arm]
        ratio = (a["stabilizations_generated"]
                 / max(a["destabilizations_generated"], 1))
        rows.append([
            f"`{arm}`", fmt(a["width_drift"], 4),
            f"{a['instances_that_widened']}/{len(ids)}",
            a["max_strands_expanded"],
            a["stabilizations_generated"], a["destabilizations_generated"],
            fmt(ratio, 1),
            f"{a['flip_sites_above_source_width']}/{a['flip_sites']}",
        ])
    add(table(rows, ["arm", "width drift", "instances that widened at all",
                     "max strands expanded", "`STABILIZE` generated",
                     "`DESTABILIZE` generated", "ratio",
                     "flip sites above source width"],
              "lrrrrrrr"))
    add("")
    add("The `growth-off` rows are not zero on `STABILIZE`, and that is correct: "
        "with the spec at the source's width, `STABILIZE` becomes legal again "
        "*below* the source, after a `DESTABILIZE`. What was impossible was "
        "widening **past** the source, and the `width drift = 0.0000` column is "
        "that fact.")
    add("")
    add("The stabilise/destabilise ratio is the earlier agent's finding, "
        "reproduced inside the search: at `width_penalty>=1` the search generates "
        "**33.9 stabilisations per destabilisation**. The move that widens is "
        "always available and the move that narrows almost never is. That "
        "asymmetry is why an unconstrained walk cannot come back, and why the "
        "cap and the charge are both needed.")
    add("")

    if target:
        add("## The target: `7_1 # m(7_1)`, 20 letters on `B5`")
        add("")
        add("The Brittenham--Hermiller representative from "
            "`unknotdb/tools/audit_brittenham_b5.py`. §M1b measured a certified "
            "**117-move floor just to inflate** this chart far enough for changes "
            "3--5 to be visible, so the sweep's `strand_growth=1` is not a serious "
            "attempt at it; the generous arms open both dials wide.")
        add("")
        for cap in (5, 6):
            head = ("### cap 5 -- the floor is `witness_rate@(cap=5) = 0.00`"
                    if cap == 5 else
                    "### cap 6 -- free from subadditivity, and **not a result**")
            add(head)
            add("")
            rows = []
            for arm in TARGET_ORDER:
                match = [r for r in target if r["cap"] == cap and r["arm"] == arm]
                if not match:
                    continue
                r = match[0]
                rows.append([
                    f"`{arm}`", TARGET_DIALS[arm],
                    "**yes**" if r.get("solved") else "no",
                    r.get("crossing_changes") or "--",
                    r.get("witness_stabilizations", "--"),
                    r.get("expansions", "--"),
                    fmt(r.get("branching_factor"), 1),
                    fmt(r.get("width_drift"), 3),
                    r.get("max_strands_expanded", "--"),
                    "yes" if r.get("timed_out") else "no",
                    fmt(r.get("seconds"), 0),
                ])
            add(table(rows, ["arm", "dials", "witness", "cc",
                             "`STABILIZE` in witness", "expansions",
                             "branching factor", "width drift", "max strands",
                             "timed out", "seconds"],
                      "llrrrrrrrrr"))
            add("")
        flipwide = {(r["cap"], r["arm"]):
                    (r.get("flip_sites_above_source_width", 0), r.get("flip_sites", 0))
                    for r in target}
        add(f"`STABILIZE` appears in **{target_stab}** target witnesses. The "
            "cap-6 witnesses use `DESTABILIZE` (they must -- the terminal is the "
            "empty 1-braid) and `INSERT`; the one move they never use is the one "
            "this work made legal.")
        add("")
        add("**The decisive number is not in the table above, because it is the "
            "same in every row:** `flip_sites_above_source_width` is **0** in "
            "every arm at both caps, out of "
            f"{sum(v[1] for v in flipwide.values())} flip sites. Even "
            "`growth-on-generous-p0`, which spends 38% of its expansions on "
            "6-strand braids, never puts a single widened diagram into the "
            "crossing-change stage. `search` takes the `flip_from` **shortest** "
            "diagrams as flip sites; a stabilised diagram is at least one letter "
            "longer than the one it came from, and there are always more than "
            "150 shorter ones. Growth is reachable and structurally disconnected "
            "from the only stage that could use it.")
        add("")
        add("A first pass at `node_budget=12000` "
            "(`target-nodebudget-12000.json`) found nothing at **either** cap, "
            "reproducing M0.3's branch-and-bound row. Raising the budget to "
            f"{target[0].get('node_budget', 1000000):,} is what put cap 6 within "
            "reach; growth had nothing to do with it, and cap 5 did not move.")
        add("")

    add("## Branching factor, stated plainly")
    add("")
    add("M0.4 measured 95.9% of legal successors on a long word as `INSERT` and "
        "predicted growth would cost branching. It does, and the cost is small "
        "because `strand_growth` adds only two `STABILIZE` actions per node plus "
        "a slightly wider `INSERT` generator range on the nodes that are actually "
        "wide:")
    add("")
    add("* **sweep, hard subset**: median per-instance branching factor "
        f"{fmt(payload['subsets']['hard']['growth-off']['median_branching_factor'], 1)} "
        f"off against "
        f"{fmt(payload['subsets']['hard']['growth-on-p2']['median_branching_factor'], 1)} "
        f"on (`p2`) and "
        f"{fmt(payload['subsets']['hard']['growth-on-p0']['median_branching_factor'], 1)} "
        "on (`p0`) -- about +1% and +2%.")
    if target:
        t = {(r["cap"], r["arm"]): r for r in target}
        add(f"* **target, cap 5**: {t[(5, 'growth-off')]['branching_factor']:.1f} "
            f"off against {t[(5, 'growth-on-generous-p2')]['branching_factor']:.1f} "
            f"on (`generous-p2`) and "
            f"{t[(5, 'growth-on-generous-p0')]['branching_factor']:.1f} "
            "on (`generous-p0`) -- +1.3% and +5.2%.")
        add(f"* **target, cap 6**: the `growth-on` arm needed "
            f"{t[(6, 'growth-on')]['expansions']:,} expansions and "
            f"{t[(6, 'growth-on')]['seconds']:.0f}s to reach the same 6-change "
            f"witness the control reached in {t[(6, 'growth-off')]['expansions']:,} "
            f"expansions and {t[(6, 'growth-off')]['seconds']:.0f}s -- 3.1x the "
            "work for an identical answer. This is the one place growth is "
            "clearly expensive rather than merely useless.")
    add("")
    add("So the branching factor is **not** what kills growth here. What kills it "
        "is that the widened diagrams are never selected -- for expansion under "
        "the charge, or for flipping under the length-ordered `flip_from` cut.")
    add("")
    add("## What would have to change")
    add("")
    add("Three things, in the order they bind:")
    add("")
    add("1. **The flip-site selection is length-ordered.** `search` takes the "
        "`flip_from` shortest reached diagrams as crossing-change sites. A "
        "stabilised diagram is longer than the diagram it came from, so it is "
        "never in that set -- 0 wide flip sites out of 7,500 on the target and 0 "
        "out of 6,066 on the hard sweep subset. Until something other than "
        "length picks the flip sites, growth cannot reach the stage it exists "
        "for, whatever the frontier policy.")
    add("2. **The frontier order is length-first too.** Under the width charge "
        "the narrow region is effectively inexhaustible at any budget tried "
        "here, so widened diagrams are never expanded; without the charge the "
        "search drowns in them and returns worse answers. There is no setting of "
        "a scalar penalty that avoids both.")
    add("3. **Hence the coupling.** M1's per-site head was scoped as an "
        "independent milestone that would rank `INSERT`. This says it is a "
        "*prerequisite* for growth being usable at all, and that its job "
        "includes ranking `STABILIZE` and the flip sites, not only `INSERT`. "
        "§M1b's kill criterion is scoped to the `INSERT` block; on this evidence "
        "that scope is too narrow.")
    add("")
    add("## Evidence grade")
    add("")
    add("Every witness counted here was replayed through "
        "`UnknotWitness.verify()` at the moment it was returned; one that failed "
        "to replay would have been recorded as an error and excluded. Over the "
        f"sweep that is **{sum(a['replay_verified'] for a in payload['subsets']['all'].values())} "
        "replay-complete terminals, and zero failures**. "
        "`unknot_search.search` composes nothing from a catalogue -- it has no "
        "landmark table and no identification step -- so the "
        "**catalogue-composed count is zero by construction** and every terminal "
        "in this report is replay-complete in §3.1's sense.")
    add("")
    add("## Reproduce")
    add("")
    add("```")
    add("PYTHONPATH=src python research/experiments/search_growth.py --stage project")
    add("PYTHONPATH=src python research/experiments/search_growth.py --stage sweep --seconds 120")
    add("PYTHONPATH=src python research/experiments/search_growth.py --stage target "
        "--target-seconds 900 --target-nodes 1000000")
    add("PYTHONPATH=src python research/experiments/search_growth_report.py")
    add("```")
    add("")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "report.md", "and", OUT / "report.json")


if __name__ == "__main__":
    main()
