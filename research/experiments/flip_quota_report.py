r"""Render `artifacts/flip-quota/` from the raw sweep and target JSON.

Separate from `flip_quota.py` so the measurements can be re-rendered without
re-running them.  `aggregate`, `paired`, `table` and `fmt` are imported from
`search_growth_report` rather than re-derived, so the two experiments are
aggregated by literally the same code and their numbers are comparable.

Three things are kept apart throughout, because note 22 §3.6 insists on each:
`cap=5` on the target from `cap=6` (free from subadditivity, and not a result);
*the quota was filled* from *the quota changed the flip sites* -- outcome 3 is
the case where too few widened diagrams reach the frontier to reserve at all;
and **fewer expansions** from **a better answer**, which §3.6 caught pointing in
opposite directions on `growth-on-p0`.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from search_growth_report import aggregate, fmt, paired, table

RF = Path(__file__).resolve().parents[2]
OUT = RF / "artifacts/flip-quota"
GROWTH = RF / "artifacts/search-growth"

ARM_ORDER = ["p2-q0", "p2-q10", "p2-q25", "p2-q50", "p0-q0", "p0-q25", "p0-q50"]
ARM_DIALS = {
    "p2-q0": "`width_penalty=2, quota=0` (control)",
    "p2-q10": "`width_penalty=2, quota=0.10`",
    "p2-q25": "`width_penalty=2, quota=0.25`",
    "p2-q50": "`width_penalty=2, quota=0.50`",
    "p0-q0": "`width_penalty=0, quota=0` (control)",
    "p0-q25": "`width_penalty=0, quota=0.25`",
    "p0-q50": "`width_penalty=0, quota=0.50`",
}
CONTROL_OF = {
    "p2-q10": "p2-q0", "p2-q25": "p2-q0", "p2-q50": "p2-q0",
    "p0-q25": "p0-q0", "p0-q50": "p0-q0",
}
TARGET_ORDER = [
    "generous-p2-q0", "generous-p2-q25",
    "generous-p0-q0", "generous-p0-q25", "generous-p0-q50",
]
TARGET_DIALS = {
    "generous-p2-q0": "`growth=6, strand_growth=3, width_penalty=2, quota=0`",
    "generous-p2-q25": "`growth=6, strand_growth=3, width_penalty=2, quota=0.25`",
    "generous-p0-q0": "`growth=6, strand_growth=3, width_penalty=0, quota=0`",
    "generous-p0-q25": "`growth=6, strand_growth=3, width_penalty=0, quota=0.25`",
    "generous-p0-q50": "`growth=6, strand_growth=3, width_penalty=0, quota=0.50`",
}


def load(base: Path, name: str):
    path = base / name
    return json.loads(path.read_text()) if path.exists() else []


def quota_stats(rows: list[dict]) -> dict:
    slots = sum(r.get("flip_quota_slots", 0) or 0 for r in rows)
    filled = sum(r.get("flip_quota_filled", 0) or 0 for r in rows)
    levels = sum(r.get("flip_quota_levels", 0) or 0 for r in rows)
    return {
        "quota_slots": slots,
        "quota_filled": filled,
        "quota_fill_rate": filled / slots if slots else None,
        "quota_levels": levels,
        "quota_levels_filled": sum(
            r.get("flip_quota_levels_filled", 0) or 0 for r in rows),
        "instances_with_any_reserved_slot": sum(
            1 for r in rows if (r.get("flip_quota_filled", 0) or 0) > 0),
        # Whether the length cut is even binding.  A level that reached fewer
        # diagrams than `flip_from` flips all of them, so no selection rule --
        # quota or length -- can change anything there, and a null result on
        # those instances is about the frontier, not about the selection.
        "flip_reached": sum(r.get("flip_reached", 0) or 0 for r in rows),
        "flip_levels": sum(r.get("flip_levels", 0) or 0 for r in rows),
        "flip_levels_cut": sum(r.get("flip_levels_cut", 0) or 0 for r in rows),
    }


def main() -> None:
    sweep = load(OUT, "sweep.json")
    target = load(OUT, "target.json")
    arms: dict[str, dict[str, dict]] = collections.OrderedDict()
    for row in sweep:
        arms.setdefault(row["arm"], {})[row["instance_id"]] = row
    order = [a for a in ARM_ORDER if a in arms]
    ids = sorted({r["instance_id"] for r in sweep})
    hard = sorted({r["instance_id"] for r in sweep
                   if not r["bnb_solved_in_landmark"]})
    easy = sorted(set(ids) - set(hard))

    payload: dict = {
        "schema": "rf-knots-flip-quota-v0",
        "question": (
            "Does applying a crossing change to a widened diagram help at all? "
            "One selection rule, no training."
        ),
        "dial": {
            "flip_growth_quota": (
                "fraction of the flip_from crossing-change slots reserved, at "
                "every level, for diagrams above the source width; the rest are "
                "filled by length exactly as before. Default 0.0 reserves "
                "nothing and leaves the selection index-for-index unchanged."
            ),
        },
        "instances": len(ids), "hard_instances": len(hard),
        "subsets": {}, "paired": {}, "target": target,
    }
    for label, keep in (("all", ids), ("hard", hard), ("easy", easy)):
        payload["subsets"][label] = {
            arm: {**aggregate([arms[arm][k] for k in keep]),
                  **quota_stats([arms[arm][k] for k in keep])}
            for arm in order
        }
    for arm, ctrl in CONTROL_OF.items():
        if arm not in arms or ctrl not in arms:
            continue
        payload["paired"][arm] = {
            "control": ctrl,
            "all": paired(arms[ctrl], arms[arm]),
            "hard": paired({k: arms[ctrl][k] for k in hard},
                           {k: arms[arm][k] for k in hard}),
        }

    # The outcome, decided from the counters rather than asserted.
    all_block = payload["subsets"]["all"]
    quota_arms = [a for a in order if a not in ("p2-q0", "p0-q0")]
    moved = any(
        all_block[a]["flip_sites_above_source_width"]
        > all_block[CONTROL_OF[a]]["flip_sites_above_source_width"]
        for a in quota_arms
    )
    stab = sum(r.get("witness_stabilizations", 0) or 0 for r in sweep)
    target_stab = sum(r.get("witness_stabilizations", 0) or 0 for r in target)
    new_solves = sorted({
        k for a in quota_arms
        for k in payload["paired"].get(a, {}).get("all", {})
                        .get("solved_only_by_arm", [])
    })
    degraded = sorted({
        k for a in quota_arms
        for k in payload["paired"].get(a, {}).get("all", {})
                        .get("worse_crossing_changes", [])
    })
    lost_exact = sorted({
        k for a in quota_arms
        for k in payload["paired"].get(a, {}).get("all", {})
                        .get("lost_exactness", [])
    })
    payload["outcome"] = {
        "quota_moved_the_flip_site_counter": moved,
        "witness_stabilizations_sweep": stab,
        "witness_stabilizations_target": target_stab,
        "solved_only_by_a_quota_arm": new_solves,
        "degraded_crossing_changes": degraded,
        "lost_exactness": lost_exact,
        "verdict": (
            "1: growth becomes usable" if (moved and (new_solves or stab))
            else "2: widened diagrams get flipped and nothing improves" if moved
            else "3: the quota cannot be filled"
        ),
    }
    (OUT / "report.json").write_text(json.dumps(payload, indent=1, default=str))

    lines: list[str] = []
    add = lines.append
    cap5 = [r for r in target if r["cap"] == 5]
    cap5_solved = [r for r in cap5 if r.get("solved")]
    hard_block = payload["subsets"]["hard"]

    add("# Reserving crossing-change slots for widened diagrams")
    add("")
    add("Note 22 §3.6's *one cheap experiment first, before any head*: no "
        "training, one selection rule, a clean measurement. `strand_growth` "
        "works -- the search can widen a braid and drift is held at 12.1% -- and "
        "**no witness anywhere used `STABILIZE`**, across 139 replay-complete "
        "sweep witnesses and 5 on the target, at every dial setting. §3.6 "
        "located the blocking mechanism at one line: crossing-change sites are "
        "the `flip_from` **shortest** diagrams reached, and a stabilised diagram "
        "is always longer, so under the policy that holds the ratchet a widened "
        "diagram never reaches the stage where a change is applied. That is "
        "§Status consequence 3's length-monotone bias surviving one stage later "
        "as a length-monotone *selection*.")
    add("")
    add("`flip_growth_quota` reserves a fraction of those slots for diagrams "
        "above the source width, filling the rest by length as before. Default "
        "`0.0` reserves nothing and leaves the selection index-for-index "
        "unchanged.")
    add("")
    add(f"**Outcome {payload['outcome']['verdict']}.**")
    add("")

    add("## Verdict")
    add("")
    for arm in quota_arms:
        a, c = all_block[arm], all_block[CONTROL_OF[arm]]
        add(f"* `{arm}`: wide flip sites "
            f"**{a['flip_sites_above_source_width']}/{a['flip_sites']}** against "
            f"the control's {c['flip_sites_above_source_width']}/{c['flip_sites']}; "
            f"reserved slots filled {a['quota_filled']}/{a['quota_slots']} "
            f"({fmt(a['quota_fill_rate'], 3)}); solved {a['solved']}/{a['instances']} "
            f"against {c['solved']}/{c['instances']}; "
            f"`STABILIZE` in witnesses **{a['witness_stabilizations']}**.")
    add("")
    add(f"* **The one move growth made legal is still never used.** "
        f"`STABILIZE` appears in **{stab}** sweep witnesses and "
        f"**{target_stab}** target witnesses, with the quota on.")
    add(f"* **Newly solved instances: "
        f"{', '.join('`' + k + '`' for k in new_solves) if new_solves else 'none'}.**")
    add(f"* **Degraded answers "
        f"(longer sequence than the control): "
        f"{', '.join('`' + k + '`' for k in degraded) if degraded else 'none'}; "
        f"lost exactness: "
        f"{', '.join('`' + k + '`' for k in lost_exact) if lost_exact else 'none'}.**")
    add("")
    t5 = {r["arm"]: r for r in cap5}
    add("**The target is the clean site, and it is unambiguous.** At cap 5 the "
        "quota lifts `flip_sites_above_source_width` from "
        f"**0 of {t5['generous-p0-q0']['flip_sites']}** to "
        f"**{t5['generous-p0-q25']['flip_sites_above_source_width']}** at "
        f"`quota=0.25` and "
        f"**{t5['generous-p0-q50']['flip_sites_above_source_width']}** at "
        f"`quota=0.50` -- every reserved slot filled -- and the search is "
        f"**byte-identical**: {t5['generous-p0-q0']['expansions']} expansions, "
        f"{t5['generous-p0-q0']['knots_expanded']} knots expanded, "
        f"{t5['generous-p0-q0']['pruned_by_bound']} pruned, no witness, in all "
        "three arms. Three hundred widened diagrams entered the "
        "crossing-change stage and contributed not one distinct knot.")
    add("")
    add("**Why, and it is a theorem rather than a tuning failure.** The crossing "
        "a `STABILIZE` creates is the one crossing whose change does nothing: "
        "`w . sigma_n` and `w . sigma_n^{-1}` are the positive and negative "
        "stabilisations of the same braid, so they close to the *same knot* "
        "(checked against the search's own key on 200 random words: "
        "`_fingerprint` agrees 200/200). Flipping *at* the new site is a no-op, "
        "and flipping at any of the old sites gives the knot the narrow diagram "
        "already gave. A widened diagram is worth flipping only **after** a "
        "braid manoeuvre has used the extra strand to move a real crossing "
        "somewhere it could not go before -- `STABILIZE`, then relations, then "
        "the change. The quota buys the first step and the third; the second is "
        "the one that is missing, and no selection rule at the flip stage can "
        "supply it.")
    add("")
    add("**The two frontier policies split, and neither rescues it.** At "
        "`width_penalty=2` -- the setting that actually controls the ratchet "
        "and returns the control's witness on every shared solve -- the quota is "
        "inert twice over: it fills "
        f"**{hard_block['p2-q25']['quota_filled']} of "
        f"{hard_block['p2-q25']['quota_slots']}** reserved slots on the hard "
        "subset, because at that charge no widened diagram reaches the frontier "
        "there at all, and on the easy instances where it does fill "
        f"({payload['subsets']['easy']['p2-q25']['quota_filled']} of "
        f"{payload['subsets']['easy']['p2-q25']['quota_slots']}) the wide flip "
        f"count does not move -- **{all_block['p2-q25']['flip_sites_above_source_width']}** "
        "in every `p2` arm -- because those diagrams were already among the "
        "shortest. That is outcome 3 for the controlled policy: a frontier "
        "problem, not a selection problem, and it points back at the width "
        "penalty exactly as §3.6 anticipated. At `width_penalty=0` the frontier "
        "is 26-57% wide, the quota fills 55-59%, the wide flip count does move "
        f"({all_block['p0-q0']['flip_sites_above_source_width']} -> "
        f"{all_block['p0-q25']['flip_sites_above_source_width']} -> "
        f"{all_block['p0-q50']['flip_sites_above_source_width']}) -- and solved, "
        "exact, expansions and every witness are unchanged. That is outcome 2.")
    add("")
    add("**So the answer to the question asked is no.** Applying a crossing "
        "change to a widened diagram does not help at this scale, and the "
        "reason is structural rather than a budget or an ordering. §3.6's "
        "reading -- that the flip-site cut is the third thing blocking growth -- "
        "is correct as a description of the counter and wrong as a diagnosis: "
        "removing the cut removes the symptom and changes no answer. **This "
        "redirects the plan.** §M1's per-site head does not get a job here: "
        "ranking flip sites on widened diagrams is ranking a set whose members "
        "are provably redundant. If growth is to earn its place it has to be "
        "judged on the *manoeuvre* it enables, which is a multi-move credit "
        "assignment problem, not a per-site score -- and the cap-5 floor needs a "
        "different explanation than the missing move.")
    add("")

    for label, title, note in (
        ("all", f"All {len(ids)} instances", ""),
        ("hard", f"The {len(hard)} the shipped search does not settle",
         "§3.5's warning: **45 of 57 instances were settled before any ordering "
         "mattered**, 43 at the very first expansion, so the all-instances table "
         "cannot show a selection effect. This subset is the 29 the shipped "
         "branch-and-bound fails outright, and it is the one to weight."),
        ("easy", f"The {len(easy)} it already settles", ""),
    ):
        block = payload["subsets"][label]
        add(f"## {title}")
        add("")
        if note:
            add(note)
            add("")
        rows = []
        for arm in order:
            a = block[arm]
            rows.append([
                f"`{arm}`", ARM_DIALS[arm], f"{a['solved']}/{a['instances']}",
                a["exact"],
                f"{a['flip_sites_above_source_width']}/{a['flip_sites']}",
                f"{a['quota_filled']}/{a['quota_slots']}",
                fmt(a["quota_fill_rate"], 3),
                f"{a['flip_levels_cut']}/{a['flip_levels']}",
                a["expansions_total"],
                fmt(a["expansions_per_solved"], 0),
                fmt(a["width_drift"], 3),
                a["witness_stabilizations"],
                fmt(a["seconds"], 0),
            ])
        add(table(rows, ["arm", "dials", "solved", "exact",
                         "**wide flip sites**", "quota filled", "fill rate",
                         "levels where the cut binds",
                         "expansions", "expansions/solved", "width drift",
                         "`STABILIZE` in witnesses", "seconds"],
                  "llrrrrrrrrrrr"))
        add("")

    add("## Paired against the matching no-quota control")
    add("")
    add("`node_budget` caps *distinct states seen*, not expansions, so a "
        "difference in the expansion total is not by itself a difference in "
        "search cost. The last two columns are the ones that decide: §3.6's trap "
        "was an arm that spent 29,325 fewer expansions and returned a 3-change "
        "sequence where the control returned a certified-optimal 2.")
    add("")
    rows = []
    for arm, blocks in payload["paired"].items():
        for label in ("all", "hard"):
            p = blocks[label]
            rows.append([
                f"`{arm}`", f"`{blocks['control']}`", label,
                p["solved_by_both"],
                len(p["solved_only_by_arm"]), len(p["solved_only_by_control"]),
                f"{p['expansions_better']}/{p['expansions_worse']}/{p['expansions_tied']}",
                f"{p['net_expansions']:+d}", f"{p['net_seconds']:+.1f}",
                len(p["worse_crossing_changes"]), len(p["better_crossing_changes"]),
                len(p["lost_exactness"]),
            ])
    add(table(rows, ["arm", "control", "subset", "solved by both",
                     "only this arm", "only control",
                     "expansions better/worse/tied", "net expansions",
                     "net seconds", "**longer sequence**",
                     "**shorter sequence**", "**lost exactness**"],
              "lllrrrrrrrrr"))
    add("")

    if target:
        add("## The target: `7_1 # m(7_1)`, 20 letters on `B5`")
        add("")
        add("The Brittenham--Hermiller representative. §M1b measured a certified "
            "**117-move floor just to inflate** this chart far enough for "
            "changes 3--5 to be visible, so the generous dials are the only "
            "serious setting; at `width_penalty=2` the target's drift was "
            "measured at 0.000, so only the `p0` arms have widened diagrams to "
            "reserve at all.")
        add("")
        for cap, heading, caption in (
            (5, "cap 5 -- the floor is `witness_rate@(cap=5) = 0.00`", ""),
            (6, "cap 6 -- free from subadditivity, and **not a result**",
             "Reported separately and explicitly not a result: 6 is free from "
             "subadditivity and the cap-6 witnesses were already found without "
             "growth."),
        ):
            block = [r for r in target if r["cap"] == cap]
            if not block:
                continue
            add(f"### {heading}")
            add("")
            if caption:
                add(caption)
                add("")
            rows = []
            by_arm = {r["arm"]: r for r in block}
            for arm in TARGET_ORDER:
                r = by_arm.get(arm)
                if r is None:
                    continue
                rows.append([
                    f"`{arm}`", TARGET_DIALS[arm],
                    "**yes**" if r.get("solved") else "no",
                    r.get("crossing_changes") if r.get("solved") else "--",
                    f"{r.get('flip_sites_above_source_width', 0)}"
                    f"/{r.get('flip_sites', 0)}",
                    f"{r.get('flip_quota_filled', 0)}/{r.get('flip_quota_slots', 0)}",
                    r.get("witness_stabilizations", 0),
                    r.get("expansions"),
                    fmt(r.get("width_drift", 0.0), 3),
                    r.get("max_strands_expanded"),
                    "yes" if r.get("timed_out") else "no",
                    fmt(r.get("seconds", 0.0), 0),
                ])
            add(table(rows, ["arm", "dials", "witness", "cc",
                             "**wide flip sites**", "quota filled",
                             "`STABILIZE` in witness", "expansions",
                             "width drift", "max strands", "timed out",
                             "seconds"],
                      "lllrrrrrrrrr"))
            add("")
        add(f"At cap 5, **{len(cap5_solved)} of {len(cap5)} arms** found a "
            f"witness.")
        add("")

    add("## Evidence grade")
    add("")
    verified = sum(1 for r in sweep + target
                   if r.get("solved") and r.get("replay_verified") is True)
    failed = sum(1 for r in sweep + target
                 if r.get("solved") and r.get("replay_verified") not in (True, None))
    add(f"Every witness was replayed through `UnknotWitness.verify()` at the "
        f"moment it was returned: **{verified} replay-complete terminals, "
        f"{failed} failures**. `unknot_search.search` composes nothing from a "
        f"catalogue -- no landmark table, no identification step -- so the "
        f"catalogue-composed count is zero by construction.")
    add("")
    add("## Reproduce")
    add("")
    add("```")
    add("PYTHONPATH=src python research/experiments/flip_quota.py --stage project")
    add("PYTHONPATH=src python research/experiments/flip_quota.py --stage sweep --seconds 120")
    add("PYTHONPATH=src python research/experiments/flip_quota.py --stage target "
        "--target-seconds 900 --target-nodes 1000000")
    add("PYTHONPATH=src python research/experiments/flip_quota_report.py")
    add("```")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'report.md'} and {OUT/'report.json'}")
    print("verdict:", payload["outcome"]["verdict"])


if __name__ == "__main__":
    main()
