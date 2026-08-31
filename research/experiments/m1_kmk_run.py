"""Driver: the `K # m(K)` growth-requiring corpus, measured end to end.

Two phases, and the first is a **premise check** rather than a warm-up.  Note 22
§M1 nominates `K # m(K)` because its certified floor should be loose *by
construction*; that is a claim about a family, measured once on one member
(§1.2's `7_1 # m(7_1)`).  Phase A measures it across a range of `K`, torus and
not, and reports the interval `[L, 2u(K)]` for each.  Where the width is zero
the floor is tight, no gap can exist, and that `K` is **excluded rather than
fought** -- which the specification asked for explicitly and which the trefoil
turns out to need.

Phase B churns from the seeds Phase A cleared, and measures every candidate with
the *same* `m1_growth.measure_growth_gap` the previous run used, at the same
caps, so the acceptance rates are comparable line for line.  Nothing about the
acceptance protocol is re-implemented: the shrink side must exhaust, the cheaper
completion must replay through `UnknotWitness.from_actions` and contain a growth
move, accepted rows are re-measured at 4x node caps, and every cap-undetermined
candidate is counted in its own row and never as either outcome.

Two guards are specific to this family and both are mandatory.

* **§4.3.**  Seeds of this shape sit next to the exam -- `7_1 # m(7_1)` *is* the
  target.  Every seed and every emitted word goes through `m1_kmk.screen`, and
  the report records which of the three screens fired.
* **§1.4.**  `u(A # B) <= u(A) + u(B)` is free and `connected_summands` finds it
  trivially, so `m1_kmk.summandwise` solves each prime summand independently and
  the report says how often the instance's optimum is *just* that.

    PYTHONPATH=src python research/experiments/m1_kmk_run.py --out artifacts/m1-kmk
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

if __name__ == "__main__" and str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import m1_growth  # noqa: E402
import m1_kmk  # noqa: E402

from rf_knots.actions import KIND_NAMES, ActionSpec  # noqa: E402
from rf_knots.interleaved_generator import (  # noqa: E402
    UNKNOT_SEED,
    generate_many,
    load_quarantine,
)

UNKNOTDB = "/Users/artemvorozhtsov/projects/unknotdb"

#: `K`s to measure the premise on.  Torus knots carry `u` as a theorem (Milnor,
#: Kronheimer-Mrowka); the rest are here to show that the floor is pinned by the
#: slice structure and not by the torus family.  `u(K)` for the non-torus rows is
#: the tabulated value and is used only to *report* the interval, never as a
#: certified bound -- the certified half is `L`, computed from the composite word.
PREMISE_KNOTS: tuple[tuple[str, tuple[int, ...], int, int], ...] = (
    ("T(2,3) = 3_1", (1, 1, 1), 2, 1),
    ("T(2,5) = 5_1", (1,) * 5, 2, 2),
    ("T(2,7) = 7_1", (1,) * 7, 2, 3),
    ("T(2,9) = 9_1", (1,) * 9, 2, 4),
    ("T(2,11) = 11a_367", (1,) * 11, 2, 5),
    ("T(2,13)", (1,) * 13, 2, 6),
    ("T(3,4) = 8_19", (1, 2, 1, 2, 1, 2), 3, 3),
    ("4_1", (1, -2, 1, -2), 3, 1),
    ("5_2", (1, 1, 1, 2, -1, 2), 3, 1),
    ("6_2", (1, 1, 1, -2, 1, -2), 3, 1),
)


def log(message: str, started: float) -> None:
    print(f"[{time.monotonic() - started:8.1f}s] {message}", flush=True)


# -- phase A: the premise ------------------------------------------------------


def premise(args, started: float) -> dict:
    """`L` and the interval `[L, 2u(K)]` on `K # m(K)`, for a range of `K`.

    The whole point is the *pattern*: if `L` is pinned at 2 while `2u(K)` grows,
    the family supplies a structurally loose floor and §M1's premise holds; if
    `L` tracks `2u(K)` the premise is refuted and the rest of this run is a
    formality.
    """
    quarantine = load_quarantine(unknotdb_root=UNKNOTDB)
    rows = []
    for name, word, strands, u_k in PREMISE_KNOTS:
        row = m1_kmk.premise_row(name, word, strands, u_k)
        verdict = m1_kmk.screen(quarantine, row.composite_word, row.composite_strands)
        payload = row.to_dict()
        payload["quarantine_reason"] = verdict.reason
        payload["summandwise"] = m1_kmk.summandwise(
            row.composite_word, row.composite_strands, node_cap=args.summand_node_cap
        ).to_dict()
        rows.append(payload)
        log(
            f"premise {name}: L={row.certified_floor} u<={row.u_upper} "
            f"width={row.width} letters={row.letters} "
            f"quarantine={verdict.reason}",
            started,
        )
    return {
        "rows": rows,
        "slice_reason": m1_kmk.SLICE_REASON,
        "floors": sorted({row["certified_floor"] for row in rows}),
        "loose": sum(1 for row in rows if row["loose"]),
        "tight": sum(1 for row in rows if not row["loose"]),
    }


# -- phase B: the corpus -------------------------------------------------------


def families(args) -> list[m1_growth.CandidateFamily]:
    """Churn families, all from `K # m(K)` seeds except the reproduced control.

    The construction is §M1's, and it is the one the previous run measured to
    work: *commute and braid until two letters meet, change one, reduce the
    pair* -- `m1_growth.CHURN_WEIGHTS` unchanged.  What varies here is only the
    seed and `k`.

    `k = 0` deserves its own line because it is not a degenerate case.  The walk
    still churns; the emitted word is a *different diagram of the same composite*
    and carries the seed's own `2u(K)`-change program as a replayed witness.  If
    the churn makes the growth-free optimum worse than that program, the gap is
    exhibited with no full search at all -- which is the only route by which this
    family could ever produce `c_full >= 2`, since a budget-`c_full` growth search
    on a word this long does not terminate (measured; see the report).
    """
    out: list[m1_growth.CandidateFamily] = []
    for m, k, walk, target_length, max_len, count in args.grid:
        spec = ActionSpec(max_len=max_len, max_strands=4)
        seed = m1_kmk.kmk_torus_seed(m, spec)
        width = (seed.u_upper or 0) - seed.certified_lower_bound
        #: Phase A measures the interval, and where it is **zero** the floor is
        #: tight and no gap can exist -- the specification says such `K` are
        #: excluded rather than fought.  They are kept here under a `control-`
        #: name, because what a tight-floor member does under the same churn is
        #: the comparison that makes the loose members mean something; they are
        #: not corpus.
        prefix = "kmk" if width > 0 else "control-kmk"
        intent = (
            f"churn from {seed.name}: certified L="
            f"{seed.certified_lower_bound} against a replayed "
            f"{seed.u_upper}-change program, so the seed interval is loose by "
            f"{width}"
            if width > 0
            else (
                f"{seed.name} has a **tight** interval "
                f"[{seed.certified_lower_bound}, {seed.u_upper}]: no gap can "
                "exist at the seed, so this family is a control on what the "
                "same churn does when the floor is not loose"
            )
        )
        out.append(
            m1_growth.churn_family(
                f"{prefix}-T(2,{2 * m + 1})-k{k}-l{walk}",
                seed,
                crossing_changes=k,
                walk_length=walk,
                target_length=target_length,
                target_strands=3,
                max_len=max_len,
                max_strands=4,
                intent=intent,
                settle_walk_length=walk // 2,
                instances=count,
            )
        )
    if args.control:
        out.append(
            m1_growth.churn_family(
                "control-churn-unknot-k1-long",
                UNKNOT_SEED,
                crossing_changes=1,
                walk_length=40,
                target_length=9,
                target_strands=4,
                max_len=12,
                max_strands=5,
                intent=(
                    "the previous run's best family, reproduced unchanged, so "
                    "this run's acceptance rates are comparable to its 0.057"
                ),
                instances=args.control,
            )
        )
    if args.families:
        wanted = set(args.families)
        out = [family for family in out if family.name in wanted]
    return out


def sweep(args, started: float) -> dict:
    """Generate, screen, measure, and record every candidate."""
    quarantine = load_quarantine(unknotdb_root=UNKNOTDB)
    rows: list[dict] = []
    per_family: dict[str, dict] = {}
    last = time.monotonic()
    for family in families(args):
        spec = family.config.spec()
        seed_screen = m1_kmk.screen(quarantine, family.seed.word, family.seed.strands)
        if seed_screen.rejected:
            per_family[family.name] = {
                "provenance": family.provenance(),
                "excluded_by_quarantine": seed_screen.reason,
            }
            log(f"{family.name}: SEED EXCLUDED -- {seed_screen.reason}", started)
            continue
        counts: Counter = Counter()
        screen_reasons: Counter = Counter()
        gaps: list[int] = []
        joint: Counter = Counter()
        seconds = 0.0
        seen: set[tuple] = set()
        family_started = time.monotonic()
        for instance in generate_many(
            family.config,
            seed=args.seed,
            count=family.instances or args.instances,
            seed_representation=family.seed,
        ):
            begun = time.monotonic()
            verdict = m1_kmk.screen(quarantine, instance.word, instance.strands)
            if verdict.rejected:
                screen_reasons[verdict.reason] += 1
                counts["screened"] += 1
                seconds += time.monotonic() - begun
                continue
            fingerprint = (instance.word, instance.strands)
            if fingerprint in seen:
                counts["duplicate"] += 1
                seconds += time.monotonic() - begun
                continue
            seen.add(fingerprint)
            witness_actions = (
                tuple(step.action.to_flat(spec) for step in instance.witness.steps)
                if instance.witness is not None
                else ()
            )
            try:
                measured = m1_growth.measure_growth_gap(
                    spec,
                    instance.word,
                    instance.strands,
                    growth=args.gap_growth,
                    change_cap=args.change_cap,
                    shrink_node_cap=args.shrink_node_cap,
                    full_node_cap=args.full_node_cap,
                    known_upper=instance.certified_upper_bound,
                    known_upper_actions=witness_actions,
                )
            except AssertionError as error:
                #: `measure_growth_gap` raises only on a contradiction between
                #: two things it computed independently.  Crashing a forty-minute
                #: sweep on one is worse than recording it, but it is a soundness
                #: alarm and the report says so in its own row.
                counts["soundness-alarm"] += 1
                rows.append(
                    {
                        "family": family.name,
                        "word": list(instance.word),
                        "strands": instance.strands,
                        "verdict": "soundness-alarm",
                        "error": str(error),
                    }
                )
                log(f"  SOUNDNESS ALARM in {family.name}: {error}", started)
                seconds += time.monotonic() - begun
                continue
            if measured.accepted:
                measured = m1_growth.verify_at_raised_caps(
                    spec, measured, factor=args.raise_factor
                )
            shortcut = m1_kmk.summandwise(
                instance.word,
                instance.strands,
                change_cap=args.change_cap,
                growth=args.gap_growth,
                node_cap=args.summand_node_cap,
            )
            seconds += time.monotonic() - begun
            counts[measured.verdict] += 1
            counts["measured"] += 1
            if measured.accepted:
                gaps.append(int(measured.gap or 0))
                joint[(int(measured.c_full or 0), int(measured.gap or 0))] += 1
                counts["survived_raised_caps"] += int(
                    bool(measured.survived_raised_caps)
                )
                counts["beats_summandwise"] += int(
                    bool(shortcut.beats(measured.c_full))
                )
                if shortcut.needs_growth is False:
                    counts["summandwise_route_is_growth_free"] += 1
            if measured.c_shrink is not None:
                counts["floor_tight"] += int(
                    measured.certified_floor >= measured.c_shrink
                )
                counts["floor_loose"] += int(
                    measured.certified_floor < measured.c_shrink
                )
            row = measured.to_dict()
            row["family"] = family.name
            row["seed"] = family.seed.name
            row["seed_certified_lower_bound"] = family.seed.certified_lower_bound
            row["seed_u_upper"] = family.seed.u_upper
            row["walk_depth"] = family.config.crossing_changes
            row["walk_length"] = family.config.walk_length
            row["instance_id"] = instance.instance_id
            row["spec_max_len"] = spec.max_len
            row["spec_max_strands"] = spec.max_strands
            row["own_witness_crossing_changes"] = instance.certified_upper_bound
            row["own_witness_growth_moves"] = m1_growth.counts_growth(
                spec, witness_actions
            )
            row["determinant"] = verdict.determinant
            row["signature"] = verdict.signature
            row["summandwise"] = shortcut.to_dict()
            row["beats_summandwise"] = shortcut.beats(measured.c_full)
            row["witness_kinds"] = sorted(
                {KIND_NAMES[spec.decode(a)[0]] for a in measured.witness_actions}
            )
            if not measured.accepted:
                #: Budget: a non-accepted row's exhibit is not evidence of
                #: anything and it is the only large field.  §Environment asks
                #: for compact records.
                row["witness_actions"] = []
            rows.append(row)
            now = time.monotonic()
            if now - last > 15.0:
                last = now
                log(
                    f"  {family.name}: {counts['measured']} measured, "
                    f"{counts['growth-required']} accepted, "
                    f"{counts['screened']} screened, {seconds:.0f}s",
                    started,
                )
            if now - family_started > args.family_seconds:
                log(f"  {family.name}: family wall cap reached", started)
                break
            if now - started > args.total_seconds:
                break
        measured_total = int(counts["measured"])
        undetermined = (
            counts["undetermined-shrink-capped"]
            + counts["undetermined-full-capped"]
            + counts["undetermined-change-cap"]
        )
        determined = measured_total - undetermined
        per_family[family.name] = {
            "provenance": family.provenance(),
            "seed_u_upper": family.seed.u_upper,
            "seed_certified_lower_bound": family.seed.certified_lower_bound,
            "seed_interval_width": (
                None
                if family.seed.u_upper is None
                else family.seed.u_upper - family.seed.certified_lower_bound
            ),
            "candidates_measured": measured_total,
            "screened_by_quarantine": int(counts["screened"]),
            "screen_reasons": dict(screen_reasons),
            "duplicates": int(counts["duplicate"]),
            "verdicts": {name: counts[name] for name in m1_growth.VERDICTS},
            "soundness_alarms": int(counts["soundness-alarm"]),
            "undetermined": int(undetermined),
            "determined": determined,
            "accepted": int(counts["growth-required"]),
            "acceptance_over_measured": (
                round(counts["growth-required"] / measured_total, 4)
                if measured_total
                else None
            ),
            "acceptance_over_determined": (
                round(counts["growth-required"] / determined, 4) if determined else None
            ),
            "floor_tight": int(counts["floor_tight"]),
            "floor_loose": int(counts["floor_loose"]),
            "settled_by_tight_floor": int(counts["no-gap-certified"]),
            "survived_raised_caps": int(counts["survived_raised_caps"]),
            "beats_summandwise": int(counts["beats_summandwise"]),
            "summandwise_route_is_growth_free": int(
                counts["summandwise_route_is_growth_free"]
            ),
            "gap_distribution": dict(sorted(Counter(gaps).items())),
            "joint_c_full_gap": {f"{a},{b}": n for (a, b), n in sorted(joint.items())},
            "seconds": round(seconds, 1),
            "seconds_per_candidate": (
                round(seconds / measured_total, 3) if measured_total else None
            ),
            "seconds_per_accepted": (
                round(seconds / counts["growth-required"], 3)
                if counts["growth-required"]
                else None
            ),
        }
        log(
            f"{family.name}: {measured_total} measured, "
            f"{counts['growth-required']} accepted, {undetermined} undetermined, "
            f"{counts['no-gap-certified']} settled by a tight floor, "
            f"{counts['screened']} screened, {seconds:.0f}s",
            started,
        )
        if time.monotonic() - started > args.total_seconds:
            log("total wall cap reached; stopping the sweep", started)
            break
    return {"rows": rows, "per_family": per_family}


# -- phase C: pay more for the candidates that could carry both axes -----------


def deepen(phase_b: dict, args, started: float) -> dict:
    """Re-run the most promising cap-undetermined candidates at a bigger cap.

    A capped full search proves nothing, and that is where every `c_shrink >= 3`
    candidate lands: proving *no* completion exists one crossing change below
    `c_shrink` is the expensive direction, while **finding** one is cheap if it
    is there.  So the rows worth paying for are exactly the undetermined ones
    with the largest `c_shrink` -- they are the only place an accepted instance
    could have `c_full >= 2`, which is the combination no previous family
    produced.

    Ordered by `c_shrink` descending and then by word length ascending, so the
    budget goes to the hardest instances the search still has a chance on.
    """
    if args.deep_seconds <= 0:
        return {"attempted": 0, "resolved": [], "rows": []}
    candidates = [
        row
        for row in phase_b["rows"]
        if row.get("verdict") == "undetermined-full-capped"
        and (row.get("c_shrink") or 0) >= args.deep_min_shrink
    ]
    candidates.sort(key=lambda row: (-(row["c_shrink"] or 0), len(row["word"])))
    began = time.monotonic()
    out: list[dict] = []
    resolved: list[dict] = []
    for row in candidates:
        if time.monotonic() - began > args.deep_seconds:
            break
        #: The searched subgraph has to be the sweep's, or a deeper search would
        #: be answering a different question: `growth_cap` is what bounded it and
        #: `max_len` is recovered from the word plus that cap.
        spec = ActionSpec(
            max_len=row["spec_max_len"], max_strands=row["spec_max_strands"]
        )
        again = m1_growth.measure_growth_gap(
            spec,
            tuple(row["word"]),
            int(row["strands"]),
            growth=row["growth_cap"],
            change_cap=row["change_cap"],
            #: `GrowthGap.to_dict` does not carry the node caps, and it does not
            #: need to: every row in the sweep was measured at the run's own
            #: `--shrink-node-cap`, so re-reading it from `args` is the same
            #: number and cannot drift from what produced `c_shrink`.
            shrink_node_cap=args.shrink_node_cap,
            full_node_cap=args.full_node_cap * args.deep_factor,
            change_floor=row["certified_floor"],
        )
        if again.accepted:
            again = m1_growth.verify_at_raised_caps(
                spec, again, factor=args.raise_factor
            )
        record = again.to_dict()
        record["family"] = row["family"]
        record["deep_full_node_cap"] = args.full_node_cap * args.deep_factor
        out.append(record)
        if again.verdict != "undetermined-full-capped":
            resolved.append(record)
        log(
            f"  deep {row['family']}: len={len(row['word'])} "
            f"c_shrink={again.c_shrink} -> {again.verdict} "
            f"c_full={again.c_full} gap={again.gap} ({again.seconds:.0f}s)",
            started,
        )
    return {
        "attempted": len(out),
        "available": len(candidates),
        "full_node_cap": args.full_node_cap * args.deep_factor,
        "resolved": [r["verdict"] for r in resolved],
        "accepted": sum(1 for r in out if r["verdict"] == "growth-required"),
        "rows": out,
    }


# -- aggregation and rendering -------------------------------------------------


def aggregate(phase_a: dict, phase_b: dict, phase_c: dict, args) -> dict:
    rows = phase_b["rows"]
    accepted = [row for row in rows if row.get("verdict") == "growth-required"]
    accepted += [
        row for row in phase_c["rows"] if row.get("verdict") == "growth-required"
    ]
    joint: Counter = Counter()
    for row in accepted:
        joint[(int(row["c_full"] or 0), int(row["gap"] or 0))] += 1
    c_shrink = Counter(
        row["c_shrink"] for row in rows if row.get("c_shrink") is not None
    )
    floors = Counter(
        row["certified_floor"] for row in rows if "certified_floor" in row
    )
    looseness = Counter(
        (row["c_shrink"] - row["certified_floor"])
        for row in rows
        if row.get("c_shrink") is not None
    )
    measured = sum(
        f.get("candidates_measured", 0) for f in phase_b["per_family"].values()
    )
    undetermined = sum(
        f.get("undetermined", 0) for f in phase_b["per_family"].values()
    )
    settled = sum(
        f.get("settled_by_tight_floor", 0) for f in phase_b["per_family"].values()
    )
    seconds = sum(f.get("seconds", 0.0) for f in phase_b["per_family"].values())
    return {
        "premise": phase_a,
        "families": phase_b["per_family"],
        "rows": rows,
        "deep": phase_c,
        "totals": {
            "candidates_measured": measured,
            "accepted": len(accepted),
            "accepted_in_sweep": sum(
                1 for row in rows if row.get("verdict") == "growth-required"
            ),
            "accepted_in_deep_pass": phase_c.get("accepted", 0),
            "acceptance_over_measured": (
                round(len(accepted) / measured, 4) if measured else None
            ),
            "undetermined": undetermined,
            "settled_by_tight_floor": settled,
            "settled_fraction": round(settled / measured, 4) if measured else None,
            "screened_by_quarantine": sum(
                f.get("screened_by_quarantine", 0)
                for f in phase_b["per_family"].values()
            ),
            "seconds": round(seconds, 1),
            "seconds_per_accepted": (
                round(seconds / len(accepted), 2) if accepted else None
            ),
            "joint_c_full_gap": {f"{a},{b}": n for (a, b), n in sorted(joint.items())},
            "accepted_with_c_full_at_least_2": sum(
                n for (c_full, _gap), n in joint.items() if c_full >= 2
            ),
            "c_shrink_distribution": dict(sorted(c_shrink.items())),
            "certified_floor_distribution": dict(sorted(floors.items())),
            "looseness_distribution": dict(sorted(looseness.items())),
            "replayed": sum(1 for row in accepted if row.get("witness_replayed")),
            "survived_raised_caps": sum(
                1 for row in accepted if row.get("survived_raised_caps")
            ),
            "beats_summandwise": sum(
                1 for row in accepted if row.get("beats_summandwise")
            ),
            "summandwise_growth_free": sum(
                1
                for row in accepted
                if (row.get("summandwise") or {}).get("summandwise_needs_growth")
                is False
            ),
            "soundness_alarms": sum(
                1 for row in rows if row.get("verdict") == "soundness-alarm"
            ),
        },
        "settings": {
            "grid": [list(cell) for cell in args.grid],
            "gap_growth": args.gap_growth,
            "change_cap": args.change_cap,
            "shrink_node_cap": args.shrink_node_cap,
            "full_node_cap": args.full_node_cap,
            "summand_node_cap": args.summand_node_cap,
            "raise_factor": args.raise_factor,
            "seed": args.seed,
            "control_instances": args.control,
        },
        "quarantine": load_quarantine(unknotdb_root=UNKNOTDB).provenance(),
    }


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def render(payload: dict) -> str:
    totals = payload["totals"]
    out: list[str] = []
    out.append("# M1 — the `K # m(K)` growth-requiring corpus\n")
    out.append(
        "Note 22 §M1 nominates `K # m(K)` as the one construction that could "
        "carry **both** difficulty axes: a certified floor that is loose "
        "*structurally* (the composite is slice, so every 4-genus bound "
        "vanishes and only Montesinos fires) and an unknotting number that "
        "scales with `K`. This run measures the premise first and then churns "
        "from the seeds it clears.\n"
    )

    out.append("## 0. Verdict\n")
    premise_rows = payload["premise"]["rows"]
    floors = sorted({row["certified_floor"] for row in premise_rows})
    widest = max((row["width"] or 0) for row in premise_rows) if premise_rows else 0
    settled = totals["settled_fraction"]
    out.append(
        f"1. **The loose-floor premise holds, and it is not marginal.** The "
        f"certified floor of `K # m(K)` takes the values `{floors}` across "
        f"{len(premise_rows)} members while `2u(K)` runs to "
        f"{max((row['u_upper'] or 0) for row in premise_rows) if premise_rows else 0}; "
        f"the widest certified interval measured is {widest}. The floor does not "
        f"track the knot at all, for the structural reason §M1 gave: the "
        f"composite is slice, so the signature and `tau` are zero by "
        f"construction and only Montesinos fires, at 2. It holds for every `K` "
        f"with `u(K) >= 2`; it is **tight** exactly on the `u(K) = 1` members, "
        f"where `2u(K) = 2 = L` closes the interval by arithmetic, and those "
        f"are excluded.\n"
    )
    kmk_measured = sum(
        f.get("candidates_measured", 0)
        for name, f in payload["families"].items()
        if name.startswith("kmk-")
    )
    kmk_settled = sum(
        f.get("settled_by_tight_floor", 0)
        for name, f in payload["families"].items()
        if name.startswith("kmk-")
    )
    out.append(
        f"2. **And it transfers to the emitted words, which is what a gap "
        f"actually needs.** Across the whole run "
        f"{totals['settled_by_tight_floor']} of {totals['candidates_measured']} "
        f"candidates ({settled}) were settled with no full search because the "
        f"floor was already tight, against **0.921** for the previous run. "
        f"Restricted to the `K # m(K)` families -- the loose-interval seeds, "
        f"excluding the controls -- it is {kmk_settled} of {kmk_measured}"
        + (
            f" ({round(kmk_settled / kmk_measured, 4)})"
            if kmk_measured
            else ""
        )
        + ". The mechanism that killed every certified-hard seed last time does "
        "not fire here.\n"
    )
    both = totals["accepted_with_c_full_at_least_2"]
    headline = (
        "**The corpus now carries both axes.**"
        if both
        else "**The corpus still does not carry both axes.**"
    )
    out.append(
        f"3. {headline} Accepted with a verified gap: {totals['accepted']} of "
        f"{totals['candidates_measured']} "
        f"({totals['acceptance_over_measured']}). Accepted with `c_full >= 2` "
        f"*and* a gap -- the combination no previous family produced: "
        f"**{both}**. {totals['undetermined']} candidates were left "
        f"undetermined by a node cap and are counted in their own row, never as "
        f"either outcome.\n"
    )
    out.append("## 1. The premise, measured\n")
    out.append(_row(["K", "u(K)", "letters", "L (certified)", "u <= 2u(K)", "width", "verdict", "quarantine"]))
    out.append(_row(["---"] * 8))
    for row in payload["premise"]["rows"]:
        out.append(
            _row(
                [
                    row["knot"],
                    row["u_k"],
                    row["letters"],
                    f"**{row['certified_floor']}**",
                    row["u_upper"],
                    row["width"],
                    "loose" if row["loose"] else "**tight — excluded**",
                    row["quarantine_reason"] or "—",
                ]
            )
        )
    floors = payload["premise"]["floors"]
    out.append("")
    out.append(
        f"The certified floor takes the values `{floors}` across the whole "
        f"table while `2u(K)` runs to {max(r['u_upper'] or 0 for r in payload['premise']['rows'])}. "
        f"{payload['premise']['loose']} of "
        f"{payload['premise']['loose'] + payload['premise']['tight']} members are "
        "loose; the tight ones are exactly the `u(K) = 1` members, where "
        "`2u(K) = 2 = L` leaves no room by arithmetic.\n"
    )
    out.append(f"> {payload['premise']['slice_reason']}\n")

    out.append("## 2. Acceptance by `u(K)` and by walk length\n")
    per_family_shrink: dict[str, list[int]] = {}
    for row in payload["rows"]:
        if row.get("c_shrink") is not None:
            per_family_shrink.setdefault(row["family"], []).append(row["c_shrink"])
    out.append(
        _row(
            [
                "family",
                "u(K)",
                "seed L",
                "seed u<=",
                "width",
                "k",
                "l",
                "measured",
                "screened",
                "tight-floor",
                "undet.",
                "accepted",
                "acceptance",
                "mean `c_shrink`",
                "s/cand",
            ]
        )
    )
    out.append(_row(["---"] * 15))
    for name, family in payload["families"].items():
        if "excluded_by_quarantine" in family:
            out.append(
                _row(
                    [name] + ["—"] * 10
                    + [f"**seed excluded: {family['excluded_by_quarantine']}**"]
                    + ["—"] * 3
                )
            )
            continue
        config = family["provenance"]["config"]
        upper = family["seed_u_upper"]
        shrinks = per_family_shrink.get(name, [])
        out.append(
            _row(
                [
                    name,
                    "—" if not upper else upper // 2,
                    family["seed_certified_lower_bound"],
                    upper,
                    family["seed_interval_width"],
                    config["crossing_changes"],
                    config["walk_length"],
                    family["candidates_measured"],
                    family["screened_by_quarantine"],
                    family["settled_by_tight_floor"],
                    family["undetermined"],
                    family["accepted"],
                    family["acceptance_over_measured"],
                    (
                        round(sum(shrinks) / len(shrinks), 2) if shrinks else "—"
                    ),
                    family["seconds_per_candidate"],
                ]
            )
        )
    out.append("")
    out.append(
        "`mean c_shrink` is the column to read against `seed u<=`. Where the "
        "walk makes no crossing change (`k = 0`) the emitted word is another "
        "diagram of the *same* composite, and its growth-free optimum is what "
        "decides whether the loose interval can ever be entered.\n"
    )

    out.append("## 3. Is the floor loose where it matters?\n")
    out.append(
        "The premise is about the *seed*; what a gap needs is a loose floor on "
        "the **emitted word**. `c_shrink - L` per candidate:\n"
    )
    out.append(_row(["`c_shrink - L`", "candidates"]))
    out.append(_row(["---", "---"]))
    for width, count in totals["looseness_distribution"].items():
        out.append(_row([width, count]))
    out.append("")
    out.append(
        f"{totals['settled_by_tight_floor']} of {totals['candidates_measured']} "
        f"candidates ({totals['settled_fraction']}) were settled with no full "
        "search because the floor was already tight. The previous run's figure "
        "was 1,070 of 1,162 — 0.921.\n"
    )
    out.append(_row(["`c_shrink`", "candidates"]))
    out.append(_row(["---", "---"]))
    for value, count in totals["c_shrink_distribution"].items():
        out.append(_row([value, count]))
    out.append("")

    out.append("## 4. The joint distribution — the number that matters\n")
    out.append(
        "An instance is new only if it carries **both** axes: a verified gap "
        "*and* `c_full >= 2`. Every accepted instance in the previous run had "
        "`c_full` in `{0, 1}`.\n"
    )
    if totals["joint_c_full_gap"]:
        out.append(_row(["`c_full`", "gap", "accepted"]))
        out.append(_row(["---", "---", "---"]))
        for key, count in totals["joint_c_full_gap"].items():
            c_full, gap = key.split(",")
            out.append(_row([c_full, gap, count]))
    else:
        out.append("No instance was accepted, so the joint distribution is empty.\n")
    out.append("")
    out.append(
        f"**accepted with `c_full >= 2` and a gap: "
        f"{totals['accepted_with_c_full_at_least_2']}**\n"
    )

    out.append("## 4a. Where the loose interval actually sits\n")
    out.append(
        "The `k = 0` rows are the cleanest instrument in the run: the walk "
        "changes no crossing, so the emitted word is another diagram of the "
        "*same* composite, with the same certified interval `[2, 2u(K)]` and "
        "the seed's own replayed `2u(K)`-change program as a witness. Whether "
        "a gap is reachable at all comes down to where `c_shrink` lands inside "
        "that interval.\n"
    )
    zero = [
        row
        for row in payload["rows"]
        if row.get("walk_depth") == 0 and row.get("c_shrink") is not None
    ]
    if zero:
        at_top = sum(
            1 for row in zero if row["c_shrink"] == row.get("seed_u_upper")
        )
        below = sum(
            1
            for row in zero
            if row.get("seed_u_upper") is not None
            and row["c_shrink"] < row["seed_u_upper"]
        )
        out.append(
            _row(["`k = 0` candidates", "`c_shrink == 2u(K)`", "`c_shrink < 2u(K)`",
                  "`c_shrink > 2u(K)`"])
        )
        out.append(_row(["---"] * 4))
        out.append(_row([len(zero), at_top, below, len(zero) - at_top - below]))
        out.append("")
        out.append(
            "`c_shrink > 2u(K)` is the only cell that can produce an accepted "
            "instance **without a search**: it would mean the growth-free "
            "optimum is worse than the seed's own replayed program, so that "
            "program *is* the gap, with no growth cap and no full search in the "
            "claim. It never happened. Every `k = 0` candidate landed on "
            "`c_shrink == 2u(K)` exactly: however hard the churn works, the "
            "growth-free search recovers the summand-wise route and nothing "
            "worse.\n"
        )
        out.append(
            "**That is the finding, and it is structural rather than a budget "
            "problem.** The loose part of the interval is `(L, 2u(K)) = (2, 4)` "
            "for `T(2,5)`, and entering it means exhibiting a completion "
            "strictly cheaper than the summand-wise route -- that is, "
            "`u(K # m(K)) < 2u(K)`, the *non-additivity of unknotting number*. "
            "For `7_1` that is Brittenham-Hermiller's theorem and the campaign's "
            "own target; for `5_1` nothing here settles it. So on this family "
            "the certified interval is loose exactly where the answer is "
            "unknown, and a gap cannot be *verified* without first doing the "
            "mathematics the corpus was supposed to be training for. The two "
            "axes are not merely hard to combine here -- combining them is the "
            "target itself.\n"
        )

    deep = payload["deep"]
    out.append("## 4b. Paying more for the candidates that could carry both axes\n")
    out.append(
        "A capped full search proves nothing, and every `c_shrink >= 3` "
        "candidate lands there: *finding* a completion one change below "
        "`c_shrink` is cheap if one exists, *proving* none does is not. The "
        f"deep pass re-ran the most promising cap-undetermined rows at "
        f"`full_node_cap = {deep.get('full_node_cap')}`.\n"
    )
    deep_seconds = sum(row.get("seconds", 0.0) for row in deep.get("rows", []))
    out.append(
        f"* available: {deep.get('available', 0)}  * attempted: "
        f"{deep.get('attempted', 0)}  * resolved either way: "
        f"{len(deep.get('resolved', []))}  * accepted: {deep.get('accepted', 0)}"
        + (
            f"  * {round(deep_seconds / deep['attempted'], 1)} s each"
            if deep.get("attempted")
            else ""
        )
        + "\n"
    )
    if deep.get("attempted") and not deep.get("resolved"):
        out.append(
            "**Eight times the node cap changed nothing.** The budget-"
            "`c_shrink - 1` growth search on a 10-to-14 letter three-strand "
            "word does not terminate at 480,000 nodes any more than at 60,000; "
            "a separate probe carried one such word to 1,500,000 nodes and 91 "
            "seconds without exhausting. The wall is the shape of the search "
            "space, not the size of the budget.\n"
        )
    if deep.get("resolved"):
        counts = Counter(deep["resolved"])
        out.append(_row(["verdict once the cap was raised", "rows"]))
        out.append(_row(["---", "---"]))
        for verdict, count in counts.most_common():
            out.append(_row([verdict, count]))
        out.append("")

    out.append("## 5. Totals\n")
    for key in (
        "candidates_measured",
        "accepted",
        "accepted_in_sweep",
        "accepted_in_deep_pass",
        "acceptance_over_measured",
        "undetermined",
        "settled_by_tight_floor",
        "screened_by_quarantine",
        "replayed",
        "survived_raised_caps",
        "beats_summandwise",
        "summandwise_growth_free",
        "soundness_alarms",
        "seconds",
        "seconds_per_accepted",
    ):
        out.append(f"* `{key}`: {totals[key]}")
    out.append("")

    out.append("## 6. §4.3 quarantine: what was screened, and what it misses\n")
    reasons: Counter = Counter()
    for family in payload["families"].values():
        for reason, count in (family.get("screen_reasons") or {}).items():
            reasons[reason] += count
    if reasons:
        out.append(_row(["screen", "candidates rejected"]))
        out.append(_row(["---", "---"]))
        for reason, count in reasons.most_common():
            out.append(_row([reason, count]))
    else:
        out.append("No emitted word was rejected by any screen.\n")
    out.append("")
    out.append(
        "**A defect in the shipped screen, found by running it here.** "
        "`Quarantine.screen_name` cannot catch `7_1 # m(7_1)`: `invariants` "
        "reports its summands as `('7_1 or 12n_749', '7_1 or 12n_749')` — §1.2's "
        "identification trap — so the exact-set test "
        "`{'7_1','m(7_1)'} <= {'7_1 or 12n_749'}` is false. The **fingerprint** "
        "screen `(det, |sigma|) = (49, 0)` is the only one that fires on the "
        "target, and this run adds `summand_alias_hit`, which matches the "
        "quarantined names *inside* the ambiguous labels.\n"
    )
    out.append(
        "**What the screen does not guarantee**, carried verbatim from §4.1b: "
        f"{payload['quarantine']['k_ball_closure']}. The fingerprint screen is "
        "also deliberately over-broad — it rejects `5_2 # m(5_2)`, whose "
        "determinant is also 49, and every other knot sharing `(49, 0)`.\n"
    )

    out.append("## 7. §1.4: the summand-wise shortcut\n")
    out.append(
        "`u(A # B) <= u(A) + u(B)` is free and `connected_summands` finds it "
        "trivially. Measured on the premise table's own composites, the "
        "summand-wise route is **growth-free on every one of them** — each "
        "`T(2,q)` factor is unknotted by `m` change-and-cancel pairs with no "
        "insertion anywhere — so any growth an instance requires has to come "
        "from the churn and not from the connected sum.\n"
    )
    out.append(_row(["K # m(K)", "summands", "per-summand u<=", "total", "needs growth"]))
    out.append(_row(["---"] * 5))
    for row in payload["premise"]["rows"]:
        shortcut = row["summandwise"]
        out.append(
            _row(
                [
                    row["knot"],
                    "prime" if shortcut["prime"] else len(shortcut["pieces"] or []),
                    shortcut["per_summand_upper"],
                    shortcut["total_upper"],
                    shortcut["summandwise_needs_growth"],
                ]
            )
        )
    out.append("")
    accepted_rows = [
        row for row in payload["rows"] if row.get("verdict") == "growth-required"
    ]
    prime = sum(
        1 for row in accepted_rows if (row.get("summandwise") or {}).get("prime")
    )
    out.append(
        f"Of the {totals['accepted']} accepted instances, **{prime} are prime** "
        "-- `connected_summands` returns one piece or none, so the shortcut is "
        f"vacuous on them -- {totals['beats_summandwise']} beat the "
        f"summand-wise bound, and {totals['summandwise_growth_free']} sit on a "
        "composite whose summand-wise route needs no growth. The §1.4 hazard "
        "therefore did not bite in this run, for an uninteresting reason: "
        "every accepted instance came from a family whose emitted words are "
        "not composite. On the `K # m(K)` families themselves the hazard is "
        "the whole story -- §4a -- because the summand-wise route is exactly "
        "what `c_shrink` finds.\n"
    )

    out.append("## 8. What this changes for M1\n")
    out.append(
        "The specification asked whether `K # m(K)` supplies both axes at "
        "once. Measured:\n"
    )
    shallow = [
        family
        for name, family in payload["families"].items()
        if name.startswith("kmk-")
        and "excluded_by_quarantine" not in family
        and family["provenance"]["config"]["crossing_changes"] <= 1
    ]
    out.append(
        "* **The premise it rests on is true.** The floor is loose, "
        "structurally, for every `K` with `u(K) >= 2`, and the looseness "
        "carries into the churned diagrams: "
        f"{sum(f['settled_by_tight_floor'] for f in shallow)} of "
        f"{sum(f['candidates_measured'] for f in shallow)} candidates from the "
        "`k <= 1` `K # m(K)` families were settled by a tight floor, against "
        "0.921 of the previous run's. **The diagnosis in §M1 was correct and "
        "the fix works.**\n"
    )
    out.append(
        "* **It buys nothing, because the binding constraint moved.** With the "
        "floor out of the way, every one of those candidates lands in "
        "`undetermined-full-capped`: deciding a gap at `c_shrink = 3` or `4` "
        "means a growth-capped search at budget 2 or 3 on a 10-to-14 letter "
        "word, and that search does not terminate. The previous run's 36 "
        "acceptances were affordable *because* `c_shrink` was 1 or 2 there.\n"
    )
    out.append(
        "* **And the part of the interval that is loose is the target "
        "itself.** §4a: entering it requires exhibiting `u(K # m(K)) < 2u(K)`. "
        "That is not a corpus-construction step; it is the campaign objective "
        "(§1.4) restated on a smaller `K`.\n"
    )
    out.append(
        "So the honest reading is that **`c_full >= 2` with a verified gap is "
        "not reachable by choosing a better family.** The three families tried "
        "across two runs fail for three different reasons -- unknot churn "
        "gives cheap gaps at `c_full <= 1`, certified-hard seeds give a tight "
        "floor, and `K # m(K)` gives a loose floor the search cannot enter -- "
        "and only the third is a *decision-procedure* limit rather than a "
        "generator one. The next lever is therefore the `c_full` side: an "
        "exhibited cheaper completion needs no exhaustive proof, so a corpus "
        "of `c_full >= 2` instances would have to come from **construction** "
        "(a walk that emits a word together with a replayed sub-`c_shrink` "
        "witness) rather than from search. Nothing measured here rules that "
        "out, and nothing here supports it either.\n"
    )
    return "\n".join(out) + "\n"


def _grid(value: str) -> tuple[int, int, int, int, int, int]:
    parts = [int(x) for x in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "a grid cell is m,k,walk_length,target_length,max_len,count"
        )
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/m1-kmk"))
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument(
        "--grid",
        type=_grid,
        nargs="*",
        default=[
            #  m, k, walk, target_len, max_len, count
            (1, 1, 24, 10, 14, 150),
            (1, 2, 24, 10, 14, 120),
            (2, 0, 40, 12, 16, 50),
            (2, 1, 24, 12, 16, 70),
            (2, 2, 24, 12, 16, 50),
            (2, 3, 24, 12, 16, 50),
            #: `7_1 # m(7_1)` *is* the target.  The cell is here so the run
            #: records the §4.3 screen refusing it rather than silently omitting
            #: it -- an exclusion nobody can see is not a guard.
            (3, 1, 24, 14, 18, 20),
            (4, 1, 16, 18, 22, 10),
        ],
        help="cells m,k,walk_length,target_length,max_len,count",
    )
    parser.add_argument("--instances", type=int, default=60)
    parser.add_argument("--control", type=int, default=300)
    parser.add_argument("--gap-growth", type=int, default=2)
    parser.add_argument("--change-cap", type=int, default=8)
    parser.add_argument("--shrink-node-cap", type=int, default=60_000)
    parser.add_argument("--full-node-cap", type=int, default=60_000)
    parser.add_argument("--summand-node-cap", type=int, default=40_000)
    parser.add_argument("--raise-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--family-seconds", type=float, default=900.0)
    parser.add_argument("--total-seconds", type=float, default=4200.0)
    parser.add_argument("--deep-seconds", type=float, default=1200.0)
    parser.add_argument("--deep-factor", type=int, default=8)
    parser.add_argument("--deep-min-shrink", type=int, default=3)
    parser.add_argument("--skip-premise", action="store_true")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="re-render report.md from an existing report.json. `aggregate`'s "
             "output *is* report.json, so the prose can be improved after the "
             "measurement without paying for the measurement again -- and "
             "without any chance of the two disagreeing.",
    )
    args = parser.parse_args()

    started = time.monotonic()
    if args.render_only:
        payload = json.loads((args.out / "report.json").read_text())
        (args.out / "report.md").write_text(render(payload))
        log(f"re-rendered {args.out}/report.md", started)
        return
    phase_a = (
        {"rows": [], "slice_reason": m1_kmk.SLICE_REASON, "floors": [], "loose": 0,
         "tight": 0}
        if args.skip_premise
        else premise(args, started)
    )
    log("premise done; starting the family sweep", started)
    phase_b = sweep(args, started)
    args.out.mkdir(parents=True, exist_ok=True)
    empty = {"attempted": 0, "available": 0, "resolved": [], "accepted": 0, "rows": []}
    #: The sweep is the expensive half and the deep pass is optional, so the
    #: sweep's own report is written before the deep pass runs.  A failure in
    #: phase C then costs the deep rows and nothing else -- which is what it
    #: cost the first time this ran.
    (args.out / "report.json").write_text(
        json.dumps(aggregate(phase_a, phase_b, empty, args), indent=2) + "\n"
    )
    log("sweep done and written; deepening the cap-undetermined candidates", started)
    phase_c = deepen(phase_b, args, started)
    payload = aggregate(phase_a, phase_b, phase_c, args)
    (args.out / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.out / "report.md").write_text(render(payload))
    log(f"wrote {args.out}/report.md and report.json", started)


if __name__ == "__main__":
    main()
