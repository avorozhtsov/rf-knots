"""Driver: retarget the M1 gate to `(cc, moves)`, then build a corpus that grows.

Two experiments, in one run, because the second only means something once the
first has been read.

**Phase 1 -- the retarget, on the gate's own states.**  Note 22 §2.1 rejects the
scalar `A*cc + B*moves` and asks for the lexicographic comparison; the gate as
built scored *shortest moves*, which is the objective the campaign threw out.
This phase takes the gate's corpus and split unchanged -- it imports them from
`m1_gate` rather than reimplementing them -- and computes **both** targets on
**the same states**, so the two baseline tables differ in the objective and in
nothing else.  The headline question is one bit: does `INSERT` enter the target
under `(cc, moves)`?

**Phase 2 -- a generator mode whose instances need growth, verified.**  For each
candidate word, `m1_growth.measure_growth_gap` measures

```text
c_shrink(W) = min crossing changes over completions that never INSERT or STABILIZE
c_full(W)   = min crossing changes over all completions, inside a growth cap
```

and keeps the instance only when `c_full < c_shrink` *with the shrink side
exhausted*.  Accepted instances are replayed and then re-measured at raised node
caps, and every capped search is reported in its own row rather than folded into
an acceptance rate.

    PYTHONPATH=src python research/experiments/m1_growth_run.py --out artifacts/m1-growth
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if __name__ == "__main__" and str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import m1_gate  # noqa: E402  (path shim above)
import m1_growth  # noqa: E402
from m1_target import (  # noqa: E402
    BASELINES,
    GateState,
    ScoredState,
    certified_change_floor,
    cluster_bootstrap,
    exhibit_target,
    score_state,
    shortest_completion,
    sites_of,
    summarise,
)

from rf_knots.actions import INSERT, KIND_NAMES, ActionSpec
from rf_knots.aplus import _Successors
from rf_knots.interleaved_generator import (
    UNKNOT_SEED,
    generate_many,
    load_quarantine,
)

KS = (1, 5, 10)
UNKNOTDB = m1_gate.UNKNOTDB


def log(message: str, started: float) -> None:
    print(f"[{time.monotonic() - started:7.1f}s] {message}", flush=True)


# -- phase 1: the same states, both objectives ---------------------------------


def retarget(built: dict, split: dict, args, started: float) -> dict:
    """Score every eval state under `moves` and under `lex`, matched pairwise.

    A state enters the matched tables only when **both** searches settled it.
    Comparing a `lex` mean taken on one set of states against a `moves` mean taken
    on another would attribute to the objective whatever the two caps did
    differently, and the two caps do behave differently -- the report says how.
    """
    corpus = built["corpus"]
    assignment = split["assignment"]
    train_states = split["train_states"]
    rankers = dict(BASELINES)
    rng = np.random.default_rng(args.seed + 2)
    rows: dict[str, list[ScoredState]] = {"moves": [], "lex": []}
    matched: dict[str, list[ScoredState]] = {"moves": [], "lex": []}
    stats: Counter = Counter()
    seconds: Counter = Counter()
    undetermined: dict[str, Counter] = {"moves": Counter(), "lex": Counter()}
    target_kinds: dict[str, Counter] = {"moves": Counter(), "lex": Counter()}
    insert_states: dict[str, int] = {"moves": 0, "lex": 0}
    exhibits: dict[str, Counter] = {"moves": Counter(), "lex": Counter()}
    optimum: list[dict] = []
    last = time.monotonic()

    work: list[tuple] = []
    for name in sorted(
        (name for name in m1_gate.CONFIGS if name in corpus),
        key=lambda name: (
            -m1_gate.CONFIGS[name]["config"].target_length,
            -m1_gate.CONFIGS[name]["config"].target_strands,
        ),
    ):
        entry = m1_gate.CONFIGS[name]
        spec = entry["config"].spec()
        for key, seed_name, spine in corpus[name]:
            if assignment.get(key) != "eval":
                continue
            succ = _Successors(spec)
            total = len(spine)
            depths = sorted(
                {
                    int(round(fraction * (total - 1)))
                    for fraction in np.linspace(0.0, 0.95, args.states_per_instance)
                }
            )
            for depth in depths:
                work.append((name, entry, spec, key, seed_name, spine, succ, depth))
    #: A seeded shuffle, for the same reason `m1_gate` goes round-robin: a wall
    #: cap that truncates in generation order deletes whole rows of the table --
    #: either the longest words or the in-distribution ones, depending on the
    #: order.  Taken at random, a cap thins every slice proportionally instead,
    #: and the shuffle is seeded so the run reproduces.
    order = np.random.default_rng(args.seed + 3).permutation(len(work))
    work = [work[index] for index in order]
    log(f"phase 1: {len(work)} eval states queued", started)

    for name, entry, spec, key, seed_name, spine, succ, depth in work:
        site = spine.sites[depth]
        stats["states_considered"] += 1
        if (site.state.word, site.state.strands) in train_states:
            stats["states_screened_out_for_appearing_in_training"] += 1
            continue
        word, strands = site.state.word, site.state.strands
        budget = site.remaining_crossing_changes
        floor = certified_change_floor(word, strands)
        found: dict[str, object] = {}
        for objective in ("moves", "lex"):
            begun = time.monotonic()
            found[objective] = shortest_completion(
                spec,
                word,
                strands,
                budget,
                growth=args.growth,
                node_cap=entry["node_cap"],
                successors=succ,
                objective=objective,
                change_floor=floor if objective == "lex" else None,
            )
            seconds[objective] += time.monotonic() - begun
        state = GateState(
            spec=spec,
            word=word,
            strands=strands,
            remaining_crossing_changes=budget,
            remaining_moves=site.remaining_moves,
            config=name,
            seed_name=seed_name,
            instance_id=key,
            identity=built["identities"][key],
            depth=depth,
            region="seed-program" if site.in_seed_program else "walk",
        )
        sites = sites_of(spec, word, strands, succ)
        scored_here: dict[str, ScoredState] = {}
        for objective in ("moves", "lex"):
            result = found[objective]
            if result.moves is None or not result.first_actions:  # type: ignore[union-attr]
                undetermined[objective][name] += 1
                continue
            claimed = list(result.first_actions)  # type: ignore[union-attr]
            #: `aplus.py`'s standard, kept: membership is *exhibited*, never
            #: inferred.  The target comes out of a mask propagation, so a
            #: sample of its sites is reconstructed as a concrete path and
            #: replayed by `UnknotWitness.from_actions`, which raises on any
            #: disagreement about either component of the optimum.  Claimed,
            #: selected and replayed are counted separately and the report
            #: prints all three.
            chosen = claimed
            if 0 < args.exhibit_per_state < len(claimed):
                picked = rng.choice(
                    len(claimed), size=args.exhibit_per_state, replace=False
                )
                chosen = [claimed[index] for index in sorted(picked)]
            replayed = exhibit_target(
                spec, word, strands, budget,
                dataclasses.replace(result, first_actions=tuple(chosen)),
            )
            exhibits[objective]["claimed"] += len(claimed)
            exhibits[objective]["selected"] += len(chosen)
            exhibits[objective]["replayed"] += len(replayed)
            scored = score_state(
                state, sites, claimed, result.moves, rankers, rng, KS  # type: ignore[union-attr]
            )
            rows[objective].append(scored)
            scored_here[objective] = scored
            inserts = sum(spec.decode(a)[0] == INSERT for a in claimed)
            insert_states[objective] += int(inserts > 0)
            for action in claimed:
                target_kinds[objective][KIND_NAMES[spec.decode(action)[0]]] += 1
        if len(scored_here) == 2:
            for objective in ("moves", "lex"):
                matched[objective].append(scored_here[objective])
            lex_result = found["lex"]
            moves_result = found["moves"]
            optimum.append(
                {
                    "config": name,
                    "word_length": len(word),
                    "strands": strands,
                    "budget": budget,
                    "certified_floor": floor,
                    "lex_crossing_changes": lex_result.crossing_changes,  # type: ignore[union-attr]
                    "lex_moves": lex_result.moves,  # type: ignore[union-attr]
                    "moves_moves": moves_result.moves,  # type: ignore[union-attr]
                    "moves_crossing_changes": moves_result.crossing_changes,  # type: ignore[union-attr]
                    "lex_target": len(lex_result.first_actions),  # type: ignore[union-attr]
                    "moves_target": len(moves_result.first_actions),  # type: ignore[union-attr]
                    "lex_target_inserts": sum(
                        spec.decode(a)[0] == INSERT
                        for a in lex_result.first_actions  # type: ignore[union-attr]
                    ),
                    "floor_is_tight": bool(
                        lex_result.crossing_changes == floor  # type: ignore[union-attr]
                    ),
                }
            )
        if time.monotonic() - last > 15.0:
            last = time.monotonic()
            log(
                f"  phase 1: {stats['states_considered']} considered, "
                f"{len(matched['lex'])} matched, "
                f"moves-undetermined {sum(undetermined['moves'].values())}, "
                f"lex-undetermined {sum(undetermined['lex'].values())} | "
                f"{seconds['moves']:.0f}s moves + {seconds['lex']:.0f}s lex",
                started,
            )
        if time.monotonic() - started > args.phase1_seconds:
            log("phase 1 wall cap reached", started)
            stats["wall_cap_reached"] = 1
            break
    return {
        "rows": rows,
        "matched": matched,
        "stats": dict(stats),
        "seconds": {key: round(value, 1) for key, value in seconds.items()},
        "undetermined": {
            objective: dict(counter) for objective, counter in undetermined.items()
        },
        "target_kinds": {
            objective: dict(counter) for objective, counter in target_kinds.items()
        },
        "insert_states": insert_states,
        "exhibits": {
            objective: dict(counter) for objective, counter in exhibits.items()
        },
        "optimum": optimum,
        "ranker_names": list(rankers),
    }


# -- phase 2: candidate families and the gap -----------------------------------


def families(args) -> list[m1_growth.CandidateFamily]:
    """Candidate constructions, each named for what it is trying to do.

    All of them use the shipped dials only; nothing about the walk is
    reimplemented here.  Three ideas are tried and the measurement separates
    them:

    * `churn-*` -- a long walk with the braid relations dominant and `INSERT`
      and `REDUCE` balanced, from an **unknot** seed.  The productive forward
      step is *commute or braid until two letters meet, change one, reduce the
      pair* -- the reduction destroys the crossing a cheap change would have
      used, and reversing it is an `INSERT` that has to happen **before** the
      change.  That is §Status's shape.
    * `shrink-*` -- the direction that suggests itself first: a long certified
      seed with a short drift target, so `_steer` multiplies `REDUCE` by
      `toward_target` and the whole walk shrinks.  Measured, it does not work,
      and the reason is in the table: the certified floor already equals
      `c_shrink`, so growth provably cannot help.
    * `control-*` -- §4.1's original path and a strong seed under the shipped
      growth-steered walk, so the report can separate "the seed did it" from
      "the walk direction did it".
    """
    small = ActionSpec(max_len=24, max_strands=5)
    fourteen = ActionSpec(max_len=14, max_strands=5)
    out = [
        m1_growth.churn_family(
            "churn-unknot-k1",
            UNKNOT_SEED,
            crossing_changes=1,
            walk_length=20,
            target_length=9,
            target_strands=4,
            max_len=12,
            max_strands=5,
            intent=(
                "one crossing change buried in a long churn: the walk reduces "
                "pairs it did not just insert, so the reversed witness has to "
                "grow before it can change"
            ),
            instances=getattr(args, "churn_instances", None),
        ),
        m1_growth.churn_family(
            "churn-unknot-k2",
            UNKNOT_SEED,
            crossing_changes=2,
            walk_length=20,
            target_length=11,
            target_strands=4,
            max_len=14,
            max_strands=5,
            intent="the same walk one change deeper and two letters longer",
            instances=getattr(args, "churn_instances", None),
        ),
        m1_growth.churn_family(
            "churn-unknot-k1-long",
            UNKNOT_SEED,
            crossing_changes=1,
            walk_length=40,
            target_length=9,
            target_strands=4,
            max_len=12,
            max_strands=5,
            intent="twice the churn at the same depth, to price `l` against `k`",
            instances=getattr(args, "churn_instances", None),
        ),
        m1_growth.churn_family(
            "churn-T(2,5)-k2",
            m1_growth.torus(2, fourteen),
            crossing_changes=2,
            walk_length=20,
            target_length=11,
            target_strands=4,
            max_len=14,
            max_strands=5,
            intent=(
                "the same churn from a seed with a certified `u = 2`, to ask "
                "whether a hard *knot* and a hard *diagram* are the same thing"
            ),
        ),
        m1_growth.shrink_family(
            "shrink-T(2,11)",
            m1_growth.torus(5, small),
            crossing_changes=3,
            walk_length=3,
            target_length=5,
            target_strands=2,
            max_len=24,
            max_strands=5,
            intent=(
                "T(2,11), certified `u = 5`, walked down to a five-letter "
                "target: the forward-REDUCE direction, tried and measured"
            ),
        ),
        m1_growth.shrink_family(
            "shrink-P(3,g=4)",
            m1_growth.positive(3, 4, small),
            crossing_changes=3,
            walk_length=3,
            target_length=5,
            target_strands=2,
            max_len=24,
            max_strands=5,
            intent="the positive-braid variant of the same direction",
        ),
        m1_growth.CandidateFamily(
            name="control-unknot",
            seed=UNKNOT_SEED,
            config=m1_gate.CONFIGS["C-small"]["config"],
            intent=(
                "§4.1's original path, and the family the published gate corpus "
                "is drawn from: the control this experiment exists to beat"
            ),
        ),
        m1_growth.CandidateFamily(
            name="control-T(2,11)-grow",
            seed=m1_growth.torus(5, small),
            config=m1_gate.CONFIGS["A-main"]["config"],
            intent=(
                "a strong seed under the *shipped* growth-steered walk, to "
                "separate 'the seed did it' from 'the walk direction did it'"
            ),
        ),
    ]
    if args.families:
        wanted = set(args.families)
        out = [family for family in out if family.name in wanted]
    return out


def sweep_families(args, started: float) -> dict:
    """Generate candidates family by family and measure every one of them."""
    quarantine = load_quarantine(unknotdb_root=UNKNOTDB)
    rows: list[dict] = []
    per_family: dict[str, dict] = {}
    last = time.monotonic()
    for family in families(args):
        spec = family.config.spec()
        reason = quarantine.screen_seed(family.seed)
        if reason is not None:  # pragma: no cover - the seeds are clean
            raise SystemExit(f"seed {family.seed.name}: {reason}")
        counts: Counter = Counter()
        gaps: list[int] = []
        seconds = 0.0
        screened = 0
        seen: set[tuple] = set()
        wanted = family.instances or args.instances
        for instance in generate_many(
            family.config,
            seed=args.seed,
            count=wanted,
            seed_representation=family.seed,
        ):
            if quarantine.screen_word(instance.word, instance.strands) is not None:
                screened += 1
                continue
            fingerprint = (instance.word, instance.strands)
            if fingerprint in seen:
                counts["duplicate"] += 1
                continue
            seen.add(fingerprint)
            begun = time.monotonic()
            #: The instance's own witness is a replayed completion, so it is a
            #: free upper bound on `c_full` -- and one that carries no growth
            #: cap, unlike the search.  It is handed in rather than trusted:
            #: `measure_growth_gap` replays it and checks its change count.
            witness_actions = (
                tuple(step.action.to_flat(spec) for step in instance.witness.steps)
                if instance.witness is not None
                else ()
            )
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
            if measured.accepted:
                measured = m1_growth.verify_at_raised_caps(
                    spec, measured, factor=args.raise_factor
                )
            seconds += time.monotonic() - begun
            counts[measured.verdict] += 1
            counts["witness_grows"] += int(
                m1_growth.counts_growth(spec, witness_actions) > 0
            )
            if measured.accepted:
                gaps.append(int(measured.gap or 0))
                counts["survived_raised_caps"] += int(
                    bool(measured.survived_raised_caps)
                )
            row = measured.to_dict()
            row["family"] = family.name
            row["seed"] = family.seed.name
            row["seed_certified_lower_bound"] = family.seed.certified_lower_bound
            row["walk_depth"] = family.config.crossing_changes
            row["walk_length"] = family.config.walk_length
            row["instance_id"] = instance.instance_id
            row["own_witness_grows"] = bool(
                m1_growth.counts_growth(spec, witness_actions)
            )
            row["own_witness_crossing_changes"] = instance.certified_upper_bound
            row["witness_kinds"] = sorted(
                {
                    KIND_NAMES[spec.decode(action)[0]]
                    for action in measured.witness_actions
                }
            )
            rows.append(row)
            if time.monotonic() - last > 15.0:
                last = time.monotonic()
                log(
                    f"  {family.name}: {sum(counts.values())} measured, "
                    f"{counts['growth-required']} accepted, {seconds:.0f}s",
                    started,
                )
            if time.monotonic() - started > args.total_seconds:
                break
        measured_total = sum(
            counts[name] for name in m1_growth.VERDICTS
        )
        determined = measured_total - (
            counts["undetermined-shrink-capped"] + counts["undetermined-full-capped"]
        )
        per_family[family.name] = {
            "provenance": family.provenance(),
            "candidates_measured": measured_total,
            "screened_by_quarantine": screened,
            "duplicates": counts["duplicate"],
            "verdicts": {name: counts[name] for name in m1_growth.VERDICTS},
            "determined": determined,
            "accepted": counts["growth-required"],
            "acceptance_over_measured": (
                round(counts["growth-required"] / measured_total, 4)
                if measured_total
                else None
            ),
            "acceptance_over_determined": (
                round(counts["growth-required"] / determined, 4) if determined else None
            ),
            "survived_raised_caps": counts["survived_raised_caps"],
            "instances_whose_own_witness_grows": counts["witness_grows"],
            "gap_distribution": dict(sorted(Counter(gaps).items())),
            "seconds": round(seconds, 1),
            "seconds_per_accepted": (
                round(seconds / counts["growth-required"], 3)
                if counts["growth-required"]
                else None
            ),
        }
        log(
            f"{family.name}: {measured_total} measured, "
            f"{counts['growth-required']} accepted, "
            f"{counts['undetermined-shrink-capped'] + counts['undetermined-full-capped']}"
            f" undetermined, {seconds:.0f}s",
            started,
        )
        if time.monotonic() - started > args.total_seconds:
            log("total wall cap reached; stopping the family sweep", started)
            break
    return {"rows": rows, "per_family": per_family}


# -- reporting -----------------------------------------------------------------


def _row(cells) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _fmt(value) -> str:
    if value is None:
        return "--"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _table(summary: dict, names: list[str], label: str = "slice") -> list[str]:
    header = [label, "states", "chance", *names]
    lines = [_row(header), _row(["---", "---:", "---:", *["---:"] * len(names)])]
    for key, cells in summary.items():
        if not cells or not cells.get("states"):
            continue
        lines.append(
            _row(
                [
                    f"`{key}`",
                    cells["states"],
                    _fmt(cells.get("uniform_chance")),
                    *[_fmt(cells.get(name)) for name in names],
                ]
            )
        )
    return lines


def aggregate(phase1: dict, phase2: dict, args) -> dict:
    names = phase1["ranker_names"]
    payload: dict = {
        "schema": "rf-knots-m1-growth-v0",
        "generated": time.strftime("%Y-%m-%d"),
        "run_metadata": {
            "seed": args.seed,
            "growth_cap_phase1": args.growth,
            "growth_cap_gap": args.gap_growth,
            "change_cap": args.change_cap,
            "shrink_node_cap": args.shrink_node_cap,
            "full_node_cap": args.full_node_cap,
            "raise_factor": args.raise_factor,
            "instances_per_family": args.instances,
            "churn_instances_per_family": getattr(args, "churn_instances", None),
            "states_per_instance": args.states_per_instance,
            "eval_fraction": args.eval_fraction,
        },
        "phase1": {
            "states_considered": phase1["stats"].get("states_considered", 0),
            "matched_states": len(phase1["matched"]["lex"]),
            "scored_states": {
                objective: len(rows) for objective, rows in phase1["rows"].items()
            },
            "undetermined": phase1["undetermined"],
            "undetermined_total": {
                objective: sum(counter.values())
                for objective, counter in phase1["undetermined"].items()
            },
            "seconds": phase1["seconds"],
            "insert_states": phase1["insert_states"],
            "exhibits": phase1["exhibits"],
            "target_kinds": phase1["target_kinds"],
            "gate_matched": {
                objective: _table_payload(phase1["matched"][objective], names)
                for objective in ("moves", "lex")
            },
            "gate_all_scored": {
                objective: _table_payload(phase1["rows"][objective], names)
                for objective in ("moves", "lex")
            },
            "bootstrap_lex_growth_aware_minus_kind_prior": cluster_bootstrap(
                phase1["matched"]["lex"], "all@1", "growth_aware", "kind_prior",
                draws=args.bootstrap, seed=args.seed,
            ),
        },
        "phase2": phase2["per_family"],
    }
    payload["phase1"]["optimum"] = _optimum_summary(phase1["optimum"])
    payload["phase2_totals"] = _phase2_totals(phase2)
    payload["phase2_rows"] = phase2["rows"]
    return payload


def _table_payload(rows: list[ScoredState], names: list[str]) -> dict:
    out = {"all": summarise(rows, "all@1", names)}
    by_config: dict[str, list[ScoredState]] = defaultdict(list)
    by_length: dict[str, list[ScoredState]] = defaultdict(list)
    for row in rows:
        by_config[row.state.config].append(row)
        by_length[m1_gate.length_bucket(len(row.state.word))].append(row)
    out["by_config"] = {
        key: summarise(value, "all@1", names) for key, value in sorted(by_config.items())
    }
    out["by_word_length"] = {
        key: summarise(value, "all@1", names) for key, value in sorted(by_length.items())
    }
    out["recall_at_5"] = summarise(rows, "all@5", names)
    out["insert_block"] = summarise(rows, "insert@1", names)
    out["rest_block"] = summarise(rows, "rest@1", names)
    out["mean_target_size"] = (
        round(statistics.fmean(len(row.target) for row in rows), 3) if rows else None
    )
    out["mean_legal_sites"] = (
        round(statistics.fmean(row.legal for row in rows), 3) if rows else None
    )
    out["states_with_an_insert_in_the_target"] = sum(
        1 for row in rows if row.target_insert
    )
    return out


def _optimum_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"states": 0}
    cheaper = [
        row for row in rows
        if row["lex_crossing_changes"] < row["moves_crossing_changes"]
    ]
    longer = [row for row in rows if row["lex_moves"] > row["moves_moves"]]
    return {
        "states": len(rows),
        "mean_lex_crossing_changes": round(
            statistics.fmean(row["lex_crossing_changes"] for row in rows), 3
        ),
        "mean_moves_objective_crossing_changes": round(
            statistics.fmean(row["moves_crossing_changes"] for row in rows), 3
        ),
        "mean_lex_moves": round(statistics.fmean(row["lex_moves"] for row in rows), 3),
        "mean_moves_objective_moves": round(
            statistics.fmean(row["moves_moves"] for row in rows), 3
        ),
        "states_where_lex_spends_fewer_changes": len(cheaper),
        "states_where_lex_spends_more_moves": len(longer),
        "states_where_the_certified_floor_is_tight": sum(
            1 for row in rows if row["floor_is_tight"]
        ),
        "lex_target_inserts": sum(row["lex_target_inserts"] for row in rows),
    }


def _phase2_totals(phase2: dict) -> dict:
    rows = phase2["rows"]
    verdicts = Counter(row["verdict"] for row in rows)
    accepted = [row for row in rows if row["verdict"] == "growth-required"]
    gaps = Counter(row["gap"] for row in accepted)
    return {
        "candidates_measured": len(rows),
        "verdicts": dict(verdicts),
        "accepted": len(accepted),
        "acceptance_over_measured": (
            round(len(accepted) / len(rows), 4) if rows else None
        ),
        "gap_distribution": dict(sorted(gaps.items())),
        "accepted_with_a_replayed_witness": sum(
            1 for row in accepted if row["witness_replayed"]
        ),
        "accepted_surviving_raised_caps": sum(
            1 for row in accepted if row["survived_raised_caps"]
        ),
        "undetermined_by_a_cap": verdicts["undetermined-shrink-capped"]
        + verdicts["undetermined-full-capped"]
        + verdicts["undetermined-change-cap"],
        "accepted_c_full_distribution": dict(
            sorted(Counter(row["c_full"] for row in accepted).items())
        ),
        "candidates_whose_own_witness_grows": sum(
            1 for row in rows if row["own_witness_grows"]
        ),
        "seconds": round(sum(row["seconds"] for row in rows), 1),
        "seconds_per_accepted": (
            round(sum(row["seconds"] for row in rows) / len(accepted), 3)
            if accepted
            else None
        ),
        "mean_growth_moves_in_an_accepted_witness": (
            round(
                statistics.fmean(
                    row["witness_growth_moves"] for row in accepted
                    if row["witness_replayed"]
                ),
                3,
            )
            if any(row["witness_replayed"] for row in accepted)
            else None
        ),
    }


def render(payload: dict, args) -> str:
    names = payload["_ranker_names"]
    phase1 = payload["phase1"]
    totals = payload["phase2_totals"]
    lines: list[str] = []
    add = lines.append
    add("# M1: the retargeted gate, and a corpus that has to grow")
    add("")
    add(
        f"`{payload['schema']}` | generated {payload['generated']} | "
        f"{phase1['matched_states']} states scored under both objectives | "
        f"{totals['candidates_measured']} growth candidates measured"
    )
    add("")
    add(
        "Two coupled defects, fixed and measured together. The gate scored "
        "**shortest moves**, which is the objective note 22 §2.1 rejects; and the "
        "generator emitted only instances a shrinking search can solve. Phase 1 "
        "retargets and re-runs on the gate's own states; phase 2 builds "
        "candidates whose optimum needs growth and *measures*, per instance, "
        "whether it does."
    )
    add("")
    add(
        "**One correction to §M1's diagnosis, before its numbers are read.** "
        "§M1 says the reversed witness *never contains an `INSERT`*, because the "
        "forward walk inserts and `REDUCE` is the inverse of `INSERT`. Measured, "
        "that is not so: the forward walk also **reduces** and **destabilises**, "
        f"so {payload['phase2_totals']['candidates_whose_own_witness_grows']} of "
        f"{totals['candidates_measured']} candidates here carry a witness that "
        "does contain a growth move. What the corpus lacks is not growth in the "
        "witness -- it is growth that is **necessary**, and those are different "
        "properties measured differently. The rest of §M1's conclusion survives "
        "the correction: the phenomenon was absent, and the reason it was absent "
        "is what §2 measures."
    )
    add("")
    add("---")
    add("")
    add("## 0. Verdict")
    add("")
    insert_lex = phase1["insert_states"]["lex"]
    insert_moves = phase1["insert_states"]["moves"]
    add(
        f"1. **The retarget alone does not put `INSERT` in the target.** Under "
        f"`(cc, moves)`, {insert_lex} of {phase1['scored_states']['lex']} scored "
        f"states have an `INSERT` in the optimal-first-action set, against "
        f"{insert_moves} of {phase1['scored_states']['moves']} under shortest "
        f"moves. The objective was one of the two defects and it was not the "
        f"binding one: **on this corpus growth never lowers the crossing-change "
        f"count, because the corpus was built by a walk that could only shrink.**"
        if insert_lex == 0 else
        f"1. **The retarget does put `INSERT` in the target, and only just.** "
        f"Under `(cc, moves)`, **{insert_lex}** of "
        f"{phase1['scored_states']['lex']} scored states have an `INSERT` in the "
        f"optimal-first-action set, against {insert_moves} of "
        f"{phase1['scored_states']['moves']} under shortest moves. So the "
        f"objective was a real defect -- under shortest moves the block is empty "
        f"by an identity, and under `(cc, moves)` it is not -- but on **this** "
        f"corpus it is empty in fact anyway, because the corpus was built by a "
        f"walk whose witnesses shrink. Fixing the objective without fixing the "
        f"generator would have moved the measured rate from 0.000 to "
        f"{insert_lex / max(phase1['scored_states']['lex'], 1):.3f}."
    )
    add("")
    add(
        f"2. **What the retarget costs, measured.** Over the same states the "
        f"lexicographic search spent {phase1['seconds'].get('lex', 0)} s against "
        f"{phase1['seconds'].get('moves', 0)} s for the move-ordered one, and "
        f"left {phase1['undetermined_total']['lex']} states undetermined against "
        f"{phase1['undetermined_total']['moves']}. Proving `c*` means proving "
        f"nothing cheaper exists, which the move objective never has to do; the "
        f"certified lower bound is what makes it affordable at all, and it was "
        f"tight -- `c* == L` -- at "
        f"{phase1['optimum'].get('states_where_the_certified_floor_is_tight', 0)} "
        f"of {phase1['optimum'].get('states', 0)} matched states."
    )
    add("")
    accepted = totals["accepted"]
    add(
        f"3. **Growth-requiring instances exist, and the generator can build "
        f"them.** {accepted} of {totals['candidates_measured']} candidates were "
        f"kept on a *measured* `c_full < c_shrink`; "
        f"{totals['accepted_with_a_replayed_witness']} carry a replayed witness "
        f"that uses growth, and {totals['accepted_surviving_raised_caps']} are "
        f"still accepted with both node caps multiplied by {args.raise_factor}. "
        f"{totals['undetermined_by_a_cap']} candidates were left undetermined by "
        f"a cap and are **not** counted as either outcome."
        if accepted else
        f"3. **No candidate construction tried here produced a proven gap.** "
        f"{totals['candidates_measured']} candidates were measured and "
        f"{accepted} were accepted; {totals['undetermined_by_a_cap']} were left "
        f"undetermined by a cap. The families and their intents are in §3, and "
        f"the negative is reported as a negative."
    )
    add("")
    add(
        f"4. **And the shape of the negative is the most useful thing here.** "
        f"{totals['verdicts'].get('no-gap-certified', 0)} of "
        f"{totals['candidates_measured']} candidates are settled with **no search "
        f"at all**: the certified lower bound already equals `c_shrink`, so "
        f"`c_full >= u >= L = c_shrink` and growth provably cannot help. Every "
        f"accepted instance has `c_full` in "
        f"{sorted(int(key) for key in totals['accepted_c_full_distribution'])} "
        f"-- they are unknots "
        f"and one-change knots wearing hard diagrams. **The gap lives in the "
        f"diagram, not in the knot**, and that is why seeding harder (§4.1b's "
        f"answer to the last defect) is the wrong lever for this one: a "
        f"certified-hard seed makes `L` large, and a large `L` is exactly what "
        f"proves the gap away."
    )
    add("")
    add("---")
    add("")
    add("## 1. Phase 1: the same states, both objectives")
    add("")
    add(
        "Every number in this section is computed on the **matched** set -- the "
        "states both searches settled -- so the only difference between the two "
        "tables is the objective. States either search left undetermined are "
        "counted in §1.3 and never scored as failures."
    )
    add("")
    for objective, title in (
        ("moves", "shortest moves -- the objective the published gate scored"),
        ("lex", "`(cc, moves)` lexicographic -- note 22 §2.1's objective"),
    ):
        block = phase1["gate_matched"][objective]
        add(f"### recall@1 under {title}")
        add("")
        add(
            f"mean legal sites {block['mean_legal_sites']}, mean target size "
            f"{block['mean_target_size']}, states with an `INSERT` in the target "
            f"**{block['states_with_an_insert_in_the_target']}**"
        )
        add("")
        summary = {"whole state, recall@1": block["all"], **block["by_config"]}
        lines.extend(_table(summary, names))
        add("")
    add("### The optimum itself, matched")
    add("")
    optimum = phase1["optimum"]
    lines.extend(
        [
            _row(["quantity", "value"]),
            _row(["---", "---:"]),
            *[
                _row([key.replace("_", " "), _fmt(value)])
                for key, value in optimum.items()
            ],
        ]
    )
    add("")
    inverted = optimum.get("states_where_lex_spends_fewer_changes", 0)
    total_matched = optimum.get("states", 0)
    add(
        f"The two optima differ on **{inverted}** of {total_matched} matched "
        f"states: that is how often the move-ordered target spends a crossing "
        f"change it did not have to. §2.1 measured that inversion costing a "
        f"whole crossing change at target scale; at these word lengths the two "
        f"objectives mostly agree, and this report says so rather than "
        f"borrowing the target's number."
        if inverted == 0 else
        f"The two optima differ on **{inverted}** of {total_matched} matched "
        f"states -- the move-ordered target spending a crossing change to save "
        f"moves, which is exactly the inversion §2.1 rejects the scalar "
        f"objective for."
    )
    add("")
    add("### 1.2 Every counted site, replayed")
    add("")
    add(
        "The target comes out of a mask propagation, so a site it names is a "
        "claim until a concrete path is reconstructed and replayed. "
        "`exhibit_target` does that and **raises** on any disagreement about "
        "either component of the optimum, so the run finishing is the assertion. "
        "Claimed, selected and replayed are separate columns because sampling is "
        "not verification of the rest."
    )
    add("")
    lines.extend(
        [
            _row(["objective", "target sites claimed", "selected", "replayed"]),
            _row(["---", "---:", "---:", "---:"]),
            *[
                _row(
                    [
                        f"`{objective}`",
                        phase1["exhibits"][objective].get("claimed", 0),
                        phase1["exhibits"][objective].get("selected", 0),
                        phase1["exhibits"][objective].get("replayed", 0),
                    ]
                )
                for objective in ("moves", "lex")
            ],
        ]
    )
    add("")
    add("### 1.3 What a cap left undetermined")
    add("")
    lines.extend(
        [
            _row(["objective", "undetermined states", "by configuration"]),
            _row(["---", "---:", "---"]),
            *[
                _row(
                    [
                        f"`{objective}`",
                        phase1["undetermined_total"][objective],
                        ", ".join(
                            f"{key} {value}"
                            for key, value in sorted(
                                phase1["undetermined"][objective].items()
                            )
                        )
                        or "--",
                    ]
                )
                for objective in ("moves", "lex")
            ],
        ]
    )
    add("")
    add("---")
    add("")
    add("## 2. Phase 2: `c_full < c_shrink`, measured per instance")
    add("")
    add("```text")
    add("c_shrink(W) = min crossing changes over completions that never INSERT or STABILIZE")
    add("c_full(W)   = min crossing changes over all completions, inside a growth cap")
    add("```")
    add("")
    add(
        "`c_full <= c_shrink` always. A gap is kept only when the **shrink** side "
        "exhausted its space -- an exhausted search is a lower bound, a capped one "
        "is not -- and the full side exhibited a completion that replays through "
        "`UnknotWitness.from_actions`. An accepted witness must contain an "
        "`INSERT` or a `STABILIZE`, since a growth-free completion at `c_full` "
        "would contradict `c_shrink`; that is asserted on every accepted instance."
    )
    add("")
    add("### Acceptance by family")
    add("")
    header = [
        "family", "seed", "seed L", "k", "l", "drawn", "measured",
        "witness grows", "accepted", "acceptance", "undetermined", "s/accepted",
    ]
    lines.append(_row(header))
    lines.append(_row(["---"] + ["---:"] * (len(header) - 1)))
    for name, block in payload["phase2"].items():
        verdicts = block["verdicts"]
        undetermined = (
            verdicts["undetermined-shrink-capped"]
            + verdicts["undetermined-full-capped"]
        )
        lines.append(
            _row(
                [
                    f"`{name}`",
                    block["provenance"]["seed"],
                    block["provenance"]["seed_certified_lower_bound"],
                    block["provenance"]["config"]["crossing_changes"],
                    block["provenance"]["config"]["walk_length"],
                    block["candidates_measured"] + block["duplicates"]
                    + block["screened_by_quarantine"],
                    block["candidates_measured"],
                    block["instances_whose_own_witness_grows"],
                    block["accepted"],
                    _fmt(block["acceptance_over_measured"]),
                    undetermined,
                    _fmt(block["seconds_per_accepted"]),
                ]
            )
        )
    add("")
    add("### Verdicts, in full")
    add("")
    header = ["family", *m1_growth.VERDICTS]
    lines.append(_row(header))
    lines.append(_row(["---"] + ["---:"] * len(m1_growth.VERDICTS)))
    for name, block in payload["phase2"].items():
        lines.append(
            _row([f"`{name}`", *[block["verdicts"][key] for key in m1_growth.VERDICTS]])
        )
    add("")
    add(
        "`no-gap-certified` is settled with no search at all: the certified lower "
        "bound already equals `c_shrink`, and `c_full >= u >= L`, so growth "
        "cannot help. `no-gap-within-growth-cap` is a proof relative to the "
        "stated growth cap and nothing more. The two `undetermined-*` rows are "
        "the honest cost of the caps."
    )
    add("")
    add("### The gap distribution")
    add("")
    if totals["gap_distribution"]:
        lines.append(_row(["`c_shrink - c_full`", "accepted instances"]))
        lines.append(_row(["---:", "---:"]))
        for gap, count in totals["gap_distribution"].items():
            lines.append(_row([gap, count]))
        add("")
        add(
            f"Mean growth moves in an accepted, replayed witness: "
            f"{totals['mean_growth_moves_in_an_accepted_witness']}."
        )
    else:
        add("No accepted instance, so no distribution to show.")
    add("")
    accepted_rows = [
        row for row in payload["phase2_rows"]
        if row["verdict"] == "growth-required" and row["witness_replayed"]
    ]
    if accepted_rows:
        add("### Accepted instances, in full")
        add("")
        add(
            "Every row below is a word, a growth-free optimum that was searched "
            "**to exhaustion**, and a replayed completion strictly cheaper than "
            "it. The `witness kinds` column is the point: the cheaper completion "
            "cannot avoid `INSERT` or `STABILIZE`, because a growth-free one at "
            "that cost is exactly what the left-hand column ruled out."
        )
        add("")
        header = [
            "family", "word", "n", "c_shrink", "c_full", "gap", "witness moves",
            "growth moves", "witness kinds", "survived raise",
        ]
        lines.append(_row(header))
        lines.append(_row(["---"] * 3 + ["---:"] * 5 + ["---", "---"]))
        for row in accepted_rows[: args.examples]:
            word = "".join(
                f"{'s' if letter > 0 else 'S'}{abs(letter)}" for letter in row["word"]
            )
            lines.append(
                _row(
                    [
                        f"`{row['family']}`",
                        f"`{word}`",
                        row["strands"],
                        row["c_shrink"],
                        row["c_full"],
                        row["gap"],
                        len(row["witness_actions"]),
                        row["witness_growth_moves"],
                        ", ".join(f"`{kind}`" for kind in row["witness_kinds"]),
                        row["survived_raised_caps"],
                    ]
                )
            )
        add("")
        add(
            "`sN` is `sigma_N` and `SN` is its inverse. Every one of these words "
            "is a state on which the shortest-moves target of the published gate "
            "would have ranked `INSERT` dead last and been right about the moves "
            "and wrong about the answer."
        )
        add("")
    add("### What the caps cost")
    add("")
    lines.extend(
        [
            _row(["quantity", "value"]),
            _row(["---", "---:"]),
            _row(["candidates measured", totals["candidates_measured"]]),
            _row(["undetermined by a cap", totals["undetermined_by_a_cap"]]),
            _row(
                [
                    "accepted, re-measured with both caps raised",
                    f"x{args.raise_factor}",
                ]
            ),
            _row(
                ["accepted surviving the raise", totals["accepted_surviving_raised_caps"]]
            ),
            _row(["seconds, phase 2", totals["seconds"]]),
            _row(["seconds per accepted instance", _fmt(totals["seconds_per_accepted"])]),
        ]
    )
    add("")
    add("---")
    add("")
    add("## 3. The candidate constructions, and what each was for")
    add("")
    for name, block in payload["phase2"].items():
        provenance = block["provenance"]
        add(f"* **`{name}`** -- seed `{provenance['seed']}` (certified "
            f"`L = {provenance['seed_certified_lower_bound']}`, "
            f"{provenance['seed_letters']} letters on "
            f"{provenance['seed_strands']} strands), `k = "
            f"{provenance['config']['crossing_changes']}`, `l = "
            f"{provenance['config']['walk_length']}`, drift target "
            f"{provenance['config']['target_length']} letters on "
            f"{provenance['config']['target_strands']} strands. "
            f"{provenance['intent']}.")
    add("")
    add(
        "Every family uses the shipped `InterleavedConfig` dials and the shipped "
        "`_steer`; nothing about the walk is reimplemented here. The three "
        "`shrink-*` families differ from the corpus the gate was built on in one "
        "respect only: the drift target is **shorter** than the seed, so `REDUCE` "
        "and `DESTABILIZE` carry the `toward_target` multiplier and the walk "
        "spends its budget shrinking. Reversing a shrinking walk is a growing "
        "witness -- but a growing witness is not a *necessary* growth, and the "
        "acceptance column above is the only thing that decides which it is."
    )
    add("")
    add("---")
    add("")
    add("## 4. Threats to these numbers")
    add("")
    add(
        f"* `c_full` is searched inside a growth cap of `+{args.gap_growth}` "
        f"letters. A completion needing more growth than that is outside the "
        f"searched subgraph, so `c_full` is an **upper** bound and every accepted "
        f"gap is a *lower* bound on the true gap. `no-gap-within-growth-cap` is "
        f"correspondingly a statement about the cap."
    )
    add(
        f"* `c_shrink` is exact only where the search **exhausted**. Where it hit "
        f"its {args.shrink_node_cap:,}-node cap the instance is "
        f"`undetermined-shrink-capped` and appears in no acceptance rate."
    )
    add(
        "* The certified floor is a theorem from `lower_bounds.computed_claims` "
        "-- signature, Montesinos, tau -- computed from the word and never read "
        "from a table. If it were unsound the lexicographic search would reach "
        "the unknot below it, which raises rather than being recorded."
    )
    add(
        "* Phase 1's corpus, split, quarantine and node caps are `m1_gate`'s, "
        "imported rather than copied, so the two reports are about the same "
        "states. What is *not* shared is the objective, which is the point."
    )
    add(
        "* The words here are 3-24 letters on 2-6 strands. Nothing measured is "
        "evidence about the target's 254 letters on 21 strands."
    )
    add("")
    add("## 5. Reproduce")
    add("")
    add("```bash")
    add(
        "OMP_NUM_THREADS=1 PYTHONPATH=src python "
        "research/experiments/m1_growth_run.py \\"
    )
    add(
        f"    --out {args.out} --instances {args.instances} "
        f"--churn-instances {getattr(args, 'churn_instances', args.instances)} "
        f"--states-per-instance {args.states_per_instance} "
        f"--gap-growth {args.gap_growth}"
    )
    add("```")
    add("")
    add(
        "Modules: `research/experiments/m1_target.py` (the retargeted target), "
        "`research/experiments/m1_growth.py` (the gap measurement and the "
        "candidate families), `research/experiments/m1_growth_run.py` (this "
        "driver); tests: `tests/test_m1_gate.py`, `tests/test_m1_growth.py`."
    )
    add("")
    add(
        f"**Run metadata.** {json.dumps(payload['run_metadata'], sort_keys=True)}"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/m1-growth"))
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--states-per-instance", type=int, default=3)
    parser.add_argument("--eval-fraction", type=float, default=0.5)
    parser.add_argument("--growth", type=int, default=3)
    parser.add_argument("--gap-growth", type=int, default=2)
    parser.add_argument(
        "--exhibit-per-state", type=int, default=3,
        help="0 replays every target site; n replays a random n of them. Every "
             "target site is demonstrated by the mask propagation either way; "
             "this is how many are additionally replayed through UnknotWitness.",
    )
    parser.add_argument("--change-cap", type=int, default=8)
    parser.add_argument("--instances", type=int, default=60)
    #: The three `churn-unknot-*` families are the cheap ones -- most candidates
    #: are settled by the certified floor with no search -- and they are the only
    #: ones that produce gaps, so they get the bigger draw.
    parser.add_argument("--churn-instances", type=int, default=300)
    parser.add_argument("--shrink-node-cap", type=int, default=60_000)
    parser.add_argument("--full-node-cap", type=int, default=60_000)
    parser.add_argument("--raise-factor", type=int, default=4)
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--phase1-seconds", type=float, default=1200.0)
    parser.add_argument("--total-seconds", type=float, default=3000.0)
    parser.add_argument("--skip-phase1", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    if args.skip_phase1:
        phase1 = {
            "rows": {"moves": [], "lex": []},
            "matched": {"moves": [], "lex": []},
            "stats": {},
            "seconds": {},
            "undetermined": {"moves": {}, "lex": {}},
            "target_kinds": {"moves": {}, "lex": {}},
            "insert_states": {"moves": 0, "lex": 0},
            "exhibits": {"moves": {}, "lex": {}},
            "optimum": [],
            "ranker_names": list(BASELINES),
        }
    else:
        built = m1_gate.build_corpus(args, started)
        split = m1_gate.split_corpus(built, args, started)
        phase1 = retarget(built, split, args, started)
    log("phase 1 done; starting the family sweep", started)
    phase2 = sweep_families(args, started)
    payload = aggregate(phase1, phase2, args)
    payload["_ranker_names"] = phase1["ranker_names"]
    args.out.mkdir(parents=True, exist_ok=True)
    report = render(payload, args)
    payload.pop("_ranker_names")
    (args.out / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.out / "report.md").write_text(report)
    log(f"wrote {args.out}/report.md and report.json", started)


if __name__ == "__main__":
    main()
