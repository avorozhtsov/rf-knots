r"""The revised M1 gate: the harness that judges the per-site head.

Note 22 §M1 records that the *original* gate was built, measured, and fell: a
ranker that reads only the action kind and never looks at the word scored
recall@10 = 0.846 against a bar of 0.80, and under the witness's own remaining
budget uniform random scored 0.953.  This is the gate §M1 revised to, built so
that a bad head fails it -- and one measurement, made while building it, that
says one of the three revisions cannot be run as written.

**What §M1's revised gate asks for, and what happened to each part.**

1. *Shortest completion, not feasibility.*  Implemented.  The target is `A*(s)`,
   the sites that begin a **shortest** completion under the crossing-change
   budget the witness had left, found by an `f`-bounded A* that exhausts the
   whole `f <= m*` layer and so is exact for the searched subgraph, and then
   replayed by `UnknotWitness.from_actions`.
2. *recall@1, split by region and word length.*  Implemented, plus strand count
   and remaining crossing changes.
3. *Gate on the `INSERT` block, with `kind_prior` as the baseline.*  **This one
   is not well posed, and the harness is what found out.**  Counting the kinds of
   any completion gives `moves = (L + n - 1)/2 + 2i + 2s + c` exactly, so every
   insertion costs two moves nothing pays back; and `INSERT(p, g, s)` is undone
   by `REDUCE(p)`, so every `INSERT` site has a completion of exactly `m* + 2`.
   Measured: the shortest-completion target contains **no** `INSERT` site
   anywhere, and the committed block optimum is `m* + 2` with the whole block
   attaining it.  The block is not hard to rank -- it is *flat*, exactly, under
   any target derived from distance to the unknot.

So the gate this script runs is recall@1 over the whole state against the
shortest-completion target, on held-out identities, with `kind_prior` as the
baseline and a transfer row at a strand count absent from training.  The
`INSERT` block is 70-84% of the legal sites and 0% of the target, so that task
*is* "where may the diagram grow" in its answerable form: recognise that it may
not, then rank what is left.  The `INSERT` block and the non-`INSERT` block are
both reported beside it, labelled, and neither is the gate.

Evidence tiers (§5.3b) stay in separate columns.  The target is tier 1:
demonstrated by search and replayed.  Tier 2, exact commutation closure, is
reported beside it and never merged.  Tier 3 (attested, a catalogue `u` reached
through an identification) is declared and empty -- nothing here consults a
catalogue.

    PYTHONPATH=src python research/experiments/m1_gate.py --out artifacts/m1-gate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path

import numpy as np

if __name__ == "__main__" and str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from m1_target import (  # noqa: E402  (path shim above)
    BASELINES,
    GateState,
    ScoredState,
    assign_split,
    block_target,
    chance_at,
    cluster_bootstrap,
    completion_path,
    exhibit_target,
    identity_key,
    order_by_scores,
    recall_at,
    score_state,
    shortest_completion,
    sites_of,
    summarise,
    undo_completion,
)

from rf_knots.actions import INSERT, KIND_NAMES
from rf_knots.aplus import _Successors, commutation_known_good, witness_spine
from rf_knots.evidence import UnknotWitness
from rf_knots.interleaved_generator import (
    UNKNOT_SEED,
    InterleavedConfig,
    generate_many,
    load_quarantine,
    torus_seed,
)

UNKNOTDB = "/Users/artemvorozhtsov/projects/unknotdb"

KS = (1, 5, 10)

#: `role` decides what a configuration is *for*.  `train-family` configurations
#: are split at the identity level; `transfer` ones are evaluation-only and are
#: additionally screened against the training identities and state keys.
CONFIGS: dict[str, dict] = {
    "C-small": {
        "role": "train-family",
        "config": InterleavedConfig(
            crossing_changes=2, walk_length=4, target_strands=3,
            target_length=8, max_len=16, max_strands=4,
        ),
        "instances": 48,
        "node_cap": 20_000,
    },
    "A-main": {
        "role": "train-family",
        "config": InterleavedConfig(
            crossing_changes=3, walk_length=8, target_strands=3,
            target_length=12, max_len=24, max_strands=4,
        ),
        "instances": 48,
        "node_cap": 20_000,
    },
    "M-mid": {
        "role": "transfer",
        "config": InterleavedConfig(
            crossing_changes=3, walk_length=12, target_strands=4,
            target_length=16, max_len=32, max_strands=5,
        ),
        "instances": 30,
        "node_cap": 20_000,
    },
    "W-wide": {
        "role": "transfer",
        "config": InterleavedConfig(
            crossing_changes=3, walk_length=10, target_strands=6,
            target_length=12, max_len=24, max_strands=7,
        ),
        "instances": 30,
        "node_cap": 20_000,
    },
    "B-long": {
        "role": "transfer",
        "config": InterleavedConfig(
            crossing_changes=4, walk_length=16, target_strands=4,
            target_length=20, max_len=40, max_strands=5,
        ),
        "instances": 12,
        #: Measured, not guessed.  Raising this to 60,000 was tried, on the
        #: argument that the cap decides which of the longest words the gate gets
        #: to see; it converted no undetermined state and made each failure three
        #: times more expensive -- 20 s to give up rather than 7.  The states this
        #: configuration cannot settle need orders of magnitude more search, not a
        #: factor of three, so the cheap cap is kept and the bias is reported in
        #: full instead of being partially bought off.
        "node_cap": 20_000,
    },
}


def log(message: str, started: float) -> None:
    print(f"[{time.monotonic() - started:7.1f}s] {message}", flush=True)


def length_bucket(length: int) -> str:
    for low, high in ((0, 4), (5, 8), (9, 12), (13, 16), (17, 20), (21, 24)):
        if low <= length <= high:
            return f"{low:02d}-{high:02d}"
    return "25+"


def seeds_for(config: InterleavedConfig) -> list:
    spec = config.spec()
    return [UNKNOT_SEED, torus_seed(1, spec), torus_seed(2, spec)]


@dataclass
class BlockRow:
    """One state's `INSERT` block, measured against the committed target."""

    config: str
    instance_id: str
    region: str
    word_length: int
    strands: int
    shortest_moves: int
    block_sites: int
    committed_best: int | None
    excess_over_shortest: int | None
    target: tuple[int, ...]
    discriminating: bool
    undo_sites_exhibited: int
    undo_all_at_shortest_plus_two: bool
    hits: dict[str, int] = field(default_factory=dict)
    chance: float = 0.0


# -- phase 1: the corpus -------------------------------------------------------


def build_corpus(args, started: float) -> dict:
    """Generate, screen, and re-index every instance the gate may look at."""
    quarantine = load_quarantine(unknotdb_root=UNKNOTDB)
    screened: Counter[str] = Counter()
    corpus: dict[str, list] = {}
    identities: dict[str, str] = {}
    for name, entry in CONFIGS.items():
        if args.configs and name not in args.configs:
            continue
        config: InterleavedConfig = entry["config"]
        rows = []
        for seed_representation in seeds_for(config):
            reason = quarantine.screen_seed(seed_representation)
            if reason is not None:  # pragma: no cover - the seeds are clean
                raise SystemExit(f"seed {seed_representation.name}: {reason}")
            count = max(1, entry["instances"] // 3)
            for instance in generate_many(
                config,
                seed=args.seed,
                count=count,
                seed_representation=seed_representation,
            ):
                rejection = quarantine.screen_word(instance.word, instance.strands)
                if rejection is not None:
                    screened[rejection] += 1
                    continue
                spine = witness_spine(instance)
                key = f"{name}/{spine.instance_id[:16]}"
                identities[key] = identity_key(instance.word, instance.strands)
                rows.append((key, seed_representation.name, spine))
        corpus[name] = rows
        log(
            f"{name}: {len(rows)} instances, "
            f"{sum(len(s) for _k, _n, s in rows)} states",
            started,
        )
    return {
        "corpus": corpus,
        "identities": identities,
        "quarantine": quarantine.provenance(),
        "screened_out": dict(screened),
    }


# -- phase 2: the identity-level split ----------------------------------------


def split_corpus(built: dict, args, started: float) -> dict:
    """Hold out whole identity groups, then screen every eval state exactly."""
    corpus = built["corpus"]
    train_names = [
        name for name, entry in CONFIGS.items()
        if entry["role"] == "train-family" and name in corpus
    ]
    payload = [
        (key, spine.states[0].word, spine.states[0].strands,
         [(state.word, state.strands) for state in spine.states])
        for name in train_names
        for key, _seed, spine in corpus[name]
    ]
    assignment, report = assign_split(payload, args.eval_fraction, seed=args.seed + 1)

    train_states: set[tuple[tuple[int, ...], int]] = set()
    train_strands: Counter[int] = Counter()
    train_groups: set[str] = set()
    for name in train_names:
        for key, _seed, spine in corpus[name]:
            if assignment.get(key) != "train":
                continue
            train_groups.add(built["identities"][key])
            for state in spine.states:
                train_states.add((state.word, state.strands))
                train_strands[state.strands] += 1

    #: A transfer instance whose identity group is a *training* group is dropped
    #: outright.  The transfer configurations were generated independently, so
    #: this is a real possibility and it has to be measured rather than assumed
    #: away.
    transfer_dropped = 0
    for name, entry in CONFIGS.items():
        if entry["role"] != "transfer" or name not in corpus:
            continue
        for key, _seed, _spine in corpus[name]:
            if built["identities"][key] in train_groups:
                assignment[key] = "dropped-identity-in-training"
                transfer_dropped += 1
            else:
                assignment[key] = "eval"
    max_train_strands = max(train_strands) if train_strands else 0
    log(
        f"split: {report.train_instances} train / {report.eval_instances} eval "
        f"in-distribution instances over {report.identities} identity groups; "
        f"{transfer_dropped} transfer instances dropped for a training identity; "
        f"training strand counts {sorted(train_strands)}",
        started,
    )
    return {
        "assignment": assignment,
        "report": report,
        "train_states": train_states,
        "train_strand_counts": tuple(sorted(train_strands)),
        "max_train_strands": max_train_strands,
        "transfer_instances_dropped_for_training_identity": transfer_dropped,
        "train_names": train_names,
    }


# -- phase 3: the target, and the scores ---------------------------------------


def _measure_insert_block(
    spec, state_word, strands, budget, sites, shortest_moves, path, args, succ, rng
) -> tuple[object, int, bool]:
    """The committed block target, plus the constructive `m* + 2` exhibits.

    The exhibits are the point: `undo_completion` builds insert-undo-go with no
    search at all, and replaying it shows -- rather than argues -- that the site
    reaches the unknot in `m* + 2` moves.  If every site in the block does, the
    block is flat and there is nothing in it for any ranker to find.
    """
    inserts = [site.action for site in sites if site.kind == INSERT]
    if len(inserts) < 2:
        return None, 0, False
    measured = block_target(
        spec, state_word, strands, budget, inserts, shortest_moves,
        growth=args.growth, node_cap=args.block_node_cap, successors=succ,
        exhibit=False,
    )
    exhibited = 0
    all_two = True
    if path is not None:
        sample = list(inserts)
        if args.undo_exhibits and len(sample) > args.undo_exhibits:
            picked = rng.choice(len(sample), size=args.undo_exhibits, replace=False)
            sample = [sample[index] for index in sorted(picked)]
        for action in sample:
            witness = UnknotWitness.from_actions(
                state_word, strands, spec,
                undo_completion(spec, state_word, strands, action, path),
            )
            exhibited += 1
            all_two &= witness.moves == shortest_moves + 2
    return measured, exhibited, all_two


def evaluate(built: dict, split: dict, args, started: float) -> dict:
    """Compute the shortest-completion target on every eval state, and score it."""
    corpus = built["corpus"]
    assignment = split["assignment"]
    train_states = split["train_states"]
    rankers = dict(BASELINES)
    head_status = None
    if args.head is not None:
        import m1_head

        #: `--head untrained` is the control the gate needs most: a `SiteHead` at
        #: initialisation, scored by the identical code on the identical states.
        #: A harness that cannot fail *that* cannot fail anything, and the
        #: original gate's failure was exactly of this kind -- it was passed by a
        #: ranker with no inputs.
        untrained = args.head == "untrained"
        checkpoint = None if untrained else args.head
        head_status = m1_head.status(args.head_geometry, args.cyclic_band_generators)
        name = "site_head_untrained" if untrained else "site_head"
        rankers[name] = m1_head.build_ranker(
            checkpoint,
            geometry=args.head_geometry,
            cyclic_band_generators=args.cyclic_band_generators,
        )
        log(
            f"learned head wired in as {name!r} under positional_geometry="
            f"{args.head_geometry!r}: {head_status.detail}",
            started,
        )
    rng = np.random.default_rng(args.seed + 2)
    rows: list[ScoredState] = []
    blocks: list[BlockRow] = []
    stats = Counter()
    seconds = Counter()
    undetermined_by: dict[str, Counter] = {"config": Counter(), "word_length": Counter()}
    considered_by: dict[str, Counter] = {"config": Counter(), "word_length": Counter()}
    last = time.monotonic()
    block_quota = {name: args.block_states_per_config for name in CONFIGS}
    #: Round-robin across configurations rather than finishing one before
    #: starting the next.  The sweep has a hard wall cap, and a cap that
    #: truncates config-by-config deletes whole rows of the table -- either the
    #: longest words or the in-distribution ones, depending on the order.  Taken
    #: round-robin, a cap thins every row proportionally instead, so a sweep that
    #: runs out of time still reports every configuration and every slice.
    queues: list[list[tuple]] = []
    for name in sorted(
        (name for name in CONFIGS if name in corpus),
        key=lambda name: (
            -CONFIGS[name]["config"].target_length,
            -CONFIGS[name]["config"].target_strands,
        ),
    ):
        entry = CONFIGS[name]
        spec = entry["config"].spec()
        items: list[tuple] = []
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
                items.append((name, entry, spec, key, seed_name, spine, succ, depth))
        queues.append(items)
        log(f"  {name}: {len(items)} eval states queued", started)
    work: list[tuple] = []
    for row in zip_longest(*queues):
        work.extend(item for item in row if item is not None)
    log(f"round-robin work list: {len(work)} states", started)

    for name, entry, spec, key, seed_name, spine, succ, depth in work:
        node_cap = entry["node_cap"]

        site = spine.sites[depth]
        stats["states_considered"] += 1
        considered_by["config"][name] += 1
        bucket = length_bucket(len(site.state.word))
        considered_by["word_length"][bucket] += 1
        if (site.state.word, site.state.strands) in train_states:
            stats["states_screened_out_for_appearing_in_training"] += 1
            continue
        begun = time.monotonic()
        found = shortest_completion(
            spec,
            site.state.word,
            site.state.strands,
            site.remaining_crossing_changes,
            growth=args.growth,
            node_cap=node_cap,
            successors=succ,
        )
        seconds["search"] += time.monotonic() - begun
        if found.moves is None or not found.first_actions:
            stats["states_dropped_undetermined"] += 1
            undetermined_by["config"][name] += 1
            undetermined_by["word_length"][bucket] += 1
            #: Log here too.  A stretch of undetermined states used to be
            #: silent, which is the one stretch worth hearing about: it
            #: is the search failing, and on a long word it can run for
            #: minutes without a single state reaching the tables.
            if time.monotonic() - last > 15.0:
                last = time.monotonic()
                log(
                    f"  {name}: {stats['states_scored']} scored, "
                    f"{stats['states_dropped_undetermined']} undetermined "
                    f"| {seconds['search']:.0f}s search",
                    started,
                )
            if time.monotonic() - started > args.seconds:
                log("wall cap reached; stopping the sweep", started)
                return _finish(rows, blocks, stats, seconds, undetermined_by,
                               considered_by, rankers, head_status, True)
            continue
        begun = time.monotonic()
        claimed = list(found.first_actions)
        chosen = claimed
        if 0 < args.exhibit_per_state < len(claimed):
            picked = rng.choice(
                len(claimed), size=args.exhibit_per_state, replace=False
            )
            chosen = [claimed[index] for index in sorted(picked)]
        replayed = exhibit_target(
            spec, site.state.word, site.state.strands,
            site.remaining_crossing_changes,
            type(found)(
                found.moves, tuple(chosen), found.legal_actions,
                found.expanded, found.rounds, found.complete, found.reason,
                found.growth, found.node_cap, found.candidates, found.commit,
            ),
        )
        seconds["exhibit"] += time.monotonic() - begun
        stats["target_sites_claimed"] += len(claimed)
        stats["target_sites_selected_for_replay"] += len(chosen)
        stats["target_sites_replayed"] += len(replayed)
        stats["target_sites_replay_gave_up"] += len(chosen) - len(replayed)
        stats["states_fully_exhibited"] += int(len(chosen) == len(claimed))
        stats["witness_action_in_target"] += int(site.action in claimed)
        stats["witness_optimal"] += int(found.moves == site.remaining_moves)
        stats["shortest_over_witness_remaining"] += int(
            found.moves > site.remaining_moves
        )
        tier2 = commutation_known_good(spine, depth, args.commutation_window)
        tier2_actions = tuple(sorted({record.action for record in tier2}))
        stats["tier2_sites"] += len(tier2_actions)
        overlap = len(set(tier2_actions) & set(claimed))
        stats["tier2_sites_in_target"] += overlap

        state = GateState(
            spec=spec,
            word=site.state.word,
            strands=site.state.strands,
            remaining_crossing_changes=site.remaining_crossing_changes,
            remaining_moves=site.remaining_moves,
            config=name,
            seed_name=seed_name,
            instance_id=key,
            identity=built["identities"][key],
            depth=depth,
            region="seed-program" if site.in_seed_program else "walk",
        )
        sites = sites_of(spec, site.state.word, site.state.strands, succ)
        scored = score_state(
            state, sites, claimed, found.moves, rankers, rng, KS
        )
        scored.exhibited = len(replayed)
        scored.tier2_sites = tier2_actions
        scored.tier2_in_target = overlap
        rows.append(scored)
        stats["states_scored"] += 1

        if block_quota[name] > 0:
            begun = time.monotonic()
            path = completion_path(
                spec, site.state.word, site.state.strands,
                site.remaining_crossing_changes, found.moves,
                min(len(site.state.word) + args.growth, spec.max_len),
            )
            measured, exhibited, all_two = _measure_insert_block(
                spec, site.state.word, site.state.strands,
                site.remaining_crossing_changes, sites, found.moves,
                path, args, succ, rng,
            )
            seconds["block"] += time.monotonic() - begun
            if measured is not None:
                block_quota[name] -= 1
                blocks.append(
                    _block_row(state, measured, exhibited, all_two,
                               sites, rankers, rng)
                )
        if time.monotonic() - last > 15.0:
            last = time.monotonic()
            log(
                f"  {name}: {stats['states_scored']} scored, "
                f"{stats['states_dropped_undetermined']} undetermined, "
                f"{len(blocks)} block rows | "
                f"{seconds['search']:.0f}s search "
                f"+ {seconds['exhibit']:.0f}s exhibit "
                f"+ {seconds['block']:.0f}s block",
                started,
            )
        if time.monotonic() - started > args.seconds:
            log("wall cap reached; stopping the sweep", started)
            return _finish(rows, blocks, stats, seconds, undetermined_by,
                           considered_by, rankers, head_status, True)
    return _finish(rows, blocks, stats, seconds, undetermined_by, considered_by,
                   rankers, head_status, False)


def _block_row(state, measured, exhibited, all_two, sites, rankers, rng) -> BlockRow:
    inserts = [site for site in sites if site.kind == INSERT]
    positives = {
        index for index, site in enumerate(inserts) if site.action in measured.target
    }
    row = BlockRow(
        config=state.config,
        instance_id=state.instance_id,
        region=state.region,
        word_length=len(state.word),
        strands=state.strands,
        shortest_moves=measured.baseline_moves,
        block_sites=len(measured.block),
        committed_best=measured.best_moves,
        excess_over_shortest=(
            None if measured.best_moves is None
            else measured.best_moves - measured.baseline_moves
        ),
        target=measured.target,
        discriminating=measured.discriminating,
        undo_sites_exhibited=exhibited,
        undo_all_at_shortest_plus_two=all_two,
    )
    if positives:
        for name, ranker in rankers.items():
            order = order_by_scores(ranker(state, inserts), rng)
            row.hits[name] = recall_at(order, positives, 1)
        row.chance = chance_at(len(inserts), len(positives), 1)
    return row


def _finish(rows, blocks, stats, seconds, undetermined_by, considered_by,
            rankers, head_status, capped):
    return {
        "rows": rows,
        "blocks": blocks,
        "stats": dict(stats),
        "seconds": {key: round(value, 1) for key, value in seconds.items()},
        "undetermined_by": {
            axis: dict(counter) for axis, counter in undetermined_by.items()
        },
        "considered_by": {
            axis: dict(counter) for axis, counter in considered_by.items()
        },
        "ranker_names": list(rankers),
        "head_status": None if head_status is None else head_status.to_dict(),
        "wall_cap_reached": capped,
    }


# -- phase 4: the tables -------------------------------------------------------


def slice_tables(rows: list[ScoredState], names: list[str], metric: str) -> dict:
    """One metric, sliced every way the gate has to be read."""
    out: dict[str, dict] = {"all": summarise(rows, metric, names)}
    axes = {
        "by_config": lambda row: row.state.config,
        "by_region": lambda row: row.state.region,
        "by_word_length": lambda row: length_bucket(len(row.state.word)),
        "by_strands": lambda row: f"n={row.state.strands}",
        "by_remaining_crossing_changes": (
            lambda row: f"cc={row.state.remaining_crossing_changes}"
        ),
    }
    for axis, key in axes.items():
        grouped: dict[str, list[ScoredState]] = defaultdict(list)
        for row in rows:
            grouped[key(row)].append(row)
        out[axis] = {
            value: summarise(subset, metric, names)
            for value, subset in sorted(grouped.items())
        }
    return out


def insert_block_rows(rows, minimum: int = 2):
    """States where the `INSERT` block is a real question.

    Two conditions, both necessary and both stated in the report.  The block must
    hold at least `minimum` sites, or there is nothing to order; and it must hold
    at least one target site, or no ranking of it can be right.  The measured
    answer is that the second condition is met at no state, which is the finding
    rather than a filter.
    """
    return [row for row in rows if row.insert_legal >= minimum and row.target_insert]


def _block_summary(blocks: list[BlockRow], names: list[str]) -> dict:
    scored = [row for row in blocks if row.hits]
    discriminating = [row for row in scored if row.discriminating]
    out = {
        "states_measured": len(blocks),
        "states_where_the_block_splits": len(discriminating),
        "mean_block_size": round(
            sum(row.block_sites for row in blocks) / max(len(blocks), 1), 1
        ),
        "committed_optimum_is_shortest_plus_two": sum(
            1 for row in blocks if row.excess_over_shortest == 2
        ),
        "committed_optimum_excess_histogram": dict(
            sorted(Counter(row.excess_over_shortest for row in blocks).items(),
                   key=lambda item: (item[0] is None, item[0]))
        ),
        "whole_block_attains_the_optimum": sum(
            1 for row in blocks if row.target and len(row.target) == row.block_sites
        ),
        "undo_sites_exhibited": sum(row.undo_sites_exhibited for row in blocks),
        "every_exhibited_undo_at_shortest_plus_two": all(
            row.undo_all_at_shortest_plus_two for row in blocks if row.undo_sites_exhibited
        ),
    }
    if discriminating:
        out["recall@1_where_the_block_splits"] = {
            "states": len(discriminating),
            "uniform_chance": round(
                float(np.mean([row.chance for row in discriminating])), 4
            ),
            **{
                name: round(
                    float(np.mean([row.hits.get(name, 0) for row in discriminating])), 4
                )
                for name in names
            },
        }
    return out


def aggregate(result: dict, split: dict, built: dict, args) -> dict:
    rows: list[ScoredState] = result["rows"]
    blocks: list[BlockRow] = result["blocks"]
    names: list[str] = result["ranker_names"]
    stats = result["stats"]
    walk = [row for row in rows if row.state.region == "walk"]
    insert_rows = insert_block_rows(rows)
    transfer_rows = [
        row for row in rows if row.state.strands > split["max_train_strands"]
    ]
    kind_histogram: Counter[str] = Counter()
    legal = insert_legal = 0
    for row in rows:
        for action in row.target:
            kind_histogram[KIND_NAMES[row.state.spec.decode(action)[0]]] += 1
        legal += row.legal
        insert_legal += row.insert_legal
    target_total = sum(len(row.target) for row in rows) or 1
    gate = {
        "definition": (
            "recall@1 over the whole state, on the shortest-completion target, "
            "over held-out identities"
        ),
        "baseline": "kind_prior",
        "states": len(rows),
        "recall@1": summarise(rows, "all@1", names),
        "recall@1_walk_region_only": summarise(walk, "all@1", names),
        "recall@5": summarise(rows, "all@5", names),
        "residual_recall@1_non_INSERT_sites_only": summarise(rows, "rest@1", names),
        "transfer_row": {
            "strands_absent_from_training": sorted(
                {row.state.strands for row in transfer_rows}
            ),
            "training_strand_counts": list(split["train_strand_counts"]),
            "states": len(transfer_rows),
            "recall@1": summarise(transfer_rows, "all@1", names),
        },
        "paired_bootstrap_over_instances": {
            f"{name}_minus_kind_prior": cluster_bootstrap(
                rows, "all@1", name, "kind_prior", draws=args.bootstrap
            )
            for name in names
            if name != "kind_prior"
        },
    }
    return {
        "gate": gate,
        "insert_block_is_not_a_gate": {
            "why": (
                "moves = (L + n - 1)/2 + 2i + 2s + c exactly, so every insertion "
                "costs two moves nothing pays back; and INSERT(p, g, s) is undone "
                "by REDUCE(p), so every INSERT site reaches the unknot in exactly "
                "m* + 2 moves. The block is flat under any target derived from "
                "distance to the unknot -- not hard to rank, but with nothing in "
                "it to rank."
            ),
            "insert_sites_in_the_shortest_completion_target": kind_histogram.get(
                "INSERT", 0
            ),
            "states_with_an_INSERT_in_the_target": sum(
                1 for row in rows if row.target_insert
            ),
            "committed_block_target": _block_summary(blocks, names),
        },
        "whole_state": {
            f"recall@{k}": slice_tables(rows, names, f"all@{k}") for k in KS
        },
        #: The table §M1's revised gate asks for, printed whatever it says.  It
        #: says nothing, because no state has an `INSERT` in the target -- and a
        #: gate whose table is empty has to show the empty table rather than
        #: quietly substitute a different one.
        "insert_block": {
            "states_with_a_scorable_insert_block": len(insert_rows),
            "states_with_at_least_two_insert_sites": sum(
                1 for row in rows if row.insert_legal >= 2
            ),
            "recall@1": slice_tables(insert_rows, names, "insert@1"),
        },
        "residual_block": {
            "note": (
                "the sites that are not INSERT, ranked among themselves. INSERT "
                "is most of the legal set and none of the target, so the whole-"
                "state number is dominated by a decision kind_prior makes for "
                "free; this is where the ordering a head could learn actually "
                "lives. It contains the crossing-change block, which §M1 "
                "excludes from the gate, so it is a diagnostic and not the gate."
            ),
            "mean_non_insert_sites": round(
                sum(row.rest_legal for row in rows) / max(len(rows), 1), 2
            ),
            **{
                f"recall@{k}": slice_tables(rows, names, f"rest@{k}") for k in (1, 5)
            },
        },
        "target": {
            "states_scored": len(rows),
            "mean_target_size": round(target_total / max(len(rows), 1), 3),
            "mean_legal_sites": round(legal / max(len(rows), 1), 2),
            "mean_insert_sites": round(insert_legal / max(len(rows), 1), 2),
            "insert_share_of_legal": round(insert_legal / max(legal, 1), 4),
            "insert_share_of_target": round(
                kind_histogram.get("INSERT", 0) / target_total, 4
            ),
            "target_share_of_legal": round(target_total / max(legal, 1), 4),
            "target_kind_histogram": dict(kind_histogram.most_common()),
            "mean_shortest_completion_moves": round(
                sum(row.shortest_moves for row in rows) / max(len(rows), 1), 2
            ),
            "mean_witness_remaining_moves": round(
                sum(row.state.remaining_moves for row in rows) / max(len(rows), 1), 2
            ),
            "witness_action_on_a_shortest_completion": round(
                stats.get("witness_action_in_target", 0) / max(len(rows), 1), 4
            ),
            "witness_itself_shortest": round(
                stats.get("witness_optimal", 0) / max(len(rows), 1), 4
            ),
            "states_where_the_growth_cap_beat_the_witness": stats.get(
                "shortest_over_witness_remaining", 0
            ),
        },
        "evidence_tiers": {
            "tier1_demonstrated": {
                "what": (
                    "a completion was found by search and replayed by "
                    "UnknotWitness.from_actions at exactly the claimed length"
                ),
                "target_sites_claimed": stats.get("target_sites_claimed", 0),
                "target_sites_selected_for_replay": stats.get(
                    "target_sites_selected_for_replay", 0
                ),
                "target_sites_replayed": stats.get("target_sites_replayed", 0),
                "target_sites_whose_path_extraction_hit_its_cap": stats.get(
                    "target_sites_replay_gave_up", 0
                ),
                "states_where_every_target_site_was_replayed": stats.get(
                    "states_fully_exhibited", 0
                ),
                "replay_disagreements": 0,
            },
            "tier2_exact_closure": {
                "what": (
                    "a later witness action bubbled to the front through adjacent "
                    "swaps, each swap decided by application; the completion is "
                    "the same multiset of moves and therefore costs what the "
                    "witness cost, which is why it lands in the shortest set only "
                    "when the witness was itself optimal"
                ),
                "sites": stats.get("tier2_sites", 0),
                "sites_also_in_the_shortest_completion_target": stats.get(
                    "tier2_sites_in_target", 0
                ),
                "merged_into_the_gate": False,
            },
            "tier3_attested": {
                "what": (
                    "a successor whose identity carries a smaller catalogue u "
                    "(§4.4); it composes to a catalogue-composed bound and is not "
                    "demonstrated in this move set"
                ),
                "sites": 0,
                "consulted": False,
                "merged_into_the_gate": False,
            },
        },
        "split": {
            **split["report"].to_dict(),
            "train_strand_counts": list(split["train_strand_counts"]),
            "transfer_strand_counts": sorted(
                {row.state.strands for row in transfer_rows}
            ),
            "transfer_instances_dropped_for_a_training_identity": split[
                "transfer_instances_dropped_for_training_identity"
            ],
            "eval_states_screened_out_for_appearing_in_training": stats.get(
                "states_screened_out_for_appearing_in_training", 0
            ),
            "eval_states_kept": stats.get("states_scored", 0),
            "quarantine": built["quarantine"],
            "quarantine_screened_out": built["screened_out"],
        },
        "search": {
            "states_considered": stats.get("states_considered", 0),
            "states_dropped_undetermined": stats.get("states_dropped_undetermined", 0),
            "undetermined_fraction": round(
                stats.get("states_dropped_undetermined", 0)
                / max(stats.get("states_considered", 1), 1),
                4,
            ),
            "undetermined_by_config": result["undetermined_by"]["config"],
            "undetermined_by_word_length": result["undetermined_by"]["word_length"],
            "states_considered_by_config": result["considered_by"]["config"],
            "states_considered_by_word_length": result["considered_by"]["word_length"],
            "seconds": result["seconds"],
            "node_caps": {name: entry["node_cap"] for name, entry in CONFIGS.items()},
            "growth_cap": args.growth,
        },
    }


# -- the written report --------------------------------------------------------


def _row(cells) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _fmt(value) -> str:
    return "--" if value is None else f"{value:.3f}"


def _table(summary: dict, names: list[str], label: str = "slice") -> list[str]:
    header = [label, "states", "chance", *names]
    lines = [_row(header), _row(["---"] + ["---:"] * (len(header) - 1))]
    for value, cell in summary.items():
        if not isinstance(cell, dict) or "states" not in cell:
            continue
        lines.append(
            _row(
                [
                    f"`{value}`",
                    cell["states"],
                    _fmt(cell.get("uniform_chance")),
                    *[_fmt(cell.get(name)) for name in names],
                ]
            )
        )
    return lines


def render(payload: dict, args) -> str:
    gate = payload["gate"]
    names = payload["ranker_names"]
    target = payload["target"]
    split = payload["split"]
    search = payload["search"]
    tiers = payload["evidence_tiers"]
    flat = payload["insert_block_is_not_a_gate"]
    block = flat["committed_block_target"]
    out: list[str] = []
    add = out.append

    add("# M1: the revised gate, and one revision that cannot be run as written")
    add("")
    add(
        f"`{payload['schema']}` | generated {payload['generated']} | "
        f"{target['states_scored']} held-out states over "
        f"{split['eval_identity_groups']} held-out identity groups | "
        f"{search['seconds'].get('search', 0):.0f} s optimal search, "
        f"{search['seconds'].get('exhibit', 0):.0f} s replay, "
        f"{search['seconds'].get('block', 0):.0f} s block search, one core"
    )
    add("")
    add(
        "Note 22 §M1 records that the original gate **was measured, and it "
        "fell**: a ranker seeing only the action kind scored recall@10 = 0.846 "
        "against a bar of 0.80, and under the witness's own remaining budget "
        "uniform random scored 0.953. §M1 then revised it in three ways. Two of "
        "them are implemented here. The third cannot be run as written, and this "
        "harness is what found out."
    )
    add("")
    add("---")
    add("")
    add("## 0. Verdict")
    add("")
    add(
        f"1. **The `INSERT`-block gate is not well posed.** The shortest-"
        f"completion target contains "
        f"**{flat['insert_sites_in_the_shortest_completion_target']}** `INSERT` "
        f"sites across all {target['states_scored']} scored states, and the "
        f"committed block optimum is `m* + 2` at "
        f"{block['committed_optimum_is_shortest_plus_two']} of "
        f"{block['states_measured']} states measured, attained by the *whole "
        f"block* at {block['whole_block_attains_the_optimum']} of them. The "
        f"block is not hard to rank; it is flat. §1 gives the reason, and it is "
        f"an identity rather than a measurement."
    )
    add(
        f"2. **The gate that is well posed is the whole state**, and `INSERT` is "
        f"{target['insert_share_of_legal']:.1%} of it. Recognising that growth is "
        f"never optimal and then ordering what is left is the same branching-"
        f"factor reduction §M1 wants, and it is measurable: chance is "
        f"{_fmt(gate['recall@1'].get('uniform_chance'))} at recall@1 and "
        f"`kind_prior` reaches {_fmt(gate['recall@1'].get('kind_prior'))}."
    )
    strongest = max(
        (name for name in names if not name.startswith("site_head")),
        key=lambda name: gate["recall@1"].get(name) or 0.0,
    )
    best = gate["recall@1"].get(strongest) or 0.0
    add(
        f"3. **The strongest hand-written baseline is `{strongest}` at "
        f"{best:.3f}**, which leaves {1 - best:.3f} of headroom. That is the "
        f"number a head has to beat, and it is close enough to the ceiling that "
        f"**this gate can fail a bad head but cannot meaningfully pass a good "
        f"one.** Under the shortest-completion target the per-site problem is "
        f"very nearly solved by a seven-line kind ordering that never looks at "
        f"the word; what is left for a head to learn at recall@1 is a few states "
        f"in a hundred."
    )
    control = next((name for name in names if name.startswith("site_head")), None)
    if control is not None:
        scored_control = gate["recall@1"].get(control) or 0.0
        add(
            f"4. **It does fail a bad head, which is the property the gate that "
            f"fell did not have.** `{control}` -- a `SiteHead` at initialisation, "
            f"scored by the identical code on the identical states through the "
            f"same `rank(state, sites) -> scores` interface -- reaches "
            f"{scored_control:.3f}, against `kind_prior`'s {best:.3f} and a "
            f"chance of {_fmt(gate['recall@1'].get('uniform_chance'))}. The old "
            f"gate was *passed* by a ranker with no inputs; this one separates an "
            f"untrained head from the hand-written baseline by "
            f"{best - scored_control:.3f}."
        )
    add("")
    add("---")
    add("")
    add("## 1. Why the `INSERT` block cannot be gated on")
    add("")
    add(
        "Count the moves of any completion by kind: `r` reductions, `i` "
        "insertions, `s` stabilisations, `d` destabilisations and `c` "
        "length-preserving moves (`COMMUTE`, `BRAID`, `CROSSING_CHANGE`). The "
        "move set forces two balances -- `n + s - d = 1` on strands and "
        "`L + 2i + s - 2r - d = 0` on letters -- and substituting them into "
        "`r + i + s + d + c` gives"
    )
    add("")
    add("```text")
    add("moves = (L + n - 1) / 2 + 2i + 2s + c")
    add("```")
    add("")
    add(
        "**exactly**, which is checked by counting the kinds of real witness "
        "suffixes rather than argued. Two consequences:"
    )
    add("")
    add(
        "* every insertion costs two moves and nothing in the identity pays them "
        "back, so an `INSERT` reaches a shortest completion only by buying back "
        "two length-preserving moves;"
    )
    add(
        "* `INSERT(p, g, s)` splices a cancelling pair before position `p` and "
        "`REDUCE(p)` deletes it again, so **every** `INSERT` site has a "
        "completion of exactly `m* + 2` -- insert, undo, take a shortest "
        "completion. That completion is built with no search at all and replayed "
        "by `UnknotWitness.from_actions`."
    )
    add("")
    add(_row(["measurement", "value"]))
    add(_row(["---", "---:"]))
    add(
        _row(
            [
                "`INSERT` sites in the shortest-completion target",
                flat["insert_sites_in_the_shortest_completion_target"],
            ]
        )
    )
    add(
        _row(
            [
                "states whose target contains an `INSERT`",
                f"{flat['states_with_an_INSERT_in_the_target']} of "
                f"{target['states_scored']}",
            ]
        )
    )
    add(_row(["`INSERT` share of legal sites", f"{target['insert_share_of_legal']:.1%}"]))
    add(_row(["states with the block measured", block["states_measured"]]))
    add(_row(["mean block size", block["mean_block_size"]]))
    add(
        _row(
            [
                "committed block optimum at `m* + 2`",
                block["committed_optimum_is_shortest_plus_two"],
            ]
        )
    )
    add(
        _row(
            [
                "whole block attains the optimum",
                block["whole_block_attains_the_optimum"],
            ]
        )
    )
    add(
        _row(
            [
                "states where the block splits at all",
                block["states_where_the_block_splits"],
            ]
        )
    )
    add(
        _row(
            [
                "insert-undo-go completions exhibited at `m* + 2`",
                f"{block['undo_sites_exhibited']} "
                f"({'all' if block['every_exhibited_undo_at_shortest_plus_two'] else 'NOT all'})",
            ]
        )
    )
    add("")
    add(
        "The committed block target is the strongest question that can be asked "
        "inside the block: the shortest completion through an `INSERT` **that "
        "never re-enters the state it came from**, so the trivial undo is "
        "excluded by construction. It still comes out flat."
    )
    add("")
    if "recall@1_where_the_block_splits" in block:
        add(
            "Where the block does split, for the record -- these are the few "
            "short-word states and the row is far too small to gate on:"
        )
        add("")
        cell = block["recall@1_where_the_block_splits"]
        add(_row(["row", "states", "chance", *names]))
        add(_row(["---", "---:", "---:", *["---:"] * len(names)]))
        add(
            _row(
                [
                    "`INSERT` block, recall@1",
                    cell["states"],
                    _fmt(cell["uniform_chance"]),
                    *[_fmt(cell.get(name)) for name in names],
                ]
            )
        )
        add("")
    add(
        "**This also explains the earlier finding it was built from.** The "
        "previous report measured that no hand-written score beat random "
        "ordering inside the `INSERT` block. Under a *feasibility* target an "
        "`INSERT` qualifies precisely by being undoable, so the block's positives "
        "carry no information; under a shortest-completion target the block has "
        "no positives at all. Both are the same fact about a move set closed "
        "under inverses."
    )
    add("")
    add("### The `INSERT`-block table §M1 asks for, printed as it comes out")
    add("")
    insert_block = payload["insert_block"]
    add(
        f"{insert_block['states_with_at_least_two_insert_sites']} of "
        f"{target['states_scored']} scored states have at least two `INSERT` "
        f"sites, so the block exists nearly everywhere. "
        f"**{insert_block['states_with_a_scorable_insert_block']}** of them have "
        f"an `INSERT` in the shortest-completion target, which is the number of "
        f"states on which recall@1 inside the block is defined at all."
    )
    add("")
    out.extend(_table(insert_block["recall@1"]["by_config"], names))
    add("")
    add(
        "Every cell is empty because every row is empty. This is the gate §M1 "
        "revised to, run exactly as written, and it has no states."
    )
    add("")
    add("---")
    add("")
    add("## 2. The gate, as implemented")
    add("")
    add(
        "**Target.** `A*(s)`: the sites that begin a completion of the *shortest* "
        "length, under the crossing-change budget the witness had left. The move "
        "budget plays no part -- slack is what made the old target vacuous, "
        "because two moves of it is exactly the cost of undoing one move."
    )
    add("")
    add(
        f"**Metric.** recall@1. The target is "
        f"{target['target_share_of_legal']:.1%} of the legal set "
        f"({target['mean_target_size']} of {target['mean_legal_sites']} sites), "
        f"against 80.5% under the feasibility target the old gate used."
    )
    add("")
    add(
        "**Interface.** `rank(state, sites) -> scores`. Ties are broken by the "
        "harness with a seeded shuffle, not by the ranker, so no score function "
        "can earn credit from the order `reference.successors` happens to emit -- "
        "and the learned head is scored by the identical code on the identical "
        "states."
    )
    add("")
    add("### The gate table")
    add("")
    out.extend(_gate_table(gate, names))
    add("")
    bootstrap = gate["paired_bootstrap_over_instances"]
    add(
        "Paired bootstrap of `ranker - kind_prior`, resampling **instances** "
        "rather than states, because states on one witness share a word, a seed "
        "and a budget:"
    )
    add("")
    add(_row(["comparison", "instances", "states", "delta", "95% CI"]))
    add(_row(["---", "---:", "---:", "---:", "---:"]))
    for key, cell in bootstrap.items():
        if not cell.get("draws"):
            continue
        add(
            _row(
                [
                    f"`{key}`",
                    cell["instances"],
                    cell["states"],
                    f"{cell['observed_delta']:+.3f}",
                    f"[{cell['ci95_low']:+.3f}, {cell['ci95_high']:+.3f}]",
                ]
            )
        )
    add("")
    add("### Sliced")
    add("")
    for axis in ("by_config", "by_word_length", "by_region", "by_strands",
                 "by_remaining_crossing_changes"):
        add(f"**recall@1 {axis.replace('_', ' ')}**")
        add("")
        out.extend(_table(payload["whole_state"]["recall@1"][axis], names))
        add("")
    add("**recall@10, for the record -- it is saturated, as §M1 said it would be**")
    add("")
    out.extend(_table(payload["whole_state"]["recall@10"]["by_config"], names))
    add("")
    add("### 2.1 The residual: the non-`INSERT` sites, ranked among themselves")
    add("")
    add(payload["residual_block"]["note"])
    add("")
    add(
        f"Mean non-`INSERT` sites per state: "
        f"{payload['residual_block']['mean_non_insert_sites']}."
    )
    add("")
    for axis in ("by_config", "by_word_length"):
        add(f"**recall@1 {axis.replace('_', ' ')}**")
        add("")
        out.extend(_table(payload["residual_block"]["recall@1"][axis], names))
        add("")
    add("---")
    add("")
    add("## 3. What the target is, measured")
    add("")
    add(_row(["quantity", "value"]))
    add(_row(["---", "---:"]))
    for label, key in (
        ("states scored", "states_scored"),
        ("mean legal sites", "mean_legal_sites"),
        ("mean `INSERT` sites", "mean_insert_sites"),
        ("mean `|A*(s)|`", "mean_target_size"),
        ("target share of the legal set", "target_share_of_legal"),
        ("`INSERT` share of legal sites", "insert_share_of_legal"),
        ("`INSERT` share of the target", "insert_share_of_target"),
        ("mean shortest completion, moves", "mean_shortest_completion_moves"),
        ("mean witness remaining moves", "mean_witness_remaining_moves"),
        ("witness's own site on a shortest completion",
         "witness_action_on_a_shortest_completion"),
        ("witness itself shortest", "witness_itself_shortest"),
    ):
        add(_row([label, target[key]]))
    add("")
    add(
        "The last two rows are why §M1's phrase *\"recall of a witness-realising "
        "site\"* had to go: the site the witness took and a good site are "
        "different targets."
    )
    add("")
    add("Kinds in the target:")
    add("")
    add(_row(["kind", "sites in the target"]))
    add(_row(["---", "---:"]))
    for kind, count in target["target_kind_histogram"].items():
        add(_row([f"`{kind}`", count]))
    add("")
    add("---")
    add("")
    add("## 4. The split, and what it guarantees")
    add("")
    add(_row(["quantity", "value"]))
    add(_row(["---", "---:"]))
    for label, key in (
        ("identity groups", "identity_groups"),
        ("held out", "eval_identity_groups"),
        ("training instances", "train_instances"),
        ("held-out instances", "eval_instances"),
        ("largest identity group, instances", "largest_identity_group"),
        ("identity groups holding one instance",
         "identity_groups_holding_one_instance"),
        ("group pairs sharing a representation",
         "group_pairs_sharing_a_representation"),
        ("transfer instances dropped for a training identity",
         "transfer_instances_dropped_for_a_training_identity"),
        ("eval states dropped for appearing in training",
         "eval_states_screened_out_for_appearing_in_training"),
        ("eval states kept", "eval_states_kept"),
    ):
        add(_row([label, split[key]]))
    add("")
    add(
        f"Training strand counts {split['train_strand_counts']}; the transfer row "
        f"is every state at a strand count strictly above them: "
        f"{split['transfer_strand_counts']}."
    )
    add("")
    add("**Guaranteed.**")
    add("")
    for line in split["guarantees"]:
        add(f"* {line}")
    add("")
    add("**Not guaranteed.**")
    add("")
    for line in split["not_guaranteed"]:
        add(f"* {line}")
    add("")
    add(
        f"§4.3's quarantine screened "
        f"{split['quarantine']['representations']} stored representations, the "
        f"names {split['quarantine']['names']} and the fingerprint "
        f"{split['quarantine']['fingerprints']} against every seed and every "
        f"emitted word; rejections: {split['quarantine_screened_out'] or 0}. Its "
        f"own stated limit is unchanged: {split['quarantine']['k_ball_closure']}"
    )
    add("")
    add(
        "`scripts/audit_embedding_validation_leakage.py` is **not** what screened "
        "this corpus and does not cover it. That script audits the `unknotdb` "
        "embedding-pair curriculum against the proof graph's anchors and their "
        "CC=0 closure, which is a different corpus with different identifiers; "
        "the states here are generated by `interleaved_generator.py` and screened "
        "by the §4.3 `Quarantine` above. Anyone reading a leakage number for this "
        "gate should read the two lists in this section, not that audit."
    )
    add("")
    add("---")
    add("")
    add("## 5. Evidence tiers, kept apart")
    add("")
    add(_row(["tier", "sites", "in the gate?"]))
    add(_row(["---", "---:", "---"]))
    add(
        _row(
            [
                "1. demonstrated and replayed",
                f"{tiers['tier1_demonstrated']['target_sites_replayed']} replayed "
                f"of {tiers['tier1_demonstrated']['target_sites_selected_for_replay']}"
                f" selected, out of "
                f"{tiers['tier1_demonstrated']['target_sites_claimed']} in the target",
                "**yes -- this is the target**",
            ]
        )
    )
    add(
        _row(
            [
                "2. exact commutation closure",
                tiers["tier2_exact_closure"]["sites"],
                "no, reported beside it",
            ]
        )
    )
    add(_row(["3. attested (catalogue `u`)", 0, "no, not consulted"]))
    add("")
    add(
        f"Of the {tiers['tier2_exact_closure']['sites']} tier-2 sites, "
        f"{tiers['tier2_exact_closure']['sites_also_in_the_shortest_completion_target']}"
        f" are also in the tier-1 target. They are not merged: a commutation "
        f"closure reorders the witness's own moves, so it costs what the witness "
        f"cost, and a gate scored on a mixture of tiers is not measuring what it "
        f"claims to (§5.3b)."
    )
    add("")
    add(
        f"Replay disagreements: "
        f"{tiers['tier1_demonstrated']['replay_disagreements']}, and a mismatch "
        f"raises rather than being recorded, so the run finishing is the "
        f"assertion. "
        f"{tiers['tier1_demonstrated']['target_sites_whose_path_extraction_hit_its_cap']}"
        f" selected sites had their path extraction hit its own cap: that is a "
        f"cap, not a disagreement, and those sites stay in the target because the "
        f"mask propagation demonstrated them."
    )
    add("")
    add("---")
    add("")
    add("## 6. What the harness could not determine")
    add("")
    add(
        f"{search['states_dropped_undetermined']} of "
        f"{search['states_considered']} states "
        f"({search['undetermined_fraction']:.1%}) hit the node cap before an "
        f"optimal completion was proved, and were **dropped**, never scored as "
        f"failures. The drop is not uniform:"
    )
    add("")
    add(_row(["word length", "considered", "undetermined", "fraction"]))
    add(_row(["---", "---:", "---:", "---:"]))
    for bucket, considered in sorted(search["states_considered_by_word_length"].items()):
        dropped = search["undetermined_by_word_length"].get(bucket, 0)
        add(_row([f"`{bucket}`", considered, dropped,
                  f"{dropped / max(considered, 1):.3f}"]))
    add("")
    add(
        "**The direction of that bias, stated plainly.** A state is dropped when "
        "its optimal-search layer is too large, which is the longest words with "
        "the most slack -- so the hardest states are under-represented and every "
        "number above, a head's included, is flattered."
    )
    add("")
    add("---")
    add("")
    add("## 7. Threats to these numbers")
    add("")
    add(
        f"* The target is *shortest known*, and known means within a word-length "
        f"growth cap of +{search['growth_cap']}. Inside that subgraph the set is "
        f"exact -- the search exhausts the whole `f <= m*` layer -- so the honest "
        f"name is exact-within-the-cap."
    )
    add("")
    add(
        "* The identity above is a statement about **this move set**. It is why "
        "`INSERT` is flat here; a move set with a different `REDUCE`/`INSERT` "
        "relationship would not have it, and the finding does not transfer to "
        "one."
    )
    add("")
    add(
        "* Every witness ends at the unknot and every instance from a seed ends "
        "in that seed's own program, so the deepest states are shared across the "
        "split by construction. The walk region is reported separately for that "
        "reason."
    )
    add("")
    add(
        "* The longest word here is far from the target's 254 letters on 24 "
        "strands, and none of the trends measured is evidence about that scale."
    )
    add("")
    add("---")
    add("")
    add("## 8. What M1 should do with this")
    add("")
    add(
        "The kill criterion §M1 states -- *inside the `INSERT` block, recall@1 no "
        "better than `kind_prior` or than random ordering* -- is met by every "
        "possible ranker, including a perfect one, because the block is flat. It "
        "cannot fail a head and it cannot pass one. Three options, and the choice "
        "belongs to the note rather than to this harness:"
    )
    add("")
    add(
        f"1. **Gate on the whole state**, as this report does, and read it as a "
        f"floor rather than a target. The `INSERT` block is most of the input and "
        f"none of the answer, so a head that has not learned that scores near "
        f"chance -- but `kind_prior` is already at {best:.3f}, so passing means "
        f"only *not being worse than a kind ordering*. That is a real kill "
        f"criterion and a useless success criterion."
    )
    add(
        "2. **Gate on a target that is not distance to the unknot.** The block is "
        "flat with respect to *that* objective only. A bounded-search objective "
        "-- which growth helps a depth-limited search escape -- would have "
        "content, but it is a different milestone and needs the search of M2."
    )
    add(
        "3. **Accept that the branching factor is reduced by a mask, not a "
        "ranking.** If no `INSERT` is ever optimal, the search's shortlist can "
        "exclude the block outright and the head's job shrinks to the remaining "
        "20% of sites -- which is a bigger win than any ranking of the block "
        "would have been, and it is available today."
    )
    add("")
    add("---")
    add("")
    add("## 9. Reproduce")
    add("")
    add("```bash")
    add("OMP_NUM_THREADS=1 PYTHONPATH=src python research/experiments/m1_gate.py \\")
    add(
        f"    --out {args.out} --states-per-instance {args.states_per_instance} "
        f"--growth {args.growth} --eval-fraction {args.eval_fraction}"
    )
    add("```")
    add("")
    add(
        "Add `--head <checkpoint>` once `rf_knots.site_head.SiteHead` and the "
        "encoder's `encode_sites` exist; the head is scored by the identical code "
        "on the identical states, through the same `rank(state, sites) -> scores` "
        "interface as every baseline. `report.json` carries every scored state -- "
        "word, strands, budget, legal sites and target -- so a head can be scored "
        "on exactly these without regenerating anything."
    )
    add("")
    add(
        f"**Run metadata.** seed {payload['run_metadata']['seed']}, eval fraction "
        f"{payload['run_metadata']['eval_fraction']}, growth cap "
        f"+{payload['run_metadata']['growth_cap']}, positional geometry "
        f"`{payload['run_metadata']['positional_geometry']}`, "
        f"cyclic band generators "
        f"{payload['run_metadata']['cyclic_band_generators']}, rotary "
        f"{json.dumps(payload['run_metadata']['rotary'])}. "
        f"{payload['run_metadata']['geometry_note']}"
    )
    add("")
    status = payload.get("head_status")
    add(
        "Learned head status at this run: "
        + ("not requested" if status is None else json.dumps(status))
        + "."
    )
    add("")
    add(
        "Modules: `research/experiments/m1_target.py` (target, interface, split), "
        "`research/experiments/m1_head.py` (the head adapter), "
        "`research/experiments/m1_gate.py` (this driver); tests: "
        "`tests/test_m1_gate.py`."
    )
    return "\n".join(out) + "\n"


def _rotary_used() -> dict | None:
    """What the encoder's rotary was doing, if an encoder was built at all."""
    try:
        import m1_head
    except Exception:  # pragma: no cover - the adapter is importable here
        return None
    return getattr(m1_head.build_ranker, "last_rotary", None)


def _gate_table(gate: dict, names: list[str]) -> list[str]:
    lines = [
        _row(["row", "states", "chance", *names]),
        _row(["---", "---:", "---:", *["---:"] * len(names)]),
    ]
    for label, key in (
        ("**whole state, recall@1 -- the gate**", "recall@1"),
        ("whole state, recall@1, walk region only", "recall@1_walk_region_only"),
        ("whole state, recall@5", "recall@5"),
    ):
        cell = gate[key]
        lines.append(
            _row(
                [
                    label,
                    cell.get("states", 0),
                    _fmt(cell.get("uniform_chance")),
                    *[_fmt(cell.get(name)) for name in names],
                ]
            )
        )
    transfer = gate["transfer_row"]
    lines.append(
        _row(
            [
                f"**transfer: strands {transfer['strands_absent_from_training']}, "
                f"training saw {transfer['training_strand_counts']}**",
                transfer["recall@1"].get("states", 0),
                _fmt(transfer["recall@1"].get("uniform_chance")),
                *[_fmt(transfer["recall@1"].get(name)) for name in names],
            ]
        )
    )
    return lines


# -- entry point ---------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/m1-gate"))
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--states-per-instance", type=int, default=4)
    parser.add_argument("--eval-fraction", type=float, default=0.5)
    parser.add_argument("--growth", type=int, default=3)
    parser.add_argument("--commutation-window", type=int, default=8)
    parser.add_argument(
        "--exhibit-per-state", type=int, default=4,
        help="0 replays every target site; n replays a random n of them. Every "
             "target site is demonstrated by the mask propagation either way; "
             "this is how many are additionally replayed through UnknotWitness, "
             "and the report prints claimed, selected and replayed separately.",
    )
    parser.add_argument("--block-states-per-config", type=int, default=8)
    parser.add_argument("--block-node-cap", type=int, default=12_000)
    parser.add_argument("--undo-exhibits", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--seconds", type=float, default=1500.0)
    parser.add_argument(
        "--head", type=str, default=None,
        help="a SiteHead checkpoint, or the literal 'untrained' to score a head "
             "at initialisation as a control",
    )
    parser.add_argument(
        "--head-geometry", type=str, default=None,
        help="positional geometry for the encoder the head reads; note 22 §6.4. "
             "The encoder's own default is 'legacy' and would silently hand the "
             "gate the wrong geometry, so this defaults to m1_head.GEOMETRY and "
             "is recorded in the report.",
    )
    parser.add_argument("--cyclic-band-generators", action="store_true")
    args = parser.parse_args()
    if args.head_geometry is None:
        import m1_head

        args.head_geometry = m1_head.GEOMETRY

    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    log("phase 1: corpus", started)
    built = build_corpus(args, started)
    log("phase 2: identity-level split", started)
    split = split_corpus(built, args, started)
    considered = sum(
        args.states_per_instance
        for _name, rows in built["corpus"].items()
        for key, _seed, _spine in rows
        if split["assignment"].get(key) == "eval"
    )
    log(
        f"phase 3: target and scores over about {considered} eval states -- at "
        f"the probe's 0.1-6 s per state that is roughly "
        f"{considered * 0.4 / 60:.0f}-{considered * 2.5 / 60:.0f} minutes, plus "
        f"{args.block_states_per_config * len(CONFIGS)} block searches",
        started,
    )
    result = evaluate(built, split, args, started)
    log("phase 4: tables", started)
    payload = aggregate(result, split, built, args)
    payload["schema"] = "rf-knots-m1-revised-gate-v0"
    payload["generated"] = time.strftime("%Y-%m-%d")
    payload["ranker_names"] = result["ranker_names"]
    payload["head_status"] = result["head_status"]
    payload["run_metadata"] = {
        "seed": args.seed,
        "eval_fraction": args.eval_fraction,
        "growth_cap": args.growth,
        "positional_geometry": args.head_geometry,
        "cyclic_band_generators": bool(args.cyclic_band_generators),
        "rotary": _rotary_used(),
        "geometry_note": (
            "note 22 §6.4. The encoder's constructor default is 'legacy' -- "
            "absolute Fourier features with a zero-padded seam, under which a "
            "cyclic shift of the word moves the knot embedding by up to 0.377 in "
            "a unit-normalised space against 2.2e-16 under 'necklace'. The gate "
            "never relies on that default; it passes this value explicitly and "
            "records it here so a reader can tell which geometry a number came "
            "from."
        ),
    }
    payload["wall_cap_reached"] = result["wall_cap_reached"]
    payload["arguments"] = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    payload["configs"] = {
        name: {"role": entry["role"], **entry["config"].provenance(),
               "node_cap": entry["node_cap"]}
        for name, entry in CONFIGS.items()
    }
    payload["insert_block_rows"] = [
        {
            "config": row.config, "region": row.region,
            "word_length": row.word_length, "strands": row.strands,
            "shortest_moves": row.shortest_moves, "block_sites": row.block_sites,
            "committed_best": row.committed_best,
            "excess_over_shortest": row.excess_over_shortest,
            "target_size": len(row.target), "discriminating": row.discriminating,
            "undo_sites_exhibited": row.undo_sites_exhibited,
            "undo_all_at_shortest_plus_two": row.undo_all_at_shortest_plus_two,
        }
        for row in result["blocks"]
    ]
    payload["eval_states"] = [
        {
            "config": row.state.config,
            "instance": row.state.instance_id,
            "depth": row.state.depth,
            "region": row.state.region,
            "word": list(row.state.word),
            "strands": row.state.strands,
            "remaining_crossing_changes": row.state.remaining_crossing_changes,
            "remaining_moves": row.state.remaining_moves,
            "legal_sites": row.legal,
            "insert_sites": row.insert_legal,
            "non_insert_sites": row.rest_legal,
            "shortest_completion_moves": row.shortest_moves,
            "target": list(row.target),
            "tier2_commutation_sites": list(row.tier2_sites),
        }
        for row in result["rows"]
    ]
    (args.out / "report.json").write_text(json.dumps(payload, indent=1))
    (args.out / "report.md").write_text(render(payload, args))
    log(f"wrote {args.out / 'report.json'} and {args.out / 'report.md'}", started)


if __name__ == "__main__":
    main()
