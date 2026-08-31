r"""Measure `|A_plus|`, and score the hand-written baselines M1 has to beat.

Note 22 §M1 gates the per-site head on "recall@10 of a witness-realising site"
and then says, in the same breath, that the gate's denominator does not exist
yet.  This script builds it and then uses it, in three phases:

1. **cheap methods at scale** -- pooling, commutation closure and reconvergent
   re-walking over every state of every generated witness;
2. **brute-force calibration** -- on a sample, every legal site tried with a
   bounded search, which is the only thing that says how much phase 1 missed;
3. **baselines** -- the hand-written rankers of §M1's kill criterion scored
   against both denominators, and against uniform chance.

Nothing here records a site as bad.  Where a search did not finish, the site is
`undetermined`, and the report says so in the same table.

    PYTHONPATH=src python research/experiments/aplus_denominator.py --out artifacts/m1-aplus
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from rf_knots import reference
from rf_knots.actions import INSERT, KIND_NAMES
from rf_knots.aplus import (
    METHODS,
    RANKERS,
    WitnessPool,
    _Successors,
    bruteforce_known_good,
    build_known_good,
    hit_at_k,
    random_hit_probability,
    rank_by_kind_order,
    witness_spine,
)
from rf_knots.interleaved_generator import (
    UNKNOT_SEED,
    InterleavedConfig,
    generate_many,
    load_quarantine,
    torus_seed,
)

UNKNOTDB = "/Users/artemvorozhtsov/projects/unknotdb"

CONFIGS: dict[str, InterleavedConfig] = {
    "C-small": InterleavedConfig(
        crossing_changes=2, walk_length=4, target_strands=3,
        target_length=8, max_len=16, max_strands=4,
    ),
    "A-main": InterleavedConfig(
        crossing_changes=3, walk_length=8, target_strands=3,
        target_length=12, max_len=24, max_strands=4,
    ),
    "B-long": InterleavedConfig(
        crossing_changes=4, walk_length=16, target_strands=4,
        target_length=20, max_len=40, max_strands=5,
    ),
}

KS = (1, 5, 10)


def reference_successors(spec, word, strands):
    return reference.successors(spec, word, strands, allow_crossing=True)


def log(message: str, started: float) -> None:
    print(f"[{time.monotonic() - started:7.1f}s] {message}", flush=True)


def seeds_for(config: InterleavedConfig) -> list:
    spec = config.spec()
    out = [UNKNOT_SEED]
    for m in (3, 5):
        out.append(torus_seed(m, spec))
    return out


def depth_bucket(depth: int, total: int) -> str:
    if total <= 1:
        return "0.0-0.2"
    fraction = depth / (total - 1)
    edges = (0.2, 0.4, 0.6, 0.8)
    names = ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")
    for index, edge in enumerate(edges):
        if fraction < edge:
            return names[index]
    return names[-1]


def length_bucket(length: int) -> str:
    for low, high in ((0, 4), (5, 8), (9, 12), (13, 16), (17, 24)):
        if low <= length <= high:
            return f"{low}-{high}"
    return "25+"


class Accumulator:
    """Sums and histograms keyed by whatever the report wants to slice on."""

    def __init__(self) -> None:
        self.rows = 0
        self.by: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0, 0, 0])
        )
        self.histogram: Counter[int] = Counter()
        self.method_totals: Counter[str] = Counter()
        self.method_unique: Counter[str] = Counter()
        self.kind_totals: Counter[str] = Counter()

    def add(self, slices: dict[str, str], size: int, legal: int, undetermined: int) -> None:
        self.rows += 1
        self.histogram[size] += 1
        for axis, value in slices.items():
            cell = self.by[axis][value]
            cell[0] += 1
            cell[1] += size
            cell[2] += legal
            cell[3] += undetermined

    def table(self, axis: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for value, (rows, size, legal, undetermined) in sorted(self.by[axis].items()):
            out[value] = {
                "states": rows,
                "mean_known_good": round(size / rows, 3),
                "mean_legal_sites": round(legal / rows, 2),
                "mean_undetermined": round(undetermined / rows, 2),
                "known_good_fraction_of_legal": round(size / max(legal, 1), 4),
            }
        return out


def phase1(args, started: float) -> dict:
    """Cheap methods over every state of every generated witness."""
    phase_started = time.monotonic()
    quarantine = load_quarantine(unknotdb_root=UNKNOTDB)
    result: dict = {
        "quarantine": quarantine.provenance(),
        "configs": {},
        "screened_out": 0,
    }
    recall_rows: list[dict] = []
    screened = 0
    for config_name, config in CONFIGS.items():
        if args.configs and config_name not in args.configs:
            continue
        spec = config.spec()
        pool = WitnessPool()
        spines: list[tuple[str, object]] = []
        for seed_representation in seeds_for(config):
            reason = quarantine.screen_seed(seed_representation)
            if reason is not None:  # pragma: no cover - the seeds are clean
                raise SystemExit(f"seed {seed_representation.name}: {reason}")
            for stream in range(args.streams):
                for instance in generate_many(
                    config,
                    seed=stream,
                    count=args.instances // args.streams,
                    seed_representation=seed_representation,
                ):
                    if quarantine.screen_word(instance.word, instance.strands):
                        screened += 1
                        continue
                    spine = witness_spine(instance)
                    pool.add(spine)
                    spines.append((seed_representation.name, spine))
        log(
            f"{config_name}: {len(spines)} spines, "
            f"{sum(len(s) for _n, s in spines)} states, "
            f"collisions {pool.emitted_word_collisions()}",
            started,
        )

        accumulators: dict[str, Accumulator] = {
            "witness_only": Accumulator(),
            "cheap_all": Accumulator(),
        }
        per_method = Counter()
        per_method_states = Counter()
        rng = np.random.default_rng(20260830)
        rank_rng = np.random.default_rng(4242)
        trials = args.rewalk_trials if config_name != "B-long" else args.rewalk_trials // 2
        last = time.monotonic()
        for position, (seed_name, spine) in enumerate(spines):
            succ = _Successors(spec)
            for index in range(len(spine)):
                site = spine.sites[index]
                known = build_known_good(
                    spine,
                    index,
                    pool=pool,
                    rng=rng,
                    commutation_window=args.commutation_window,
                    rewalk_trials=trials,
                    rewalk_detour=args.rewalk_detour,
                    successors=succ,
                )
                slices = {
                    "seed": seed_name,
                    "seed_certified_lower_bound": str(
                        spine.seed_certified_lower_bound
                    ),
                    "depth_fraction": depth_bucket(index, len(spine)),
                    "raw_depth": str(min(index, 9)),
                    "word_length": length_bucket(len(site.state.word)),
                    "strands": str(site.state.strands),
                    "region": "seed-program" if site.in_seed_program else "walk",
                    "remaining_crossing_changes": str(site.remaining_crossing_changes),
                }
                accumulators["cheap_all"].add(
                    slices, known.size, len(known.legal_actions), known.sites_undetermined
                )
                accumulators["witness_only"].add(slices, 1, len(known.legal_actions),
                                                 len(known.legal_actions) - 1)
                for name in METHODS:
                    demonstrated = known.demonstrated_by_method.get(name, set())
                    per_method[name] += len(demonstrated)
                    if demonstrated:
                        per_method_states[name] += 1
                for site_record in known.sites.values():
                    accumulators["cheap_all"].kind_totals[
                        KIND_NAMES[site_record.kind]
                    ] += 1

                moves = succ(site.state.word, site.state.strands)
                row = {
                    "config": config_name,
                    "seed": seed_name,
                    "L": spine.seed_certified_lower_bound,
                    "depth_fraction": slices["depth_fraction"],
                    "region": slices["region"],
                    "legal": len(moves),
                    "witness_action": site.action,
                    "known_good": sorted(known.sites),
                    "insert_fraction": round(
                        sum(1 for a, _w, _s in moves if spec.decode(a)[0] == INSERT)
                        / max(len(moves), 1),
                        4,
                    ),
                    "rankings": {},
                    "kinds_known_good": sorted(
                        {KIND_NAMES[spec.decode(a)[0]] for a in known.sites}
                    ),
                }
                for ranker_name, ranker in RANKERS.items():
                    order = ranker(
                        spec, site.state.word, site.state.strands, moves, rank_rng
                    )
                    row["rankings"][ranker_name] = order[: max(KS)]
                recall_rows.append(row)
            if time.monotonic() - last > 15.0:
                last = time.monotonic()
                log(
                    f"  {config_name}: spine {position + 1}/{len(spines)}",
                    started,
                )
            if time.monotonic() - phase_started > args.phase1_seconds:
                log(f"  {config_name}: wall cap reached, stopping phase 1", started)
                break

        result["configs"][config_name] = {
            "config": config.provenance(),
            "spines": len(spines),
            "states": accumulators["cheap_all"].rows,
            "emitted_word_collisions": pool.emitted_word_collisions(),
            "sites_by_method": dict(per_method),
            "states_with_a_site_from_method": dict(per_method_states),
            "known_good_size_histogram": dict(
                sorted(accumulators["cheap_all"].histogram.items())
            ),
            "known_good_kind_histogram": dict(
                accumulators["cheap_all"].kind_totals.most_common()
            ),
            "by_depth_fraction": accumulators["cheap_all"].table("depth_fraction"),
            "by_raw_depth": accumulators["cheap_all"].table("raw_depth"),
            "by_word_length": accumulators["cheap_all"].table("word_length"),
            "by_seed": accumulators["cheap_all"].table("seed"),
            "by_seed_certified_lower_bound": accumulators["cheap_all"].table(
                "seed_certified_lower_bound"
            ),
            "by_region": accumulators["cheap_all"].table("region"),
            "by_remaining_crossing_changes": accumulators["cheap_all"].table(
                "remaining_crossing_changes"
            ),
        }
    result["screened_out"] = screened
    result["recall_rows"] = recall_rows
    return result


def phase2(args, started: float) -> dict:
    """Brute force on a sample: the only measurement of what phase 1 missed."""
    phase_started = time.monotonic()
    quarantine = load_quarantine(unknotdb_root=UNKNOTDB)
    out: dict = {"configs": {}}
    for config_name, config in CONFIGS.items():
        if args.configs and config_name not in args.configs:
            continue
        spec = config.spec()
        rows: list[dict] = []
        pool = WitnessPool()
        sampled: list[tuple[str, object]] = []
        for seed_representation in seeds_for(config):
            for instance in generate_many(
                config,
                seed=1234,
                count=args.bruteforce_instances,
                seed_representation=seed_representation,
            ):
                if quarantine.screen_word(instance.word, instance.strands):
                    continue
                spine = witness_spine(instance)
                pool.add(spine)
                sampled.append((seed_representation.name, spine))
        rng = np.random.default_rng(99)
        rank_rng = np.random.default_rng(7)
        last = time.monotonic()
        budget_exhausted = False
        for seed_name, spine in sampled:
            total = len(spine)
            depths = sorted(
                {
                    int(round(fraction * (total - 1)))
                    for fraction in np.linspace(0, 0.95, args.bruteforce_depths)
                }
            )
            succ = _Successors(spec)
            for index in depths:
                site = spine.sites[index]
                cheap = build_known_good(
                    spine,
                    index,
                    pool=pool,
                    rng=rng,
                    commutation_window=args.commutation_window,
                    rewalk_trials=args.rewalk_trials,
                    rewalk_detour=args.rewalk_detour,
                    successors=succ,
                )
                sites, report = bruteforce_known_good(
                    spine, index, node_cap=args.node_cap, successors=succ
                )
                found = {s.action: s for s in sites}
                shortest = min(
                    (s.completion_moves for s in sites), default=None
                )
                shortest_set = (
                    {a for a, s in found.items() if s.completion_moves == shortest}
                    if shortest is not None
                    else set()
                )
                moves = succ(site.state.word, site.state.strands)
                rankings = {}
                for ranker_name, ranker in RANKERS.items():
                    rankings[ranker_name] = ranker(
                        spec, site.state.word, site.state.strands, moves, rank_rng
                    )[: max(KS)]
                rows.append(
                    {
                        "config": config_name,
                        "seed": seed_name,
                        "L": spine.seed_certified_lower_bound,
                        "instance": spine.instance_id[:12],
                        "depth": index,
                        "depth_fraction": depth_bucket(index, total),
                        "region": "seed-program" if site.in_seed_program else "walk",
                        "word_length": len(site.state.word),
                        "strands": site.state.strands,
                        "remaining_moves": site.remaining_moves,
                        "remaining_crossing_changes": site.remaining_crossing_changes,
                        "legal": report.legal_actions,
                        "cheap_known_good": sorted(cheap.sites),
                        "bruteforce_known_good": sorted(found),
                        "bruteforce_undetermined": report.undetermined,
                        "shortest_completion_moves": shortest,
                        "shortest_known_set": sorted(shortest_set),
                        "completion_moves": sorted(
                            [a, site.completion_moves, site.completion_crossing_changes]
                            for a, site in found.items()
                        ),
                        "witness_action": site.action,
                        "witness_in_shortest": site.action in shortest_set,
                        "word": list(site.state.word),
                        "seconds": round(report.seconds, 3),
                        "rankings": rankings,
                    }
                )
                if time.monotonic() - last > 15.0:
                    last = time.monotonic()
                    log(
                        f"  {config_name} brute force: {len(rows)} states, "
                        f"last {report.seconds:.2f}s "
                        f"({report.demonstrated}/{report.legal_actions})",
                        started,
                    )
                if time.monotonic() - phase_started > args.phase2_seconds:
                    budget_exhausted = True
                    break
            if budget_exhausted:
                log(f"  {config_name} brute force: wall cap reached", started)
                break
        out["configs"][config_name] = rows
        log(f"{config_name}: brute force done, {len(rows)} states", started)
    return out


def recall_table(
    rows: list[dict], key: str, ranker_names: list[str], min_legal: int = 0
) -> dict:
    """recall@k against whatever site set `key` names, plus uniform chance.

    `min_legal` drops states with so few legal sites that a top-10 shortlist is
    the whole action set: there, every ranker scores 1.0 and the number says
    nothing about ranking.  Both variants are reported.
    """
    subset = [r for r in rows if r["legal"] >= min_legal and r.get(key)]
    out: dict = {"states": len(subset)}
    for name in ranker_names:
        cells = {}
        for k in KS:
            hits = sum(
                int(hit_at_k(row["rankings"][name], row[key], k)) for row in subset
            )
            cells[f"recall@{k}"] = round(hits / len(subset), 4) if subset else None
        out[name] = cells
    chance = {}
    for k in KS:
        values = [
            random_hit_probability(row["legal"], len(row[key]), k) for row in subset
        ]
        chance[f"recall@{k}"] = round(sum(values) / len(values), 4) if values else None
    out["uniform_chance_closed_form"] = chance
    out["mean_known_good"] = (
        round(sum(len(r[key]) for r in subset) / len(subset), 3) if subset else None
    )
    out["mean_legal_sites"] = (
        round(sum(r["legal"] for r in subset) / len(subset), 2) if subset else None
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/m1-aplus"))
    parser.add_argument("--instances", type=int, default=36)
    parser.add_argument("--streams", type=int, default=6)
    parser.add_argument("--rewalk-trials", type=int, default=64)
    parser.add_argument("--rewalk-detour", type=int, default=4)
    parser.add_argument("--commutation-window", type=int, default=8)
    parser.add_argument("--bruteforce-instances", type=int, default=3)
    parser.add_argument("--bruteforce-depths", type=int, default=6)
    parser.add_argument("--node-cap", type=int, default=400)
    parser.add_argument("--phase1-seconds", type=float, default=2400.0)
    parser.add_argument("--phase2-seconds", type=float, default=1800.0)
    parser.add_argument("--configs", nargs="*", default=None)
    args = parser.parse_args()

    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)

    log("phase 1: cheap methods", started)
    one = phase1(args, started)
    rows = one.pop("recall_rows")
    log(f"phase 1 done: {len(rows)} states", started)

    log("phase 2: brute-force calibration", started)
    two = phase2(args, started)
    log("phase 2 done", started)

    # -- phase 3: baselines ---------------------------------------------------
    ranker_names = list(RANKERS)
    kind_counts: Counter[str] = Counter()
    for row in rows:
        for kind in row["kinds_known_good"]:
            kind_counts[kind] += 1
    fitted_order = [
        KIND_NAMES.index(name) for name, _count in kind_counts.most_common()
    ]
    fitted = rank_by_kind_order(fitted_order)
    fitted_name = "kind_prior_fitted_on_the_eval_set"
    rank_rng = np.random.default_rng(31337)

    baselines: dict = {}
    for scope_name, subset in (
        ("all", rows),
        *[
            (f"config:{name}", [r for r in rows if r["config"] == name])
            for name in sorted({r["config"] for r in rows})
        ],
        *[
            (f"region:{name}", [r for r in rows if r["region"] == name])
            for name in ("walk", "seed-program")
        ],
    ):
        if not subset:
            continue
        witness_rows = [
            {**r, "witness_only": [r["witness_action"]]} for r in subset
        ]
        baselines[scope_name] = {
            "states": len(subset),
            "mean_legal_sites": round(
                sum(r["legal"] for r in subset) / len(subset), 2
            ),
            "against_witness_only": recall_table(
                witness_rows, "witness_only", ranker_names
            ),
            "against_cheap_A_plus": recall_table(subset, "known_good", ranker_names),
            "against_cheap_A_plus_states_with_over_10_legal_sites": recall_table(
                subset, "known_good", ranker_names, min_legal=11
            ),
        }

    brute_rows = [r for rows_ in two["configs"].values() for r in rows_]
    # How the set behaves as the move budget is loosened from the shortest
    # completion the search demonstrated up to the witness's own remaining moves.
    # This is the dial M1 has to set, and the curve is what says where to set it.
    SLACKS = (0, 1, 2, 4, 8, 16, 10_000)
    for row in brute_rows:
        best = row["shortest_completion_moves"]
        row["slack_sets"] = {}
        for slack in SLACKS:
            if best is None:
                row["slack_sets"][str(slack)] = []
                continue
            row["slack_sets"][str(slack)] = sorted(
                a for a, moves, _cc in row["completion_moves"] if moves <= best + slack
            )
        row["A_plus_shortest_plus_2"] = row["slack_sets"]["2"]
    # The fitted kind order is an upper bound on what a kind-only ranker can do,
    # and it is fitted on the very set it is scored against.  It is reported as a
    # ceiling, never as a baseline anyone is allowed to claim.
    for row in brute_rows:
        spec = CONFIGS[row["config"]].spec()
        word = tuple(row["word"])
        moves = reference_successors(spec, word, row["strands"])
        row["rankings"][fitted_name] = fitted(
            spec, word, row["strands"], moves, rank_rng
        )[: max(KS)]
    brute_ranker_names = [*ranker_names, fitted_name]
    calibration = {}
    if brute_rows:
        cheap_total = sum(len(r["cheap_known_good"]) for r in brute_rows)
        brute_total = sum(len(r["bruteforce_known_good"]) for r in brute_rows)
        missed = sum(
            len(set(r["bruteforce_known_good"]) - set(r["cheap_known_good"]))
            for r in brute_rows
        )
        cheap_only = sum(
            len(set(r["cheap_known_good"]) - set(r["bruteforce_known_good"]))
            for r in brute_rows
        )
        calibration = {
            "states": len(brute_rows),
            "mean_legal_sites": round(
                sum(r["legal"] for r in brute_rows) / len(brute_rows), 2
            ),
            "mean_cheap_A_plus": round(cheap_total / len(brute_rows), 3),
            "mean_bruteforce_A_plus": round(brute_total / len(brute_rows), 3),
            "mean_bruteforce_undetermined": round(
                sum(r["bruteforce_undetermined"] for r in brute_rows)
                / len(brute_rows),
                3,
            ),
            "sites_bruteforce_found_that_cheap_missed": missed,
            "sites_cheap_found_that_bruteforce_missed": cheap_only,
            "cheap_coverage_of_bruteforce": round(
                1.0 - missed / max(brute_total, 1), 4
            ),
            "bruteforce_saturation": round(
                brute_total / max(sum(r["legal"] for r in brute_rows), 1), 4
            ),
            "witness_on_a_shortest_known_completion": round(
                sum(int(r["witness_in_shortest"]) for r in brute_rows)
                / len(brute_rows),
                4,
            ),
            "mean_shortest_completion_moves": round(
                sum(r["shortest_completion_moves"] or 0 for r in brute_rows)
                / len(brute_rows),
                2,
            ),
            "mean_witness_remaining_moves": round(
                sum(r["remaining_moves"] for r in brute_rows) / len(brute_rows), 2
            ),
            "mean_shortest_known_set_size": round(
                sum(len(r["shortest_known_set"]) for r in brute_rows)
                / len(brute_rows),
                3,
            ),
            "seconds": round(sum(r["seconds"] for r in brute_rows), 1),
            "baselines_against_bruteforce_A_plus": recall_table(
                brute_rows, "bruteforce_known_good", brute_ranker_names
            ),
            "baselines_against_bruteforce_A_plus_over_10_legal": recall_table(
                brute_rows, "bruteforce_known_good", brute_ranker_names, min_legal=11
            ),
            "baselines_against_shortest_known_A_plus_over_10_legal": recall_table(
                brute_rows, "shortest_known_set", brute_ranker_names, min_legal=11
            ),
            "baselines_against_shortest_known_A_plus": recall_table(
                brute_rows, "shortest_known_set", brute_ranker_names
            ),
            "baselines_against_cheap_A_plus": recall_table(
                brute_rows, "cheap_known_good", brute_ranker_names
            ),
            "baselines_against_witness_only": recall_table(
                [{**r, "witness_only": [r["witness_action"]]} for r in brute_rows],
                "witness_only",
                brute_ranker_names,
            ),
            "baselines_against_A_plus_shortest_plus_2": recall_table(
                brute_rows, "A_plus_shortest_plus_2", brute_ranker_names
            ),
            "move_budget_slack_curve": {
                str(slack): {
                    "mean_A_plus": round(
                        sum(len(r["slack_sets"][str(slack)]) for r in brute_rows)
                        / len(brute_rows),
                        3,
                    ),
                    "fraction_of_legal": round(
                        sum(len(r["slack_sets"][str(slack)]) for r in brute_rows)
                        / max(sum(r["legal"] for r in brute_rows), 1),
                        4,
                    ),
                    "uniform_chance_recall@10": round(
                        sum(
                            random_hit_probability(
                                r["legal"], len(r["slack_sets"][str(slack)]), 10
                            )
                            for r in brute_rows
                        )
                        / len(brute_rows),
                        4,
                    ),
                    "witness_site_in_the_set": round(
                        sum(
                            int(r["witness_action"] in r["slack_sets"][str(slack)])
                            for r in brute_rows
                        )
                        / len(brute_rows),
                        4,
                    ),
                }
                for slack in SLACKS
            },
        }

    payload = {
        "note": (
            "Every count is a lower bound on |A_plus|. A site absent from a set "
            "was never demonstrated to complete; it is UNEXPLORED, never negative."
        ),
        "generated": time.strftime("%Y-%m-%d"),
        "arguments": {k: (str(v) if isinstance(v, Path) else v)
                      for k, v in vars(args).items()},
        "phase1_cheap_methods": one,
        "phase2_bruteforce_calibration": calibration,
        "phase3_baselines": baselines,
        "bruteforce_sample_rows": [
            {
                k: v
                for k, v in row.items()
                if k not in ("rankings", "slack_sets", "completion_moves")
            }
            for row in brute_rows
        ],
        "fitted_kind_order": [KIND_NAMES[k] for k in fitted_order],
    }
    (args.out / "report.json").write_text(json.dumps(payload, indent=1))
    log(f"wrote {args.out / 'report.json'}", started)


if __name__ == "__main__":
    main()
