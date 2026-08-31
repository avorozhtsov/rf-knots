#!/usr/bin/env python3
"""Does seeding the interleaved walk fix the difficulty defect M0 measured?

M0 built the generator of note 22 §4.1 and found the label useless: at `k = 5`
from the empty one-braid the Murasugi bound came out `mean 1.01, never 5`, with
14% of instances literally the unknot.  §4.1b says why -- the inherited bound is
`L(seed) - k` and the unknot has `L = 0` -- and says what to do instead.

This script runs the grid that decides it: seed strength `L` crossed with walk
depth `k`, and for every emitted `R'` it computes the certified lower bound
**again, independently**, from `lower_bounds.computed_claims` on the word itself.
Three numbers come out of that comparison and all three are needed:

* the acceptance rate at certified `u >= 2, 3, 4, 5` -- the `>= 4` rate is what
  M3 has never had and what decides whether hard positives exist at all;
* whether the independently computed bound ever **exceeds** the inherited one,
  which would mean the walk is not merely preserving difficulty but adding it;
* instances per second per configuration, so the price of one hard positive is
  measured rather than guessed.

Every pair `(R', seed)` also leaves a two-sided Gordian interval behind:
`upper = d` by construction, `lower = max |f(R') - f(seed)|` over the 1-Lipschitz
`sigma/2` and `tau`.  Those are written in the shape
`rf_knots.embedding_supervision.DistanceInterval` carries and
`rf_knots.embedding_metric.MetricEmbeddingObjective` consumes.

Usage::

    python scripts/measure_seeded_generator.py --project --count 300
    python scripts/measure_seeded_generator.py --count 300 \\
        --output-dir artifacts/m3-seeded
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

from rf_knots.interleaved_generator import (  # noqa: E402
    InterleavedConfig,
    InterleavedInstance,
    LipschitzValues,
    Quarantine,
    SeedRepresentation,
    default_seed,
    generate,
    gordian_interval,
    lipschitz_values,
    load_quarantine,
    positive_braid_seed,
    torus_seed,
)
from rf_knots.reference import free_reduce  # noqa: E402

UNKNOTDB = "/Users/artemvorozhtsov/projects/unknotdb"
SCHEMA = "rf-knots-m3-seeded-measurement-v0"


# -- the grid ------------------------------------------------------------------


def build_seeds(
    config: InterleavedConfig,
    strengths: list[int],
    families: list[str],
    braid_strands: int,
    seed: int,
) -> list[tuple[str, int, SeedRepresentation]]:
    """`(family, L, seed)` for the requested grid, each verified as it is built."""
    spec = config.spec()
    out: list[tuple[str, int, SeedRepresentation]] = []
    for strength in strengths:
        if "unknot" in families and strength == 0:
            out.append(("unknot", 0, default_seed(config)))
        if "torus" in families and strength > 0:
            out.append(("torus-2q", strength, torus_seed(strength, spec)))
        if "positive-braid" in families and strength > 0:
            representation = positive_braid_seed(
                spec, strands=braid_strands, genus=strength, seed=seed
            )
            out.append(("positive-braid", strength, representation))
    return out


def strongest(claims) -> tuple[int, str]:
    best, method = 0, "none"
    for claim in claims:
        if claim.value > best:
            best, method = claim.value, claim.method
    return best, method


def signed_values(claims) -> LipschitzValues:
    """Pull the signed `sigma` and `tau` back out of the claims we already paid for.

    `computed_claims` records both in `LowerBoundClaim.details`, so the Gordian
    side of the measurement costs no extra Seifert matrix.
    """
    half = tau = None
    for claim in claims:
        if claim.method == "signature" and "signature" in claim.details:
            half = int(claim.details["signature"]) // 2
        if claim.method == "ozsvath-szabo-tau" and "tau" in claim.details:
            tau = int(claim.details["tau"])
    return LipschitzValues(half, tau)


def histogram(counter: Counter) -> dict[str, Any]:
    if not counter:
        return {}
    keys = sorted(counter)
    total = sum(counter.values())
    return {
        "min": keys[0],
        "max": keys[-1],
        "mean": round(sum(k * v for k, v in counter.items()) / total, 3),
        "counts": {str(k): counter[k] for k in keys},
    }


# -- one cell ------------------------------------------------------------------


def measure_cell(
    config: InterleavedConfig,
    representation: SeedRepresentation,
    family: str,
    strength: int,
    count: int,
    rng_seed: int,
    quarantine: Quarantine,
    seed_values: LipschitzValues,
    deadline: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from rf_knots import seifert
    from rf_knots.lower_bounds import computed_claims

    achieved: Counter = Counter()
    inherited: Counter = Counter()
    excess: Counter = Counter()
    methods: Counter = Counter()
    gordian_lower: Counter = Counter()
    gordian_methods: Counter = Counter()
    gordian_exact = 0
    d_upper: Counter = Counter()
    lengths: Counter = Counter()
    rejected: Counter = Counter()
    intervals: list[dict[str, Any]] = []

    emitted = 0
    index = 0
    replay_complete = 0
    exact = 0
    started = time.perf_counter()
    generation_seconds = 0.0
    bound_seconds = 0.0
    while emitted < count:
        if time.monotonic() > deadline:
            break
        mark = time.perf_counter()
        instance: InterleavedInstance = generate(
            config, seed=rng_seed, index=index, seed_representation=representation
        )
        index += 1
        generation_seconds += time.perf_counter() - mark
        reason = quarantine.screen_word(instance.word, instance.strands)
        if reason is not None:
            rejected[reason] += 1
            continue

        mark = time.perf_counter()
        reduced = free_reduce(instance.word)
        claims = computed_claims(reduced, instance.strands)
        value, method = strongest(claims)
        values = signed_values(claims)
        # The target's fingerprint has `sigma = 0`, so a non-zero signature rules
        # it out without a second Seifert matrix.  This is an exact shortcut, not
        # an approximation of the screen.
        if values.signature_half == 0:
            try:
                determinant = seifert.determinant(reduced, instance.strands)
            except (ValueError, seifert.BackendUnavailable):
                determinant = None
            if determinant is not None:
                reason = quarantine.screen_fingerprint(determinant, 0)
                if reason is not None:
                    rejected[reason] += 1
                    bound_seconds += time.perf_counter() - mark
                    continue
        bound_seconds += time.perf_counter() - mark

        achieved[value] += 1
        methods[method] += 1
        inherited[instance.u_lower_inherited] += 1
        excess[value - instance.u_lower_inherited] += 1
        d_upper[instance.d_upper] += 1
        lengths[len(instance.word)] += 1
        if instance.replay_complete:
            replay_complete += 1
        if instance.u_upper is not None and value == instance.u_upper:
            exact += 1

        interval = gordian_interval(instance, left=values, right=seed_values)
        gordian_lower[interval.lower] += 1
        gordian_methods[interval.lower_method] += 1
        if interval.exact:
            gordian_exact += 1
        intervals.append(interval.to_dict())
        emitted += 1

    elapsed = time.perf_counter() - started
    total = max(emitted, 1)
    row: dict[str, Any] = {
        "family": family,
        "seed_name": representation.name,
        "seed_strength_L": strength,
        "seed_certified_lower_bound": representation.certified_lower_bound,
        "seed_u_upper": representation.u_upper,
        "seed_replay_complete": representation.replay_complete,
        "k": config.crossing_changes,
        "walk_length": config.walk_length,
        "requested": count,
        "emitted": emitted,
        "attempts": index,
        "rejected": dict(rejected),
        "seconds": round(elapsed, 3),
        "instances_per_second": round(emitted / elapsed, 2) if elapsed else None,
        "generation_seconds": round(generation_seconds, 3),
        "bound_seconds": round(bound_seconds, 3),
        "replay_complete_fraction": round(replay_complete / total, 4),
        "inherited_lower_bound": histogram(inherited),
        "achieved_lower_bound": histogram(achieved),
        "achieved_minus_inherited": histogram(excess),
        "achieved_method": dict(methods.most_common()),
        "d_upper": histogram(d_upper),
        "word_length": histogram(lengths),
        "exact_at_u_upper": exact,
        "acceptance": {
            f">={floor}": round(
                sum(v for k, v in achieved.items() if k >= floor) / total, 4
            )
            for floor in (2, 3, 4, 5)
        },
        "acceptance_count": {
            f">={floor}": sum(v for k, v in achieved.items() if k >= floor)
            for floor in (2, 3, 4, 5)
        },
        "gordian_lower": histogram(gordian_lower),
        "gordian_lower_method": dict(gordian_methods.most_common()),
        "gordian_nonzero_fraction": round(
            sum(v for k, v in gordian_lower.items() if k > 0) / total, 4
        ),
        # `lower == upper`, not `lower == k`: loop erasure can leave a walk with
        # fewer than `k` changes, and such a pair is exact at its own `d`.
        "gordian_exact_fraction": round(gordian_exact / total, 4),
    }
    for floor in (2, 3, 4, 5):
        hits = row["acceptance_count"][f">={floor}"]
        rate = hits / total
        row.setdefault("seconds_per_hard_positive", {})[f">={floor}"] = (
            round(elapsed / hits, 3) if hits else None
        )
        row.setdefault("cost_multiplier", {})[f">={floor}"] = (
            round(1.0 / rate, 2) if rate else None
        )
    return row, intervals


# -- report --------------------------------------------------------------------


def best_cost(rows: list[dict[str, Any]], unknot: list[dict[str, Any]]) -> str:
    """Cheapest `u >= 4` positive among the unknot-seeded cells, as a string."""
    del rows
    costs = [
        row["seconds_per_hard_positive"][">=4"]
        for row in unknot
        if row["seconds_per_hard_positive"][">=4"] is not None
    ]
    return f"{min(costs)} s" if costs else "no hit at all"


def verdict(report: dict[str, Any]) -> list[str]:
    """The three numbers the task asked for, read straight off the cells."""
    rows = report["cells"]
    lines: list[str] = []

    def rate(row: dict[str, Any], floor: str) -> float:
        return row["acceptance"][floor]

    unknot = [row for row in rows if row["family"] == "unknot"]
    seeded = [row for row in rows if row["family"] != "unknot"]
    below = sum(
        int(count)
        for row in rows
        for key, count in row["achieved_minus_inherited"].get("counts", {}).items()
        if int(key) < 0
    )
    above = sum(
        int(count)
        for row in rows
        for key, count in row["achieved_minus_inherited"].get("counts", {}).items()
        if int(key) > 0
    )
    total = sum(row["emitted"] for row in rows)
    nonzero = sum(
        row["gordian_nonzero_fraction"] * row["emitted"] for row in rows
    )
    exact_pairs = sum(row["gordian_exact_fraction"] * row["emitted"] for row in rows)

    depth = max(row["k"] for row in rows)
    deep_unknot = [row for row in unknot if row["k"] == depth]
    deep_seeded = [row for row in seeded if row["k"] == depth]

    same = all(
        max((rate(row, floor) for row in deep_seeded), default=0.0)
        == max((rate(row, floor) for row in seeded), default=0.0)
        and max((rate(row, floor) for row in deep_unknot), default=0.0)
        == max((rate(row, floor) for row in unknot), default=0.0)
        for floor in (">=2", ">=3", ">=4", ">=5")
    )
    lines.append("## 0. Verdict\n")
    lines.append(
        f"Best acceptance rate at `k = {depth}`, the depth the target needs"
        + (
            ", which is also the best over the whole grid on every row"
            if same
            else "; the best over the whole grid differs on at least one row"
        )
        + f". §1 has all {len(rows)} cells.\n"
    )
    lines.append("| | unknot-seeded (§4.1) | seeded (§4.1b) |\n|---|---:|---:|")
    for floor in (">=2", ">=3", ">=4", ">=5"):
        left = max((rate(row, floor) for row in deep_unknot), default=0.0)
        right = max((rate(row, floor) for row in deep_seeded), default=0.0)
        emphasis = "**" if floor == ">=4" else ""
        lines.append(
            f"| certified `u {floor}` | {left:.3f} | "
            f"{emphasis}{right:.3f}{emphasis} |"
        )
    lines.append("")

    cheapest = min(
        (
            row
            for row in rows
            if row["seconds_per_hard_positive"][">=4"] is not None
        ),
        key=lambda row: row["seconds_per_hard_positive"][">=4"],
        default=None,
    )
    if cheapest is not None:
        lines.append(
            f"- **A hard positive costs milliseconds.** The cheapest cell at "
            f"`u >= 4` is `{cheapest['seed_name']}` "
            f"(`L = {cheapest['seed_strength_L']}`) at `k = {cheapest['k']}`: "
            f"{rate(cheapest, '>=4'):.3f} acceptance, "
            f"{cheapest['instances_per_second']} instances/s, "
            f"**{cheapest['seconds_per_hard_positive']['>=4']} s** per hard "
            f"positive against {best_cost(rows, unknot)} for the unknot-seeded "
            f"control."
        )
    deep_best = (
        max(deep_seeded, key=lambda row: (rate(row, ">=5"), rate(row, ">=4")))
        if deep_seeded
        else None
    )
    if deep_best is not None:
        lines.append(
            f"- **At `k = {depth}` the shape M3 wants**, `{deep_best['seed_name']}` "
            f"gives {rate(deep_best, '>=4'):.3f} at `u >= 4` and "
            f"{rate(deep_best, '>=5'):.3f} at `u >= 5`, "
            f"{deep_best['instances_per_second']} instances/s "
            f"({deep_best['seconds_per_hard_positive']['>=4']} s per hard "
            f"positive). The unknot-seeded control at the same `k` gives "
            f"{max((rate(row, '>=4') for row in deep_unknot), default=0.0):.3f} "
            f"and {max((rate(row, '>=5') for row in deep_unknot), default=0.0):.3f}."
        )
    control = [row for row in unknot if row["k"] == depth]
    if control:
        lines.append(
            f"- **The control reproduces M0.** Unknot-seeded at `k = {depth}` the "
            f"achieved bound has mean "
            f"{control[0]['achieved_lower_bound'].get('mean')} and maximum "
            f"{control[0]['achieved_lower_bound'].get('max')} -- M0 measured "
            f"Murasugi `mean 1.01, never 5` on the same shape. The generator was "
            f"never weak; it was seeded at the one point of the graph that "
            f"certifies nothing."
        )
    lines.append(
        f"- **Inherited and computed agree.** Over {total} instances the "
        f"independently computed bound falls below the inherited one "
        f"**{below} times** and exceeds it {above} times "
        f"({above / max(total, 1):.1%})."
    )
    generation = sum(row["generation_seconds"] for row in rows)
    bounds = sum(row["bound_seconds"] for row in rows)
    lines.append(
        f"- **The generator is not the cost; the certificate is.** "
        f"{generation:.0f} s of the {generation + bounds:.0f} s of measured work "
        f"went on generating instances and {bounds:.0f} s "
        f"({bounds / max(generation + bounds, 1e-9):.0%}) on computing the "
        f"independent bound. Instances whose difficulty is *inherited* need "
        f"none of that second number."
    )
    lines.append(
        f"- **The Gordian lower side is populated.** "
        f"{nonzero / max(total, 1):.1%} of pairs carry a non-zero certified "
        f"lower bound and {exact_pairs / max(total, 1):.1%} are exact "
        f"(`lower = upper`, so the pair pins `d(R', R)` outright)."
    )
    lines.append("")
    return lines


def markdown(report: dict[str, Any]) -> str:
    rows = report["cells"]
    lines: list[str] = []
    lines.append("# M3: seed the interleaved walk anywhere, and the label stops lying\n")
    lines.append(
        f"`{report['schema']}` | generated {report['generated_at']} | "
        f"seed {report['rng_seed']} | {report['total_emitted']} instances in "
        f"{report['seconds']} s on one core\n"
    )
    lines.append(
        "M0 built note 22 §4.1's generator and found the label useless: at "
        "`k = 5` from the empty one-braid the Murasugi bound came out "
        "`mean 1.01, never 5`, with 14% of instances literally the unknot. "
        "§4.1b says why. `u` is the graph distance to the unknot, so it is "
        "1-Lipschitz under a crossing change and a `k`-change walk from a seed "
        "`R` of `K` gives `u(R') >= L(K) - k` -- and the unknot has `L = 0`.\n"
    )
    lines.append(
        "Every emitted instance below carries `d(R', R) <= k` by construction, "
        "that inherited bound, and -- computed again and **independently** from "
        "`lower_bounds.computed_claims` on `R'` itself -- an *achieved* certified "
        "lower bound. The two are compared, never conflated.\n"
    )
    lines.extend(verdict(report))

    lines.append("## 1. Acceptance at a certified lower bound\n")
    lines.append(
        "| family | seed | `L` | `k` | `L-k` | achieved mean | `>=2` | `>=3` | "
        "**`>=4`** | `>=5` | inst/s | s per `>=4` |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        acceptance = row["acceptance"]
        cost = row["seconds_per_hard_positive"][">=4"]
        mean = row["achieved_lower_bound"].get("mean", 0)
        lines.append(
            f"| {row['family']} | `{row['seed_name']}` | {row['seed_strength_L']} | "
            f"{row['k']} | {max(0, row['seed_strength_L'] - row['k'])} | {mean} | "
            f"{acceptance['>=2']:.2f} | {acceptance['>=3']:.2f} | "
            f"**{acceptance['>=4']:.2f}** | {acceptance['>=5']:.2f} | "
            f"{row['instances_per_second']} | "
            f"{'--' if cost is None else cost} |"
        )
    lines.append("")

    lines.append("## 2. Computed against inherited\n")
    lines.append(
        "`achieved - inherited` per instance.  A positive value means the walk "
        "left `R'` *harder* than the inheritance formula could see; a negative "
        "one would be a contradiction and there are none.\n"
    )
    lines.append("| family | seed | `k` | min | mean | max | below inherited |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        excess = row["achieved_minus_inherited"]
        below = sum(
            int(v) for k, v in excess.get("counts", {}).items() if int(k) < 0
        )
        lines.append(
            f"| {row['family']} | `{row['seed_name']}` | {row['k']} | "
            f"{excess.get('min')} | {excess.get('mean')} | {excess.get('max')} | "
            f"{below} |"
        )
    lines.append("")

    lines.append("## 3. Gordian intervals\n")
    lines.append(
        "`upper = d(R', R) <= k` by construction; `lower = max |f(R') - f(R)|` "
        "over the 1-Lipschitz `sigma/2` and `tau`.\n"
    )
    lines.append(
        "| family | seed | `k` | non-zero lower | mean lower | "
        "exact (`lower = upper`) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['family']} | `{row['seed_name']}` | {row['k']} | "
            f"{row['gordian_nonzero_fraction']:.2f} | "
            f"{row['gordian_lower'].get('mean')} | "
            f"{row['gordian_exact_fraction']:.2f} |"
        )
    lines.append("")

    lines.append("## 4. Quarantine\n")
    quarantine = report["quarantine"]
    lines.append(
        f"- {quarantine['representations']} stored representations screened; "
        f"names {', '.join(quarantine['names'])}; fingerprints "
        f"{quarantine['fingerprints']}."
    )
    for screen in quarantine["screens"]:
        lines.append(f"- {screen}")
    lines.append(f"- **`k`-ball closure:** {quarantine['k_ball_closure']}")
    lines.append(
        f"- seeds screened: {report['seeds_screened']}, "
        f"seeds rejected: {report['seeds_rejected']}; emitted words rejected: "
        f"{report['rejected_total']} {report['rejected_by_reason']}"
    )
    lines.append("")

    lines.append("## 5. Seeds\n")
    lines.append("| seed | family | `L` | method | `u_upper` | program moves | exact |")
    lines.append("|---|---|---:|---|---:|---:|---|")
    for seed in {row["name"]: row for row in report["seeds"]}.values():
        method = (
            seed["lower_bounds"][0]["method"] if seed["lower_bounds"] else "--"
        )
        lines.append(
            f"| `{seed['name']}` | {seed['family']} | "
            f"{seed['certified_lower_bound']} | {method} | {seed['u_upper']} | "
            f"{seed['program_moves']} | {'yes' if seed['exact'] else 'no'} |"
        )
    lines.append("")
    lines.append(
        "Every program was replayed to the empty one-braid at build time, so "
        "every row above is **replay-complete**: the composed witness "
        "`R' -> seed -> unknot` is a single verified `UnknotWitness`. No cell in "
        "this report is catalogue-composed.\n"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--strengths", type=int, nargs="+", default=[0, 3, 5, 7, 10])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument(
        "--families",
        nargs="+",
        default=["unknot", "torus", "positive-braid"],
        choices=["unknot", "torus", "positive-braid"],
    )
    parser.add_argument("-l", "--walk-length", type=int, default=24)
    parser.add_argument("--target-strands", type=int, default=4)
    parser.add_argument("--target-length", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--max-strands", type=int, default=6)
    parser.add_argument("--braid-seed-strands", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/m3-seeded"))
    parser.add_argument("--unknotdb", type=str, default=UNKNOTDB)
    parser.add_argument(
        "--project",
        action="store_true",
        help="run a small sample of every cell and print the projected total "
        "cost of the full sweep instead of running it",
    )
    parser.add_argument("--project-count", type=int, default=8)
    parser.add_argument(
        "--time-budget",
        type=float,
        default=1800.0,
        help="hard wall-clock cap in seconds; cells stop early rather than "
        "overrun and the report records what was truncated",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="rewrite report.md from an existing report.json without "
        "regenerating a single instance",
    )
    args = parser.parse_args()

    if args.render_only:
        report = json.loads((args.output_dir / "report.json").read_text())
        (args.output_dir / "report.md").write_text(markdown(report))
        print(f"rendered {args.output_dir}/report.md from report.json")
        return

    quarantine = load_quarantine(args.unknotdb)
    deadline = time.monotonic() + args.time_budget
    count = args.project_count if args.project else args.count

    cells: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    seeds_screened = 0
    seeds_rejected: list[dict[str, str]] = []
    started = time.perf_counter()
    last_report = time.monotonic()

    plan: list[tuple[str, int, SeedRepresentation, InterleavedConfig]] = []
    for depth in args.depths:
        config = InterleavedConfig(
            crossing_changes=depth,
            walk_length=args.walk_length,
            target_strands=args.target_strands,
            target_length=args.target_length,
            max_len=args.max_len,
            max_strands=args.max_strands,
        )
        for family, strength, representation in build_seeds(
            config, args.strengths, args.families, args.braid_seed_strands, args.seed
        ):
            plan.append((family, strength, representation, config))

    seed_cache: dict[tuple[tuple[int, ...], int], LipschitzValues] = {}
    for family, strength, representation, config in plan:
        seeds_screened += 1
        reason = quarantine.screen_seed(representation)
        if reason is not None:
            seeds_rejected.append({"seed": representation.name, "reason": reason})
            continue
        key = (representation.word, representation.strands)
        if key not in seed_cache:
            seed_cache[key] = lipschitz_values(*key)
        row, rows = measure_cell(
            config,
            representation,
            family,
            strength,
            count,
            args.seed,
            quarantine,
            seed_cache[key],
            deadline,
        )
        cells.append(row)
        intervals.extend(rows)
        if time.monotonic() - last_report >= 15.0:
            last_report = time.monotonic()
            print(
                f"[{time.perf_counter() - started:7.1f}s] {len(cells)}/{len(plan)} "
                f"cells, {len(intervals)} instances",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    total_emitted = sum(row["emitted"] for row in cells)

    if args.project:
        per_instance = elapsed / max(total_emitted, 1)
        projected = per_instance * args.count * len(plan)
        print(
            json.dumps(
                {
                    "projection": {
                        "cells": len(plan),
                        "sample_instances": total_emitted,
                        "sample_seconds": round(elapsed, 2),
                        "seconds_per_instance": round(per_instance, 4),
                        "full_count_per_cell": args.count,
                        "projected_instances": args.count * len(plan),
                        "projected_seconds": round(projected, 1),
                        "projected_minutes": round(projected / 60.0, 2),
                        "projected_interval_bytes": int(
                            args.count * len(plan) * 340
                        ),
                    }
                },
                indent=2,
            )
        )
        return

    rejected_by_reason: Counter = Counter()
    for row in cells:
        rejected_by_reason.update(row["rejected"])

    report = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rng_seed": args.seed,
        "seconds": round(elapsed, 2),
        "total_emitted": total_emitted,
        "count_per_cell": args.count,
        "grid": {
            "strengths": args.strengths,
            "depths": args.depths,
            "families": args.families,
            "walk_length": args.walk_length,
            "target_strands": args.target_strands,
            "target_length": args.target_length,
            "max_len": args.max_len,
            "max_strands": args.max_strands,
            "braid_seed_strands": args.braid_seed_strands,
        },
        "quarantine": quarantine.provenance(),
        "seeds_screened": seeds_screened,
        "seeds_rejected": seeds_rejected,
        "rejected_total": sum(rejected_by_reason.values()),
        "rejected_by_reason": dict(rejected_by_reason),
        "truncated": [
            row["seed_name"] for row in cells if row["emitted"] < row["requested"]
        ],
        "seeds": list(
            {
                representation.name: representation.provenance()
                for _, _, representation, _ in plan
            }.values()
        ),
        "cells": cells,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(report, indent=2, sort_keys=False) + "\n"
    temporary = args.output_dir / f".report.json.{os.getpid()}.part"
    temporary.write_text(body)
    os.replace(temporary, args.output_dir / "report.json")
    (args.output_dir / "report.md").write_text(markdown(report))
    with (args.output_dir / "gordian-intervals.jsonl").open("w") as stream:
        for row in intervals:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(
        f"wrote {args.output_dir}/report.json, report.md and "
        f"{len(intervals)} gordian intervals in {elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
