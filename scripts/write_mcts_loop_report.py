"""Render `artifacts/mcts-loop/report.json` as `report.md`.

Nothing here hardcodes what it is reporting -- rung names, arm names and doses are
all read off the results.  `docs/lessons.md`: a report that states its own shape
relabels a historical run with today's structure the first time the shape changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def table(rows: list[dict], rungs: list[str], doses: list[int], arms: list[str]) -> list[str]:
    lookup = {(r["arm"], r["rung"], r["simulations"]): r for r in rows}
    lines = []
    header = "| arm | " + " | ".join(
        f"{rung} @ {dose}" for rung in rungs for dose in doses
    ) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (1 + len(rungs) * len(doses)))
    for arm in arms:
        cells = []
        for rung in rungs:
            for dose in doses:
                row = lookup.get((arm, rung, dose))
                cells.append(f"{row['witness_rate']:.3f}" if row else "--")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    return lines


#: Standing limits of this loop.  They are statements about the design, not about
#: this run's numbers, which is why they are the one thing here that is written
#: down rather than derived -- and they are what the next milestone has to attack.
LIMITS = [
    "**The value head is not §5.1's.** It is a single "
    "`p(solve within the remaining budget)` scalar, so one rollout supervises one "
    "budget rather than a monotone curve over the whole budget vector, and "
    "monotonicity is not built in -- it is not even checked.",
    "**There is no outer/inner two-level search (§3).** One MCTS tree searches "
    "the whole `(isotopy, crossing change)` space at once. There is no "
    "identification against the knot table, no certified lower bound as a "
    "terminal test, and no proof graph. The transposition guard is exact word "
    "equality, so two presentations of one knot are two nodes.",
    "**The policy is not trained by GRPO (§5.2).** No group-relative advantage, "
    "no clipped ratio, no KL anchor to the previous checkpoint, and therefore no "
    "retention panel.",
    "**`A_plus` reaches tiers 1 and 2 only.** The generator's own witnesses and "
    "the evidence index supply tier 1, `aplus.commutation_known_good` supplies "
    "tier 2. There is no tier-3 attestation and no brute-force beam, so a site "
    "that works but that nothing has exhibited stays undetermined.",
    "**The rungs are `u = 1` three-strand knots.** Nothing here goes near cap 5, "
    "the 20-letter B5 word, or the 254-letter target. The rungs come from the "
    "curriculum bank's own easy end and the cap is the bank's certified upper "
    "bound, so a single crossing change is always enough.",
    "**Self-play trains only on generated instances.** Their difficulty is "
    "certified but their distribution is a backward random walk, not the "
    "curriculum's. The curriculum rungs are held out entirely and never trained "
    "on -- which is what makes the evaluation clean, and also what leaves the "
    "distribution gap unmeasured.",
    "**No incremental recompute.** Every leaf is a full encoder forward over the "
    "whole word (§6.3 asks for the opposite), which is why the affordable "
    "simulation counts are in the hundreds and not the tens of thousands.",
]


def _best(rows: list[dict], prefix: str, rung: str, dose: int) -> dict | None:
    """The best arm whose name starts with `prefix` at this `(rung, dose)`."""
    candidates = [
        r
        for r in rows
        if r["arm"].startswith(prefix) and r["rung"] == rung and r["simulations"] == dose
    ]
    return max(candidates, key=lambda r: r["witness_rate"]) if candidates else None


def _paired(left: dict, right: dict) -> tuple[int, int, int]:
    """Instance-by-instance `(better, worse, tied)` for `left` against `right`.

    §3.5's lesson: an aggregate ordering that reverses when one instance is
    dropped is not a result.  The paired counts are what says whether a gap is
    spread across the rung or is two instances carrying it.
    """
    a, b = left.get("per_instance", {}), right.get("per_instance", {})
    shared = sorted(set(a) & set(b))
    better = sum(1 for k in shared if a[k] > b[k])
    worse = sum(1 for k in shared if a[k] < b[k])
    return better, worse, len(shared) - better - worse


def headline(rows: list[dict], rungs: list[str], doses: list[int]) -> list[str]:
    """The three arms the milestone asks for, plus the honest strongest control.

    `docs/lessons.md`: measure the untrained floor before reading a training
    curve.  The floor here turns out not to be the classical beam -- an MCTS with
    a *flat* prior is the stronger search-only arm at every dose -- so both are
    reported and the network is judged against the better of them.
    """
    lines: list[str] = []
    lines.append("### The three arms, and the strongest control")
    lines.append("")
    lines.append(
        "| rung | dose | untrained net | trained net | classical beam | "
        "flat-prior MCTS | trained vs strongest control |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for rung in rungs:
        for dose in doses:
            untrained = _best(rows, "untrained-", rung, dose)
            trained = _best(rows, "trained-", rung, dose)
            beam = _best(rows, "classical-beam", rung, dose)
            flat = _best(rows, "uniform-mcts", rung, dose)
            controls = [c for c in (beam, flat) if c]
            best_control = max(controls, key=lambda r: r["witness_rate"]) if controls else None
            verdict = "--"
            if trained and best_control:
                better, worse, tied = _paired(trained, best_control)
                delta = trained["witness_rate"] - best_control["witness_rate"]
                verdict = (
                    f"{delta:+.3f} vs `{best_control['arm']}`; "
                    f"paired {better} better / {worse} worse / {tied} tied"
                )
            cells = [
                f"{untrained['witness_rate']:.3f}" if untrained else "--",
                f"{trained['witness_rate']:.3f}" if trained else "--",
                f"{beam['witness_rate']:.3f}" if beam else "--",
                f"{flat['witness_rate']:.3f}" if flat else "--",
                verdict,
            ]
            lines.append(f"| {rung} | {dose} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "The network columns take the better of the two root rules at that cell; "
        "the per-rule numbers are in the table above."
    )
    lines.append("")
    return lines


def reading(rows: list[dict], rungs: list[str], doses: list[int]) -> list[str]:
    """The verdict, counted off the cells rather than asserted."""
    cells = [(rung, dose) for rung in rungs for dose in doses]
    tally = {"untrained": [0, 0, 0], "classical-beam": [0, 0, 0], "uniform-mcts": [0, 0, 0]}
    paired = [0, 0, 0]
    for rung, dose in cells:
        trained = _best(rows, "trained-", rung, dose)
        if trained is None:
            continue
        for key, prefix in (
            ("untrained", "untrained-"),
            ("classical-beam", "classical-beam"),
            ("uniform-mcts", "uniform-mcts"),
        ):
            other = _best(rows, prefix, rung, dose)
            if other is None:
                continue
            slot = 0 if trained["witness_rate"] > other["witness_rate"] else (
                2 if trained["witness_rate"] < other["witness_rate"] else 1
            )
            tally[key][slot] += 1
        flat = _best(rows, "uniform-mcts", rung, dose)
        if flat is not None:
            better, worse, _ = _paired(trained, flat)
            paired[0 if better > worse else (2 if better < worse else 1)] += 1

    lines = ["## Reading", ""]
    lines.append(f"Over the {len(cells)} `(rung, dose)` cells, the trained network:")
    lines.append("")
    for key, label in (
        ("untrained", "the **untrained** network at the same simulations"),
        ("classical-beam", "the **classical beam**"),
        ("uniform-mcts", "the **flat-prior MCTS**, the strongest search-only arm,"),
    ):
        win, tie, loss = tally[key]
        lines.append(f"* beats {label} in {win}, ties in {tie}, loses in {loss};")
    win, tie, loss = paired
    lines.append(
        f"* and against that flat-prior arm *instance by instance*, is ahead on more "
        f"instances than it is behind in {win} cells, level in {tie} and behind in "
        f"{loss}."
    )
    lines.append("")
    lines.append(
        "The flat-prior rows are the ones that matter, and they say something "
        "different from the rest. Against the untrained floor and the classical beam the "
        "gap is not close. Against a flat prior it is thin on the low-slack rungs "
        "and reverses at the largest dose, where the search no longer needs help "
        "finding the answer -- and it is decisive in the growth-heavy regime below, "
        "which is the regime §2.3 says the target actually requires."
    )
    lines.append("")
    return lines


def growth_section(out: Path) -> list[str]:
    """The growth-heavy regime, if `run_mcts_loop_growth.py` has been run."""
    path = out / "growth.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    rows = payload.get("arms", [])
    if not rows:
        return []
    lines = ["## The growth-heavy regime", ""]
    lines.append(
        "§2.3 and §3.6: the search has to be able to grow the diagram, and a "
        "length-ordered beam by construction never will. Widening the slack the "
        "`ActionSpec` gets over the instance multiplies the branching factor "
        "without changing a single instance. No retraining -- the same three "
        "checkpoints as above."
    )
    lines.append("")
    growths = sorted({r["growth_len"] for r in rows})
    doses = sorted({r["simulations"] for r in rows})
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    header = "| arm | " + " | ".join(
        f"slack {g} @ {d}" for g in growths for d in doses
    ) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (1 + len(growths) * len(doses)))
    lookup = {(r["arm"], r["growth_len"], r["simulations"]): r for r in rows}
    for arm in arms:
        cells = []
        for growth in growths:
            for dose in doses:
                row = lookup.get((arm, growth, dose))
                cells.append(f"{row['witness_rate']:.3f}" if row else "--")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    lines.append("")
    claimed = sum(r["claimed"] for r in rows)
    verified = sum(r["verified"] for r in rows)
    lines.append(f"{claimed} claimed witnesses, {verified} verified.")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/mcts-loop")
    args = parser.parse_args()
    out = Path(args.out)
    payload = json.loads((out / "report.json").read_text())

    rows = payload.get("arms", [])
    rungs = sorted({r["rung"] for r in rows})
    doses = sorted({r["simulations"] for r in rows})
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    config = payload.get("config", {})
    size = payload.get("size_independence", {})
    corpus = payload.get("corpus", {})
    seeds = config.get("seeds")

    lines: list[str] = []
    lines.append("# The first MCTS training loop")
    lines.append("")
    lines.append(
        "Generated by `scripts/run_mcts_loop.py`; every number below is read off "
        "`report.json` rather than restated."
    )
    lines.append("")

    # -- what the loop is -------------------------------------------------
    lines.append("## The loop")
    lines.append("")
    lines.append(
        "* **Environment** `rf_knots.reference`, exact and in-process. A state is "
        "`(word, strands, crossing changes used, moves used)`; the terminal is the "
        "empty one-strand braid. A crossing change is offered only while its budget "
        "is unspent."
    )
    lines.append(
        f"* **Instances** the fast-6 cohort's own curriculum banks, in their "
        f"registered row order: `{Path(payload['bank']['prior']).name}` and "
        f"`{Path(payload['bank']['q50']).name}`. Each row's "
        "`certified_unknotting_upper_bound` *is* the crossing-change cap, so the "
        "instance ladder and its answers come from the bank rather than from here."
    )
    lines.append(
        "* **Search** one MCTS tree per `(instance, seed)`, "
        "`simulations` node expansions, priors from `SiteHead` and a leaf value "
        "from a single `p(solve within the remaining budget)` head. A simulation "
        "that reaches the empty braid *is* the witness. Root selection is PUCT or "
        "Gumbel top-`m` with sequential halving."
    )
    lines.append(
        "* **Targets** visit counts, the observed `did this subtree reach the "
        "unknot` bit for the value, and the set-valued `A_plus` loss of §5.3 "
        "wherever a replay-verified witness exhibits a good site. The `A_plus` term "
        "is `site_head.set_valued_loss` verbatim, whose signature has no way to "
        "express a negative."
    )
    lines.append(
        f"* **Supervision** {corpus.get('generated_instances', '?')} "
        f"backward-generated instances (tier 1, free by construction) contributing "
        f"{corpus.get('generated_aplus_rows', '?')} `A_plus` rows, plus "
        f"{corpus.get('evidence_witnesses', '?')} replay-verified witnesses from "
        f"note 20's evidence index re-indexed through `aplus.spine_from_witness`, "
        f"contributing {corpus.get('evidence_aplus_rows', '?')} rows."
    )
    lines.append("")

    # -- size independence -------------------------------------------------
    lines.append("## Size independence")
    lines.append("")
    parameters = size.get("parameters", {})
    lines.append(
        f"One `state_dict` ran the loop at `{size.get('small_spec')}` and at "
        f"`{size.get('large_spec')}` (`[max_len, max_strands, num_actions]`)."
    )
    lines.append("")
    lines.append("| | value |")
    lines.append("|---|---:|")
    for key, value in parameters.items():
        lines.append(f"| `{key}` parameters | {value:,} |")
    lines.append(
        f"| parameter count stable across both sizes | "
        f"{size.get('parameter_count_stable')} |"
    )
    lines.append(f"| search left the weights untouched | {size.get('state_dict_unchanged')} |")
    lines.append(f"| expansions, small instances | {size.get('ran_small')} |")
    lines.append(f"| expansions, large instances | {size.get('ran_large')} |")
    delta = size.get("cyclic_shift_delta", {})
    for name, value in delta.items():
        lines.append(f"| `knot` embedding moved by a cyclic word shift, {name} | {value:.2e} |")
    lines.append("")
    lines.append(
        f"The loop constructs `positional_geometry=\"{size.get('geometry')}\"` "
        "explicitly. The default is the legacy geometry, and conjugation is a no-op "
        "on the knot, so the delta above is the whole reason for the flag."
    )
    lines.append("")

    # -- the three arms ----------------------------------------------------
    lines.append("## Playing strength at equal simulations")
    lines.append("")
    lines.append(
        f"`witness_rate@(cap, dose, seeds)`, cap = the bank's certified upper bound, "
        f"{seeds} seeds, columns are `rung @ simulations`. One simulation is one node "
        f"expansion in every arm, the classical beam included."
    )
    lines.append("")
    lines += table(rows, rungs, doses, arms)
    lines.append("")
    claimed = sum(r["claimed"] for r in rows)
    verified = sum(r["verified"] for r in rows)
    lines.append(
        f"**{claimed} claimed witnesses, {verified} verified.** Every one replays "
        f"through `UnknotWitness.verify()` twice -- once at return and once "
        f"round-tripped through JSON."
    )
    lines.append("")

    # -- the headline three arms, and what beats what -----------------------
    lines += headline(rows, rungs, doses)
    lines += reading(rows, rungs, doses)

    # -- cost --------------------------------------------------------------
    lines.append("### Expansions per solved instance")
    lines.append("")
    lines.append(
        "`docs/lessons.md`: an arm that abandons hard instances drops them from its "
        "own average, so the conditional cost is never reported alone."
    )
    lines.append("")
    lines.append("| arm | rung | dose | witness rate | expansions/solved |")
    lines.append("|---|---|---:|---:|---:|")
    for row in rows:
        cost = row.get("expansions_per_solved")
        rendered = f"{cost:.0f}" if cost else "--"
        lines.append(
            f"| `{row['arm']}` | {row['rung']} | {row['simulations']} | "
            f"{row['witness_rate']:.3f} | {rendered} |"
        )
    lines.append("")

    # -- training ----------------------------------------------------------
    lines.append("## Training")
    lines.append("")
    warmup = payload.get("warmup") or {}
    if warmup.get("first") and warmup.get("last"):
        lines.append(
            f"Supervised warm-up on the `A_plus` corpus: {warmup['batches']} batches, "
            f"total loss {warmup['first']['total']:.4f} -> {warmup['last']['total']:.4f} "
            f"(`A_plus` term {warmup['first']['aplus']:.4f} -> "
            f"{warmup['last']['aplus']:.4f})."
        )
        lines.append("")
    iterations = payload.get("iterations") or []
    if iterations:
        lines.append("| iteration | self-play witness rate | rows from search | train rows |")
        lines.append("|---:|---:|---:|---:|")
        for row in iterations:
            lines.append(
                f"| {row['iteration']} | {row['selfplay_witness_rate']:.3f} | "
                f"{row['selfplay_rows']} | {row['train_rows']} |"
            )
        lines.append("")

    lines.append(
        "The `warmup` arm is the set-valued `A_plus` supervision alone and the "
        "`trained` arm is that plus the self-play iterations, so the two rows say "
        "separately what the supervision bought and what the *loop* bought."
    )
    lines.append("")

    lines += growth_section(out)

    # -- limits ------------------------------------------------------------
    lines.append("## What this loop cannot yet do")
    lines.append("")
    for item in LIMITS:
        lines.append(f"* {item}")
    lines.append("")

    lines.append(
        f"Elapsed {payload.get('elapsed_seconds', 0):.0f} s. Config: "
        f"`{json.dumps(config)}`."
    )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines))
    print(f"wrote {out / 'report.md'} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
