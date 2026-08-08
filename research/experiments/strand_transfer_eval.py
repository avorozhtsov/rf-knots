r"""Zero-shot evaluation on a strand count the checkpoint never trained on.

## Why the gate alone cannot answer the question

`strand_architecture_gate` trains *and* evaluates inside the same strand range.
The probes in [18 §6](../18-raster-representation.md) already found both encoders
at 1.000 there, so the gate's honest prediction is parity. The raster's measured
advantage is **transfer**: 0.284 -> 0.981 recall on braids wider than anything it
saw.

This script tests that end to end. It takes a checkpoint the gate trained on
stages up to `P(4,5)#0` — at most four strands — and evaluates it, with no further
training, on a source with **five**, then reports the two side by side.

## Why five and not eight

The ladder config runs at `max_strands = 5`. Going wider is not a data change but
an architecture change: rebuilding at `max_strands = 8` moves the action space from
98 to 140 and changes the shape of

* `positional.weight` / `positional.bias` — the per-offset policy head, whose width
  is `3 + 2(N-1) + 1`, in **both** architectures; and
* `representation.net.0.weight` — the input convolution, in the **one-hot**
  architecture only, because the raster trunk reads four channels whatever `N` is.

So a checkpoint simply cannot be loaded past five strands today. That is not a
limitation of this script; it is the strand dependence the note is about, and it
is why [18 §4.1](../18-raster-representation.md) argues the cell-indexed action
space is the other half of the fix rather than a refinement. Four-to-five is the
whole headroom the current environment has, and it is what is measured here.

## Usage

    cd ../../pgx-mcts-bench
    PYTHONPATH=<this dir> .venv/bin/python <this file> \
        --gate artifacts/strand-gate-raster-vs-window-20260808 \
        --output artifacts/strand-transfer-20260808
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

TRAINED_ON = ("P(4,5)#0", 0)  # four strands: the widest stage the gate reaches
HELD_OUT = ("P(5,6)#0", 0)  # five strands: never seen, still inside capacity


def _latest_checkpoint(run_dir: Path) -> Path | None:
    """The newest `stageNN-after.pt`, falling back to `stageNN-before.pt`.

    `-after` is a stage the run finished; `-before` is one it started. Preferring
    the former means a partially complete gate is still readable, and the stage
    index is reported so a half-trained checkpoint is never quoted as a full one.
    """
    for pattern in ("stage*-after.pt", "stage*-before.pt"):
        found = sorted(run_dir.glob(pattern))
        if found:
            return found[-1]
    return None


def evaluate(checkpoint: Path, candidate_name: str, stage, games: int, seed: int) -> dict:
    # `make_game` is the ladder's own factory: it returns the moving-window
    # adapter for serial candidates and the parallel one otherwise. Building
    # `BraidUnknotGame` directly gives a serial candidate the wrong
    # observation shape, which fails loudly rather than silently -- but only
    # because the raster widens the channel count.
    from pgx_mcts_bench.game import make_game
    from pgx_mcts_bench.ladder import _config, candidates, evaluate_stage
    from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network

    candidate = {c.name: c for c in candidates()}[candidate_name]
    config = _config(candidate, stage, seed=seed, device="cpu")
    game = make_game(config.game)
    network = make_braid_network(config.game, config.model)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_policy_value_state_dict(network, saved["network"])
    network.eval()
    started = time.perf_counter()
    per_ratio = evaluate_stage(game, network, config, games=games, seed=seed)
    # The ladder reports per A:B ratio; the headline here is whether the instance
    # was solved at all, so the ratios are averaged rather than picked from.
    rates = [row["solved"] for row in per_ratio.values()]
    return {
        "checkpoint": str(checkpoint),
        "trained_through_stage": int(re.search(r"stage(\d+)", checkpoint.name).group(1)),
        "phase": "after" if "after" in checkpoint.name else "before",
        "source": stage[0],
        "scramble": stage[1],
        "solve_rate": float(np.mean(rates)),
        "per_ratio": {str(k): v["solved"] for k, v in per_ratio.items()},
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=12)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "run.log"

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        with log_path.open("a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    rows: list[dict] = []
    for candidate_dir in sorted(p for p in args.gate.iterdir() if p.is_dir()):
        for seed_dir in sorted(candidate_dir.glob("seed-*")):
            seed = int(seed_dir.name.split("-")[1])
            run_dir = seed_dir / candidate_dir.name
            checkpoint = _latest_checkpoint(run_dir)
            if checkpoint is None:
                log(f"  {candidate_dir.name} seed {seed}: no checkpoint yet")
                continue
            for label, stage in (("trained", TRAINED_ON), ("held-out", HELD_OUT)):
                row = evaluate(checkpoint, candidate_dir.name, stage, args.games, seed)
                row |= {"candidate": candidate_dir.name, "seed": seed, "split": label}
                rows.append(row)
                log(f"  {candidate_dir.name:18s} seed {seed} {label:8s} "
                    f"{stage[0]:10s} solve={row['solve_rate']:.3f} "
                    f"(from stage{row['trained_through_stage']:02d}-{row['phase']}, "
                    f"{row['seconds']:.0f}s)")
                (args.output / "rows.json").write_text(json.dumps(rows, indent=2))
    (args.output / "report.md").write_text(report(rows))
    log("done")


def report(rows: list[dict]) -> str:
    lines = [
        "# Zero-shot transfer to an unseen strand count",
        "",
        f"Trained through `{TRAINED_ON[0]}` (four strands), evaluated with no",
        f"further training on `{HELD_OUT[0]}` (five strands). Solve rate averaged",
        "over the three A:B ratios. Five is the whole headroom the environment has",
        "at `max_strands = 5`; see the module docstring for why eight is not",
        "reachable without changing the policy head.",
        "",
        "| candidate | seed | trained-on | held-out | drop |",
        "|---|---:|---:|---:|---:|",
    ]
    by = {}
    for row in rows:
        by.setdefault((row["candidate"], row["seed"]), {})[row["split"]] = row["solve_rate"]
    for (candidate, seed), got in sorted(by.items()):
        trained, held = got.get("trained"), got.get("held-out")
        if trained is None or held is None:
            continue
        lines.append(
            f"| `{candidate}` | {seed} | {trained:.3f} | {held:.3f} | {held - trained:+.3f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
