r"""How much of the raster canvas is padding, and what would removing it buy?

The raster is a fixed `max_len x max_strands` canvas, but a real instance fills
only `length x strands` of it. Everything outside is inactive: it costs FLOPs in
the trunk, it dilutes any pooling that is not masked, and it is the reason the
raster costs roughly `max_strands` times the one-hot strip
([18 §4.5](../18-raster-representation.md)).

The proposal is to drop the padding and let actions grow the tensor, keeping
padding only to round the shape up to a multiple of the tile size so a jitted
environment sees a handful of shapes instead of one per length. This measures the
three numbers that decide whether that is worth building:

1. **Occupancy** -- what fraction of the canvas real ladder instances actually
   use. This is the size of the prize.
2. **Bucket count** -- how many distinct shapes survive rounding to a tile
   multiple. This is the cost, in JIT compilations.
3. **Wall clock** -- what the trunk actually costs at full, bucketed and exact
   shapes, at matched parameters. This is the prize in seconds rather than cells.

It deliberately does *not* measure whether dense input improves accuracy; that
needs the probe suite rerun with cropped rather than masked input, and is a
separate and larger experiment.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


def sample_instances(count: int, seed: int, max_strands: int, max_crossings: int):
    """`(length, strands)` for instances the ladder's own generator emits."""
    from rf_knots.config import BraidConfig
    from rf_knots.generator import GradedGenerator

    config = BraidConfig(max_len=48, max_strands=max_strands,
                         scramble_budget=12, simplify_budget=48)
    generator = GradedGenerator(config, max_crossings=max_crossings)
    rng = np.random.default_rng(seed)
    sources = generator.sources
    out = []
    for _ in range(count):
        source = sources[int(rng.integers(0, len(sources)))]
        moves = int(rng.integers(0, 6))
        instance = generator.generate(source, moves, rng)
        letters = [x for x in instance.word if x]
        out.append((len(letters), instance.strands))
    return out


def occupancy(instances, max_len: int, max_strands: int) -> dict:
    canvas = max_len * max_strands
    fills = [(length * strands) / canvas for length, strands in instances]
    rows = [length / max_len for length, _ in instances]
    columns = [strands / max_strands for _, strands in instances]
    return {
        "canvas_cells": canvas,
        "mean_occupancy": statistics.mean(fills),
        "median_occupancy": statistics.median(fills),
        "mean_row_fill": statistics.mean(rows),
        "mean_column_fill": statistics.mean(columns),
        "mean_wasted_fraction": 1 - statistics.mean(fills),
    }


def buckets(instances, tile_rows: int, tile_columns: int, max_len: int,
            max_strands: int) -> dict:
    """Distinct shapes after rounding up to a tile multiple, and the waste left."""
    def up(value, multiple):
        return ((value + multiple - 1) // multiple) * multiple

    shapes, wasted = set(), []
    for length, strands in instances:
        rows = min(up(max(length, 1), tile_rows), up(max_len, tile_rows))
        columns = min(up(max(strands, 1), tile_columns), up(max_strands, tile_columns))
        shapes.add((rows, columns))
        wasted.append(1 - (length * strands) / (rows * columns))
    return {
        "tile": [tile_rows, tile_columns],
        "distinct_shapes": len(shapes),
        "shapes": sorted(shapes),
        "mean_wasted_fraction": statistics.mean(wasted),
        "per_length_shapes": len({length for length, _ in instances}),
    }


def trunk_cost(shapes, width: int = 36, blocks: int = 4, batch: int = 64,
               repeats: int = 20) -> list[dict]:
    """Forward+backward wall clock for the cylinder trunk at each shape."""
    import torch
    from probe_models import Block

    torch.set_num_threads(1)
    rows = []
    for label, (height, columns) in shapes:
        net = torch.nn.Sequential(
            torch.nn.Conv2d(4, width, 1, bias=False),
            *[Block(width, wrap_position=True, wrap_strands=False)
              for _ in range(blocks)],
        )
        x = torch.randn(batch, 4, height, columns)
        for _ in range(3):
            net(x).sum().backward()
        started = time.perf_counter()
        for _ in range(repeats):
            net.zero_grad()
            net(x).sum().backward()
        elapsed = (time.perf_counter() - started) / repeats
        rows.append({"label": label, "strands": height, "positions": columns,
                     "cells": height * columns, "ms_per_step": elapsed * 1000})
    return rows


def jit_cost(max_lens, max_strands: int) -> list[dict]:
    """What one extra jitted environment shape costs, which is the bucketing bill."""
    import jax

    from rf_knots.config import BraidConfig
    from rf_knots.env import BraidUnknot

    rows = []
    for max_len in max_lens:
        config = BraidConfig(max_len=max_len, max_strands=max_strands,
                             scramble_budget=12, simplify_budget=48)
        env = BraidUnknot(config)
        step = jax.jit(env.step)
        state = env.init(jax.random.PRNGKey(0))
        legal = np.asarray(state.legal_action_mask)
        action = int(np.flatnonzero(legal)[0])
        started = time.perf_counter()
        out = step(state, np.int32(action))
        out.terminated.block_until_ready()
        compile_seconds = time.perf_counter() - started
        started = time.perf_counter()
        for _ in range(50):
            out = step(state, np.int32(action))
        out.terminated.block_until_ready()
        rows.append({"max_len": max_len, "compile_seconds": compile_seconds,
                     "steady_ms": (time.perf_counter() - started) / 50 * 1000})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=600)
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--max-strands", type=int, default=5)
    parser.add_argument("--max-crossings", type=int, default=22)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    print("sampling instances from the graded generator...", flush=True)
    instances = sample_instances(args.instances, 0, args.max_strands, args.max_crossings)
    report["instances"] = len(instances)
    report["occupancy"] = occupancy(instances, args.max_len, args.max_strands)
    print(json.dumps(report["occupancy"], indent=2), flush=True)

    report["buckets"] = [
        buckets(instances, r, c, args.max_len, args.max_strands)
        for r, c in ((7, 5), (8, 4), (16, 4), (4, 2))
    ]
    for row in report["buckets"]:
        print(f"tile {row['tile']}: {row['distinct_shapes']} shapes "
              f"(vs {row['per_length_shapes']} per-length), "
              f"waste {row['mean_wasted_fraction']:.1%}", flush=True)

    median_length = int(statistics.median(length for length, _ in instances))
    median_strands = int(statistics.median(strands for _, strands in instances))
    shapes = [
        ("full canvas", (args.max_strands, args.max_len)),
        ("bucketed 7x5", (((median_strands + 4) // 5) * 5,
                          ((median_length + 6) // 7) * 7)),
        ("exact", (median_strands, median_length)),
    ]
    print("\ntrunk wall clock (batch 64, width 36, 4 blocks, 1 thread):", flush=True)
    report["trunk"] = trunk_cost(shapes)
    base = report["trunk"][0]["ms_per_step"]
    for row in report["trunk"]:
        print(f"  {row['label']:14s} {row['strands']}x{row['positions']:3d} "
              f"{row['cells']:4d} cells  {row['ms_per_step']:7.1f} ms  "
              f"{base / row['ms_per_step']:4.2f}x", flush=True)

    print("\njit cost per environment shape:", flush=True)
    report["jit"] = jit_cost([16, 24, 32, 48], args.max_strands)
    for row in report["jit"]:
        print(f"  max_len={row['max_len']:3d}  compile {row['compile_seconds']:5.2f}s  "
              f"steady {row['steady_ms']:.3f} ms/step", flush=True)

    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
