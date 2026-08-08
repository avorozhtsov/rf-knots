r"""Shape bucketing on the network side: batching, the live wrap, and the payoff.

`rf_knots.torus.bucket_shape` rounds an instance up to a `4 x 2` tile instead of
to the global `48 x 5` canvas. That is worth about 4x the trunk throughput
(`padding_overhead.py`), and it needs two things here that the encoder cannot
provide on its own.

## 1. Batches have to be homogeneous

A batch is one tensor. Mixing shapes means either padding back up to the largest
member -- which undoes the saving -- or running one instance at a time. So
instances are grouped by bucket shape and each group forms its own batch. The cost
is that a batch is smaller than it would otherwise be; the measurement below shows
it still wins comfortably.

## 2. The cyclic wrap has to close at the live length, not at the canvas edge

This is the subtle one. Circular padding on the position axis is what makes the
network invariant to conjugation ([18 §2.2](../18-raster-representation.md)). But
bucketing leaves up to three unused rows, so `F.pad(mode="circular")` wraps row 0
onto the last *padding* row rather than onto the last real one. The knot is
unchanged -- identity rows are no-ops -- but the exact rotation equivariance the
padding existed to provide is gone: rotating the word by one is no longer the same
as rotating the array by one.

The fix is the same per-sample gather used for the strand axis in
`torus_probe.py`, applied to positions: wrap at the live length. That restores
exact equivariance and costs one gather per block.

This is the general lesson from both: **a cyclic axis has to wrap at the live
extent, never at the canvas.** Getting it wrong on the strand axis silently made
the torus arm test a different hypothesis than the one intended; getting it wrong
here would silently discard the conjugation invariance while appearing to keep it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rf_knots.torus import TILE, bucket_batches, bucketed_raster  # noqa: E402


def wrap_positions_at(x: Tensor, lengths: Tensor) -> Tensor:
    """Circular pad the position axis at `lengths[b]`, per sample.

    `x` is `(batch, channels, strands, positions)`. Row `-1` of the pad is the
    last *live* position and row `+1` is the first, so a `3x3` kernel at position
    `i` sees `(i-1) mod L`, `i`, `(i+1) mod L` for every `i < L`. Positions at or
    beyond `L` are padding and are masked downstream.
    """
    batch, channels, strands, width = x.shape
    source = torch.cat([x, x.new_zeros(batch, channels, strands, 1)], dim=3)
    columns = torch.arange(width + 2, device=x.device)[None, :]
    live = lengths.clamp(min=1, max=width)[:, None]
    index = torch.where(
        columns == 0,
        live - 1,
        torch.where(
            columns <= live,
            columns - 1,
            torch.where(columns == live + 1, torch.zeros_like(columns),
                        torch.full_like(columns, width)),
        ),
    ).clamp(0, width)
    gather = index[:, None, None, :].expand(batch, channels, strands, width + 2)
    return source.gather(3, gather)


def _canvas_wrap(x: Tensor) -> Tensor:
    """What `F.pad(mode='circular')` does: wrap at the canvas edge."""
    return F.pad(x, (1, 1, 0, 0), mode="circular")


def equivariance_check(width: int = 12, live: int = 9) -> dict:
    """Does the necklace actually close at the live length?

    The first version of this compared the two wraps on the slice `[1:live+1]`,
    which is exactly the interior -- the columns the padding never touches -- so
    both scored a perfect zero and the check discriminated nothing. The wrap is
    only visible *at the pad columns*, so that is where to look:

    * the left pad must be the **last live** position, so position 0's kernel sees
      its true cyclic predecessor;
    * the right pad, at index `live + 1`, must be position 0.

    Under `F.pad(mode="circular")` on a bucketed canvas both are padding cells
    instead, which is the failure this exists to catch.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 3, 2, width)
    x[:, :, :, live:] = 0.0  # the bucket's unused tail
    lengths = torch.tensor([live])

    live_pad = wrap_positions_at(x, lengths)
    canvas_pad = _canvas_wrap(x)
    last, first = x[:, :, :, live - 1], x[:, :, :, 0]

    def gap(padded: Tensor) -> float:
        left = (padded[:, :, :, 0] - last).abs().max()
        right = (padded[:, :, :, live + 1] - first).abs().max()
        return float(torch.maximum(left, right))

    return {
        "live_wrap_error": gap(live_pad),
        "canvas_wrap_error": gap(canvas_pad),
        "note": "error is distance from the true cyclic neighbour at both pad columns",
    }


def throughput(instances, max_len: int, max_strands: int, width: int = 36,
               blocks: int = 4, batch: int = 64, repeats: int = 6) -> dict:
    """Full canvas in one batch, against bucketed groups, on the same instances."""
    from probe_models import Block

    torch.set_num_threads(1)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(4, width, 1, bias=False),
        *[Block(width, wrap_position=True, wrap_strands=False) for _ in range(blocks)],
    )

    def run(shapes_and_counts) -> float:
        started = time.perf_counter()
        for _ in range(repeats):
            for (rows, columns), count in shapes_and_counts:
                for start in range(0, count, batch):
                    size = min(batch, count - start)
                    net(torch.randn(size, 4, columns, rows))
        return (time.perf_counter() - started) / repeats

    full = [((max_len, max_strands), len(instances))]
    groups = bucket_batches(instances, max_len=max_len, max_strands=max_strands)
    bucketed = [(shape, len(indexes)) for shape, indexes in groups.items()]
    with torch.no_grad():
        full_seconds = run(full)
        bucketed_seconds = run(bucketed)
    return {
        "instances": len(instances),
        "distinct_shapes": len(groups),
        "shapes": sorted(f"{r}x{c}:{len(i)}" for (r, c), i in groups.items()),
        "full_seconds": full_seconds,
        "bucketed_seconds": bucketed_seconds,
        "speedup": full_seconds / bucketed_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=600)
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--max-strands", type=int, default=5)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    from padding_overhead import sample_instances

    print(f"tile {TILE}", flush=True)
    sampled = sample_instances(args.instances, 0, args.max_strands, 22)

    print("\nrotation equivariance of the position wrap:", flush=True)
    checks = equivariance_check()
    print(f"  wrap at the live length : max error {checks['live_wrap_error']:.2e}",
          flush=True)
    print(f"  wrap at the canvas edge : max error {checks['canvas_wrap_error']:.2e}",
          flush=True)

    # Rebuild words so the encoder path is exercised, not just the shapes.
    rng = np.random.default_rng(0)
    instances = []
    for length, strands in sampled:
        word = tuple(
            int(rng.choice((-1, 1))) * int(rng.integers(1, max(strands, 2)))
            for _ in range(length)
        )
        instances.append((word, strands))
    encoded = [bucketed_raster(w, s, max_len=args.max_len, max_strands=args.max_strands)
               for w, s in instances if w]
    print(f"\nencoded {len(encoded)} instances; mean cells "
          f"{statistics.mean(e.shape[0] * e.shape[1] for e in encoded):.0f} "
          f"against {args.max_len * args.max_strands} on the full canvas", flush=True)

    print("\nend-to-end trunk throughput on the same instances:", flush=True)
    result = throughput(instances, args.max_len, args.max_strands)
    print(f"  full canvas : {result['full_seconds']:.2f} s", flush=True)
    print(f"  bucketed    : {result['bucketed_seconds']:.2f} s "
          f"over {result['distinct_shapes']} shapes", flush=True)
    print(f"  speedup     : {result['speedup']:.2f}x", flush=True)

    report = {"tile": list(TILE), "equivariance": checks, "throughput": result}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
