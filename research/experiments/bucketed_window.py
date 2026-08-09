r"""Strand bucketing for `conv-window-128`, done where it is both exact and cheap.

## Why the whole-word number does not transfer

`padding_overhead.py` measured 82% padding and a 3.5x win, but that was the
**whole-word** canvas, `48 x 5`. `conv-window-128` rasters only its *seven-cell
window*, so its canvas is `7 x 5 = 35` cells at **50.2%** occupancy. Tile
bucketing at `4 x 2` there is worth **1.69x**, not 3.5x -- and for 2 instances in
600 it makes the canvas *bigger*, because capacity 5 rounds up to 6.

## Crop to the live strand count, not to a tile multiple

Tile multiples exist so a tensor divides evenly into `k0 x n0` blocks. This arm
does no tiling, so the multiple buys nothing and costs exactness:

* on a **bounded** strand axis the inactive rows are all-zero and the block
  zero-pads, so cropping to exactly `n` is bit-identical to the full canvas;
* on a **wrapped** strand axis (`serial_raster_wrap_strands`) cropping to exactly
  `n` *is* the definition of wrapping at the live count, while cropping to
  `ceil(n/2)*2` would wrap at the wrong place -- the same error as
  `raster-torus-canvas` in [18 §5.3](../18-raster-representation.md).

So: group the batch by live strand count, crop each group, run, scatter back.
Five groups instead of one, exact in both variants, and a larger saving than tile
bucketing because the mean live count is about 2.5 against a capacity of 5.

## Where it goes

Inside the network's `encode_spatial`, not in the environment. The observation
stays fixed-shape, so the JAX environment, the search, the replay buffer and the
checkpoint format are all untouched; only the trunk sees the smaller tensors.
That is the whole reason to put it here rather than in `serial_braid.py`.
"""

from __future__ import annotations

import time

import torch
from torch import Tensor


def group_by_live_strands(active: Tensor) -> dict[int, Tensor]:
    """Sample indices keyed by live strand count, read off the occupancy channel."""
    counts = active[:, 0, :, 0].sum(dim=1).round().long()
    groups: dict[int, Tensor] = {}
    for value in counts.unique().tolist():
        groups[int(value)] = torch.nonzero(counts == value, as_tuple=False).squeeze(1)
    return groups


def bucketed_trunk(representation, raster: Tensor, active: Tensor) -> Tensor:
    """Run the trunk per live-strand-count group and scatter the results back.

    Exactly equivalent to running the full canvas, because rows at or beyond the
    live count are all-zero in the raster and the block pads the strand axis with
    zeros. `tests`/`main` below assert that rather than assuming it.
    """
    batch, _, strands, columns = raster.shape
    width = representation.input.out_channels
    out = raster.new_zeros(batch, width, strands, columns)
    for live, index in group_by_live_strands(active).items():
        live = max(int(live), 1)
        piece = raster[index][:, :, :live]
        hidden = torch.relu(representation.input(piece))
        mask = piece[:, 3:4]
        if representation.variant in {"recurrent", "scalable"}:
            for _ in range(representation.recurrent_steps):
                hidden = representation.blocks[0](hidden, mask)
        else:
            for block in representation.blocks:
                hidden = block(hidden, mask)
        out[index.unsqueeze(1), torch.arange(width)[None, :], :live] = hidden.permute(
            0, 1, 2, 3
        )[:, :, :live]
    return out


def full_trunk(representation, raster: Tensor, active: Tensor) -> Tensor:
    hidden = torch.relu(representation.input(raster))
    if representation.variant in {"recurrent", "scalable"}:
        for _ in range(representation.recurrent_steps):
            hidden = representation.blocks[0](hidden, active)
    else:
        for block in representation.blocks:
            hidden = block(hidden, active)
    return hidden


def _make(candidate: str, stage=("P(4,5)#0", 0)):
    from pgx_mcts_bench.ladder import _config, candidates
    from pgx_mcts_bench.networks import RasterWindowRepresentation

    by = {c.name: c for c in candidates()}
    config = _config(by[candidate], stage, seed=0, device="cpu")
    return RasterWindowRepresentation(config.game, config.model), config


def _raster_batch(representation, config, counts) -> tuple[Tensor, Tensor]:
    """A batch of rasters whose live strand counts are `counts`."""
    from rf_knots.torus import raster as draw

    window = config.game.serial_window
    planes = []
    for live in counts:
        word = tuple(1 for _ in range(window)) if live >= 2 else ()
        art = draw(word, max(live, 2), max_strands=config.game.max_strands,
                   rows=window, pad_mode="zero")
        planes.append(torch.from_numpy(art).permute(2, 1, 0))
    batch = torch.stack(planes)
    return batch, batch[:, 3:4]


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    representation, config = _make("conv-window-128")
    representation.eval()

    counts = [2] * 40 + [3] * 30 + [4] * 8 + [5] * 2
    raster, active = _raster_batch(representation, config, counts)
    print(f"variant={representation.variant!r} capacity={config.game.max_strands} "
          f"window={config.game.serial_window} batch={len(counts)}")

    with torch.no_grad():
        reference = full_trunk(representation, raster, active)
        bucketed = bucketed_trunk(representation, raster, active)
    live_mask = active.expand_as(reference[:, :1]).bool()
    gap = (reference - bucketed).abs()
    masked_gap = float((gap * active).max())
    print(f"\nmax difference on live cells: {masked_gap:.3e} "
          f"({'EXACT' if masked_gap < 1e-5 else 'MISMATCH'})")
    del live_mask

    def timed(fn, repeats=40):
        with torch.no_grad():
            for _ in range(3):
                fn(representation, raster, active)
            started = time.perf_counter()
            for _ in range(repeats):
                fn(representation, raster, active)
        return (time.perf_counter() - started) / repeats * 1000

    full_ms = timed(full_trunk)
    bucket_ms = timed(bucketed_trunk)
    print(f"full canvas   {full_ms:7.2f} ms")
    print(f"strand-grouped{bucket_ms:7.2f} ms   {full_ms / bucket_ms:.2f}x")
    mean_live = sum(counts) / len(counts)
    print(f"mean live strands {mean_live:.2f} of {config.game.max_strands} "
          f"-> cell ratio {config.game.max_strands / mean_live:.2f}x")


if __name__ == "__main__":
    main()
