r"""A policy head with no strand-count dependence at all.

## The last blocking tensor

[18 §2.1](../18-raster-representation.md) measured which parameters stop a
checkpoint from being loaded at a larger `max_strands`:

| | `s-window-128` | `conv-window-128` (raster) |
|---|---:|---:|
| tensors that change shape, `N = 5 -> 8` | 3 | **2** |

The raster fixes the input convolution -- it reads four channels whatever `N` is --
and the two survivors are both the per-offset policy convolution. Its width is
`per_offset = 3 + 2(N-1) + 1`, because the serial action layout is

```
offset j:  REDUCE, COMMUTE, BRAID, INSERT(g_1,+), INSERT(g_1,-), ...,
           INSERT(g_G,+), INSERT(g_G,-), CROSSING_CHANGE          (G = N-1)
```

and only the `INSERT` block grows. So a single `1x1` convolution emitting
`per_offset` channels has to learn "insert `sigma_3`" as a separate output from
"insert `sigma_4`", at a separate parameter, even though they are the same
operation performed one strand higher.

## The fix

`INSERT(g, s)` inserts a cancelling pair between strands `g-1` and `g`. On the
raster that is a **cell**, and the cell already says which strands. So read the
two insert logits off the strand axis with a shared `2x1` kernel:

```
Conv2d(width, 2, kernel_size=(2, 1))  applied to (width, strand, position)
    -> (2, N-1, window) = (sign, generator, offset)
```

which is exactly the `INSERT` block, in exactly the layout `underlying_action`
decodes, with `2 * 2 * width + 2` parameters **for every `N`**. The remaining four
per-offset channels are position-only, so they come from a strand-pooled `1x1`
convolution: `4 * width + 4` parameters, also for every `N`.

Head parameters go from `(2N + 2) * width + ...` to a flat `8 * width + 6`, and
the count of tensors blocking a checkpoint transfer goes from two to **zero**.

## What this is and is not

The action *space* is unchanged -- same actions, same order, same size -- so the
environment, the search and the replay buffer are untouched. Only the way the
logits are produced changes. The action space itself still has `O(N)` entries;
making *that* independent of `N` needs a per-cell action space, which is an
environment change and a larger piece of work.

This module is standalone rather than a patch to `pgx_mcts_bench.networks`
because that repository has training jobs running against it; integration is one
branch in `make_braid_network`.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _import_bench():
    from pgx_mcts_bench.networks import CylinderResidualBlock

    return CylinderResidualBlock


class StrandSharedSerialNet(nn.Module):
    """`s-window`'s controller with a strand-shared insert head.

    Deliberately not a subclass of `SerialBraidNet`: that class's
    `representation` has already pooled the strand axis away by the time the head
    runs, and the whole point here is to read the head *off* that axis.
    """

    def __init__(self, game, model):
        super().__init__()
        block = _import_bench()
        width = model.channels
        self.max_strands = game.max_strands
        self.window = game.serial_window
        self.act_width = game.serial_act_width

        # Where the raster sits inside the observation, mirroring the layout in
        # `RasterWindowRepresentation`: letters, environment scalars, raster, then
        # any trailing conditioning planes.
        trailing = int(game.serial_internal_horizon > 0) + int(
            game.objective_budget_channel
        )
        self.raster_end = game.observation_channels - trailing
        self.raster_start = self.raster_end - 4 * game.max_strands
        self.letter_channels = 2 * (game.max_strands - 1) + 2
        metadata_channels = self.raster_start - self.letter_channels + trailing

        self.input = nn.Conv2d(4, width, 1, bias=False)
        self.blocks = nn.Sequential(
            *[block(width) for _ in range(model.residual_blocks)]
        )
        # (sign, generator, offset) from a kernel that spans one adjacent strand
        # pair. Shared across generators: this is the tensor that used to grow.
        self.insert = nn.Conv2d(width, 2, kernel_size=(2, 1))
        # REDUCE, COMMUTE, BRAID, CROSSING_CHANGE: position-only, so strand-pooled.
        self.positional = nn.Conv1d(width, 4, 1)
        self.metadata = nn.Conv1d(metadata_channels, width, 1)
        self.summary = nn.Sequential(nn.Linear(3 * width, 64), nn.ReLU())
        self.n_global = game.action_size - self.act_width * (
            3 + 2 * (game.max_strands - 1) + 1
        )
        self.global_policy = nn.Linear(64, self.n_global)
        self.value = nn.Sequential(nn.Linear(64, 1), nn.Tanh())
        self.act_start = (game.serial_window - self.act_width) // 2

    def _raster(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        batch, _, _, columns = observation.shape
        raster = observation[:, self.raster_start : self.raster_end, 0, :]
        raster = raster.reshape(batch, self.max_strands, 4, columns).permute(0, 2, 1, 3)
        return raster, raster[:, 3:4]

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        raster, active = self._raster(observation)
        hidden = self.blocks(torch.relu(self.input(raster)))

        metadata = torch.cat(
            [
                observation[:, self.letter_channels : self.raster_start, 0, :],
                observation[:, self.raster_end :, 0, :],
            ],
            dim=1,
        )
        pooled = (hidden * active).sum(dim=2) / active.sum(dim=2).clamp(min=1.0)
        pooled = pooled + self.metadata(metadata)

        inserts = self.insert(hidden)  # (B, 2, N-1, window)
        others = self.positional(pooled)  # (B, 4, window)

        window = slice(self.act_start, self.act_start + self.act_width)
        insert_block = inserts[:, :, :, window]  # (B, 2, G, W)
        other_block = others[:, :, window]  # (B, 4, W)

        # Assemble the per-offset layout exactly as `underlying_action` decodes it:
        # REDUCE, COMMUTE, BRAID, then INSERT(g, sign) generator-major, then
        # CROSSING_CHANGE. Getting this order wrong trains perfectly happily and
        # means nothing, which is why `test_policy_head_blocks_align_with_the
        # _action_space` exists in the parent repository.
        batch = observation.shape[0]
        per_offset = []
        per_offset.append(other_block[:, 0:3, :])                      # 3 x W
        per_offset.append(
            insert_block.permute(0, 2, 1, 3).reshape(batch, -1, self.act_width)
        )                                                              # 2G x W
        per_offset.append(other_block[:, 3:4, :])                      # 1 x W
        stacked = torch.cat(per_offset, dim=1)                         # (B, P, W)
        positional = stacked.permute(0, 2, 1).flatten(1)               # offset-major

        peak = (hidden + (active - 1.0) * 1e4).amax(dim=(2, 3))
        features = self.summary(
            torch.cat([pooled.mean(dim=2), pooled.amax(dim=2), peak], dim=1)
        )
        logits = torch.cat([positional, self.global_policy(features)], dim=1)
        return logits, self.value(features).squeeze(-1)


def blocking_tensors(build, narrow: int, wide: int) -> list[str]:
    """Names of parameters whose shape changes between two strand capacities."""
    a, b = build(narrow), build(wide)
    sa = {k: tuple(v.shape) for k, v in a.state_dict().items()}
    sb = {k: tuple(v.shape) for k, v in b.state_dict().items()}
    return sorted(k for k in sa if k in sb and sa[k] != sb[k])


def main() -> None:
    from pgx_mcts_bench.ladder import _config, candidates
    from pgx_mcts_bench.networks import make_braid_network

    by = {c.name: c for c in candidates()}
    stage = ("P(4,5)#0", 0)

    def existing(name):
        def build(strands):
            config = _config(by[name], stage, seed=0, device="cpu")
            return make_braid_network(replace(config.game, max_strands=strands),
                                      config.model)
        return build

    def shared(strands):
        config = _config(by["conv-window-128"], stage, seed=0, device="cpu")
        return StrandSharedSerialNet(replace(config.game, max_strands=strands),
                                     config.model)

    print(f"{'network':32s} {'params N=5':>11s} {'params N=8':>11s} {'blocking':>9s}  which")
    for label, build in (("s-window-128", existing("s-window-128")),
                         ("conv-window-128", existing("conv-window-128")),
                         ("strand-shared (this file)", shared)):
        blocking = blocking_tensors(build, 5, 8)
        pa = sum(p.numel() for p in build(5).parameters())
        pb = sum(p.numel() for p in build(8).parameters())
        names = ", ".join(b.split(".")[0] for b in blocking) or "-- none --"
        print(f"{label:32s} {pa:11,d} {pb:11,d} {len(blocking):9d}  {names}")

    # The property that matters: a narrow checkpoint loads into a wide network.
    narrow, wide = shared(5), shared(8)
    missing = wide.load_state_dict(narrow.state_dict(), strict=True)
    print(f"\nload_state_dict(N=5 weights into N=8 network, strict=True): OK {missing}")


if __name__ == "__main__":
    main()
