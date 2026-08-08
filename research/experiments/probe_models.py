"""The family of encoders under test, and the parameter matching that makes the
comparison mean anything.

Every arm is the same trunk -- an input projection, `residual_blocks` residual
blocks, a masked global pool, a two-layer head -- and differs only in

* **what it reads**: one-hot letters over a `1 x L` strip, or the `k x L` raster;
* **how it treats the edges of the picture**: the position axis is a circle
  (conjugation) or a line; the strand axis is a circle (the affine braid group,
  which is not the group we are in) or a bounded interval;
* **whether the boundary is marked**, since a bounded axis has two ends and one
  of them is where `DESTABILIZE` lives;
* **whether it sees a pyramid** over 2x2 and 4x4 blocks of cells, which is the
  hierarchical combiner in convolutional clothing;
* **whether far-commuting letters are packed into one row** before it looks.

Widths are then solved per arm so that every arm has the same parameter count to
within a few percent. Without that step the comparison silently becomes "which
encoder happened to get more parameters at a fixed channel width" -- the raster
arms read four input channels where the one-hot arm reads fifteen, so a shared
`channels=32` is not a shared budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rf_knots.torus import raster

MAX_STRANDS = 8


# --------------------------------------------------------------------------- #
# encoders: sample -> (C, H, W)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Encoding:
    """How a `(word, strands)` pair becomes an array."""

    kind: str  # "word" | "raster"
    edges: bool = False
    pack: bool = False
    pad_mode: str = "identity"

    @property
    def channels(self) -> int:
        if self.kind == "word":
            # positive and negative one-hot per generator, plus a padding plane.
            # Deliberately *not* the environment's `top_generator` plane: that
            # plane is the answer to the `destab` probe, computed by a human.
            return 2 * (MAX_STRANDS - 1) + 1
        return 4 + (2 if self.edges else 0)

    @property
    def height(self) -> int:
        return 1 if self.kind == "word" else MAX_STRANDS

    def __call__(self, word, strands: int, rows: int) -> np.ndarray:
        if self.kind == "word":
            planes = np.zeros((1, rows, self.channels), dtype=np.float32)
            planes[0, :, 2 * (MAX_STRANDS - 1)] = 1.0
            for position, letter in enumerate(word[:rows]):
                index = abs(letter) - 1 + (0 if letter > 0 else MAX_STRANDS - 1)
                planes[0, position, index] = 1.0
                planes[0, position, 2 * (MAX_STRANDS - 1)] = 0.0
            return planes.transpose(2, 0, 1)
        planes = raster(
            tuple(word),
            strands,
            max_strands=MAX_STRANDS,
            rows=rows,
            pack=self.pack,
            edges=self.edges,
            pad_mode=self.pad_mode,
        )
        return planes.transpose(2, 1, 0)  # (channels, strand, position)


# --------------------------------------------------------------------------- #
# trunk
# --------------------------------------------------------------------------- #

def _groups(width: int) -> int:
    """The largest group count up to 8 that actually divides the width."""
    for groups in range(min(8, width), 0, -1):
        if width % groups == 0:
            return groups
    return 1


class Block(nn.Module):
    """A residual block whose padding says which axes are circles.

    `wrap_position` is the conjugation symmetry and is very likely correct.
    `wrap_strands` glues strand 0 to strand `k-1`, which is the affine braid
    group rather than `B_n`; it is here to be measured, not because it is right.
    """

    def __init__(self, width: int, *, wrap_position: bool, wrap_strands: bool):
        super().__init__()
        self.wrap_position = wrap_position
        self.wrap_strands = wrap_strands
        self.conv1 = nn.Conv2d(width, width, 3, padding=0, bias=False)
        self.norm1 = nn.GroupNorm(_groups(width), width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=0, bias=False)
        self.norm2 = nn.GroupNorm(_groups(width), width)

    def _pad(self, x: Tensor) -> Tensor:
        x = F.pad(x, (1, 1, 0, 0), mode="circular" if self.wrap_position else "constant")
        if x.shape[2] == 1:
            return F.pad(x, (0, 0, 1, 1))
        return F.pad(x, (0, 0, 1, 1), mode="circular" if self.wrap_strands else "constant")

    def forward(self, x: Tensor) -> Tensor:
        hidden = torch.relu(self.norm1(self.conv1(self._pad(x))))
        return torch.relu(x + self.norm2(self.conv2(self._pad(hidden))))


class Pyramid(nn.Module):
    """The user's `2x2` and `4x4` block combiners, written as what they are.

    Pooling to half resolution, convolving, and adding the result back is a
    feature pyramid; two levels of it give a cell at the finest scale a view of
    the `4x4` neighbourhood of blocks around it. Naming the levels "interaction
    blocks" and "combiners" describes the same computation, so this is the cheap
    version of that proposal rather than a different one.
    """

    def __init__(self, width: int, *, wrap_position: bool, wrap_strands: bool):
        super().__init__()
        self.levels = nn.ModuleList(
            Block(width, wrap_position=wrap_position, wrap_strands=wrap_strands)
            for _ in range(2)
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for level in self.levels:
            size = (max(out.shape[2] // 2, 1), max(out.shape[3] // 2, 1))
            coarse = level(F.adaptive_avg_pool2d(out, size))
            out = out + F.interpolate(coarse, size=out.shape[2:], mode="nearest")
        return out


class ProbeNet(nn.Module):
    """Input projection, trunk, masked global pool, head.

    The pool is masked by strand occupancy so that the padding capacity of a
    narrow braid inside a wide array does not dilute the average -- otherwise a
    two-strand word inside an eight-strand canvas is three-quarters silence, and
    the transfer split would measure how much silence each arm was trained on.
    """

    def __init__(
        self,
        encoding: Encoding,
        *,
        width: int,
        blocks: int,
        outputs: int,
        wrap_position: bool,
        wrap_strands: bool,
        pyramid: bool,
    ):
        super().__init__()
        self.encoding = encoding
        self.input = nn.Conv2d(encoding.channels, width, 1, bias=False)
        self.blocks = nn.Sequential(
            *[
                Block(width, wrap_position=wrap_position, wrap_strands=wrap_strands)
                for _ in range(blocks)
            ]
        )
        self.pyramid = (
            Pyramid(width, wrap_position=wrap_position, wrap_strands=wrap_strands)
            if pyramid
            else None
        )
        self.head = nn.Sequential(
            nn.Linear(2 * width, 64), nn.ReLU(), nn.Linear(64, outputs)
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.encoding.kind == "raster":
            mask = x[:, 3:4]  # the occupancy channel
        else:
            # The complement of the padding plane, so both encodings pool over
            # exactly the cells that carry a diagram.
            mask = 1.0 - x[:, 2 * (MAX_STRANDS - 1) : 2 * (MAX_STRANDS - 1) + 1]
        hidden = torch.relu(self.input(x))
        hidden = self.blocks(hidden)
        if self.pyramid is not None:
            hidden = self.pyramid(hidden)
        weight = mask.sum(dim=(2, 3)).clamp(min=1.0)
        mean = (hidden * mask).sum(dim=(2, 3)) / weight
        peak = (hidden + (mask - 1.0) * 1e4).amax(dim=(2, 3))
        return self.head(torch.cat([mean, peak], dim=1))


# --------------------------------------------------------------------------- #
# the arms
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Arm:
    name: str
    rationale: str
    encoding: Encoding
    wrap_position: bool = False
    wrap_strands: bool = False
    pyramid: bool = False
    blocks: int = 4
    # Matched parameters are not matched compute. The raster's canvas is
    # `max_strands` times taller than the one-hot strip, so at equal parameter
    # count it does roughly eight times the arithmetic per step. `scale` exists
    # for the control that spends that arithmetic on the word encoding instead,
    # which is the honest question: is the picture better, or just bigger?
    scale: float = 1.0
    width: int = field(default=0, compare=False)  # solved by `match_parameters`

    def build(self, outputs: int, width: int | None = None) -> ProbeNet:
        return ProbeNet(
            self.encoding,
            width=width or self.width,
            blocks=self.blocks,
            outputs=outputs,
            wrap_position=self.wrap_position,
            wrap_strands=self.wrap_strands,
            pyramid=self.pyramid,
        )


def arms() -> list[Arm]:
    """An ablation ladder, not a grid: each arm adds one claim to the one above."""
    return [
        Arm(
            "word-onehot",
            "what every network in this project reads today",
            Encoding("word"),
        ),
        Arm(
            "word-onehot-cyclic",
            "the same letters, but the word treated as a necklace",
            Encoding("word"),
            wrap_position=True,
        ),
        Arm(
            "word-onehot-cyclic-8x",
            "the control: the same letters, given the raster's compute instead",
            Encoding("word"),
            wrap_position=True,
            scale=float(MAX_STRANDS),
        ),
        Arm(
            "raster-flat",
            "the picture, with no symmetry claimed at all",
            Encoding("raster", pad_mode="zero"),
        ),
        Arm(
            "raster-cyclic",
            "+ position is a circle: conjugation is free",
            Encoding("raster", pad_mode="zero"),
            wrap_position=True,
        ),
        Arm(
            "raster-cyclic-idpad",
            "+ padding rows are identity braids rather than blank",
            Encoding("raster", pad_mode="identity"),
            wrap_position=True,
        ),
        Arm(
            "raster-torus",
            "+ strand 0 glued to strand k-1: the proposal taken literally",
            Encoding("raster", pad_mode="identity"),
            wrap_position=True,
            wrap_strands=True,
        ),
        Arm(
            "raster-cylinder-edge",
            "+ the two strand boundaries marked instead of glued",
            Encoding("raster", edges=True, pad_mode="identity"),
            wrap_position=True,
        ),
        Arm(
            "raster-cylinder-pyramid",
            "+ 2x2 and 4x4 block combiners over the cylinder",
            Encoding("raster", edges=True, pad_mode="identity"),
            wrap_position=True,
            pyramid=True,
        ),
        Arm(
            "raster-cylinder-packed",
            "+ far-commuting letters share a row, quotienting out COMMUTE",
            Encoding("raster", edges=True, pack=True, pad_mode="identity"),
            wrap_position=True,
            pyramid=True,
        ),
    ]


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def match_parameters(arm: Arm, target: int, outputs: int) -> tuple[Arm, int]:
    """Pick the width whose parameter count lands closest to `target`.

    Reported alongside every result: a claim about a representation that is
    really a claim about capacity is not a claim about a representation.
    """
    goal = target * arm.scale
    best, best_gap = 8, None
    for width in range(8, 513, 2):
        total = count_parameters(arm.build(outputs, width=width))
        gap = abs(total - goal)
        if best_gap is None or gap < best_gap:
            best, best_gap = width, gap
        if total > goal * 1.6:
            break
    matched = Arm(
        arm.name, arm.rationale, arm.encoding, arm.wrap_position, arm.wrap_strands,
        arm.pyramid, arm.blocks, arm.scale, best,
    )
    return matched, count_parameters(matched.build(outputs))
