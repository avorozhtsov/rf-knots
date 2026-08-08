"""Bucketing is only safe if the cyclic wrap still closes at the live length.

The compute win is measured in `research/experiments/`; what needs a test is the
invariant that makes it *correct*. Rounding a shape up to a tile leaves unused
positions, and a naive circular pad then wraps position 0 onto a padding cell
instead of onto the true last letter -- silently discarding the conjugation
invariance the cyclic padding exists to provide.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the wrap helper is torch-side")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))

from bucketing import _canvas_wrap, wrap_positions_at  # noqa: E402


def test_the_live_wrap_closes_the_necklace_and_the_canvas_wrap_does_not():
    """Position 0's predecessor must be the last *live* letter, not a padding cell."""
    torch.manual_seed(0)
    width, live = 12, 9
    x = torch.randn(1, 3, 2, width)
    x[:, :, :, live:] = 0.0
    lengths = torch.tensor([live])

    live_pad = wrap_positions_at(x, lengths)
    assert torch.allclose(live_pad[:, :, :, 0], x[:, :, :, live - 1])
    assert torch.allclose(live_pad[:, :, :, live + 1], x[:, :, :, 0])

    canvas_pad = _canvas_wrap(x)
    assert not torch.allclose(canvas_pad[:, :, :, 0], x[:, :, :, live - 1])


@pytest.mark.parametrize("live", [1, 2, 5, 11, 12])
def test_every_live_position_sees_its_cyclic_neighbours(live):
    width = 12
    x = torch.arange(1, width + 1, dtype=torch.float32)[None, None, None, :].clone()
    x[:, :, :, live:] = 0.0
    padded = wrap_positions_at(x, torch.tensor([live]))[0, 0, 0]
    for i in range(live):
        window = padded[i : i + 3].tolist()
        expected = [float((i - 1) % live + 1), float(i + 1), float((i + 1) % live + 1)]
        assert window == expected, f"live={live} position {i}: {window} != {expected}"


def test_the_wrap_is_per_sample_so_a_mixed_batch_is_still_correct():
    """The whole point of bucketing is batches, so the wrap must vary within one."""
    width = 8
    rows = []
    for live in (3, 5, 8):
        row = torch.arange(1, width + 1, dtype=torch.float32)
        row[live:] = 0.0
        rows.append(row)
    x = torch.stack(rows)[:, None, None, :]
    padded = wrap_positions_at(x, torch.tensor([3, 5, 8]))
    for index, live in enumerate((3, 5, 8)):
        assert padded[index, 0, 0, 0].item() == float(live)
        assert padded[index, 0, 0, live + 1].item() == 1.0


def test_the_shape_count_is_bounded_by_the_tile_not_by_the_data():
    """The JIT budget must hold for *any* distribution, not the current one.

    `padding_overhead.py` measured 17 shapes, but that is a property of the
    ladder's present instance mix -- a uniform spread over the same capacity gives
    33. The number that can be promised is the structural ceiling: one shape per
    (row bucket, column bucket) pair, which at `48 x 5` capacity and a `4 x 2` tile
    is 12 x 3 = 36. At ~0.4 s to jit one shape that is under fifteen seconds of
    compilation even in the worst case, which is why the trade-off survives a
    broader instance distribution.
    """
    import numpy as np

    from rf_knots.torus import TILE, bucket_batches

    max_len, max_strands = 48, 5
    ceiling = (-(-max_len // TILE[0])) * (-(-max_strands // TILE[1]))
    assert ceiling == 36

    rng = np.random.default_rng(0)
    instances = []
    for _ in range(600):
        strands = int(rng.integers(2, 6))
        length = int(rng.integers(1, max_len))
        word = tuple(
            int(rng.choice((-1, 1))) * int(rng.integers(1, strands))
            for _ in range(length)
        )
        instances.append((word, strands))
    groups = bucket_batches(instances, max_len=max_len, max_strands=max_strands)
    assert len(groups) <= ceiling, f"{len(groups)} shapes exceeds the ceiling"
    assert sum(len(v) for v in groups.values()) == len(instances)
