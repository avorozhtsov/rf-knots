import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def tiles(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("pretrained_raster_tiles")


def test_positive_negative_crossing_planes_and_short_word_wrap(tiles):
    raster, origins = tiles.compile_tiles([1, -1], 2, [0, 1])
    assert raster.shape == (2, 4, 12, 7) and origins.tolist() == [0, 0]
    assert raster[0, :3, 0, 3].tolist() == [0, 1, 1]
    assert raster[0, :3, 1, 3].tolist() == [1, 0, 0]
    assert raster[1, :3, 0, 3].tolist() == [0, 0, 1]
    assert raster[1, :3, 1, 3].tolist() == [1, 1, 0]
    assert not raster[:, :, 2:].any()
    assert torch.equal(raster[:, :, :, 1], raster[:, :, :, 3])


def test_large_strand_index_uses_bounded_crop_and_keeps_central_pair(tiles):
    word, n = [1, 500000, -999999], 1000000
    raster, origins = tiles.compile_tiles(word, n, [0, 1, 2])
    assert raster.numel() == 3 * 4 * 12 * 7
    assert origins.tolist() == [0, 499994, 999988]
    assert raster[:, 3].eq(1).all()
    for i, g in enumerate(word):
        upper = abs(g) - 1 - origins[i].item()
        assert 0 <= upper < 11
        assert raster[i, 2, upper, 3] == 1
        assert raster[i, 0, upper + 1, 3] == 1


def test_head_context_is_covariant_and_does_not_require_backbone(tiles):
    word = torch.tensor([1, 2, -3, 4, -5, 1, 3])
    a = tiles.head_context(word, 6, 2)
    b = tiles.head_context(torch.roll(word, -3), 6, (2 - 3) % 7)
    torch.testing.assert_close(b, torch.roll(a, -3, dims=0))
    shifted = tiles.head_context(word, 6, 3)
    assert not torch.equal(a[:, :8], shifted[:, :8])
    assert torch.equal(a[:, 8:], shifted[:, 8:])
    assert a.shape == (7, 16)


def test_invalid_words_and_empty_cases(tiles):
    for word, n in [([0], 3), ([3], 3), ([1], 1), ([1.5], 3)]:
        with pytest.raises(ValueError):
            tiles.compile_tiles(word, n, [0])
    with pytest.raises(ValueError):
        tiles.compile_tiles([1], 2, [1])
    assert tiles.compile_tiles([], 1, [])[0].shape == (0, 4, 12, 7)
    assert tiles.head_context([], 1, 0).shape == (0, 16)


def test_shared_encoder_validates_word_once_and_preserves_gradients(tiles, monkeypatch):
    class Block(torch.nn.Module):
        wrap_strands = False

        def forward(self, x, active):
            return x * active

    representation = SimpleNamespace(
        variant="axial",
        recurrent_steps=0,
        max_strands=12,
        input=torch.nn.Conv2d(4, 4, 1, bias=False),
        blocks=torch.nn.ModuleList([Block()]),
    )
    net = tiles.PretrainedRasterTiles(representation, tile_batch=2)
    original = tiles.checked_word
    calls = []

    def counted(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(tiles, "checked_word", counted)
    result = net([1, 2, -1, -2, 1, 2], 3)
    assert result.shape == (6, 8) and calls == [6]
    assert net.largest_tile_batch == 2
    result.sum().backward()
    assert torch.isfinite(net.input.weight.grad).all()
    assert representation.input.weight.grad is None
    assert torch.equal(net.input.weight, representation.input.weight)
    assert net([], 1).shape == (0, 8)
