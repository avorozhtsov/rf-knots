import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def model(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = importlib.import_module("pretrained_raster_graph")
    tile_module = importlib.import_module("pretrained_raster_tiles")

    class Block(torch.nn.Module):
        wrap_strands = False

        def forward(self, state, active):
            return state * active

    torch.manual_seed(20261044)
    teacher = SimpleNamespace(
        variant="axial",
        recurrent_steps=0,
        max_strands=12,
        input=torch.nn.Conv2d(4, 8, 1, bias=False),
        blocks=torch.nn.ModuleList([Block()]),
    )
    return module.RasterGraphPolicy(tile_module.PretrainedRasterTiles(teacher), width=16).eval()


def test_padding_and_empty_word_do_not_change_valid_global_or_local_features(model):
    word = torch.tensor([[1, -2, 1, 2, 1]])
    with torch.no_grad():
        alone = model.encoder.encode_sites(word, torch.tensor([5]), torch.tensor([3]))
        batch = torch.full((2, 17), 999999, dtype=torch.long)
        batch[0, :5] = word[0]
        mixed = model.encoder.encode_sites(batch, torch.tensor([5, 0]), torch.tensor([3, 1]))
    for name in ("knot", "layout", "isotopy"):
        torch.testing.assert_close(alone[name][0], mixed[name][0], rtol=1e-5, atol=1e-6)
        assert torch.isfinite(mixed[name]).all()
    torch.testing.assert_close(alone["tokens"][0], mixed["tokens"][0, :5], rtol=1e-5, atol=1e-6)
    assert not mixed["tokens"][1].any()
    assert not mixed["word"][1].any()


def test_head_rescoring_does_not_reencode_and_owns_input_geometry(model):
    word = torch.tensor([[1, 1, 1]])
    encoded = model.encoder.encode_sites(word, torch.tensor([3]), torch.tensor([2]))
    word[0, 0] = 99999
    assert encoded["word"][0, 0] == 1
    sites = torch.tensor([[[8, 0, 1, 1], [8, 1, 1, 1], [8, 2, 1, 1]]])
    calls = []
    handle = model.encoder.tiles.register_forward_hook(lambda *args: calls.append(1))
    globals_before = {k: encoded[k].clone() for k in ("knot", "layout", "isotopy")}
    mask, budgets = torch.ones(1, 3, dtype=torch.bool), torch.tensor([[32.0, 2.0, 8.0, 4.0, 10.0]])
    a, valid = model.score_encoded(encoded, sites, mask, budgets, heads=torch.tensor([0]))
    b, _ = model.score_encoded(encoded, sites, mask, budgets, heads=torch.tensor([1]))
    handle.remove()
    assert not calls and valid.all() and not torch.equal(a, b)
    for k, before in globals_before.items():
        assert torch.equal(encoded[k], before)


def test_active_invalid_geometry_is_rejected(model):
    with pytest.raises(ValueError):
        model.encoder(torch.tensor([[3]]), torch.tensor([1]), torch.tensor([3]))
    with pytest.raises(ValueError):
        model.encoder(torch.tensor([[1.5]]), torch.tensor([1]), torch.tensor([3]))
