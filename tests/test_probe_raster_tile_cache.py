import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def cache_parts(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    monkeypatch.syspath_prepend(str(root.parent / "pgx-mcts-bench/src"))
    native = pytest.importorskip("pgx_mcts_bench.networks")
    module = importlib.import_module("probe_raster_tile_cache")
    with torch.random.fork_rng():
        torch.manual_seed(20261055)
        teacher = SimpleNamespace(
            variant="axial",
            recurrent_steps=0,
            max_strands=12,
            input=torch.nn.Conv2d(4, 8, 1, bias=False),
            blocks=torch.nn.ModuleList([native.MaskedAxialCylinderResidualBlock(8)]),
        )
        net = module.PretrainedRasterTiles(teacher, tile_batch=4).eval()
    return module, net


def test_parity_repeat_cc_and_cyclic_reuse(cache_parts):
    module, net = cache_parts
    probe = module.TileCacheProbe(net)
    word = [1, 2, -1, 3, 2, -2, 1, -3, 2]
    variants = [word, word, [-word[0], *word[1:]], word[2:] + word[:2], [], [1]]
    with torch.no_grad():
        signature = probe.signature()
        for index, row in enumerate(variants):
            before = probe.computed_tiles
            actual = probe(row, 4)
            torch.testing.assert_close(actual, net(row, 4), atol=1e-5, rtol=1e-5)
            if index == 1:
                assert probe.computed_tiles == before
        assert probe.signature() == signature
    assert probe.invalidations == 0
    assert probe.hits + probe.computed_tiles == probe.logical_tiles


@pytest.mark.parametrize("change", ["inplace", "data", "replace", "load", "norm"])
def test_weight_and_configuration_changes_invalidate(cache_parts, change):
    module, net = cache_parts
    probe = module.TileCacheProbe(net)
    word = [1, 2, -1, 2]
    with torch.no_grad():
        probe(word, 3)
        previous = probe.computed_tiles
        if change == "inplace":
            net.input.weight.add_(0.25)
        elif change == "data":
            net.input.weight.data.add_(0.25)
        elif change == "replace":
            net.input.weight = torch.nn.Parameter(net.input.weight.detach().clone() + 0.25)
        elif change == "load":
            state = {k: v.clone() for k, v in net.state_dict().items()}
            state["input.weight"].add_(0.25)
            net.load_state_dict(state)
        else:
            net.blocks[0].norm.eps = 0.1
        actual = probe(word, 3)
        assert probe.invalidations == 1 and probe.computed_tiles > previous
        torch.testing.assert_close(actual, net(word, 3), atol=1e-5, rtol=1e-5)


def test_capacity_eviction_and_returned_storage_are_safe(cache_parts):
    module, net = cache_parts
    probe = module.TileCacheProbe(net, capacity=1)
    word = [1, 2, -1, 3, 2, -2, 1, -3, 2]
    with torch.no_grad():
        actual = probe(word, 4)
        torch.testing.assert_close(actual, net(word, 4), atol=1e-5, rtol=1e-5)
        assert len(probe.entries) <= 1
        actual.fill_(999)
        torch.testing.assert_close(probe(word, 4), net(word, 4), atol=1e-5, rtol=1e-5)


def test_training_and_nonfloat32_are_rejected(cache_parts):
    module, net = cache_parts
    probe = module.TileCacheProbe(net)
    with pytest.raises(ValueError, match="disabled gradients"):
        probe([1], 2)
    with torch.no_grad():
        net.train()
        with pytest.raises(ValueError, match="eval mode"):
            probe([1], 2)
        net.eval().double()
        with pytest.raises(ValueError, match="CPU float32"):
            probe([1], 2)


def test_unknown_module_invalid_geometry_and_nonfinite_are_rejected(cache_parts):
    module, net = cache_parts
    probe = module.TileCacheProbe(net)
    with torch.no_grad():
        with pytest.raises(ValueError):
            probe([0], 3)
        net.input.weight[0, 0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="nonfinite"):
            probe([1], 2)
        net.input.weight[0, 0, 0, 0] = 0
        net.blocks.append(torch.nn.Dropout().eval())
        with pytest.raises(ValueError, match="unsupported"):
            probe([1], 2)
    for capacity in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            module.TileCacheProbe(net, capacity=capacity)
