import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("raster_resource_units")


def test_explicit_units_preserve_globals_flags_and_missingness(module):
    head = module.ResourceUnitsHead(width=8, embedding_dim=4, budget_dim=5)
    globals_ = torch.arange(12, dtype=torch.float32)[None]
    units = torch.tensor([module.RESOURCE_UNITS])
    features = head._context_features(globals_, units, 1, "cpu", torch.float32)
    assert torch.equal(features[:, :12], globals_)
    torch.testing.assert_close(features[:, 12:22], torch.ones(1, 10))
    assert torch.equal(features[:, 22:], torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))
    missing = head._context_features(globals_, None, 1, "cpu", torch.float32)
    assert torch.equal(missing[:, 12:22], torch.zeros(1, 10))
    assert torch.equal(missing[:, 22:], torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]))
    zero = head._context_features(globals_, units * 0, 1, "cpu", torch.float32)
    assert torch.equal(zero[:, 22:], torch.ones(1, 6))


def test_unbounded_monotone_differentiable_and_fail_closed(module):
    head = module.ResourceUnitsHead(width=8, embedding_dim=4, budget_dim=5)
    x = (torch.tensor([module.RESOURCE_UNITS]) * 100).requires_grad_()
    f = head._context_features(torch.zeros(1, 12), x, 1, "cpu", torch.float32)
    assert (f[:, 12:22] > 1).all()
    f.sum().backward()
    assert torch.isfinite(x.grad).all() and (x.grad > 0).all()
    for value in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            head._context_features(
                torch.zeros(1, 12), torch.full((1, 5), value), 1, "cpu", torch.float32
            )


def test_conversion_owns_weights_and_preserves_initial_embeddings(module):
    from pretrained_raster_graph import RasterGraphPolicy
    from pretrained_raster_tiles import PretrainedRasterTiles

    teacher = SimpleNamespace(
        variant="axial",
        recurrent_steps=0,
        max_strands=12,
        input=torch.nn.Conv2d(4, 8, 1, bias=False),
        blocks=torch.nn.ModuleList(),
    )
    net = RasterGraphPolicy(PretrainedRasterTiles(teacher), width=16).eval()
    units = module.with_resource_units(net, "units")
    raw = module.with_resource_units(net, "raw")
    for key, value in net.state_dict().items():
        assert torch.equal(value, units.state_dict()[key])
        assert value.data_ptr() != units.state_dict()[key].data_ptr()
        assert torch.equal(value, raw.state_dict()[key])
    args = (torch.tensor([[1, 1, 1]]), torch.tensor([3]), torch.tensor([2]))
    for name, value in net.encoder(*args).items():
        assert torch.equal(value, units.encoder(*args)[name])


def test_checkpoint_roundtrip_rejects_feature_contract_drift(module, tmp_path):
    runner = importlib.import_module("train_raster_resource_pair")
    payload = dict(
        config=runner.CONFIG,
        contract_sha256="contract",
        arm="units",
        steps=16,
        model={"weight": torch.tensor([1.0, 2.0])},
        rng=(1, 2, 3),
    )
    entry = runner.save_checkpoint(tmp_path, payload)
    restored = runner.verify_checkpoint(tmp_path, entry, "contract", "units")
    assert torch.equal(restored["model"]["weight"], payload["model"]["weight"])
    for contract, arm in (("different", "units"), ("contract", "raw")):
        with pytest.raises(ValueError, match="contract differs"):
            runner.verify_checkpoint(tmp_path, entry, contract, arm)
    with pytest.raises(ValueError, match="contract differs"):
        runner.verify_checkpoint(tmp_path, {**entry, "steps": 32}, "contract", "units")
    with pytest.raises(ValueError, match="identity differs"):
        runner.verify_checkpoint(tmp_path, {**entry, "sha256": "wrong"}, "contract", "units")


def test_cpu_snapshot_owns_optimizer_tensors(module):
    runner = importlib.import_module("train_raster_resource_pair")
    live = {"state": [torch.tensor([2.0], requires_grad=True)], "nested": (torch.tensor(16),)}
    saved = runner.cpu_tree(live)
    assert not saved["state"][0].requires_grad
    with torch.no_grad():
        live["state"][0].add_(7)
    assert saved["state"][0].item() == 2
    assert type(saved["nested"]) is tuple
