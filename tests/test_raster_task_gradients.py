import importlib
from pathlib import Path

import pytest
import torch


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("diagnose_raster_task_gradients")


def test_exact_vector_geometry_and_zero_cosine(probe):
    stats = probe.vector_stats(
        dict(a=torch.tensor([3.0, 4.0]), b=torch.tensor([-4.0, 3.0]), z=torch.zeros(2))
    )
    assert stats["norms"] == dict(a=5.0, b=5.0, z=0.0)
    assert stats["pairs"]["a__b"] == dict(dot=0.0, cosine=0.0)
    assert stats["pairs"]["a__z"]["cosine"] is None
    with pytest.raises(ValueError):
        probe.vector_stats(dict(a=torch.tensor([float("nan")])))


def test_snapshot_disconnected_gradients_and_no_aliasing(probe):
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0]))
    first.square().sum().backward()
    vector, layout = probe.gradient_snapshot(
        [("net.encoder.tiles.w", first), ("objective.scale", second)]
    )
    torch.testing.assert_close(vector, torch.tensor([2.0, 4.0, 0.0]))
    assert [p["absent"] for p in layout] == [False, True]
    first.grad.zero_()
    assert vector[0] == 2
    with pytest.raises(ValueError):
        probe.block("unknown.w")


def test_sum_clip_and_conflict_are_weighted(probe):
    vectors = dict(
        metric=torch.tensor([1.0, 0.0]),
        positive=torch.tensor([0.0, 2.0]),
        relative0=torch.tensor([-3.0, 0.0]),
    )
    layout = [dict(start=0, stop=1, block="local"), dict(start=1, stop=2, block="policy")]
    summary = probe.summarize(vectors, layout)
    assert summary["sums"]["relative0"]["dot_with_metric"] == -2
    assert summary["sums"]["ordinary"]["norm_before_clip"] == pytest.approx(5**0.5)
    assert summary["sums"]["relative0"]["hypothetical_clip_factor"] == pytest.approx(
        1 / (8**0.5 + 1e-6)
    )
    assert summary["blocks"]["local"]["pairs"]["metric__relative0"]["cosine"] == -1
    with pytest.raises(ValueError):
        probe.summarize(vectors, [dict(start=0, stop=1, block="missing")])


def test_fixed_scope_and_group_sizes(probe):
    assert probe.GROUP_SIZES == [334, 52, 74, 54]
    assert probe.CONFIG["relative_weight"] * probe.CONFIG["relative_every"] == 0.16
    assert probe.block("net.encoder.heads.knot.weight") == "global_knot"
    assert probe.block("net.encoder.project.0.weight") == "shared_graph_pool"
    assert probe.block("net.position.0.weight") == "position"
    assert probe.block("net.head.weight") == "policy_head"
