import importlib
from pathlib import Path

import pytest
import torch


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    torch.set_num_threads(1)
    return importlib.import_module("benchmark_policy_training_device")


def test_numerical_gate_does_not_hide_missing_gradients_or_large_errors(probe):
    reference = {"loss": torch.tensor(1.0), "grad/p": torch.tensor([0.5])}
    assert probe.difference(reference, reference, exact=True)["passed"]
    with pytest.raises(ValueError, match="coverage"):
        probe.difference(reference, {"loss": reference["loss"]}, exact=False)
    with pytest.raises(AssertionError):
        probe.difference(reference, {**reference, "loss": torch.tensor(2.0)}, exact=False)


def test_cpu_training_probe_does_not_update_weights(probe, monkeypatch):
    from rf_knots.actions import REDUCE

    torch.manual_seed(61)
    architecture = {
        "width": 16,
        "depth": 2,
        "embedding_dim": 8,
        "budget_dim": 5,
        "encoder_kind": "strand_graph",
        "normalized_knot": True,
    }
    net = probe.M.SitePolicyValueNet(**architecture)
    checkpoint = {"architecture": architecture, "model": net.state_dict()}
    instance = probe.M.Instance("probe", (1, -1, 1), 2, 1, cap_moves=8)
    state = probe.M.SearchState(instance.word, instance.strands, 0, 0)
    actions = tuple(a for a, _ in probe.M.successors_of(state, instance, instance.spec))
    selected = [((instance, state, actions), (actions.index(instance.spec.encode(REDUCE, 0)),))]
    monkeypatch.setattr(probe, "CONFIG", {**probe.CONFIG, "warmup": 1, "repeats": 1})
    result, values = probe.benchmark_arm(
        checkpoint, selected, "cpu", probe.graph.crossing_adjacency
    )
    assert result["weights_unchanged"]
    assert result["median_seconds"] > 0
    assert any(k.startswith("grad/") for k in values)
