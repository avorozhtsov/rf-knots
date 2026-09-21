import copy
import importlib
from pathlib import Path

import pytest
import torch

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.embedding_metric import MetricEmbeddingObjective


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    torch.set_num_threads(1)
    return (
        importlib.import_module("audit_joint_training_pair"),
        importlib.import_module("run_strand_graph_pilot"),
        importlib.import_module("mcts_joint_training"),
    )


def train_one_step(modules, mode):
    _, pilot, helper = modules
    architecture = dict(
        width=16,
        depth=2,
        embedding_dim=8,
        encoder_kind="strand_graph",
        normalized_knot=True,
        budget_dim=5,
    )
    torch.manual_seed(800)
    net = M.SitePolicyValueNet(**architecture)
    net.value.requires_grad_(False)
    objective = MetricEmbeddingObjective(cc_rank_weight=0.1, witness_priority=False)
    initial = {"architecture": architecture, "model": copy.deepcopy(net.state_dict())}
    optimizer = torch.optim.AdamW(
        [p for p in [*net.parameters(), *objective.parameters()] if p.requires_grad],
        lr=2.5e-5,
        weight_decay=0.01,
    )
    row = pilot.pair_row(0, "mirror", ((1, 1, 1), 2), ((-1, -1, -1), 2), 2, 2, 6)
    loss, _ = pilot.metric_loss(net, objective, [row], "cpu")
    loss.backward()
    if mode == "joint":
        instance = M.Instance("t", (1, 1, 1), 2, 1, cap_moves=12)
        state = M.SearchState(instance.word, instance.strands, 0, 0)
        actions = tuple(instance.spec.encode(CROSSING_CHANGE, p) for p in range(3))
        helper.policy_backward(net, [((instance, state, actions), (0,))], None)
    optimizer.step()
    checkpoint = {
        "steps": 1,
        "architecture": architecture,
        "model": net.state_dict(),
        "objective": objective.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    return checkpoint, initial


@pytest.mark.parametrize("mode", ["joint", "metric-control"])
def test_audits_real_optimizer_step_and_expected_head_updates(modules, mode):
    auditor, _, _ = modules
    checkpoint, initial = train_one_step(modules, mode)
    result = auditor.inspect_checkpoint(checkpoint, initial, mode=mode, steps=1)
    assert result["changed_model_tensors"]["encoder"] > 0
    assert result["changed_model_tensors"]["value"] == 0
    assert (result["changed_model_tensors"]["head"] > 0) == (mode == "joint")
    assert result["optimizer_step_range"] == [1, 1]


@pytest.mark.parametrize("fault", ["value", "steps", "moment", "lr", "mode"])
def test_rejects_corrupted_or_wrong_arm_checkpoint(modules, fault):
    auditor, _, _ = modules
    checkpoint, initial = train_one_step(modules, "metric-control")
    mode = "metric-control"
    if fault == "value":
        checkpoint["model"]["value.0.weight"].add_(1)
    elif fault == "steps":
        checkpoint["steps"] = 0
    elif fault == "moment":
        next(iter(checkpoint["optimizer"]["state"].values()))["exp_avg_sq"].fill_(-1)
    elif fault == "lr":
        checkpoint["optimizer"]["param_groups"][0]["lr"] = 1
    else:
        mode = "joint"
    with pytest.raises(ValueError):
        auditor.inspect_checkpoint(checkpoint, initial, mode=mode, steps=1)
