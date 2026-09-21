import importlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.embedding_metric import MetricEmbeddingObjective
from rf_knots.rollout_value import clipped_relative_loss


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    torch.set_num_threads(1)
    return importlib.import_module("mcts_joint_training"), importlib.import_module(
        "train_mcts_policy_metric"
    )


def inputs():
    instance = M.Instance("t", (1, 1, 1), 2, 1, cap_moves=20)
    state = M.SearchState(instance.word, instance.strands, 0, 0)
    actions = tuple(instance.spec.encode(CROSSING_CHANGE, n) for n in range(3))
    return (instance, state, actions), (0, 2)


def network():
    torch.manual_seed(410)
    net = M.SitePolicyValueNet(
        width=16,
        depth=2,
        embedding_dim=8,
        encoder_kind="strand_graph",
        budget_dim=5,
        normalized_knot=True,
        window_size=4,
        hierarchy_levels=2,
    )
    net.value.requires_grad_(False)
    return net


def test_joint_updates_encoder_and_policy_but_not_value(modules):
    helper, _ = modules
    pilot = importlib.import_module("run_strand_graph_pilot")
    net = network()
    objective = MetricEmbeddingObjective()
    value = {k: v.clone() for k, v in net.value.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    row = pilot.pair_row(0, "mirror", ((1, 1, 1), 2), ((-1, -1, -1), 2), 2, 2, 6)
    metric, _ = pilot.metric_loss(net, objective, [row], "cpu")
    metric.backward()
    report = helper.policy_backward(net, [inputs()], None)
    assert report["positive_loss"] > 0
    for module in (net.encoder, net.head):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
    assert all(p.grad is None for p in net.value.parameters())
    optimizer.step()
    for name, tensor in net.value.state_dict().items():
        assert torch.equal(tensor, value[name])


def test_metric_control_leaves_unused_policy_parameters_unchanged(modules):
    pilot = importlib.import_module("run_strand_graph_pilot")
    net = network()
    objective = MetricEmbeddingObjective()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    optimizer = torch.optim.AdamW(
        [p for p in [*net.parameters(), *objective.parameters()] if p.requires_grad],
        lr=2.5e-5,
        weight_decay=0.01,
    )
    row = pilot.pair_row(0, "mirror", ((1, 1, 1), 2), ((-1, -1, -1), 2), 2, 2, 6)
    loss, _ = pilot.metric_loss(net, objective, [row], "cpu")
    loss.backward()
    assert all(p.grad is None for p in net.head.parameters())
    optimizer.step()
    for name, tensor in net.state_dict().items():
        if name.startswith(("head.", "value.")):
            assert torch.equal(tensor, before[name])
    assert any(
        not torch.equal(tensor, before[name])
        for name, tensor in net.state_dict().items()
        if name.startswith("encoder.")
    )


@pytest.mark.parametrize("micro_batch", [1, 2, 5])
def test_relative_microbatches_match_summed_objective(modules, micro_batch):
    helper, _ = modules
    net = network()
    request, good = inputs()
    group = [(request, 0, 1.0), (request, 1, -0.5), (request, 2, -0.5)]
    reference = helper.reference_preferences(net, [group])[0]
    assert not reference[1].requires_grad
    logp = helper.policy_forward(net, [request])
    positive_loss = -torch.logsumexp(logp[0, list(good)], 0)
    relative_loss = clipped_relative_loss(logp[0], reference[1], torch.tensor([1.0, -0.5, -0.5]))
    (0.1 * positive_loss + 0.01 * relative_loss).backward()
    expected = {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None}
    net.zero_grad(set_to_none=True)
    report = helper.policy_backward(net, [inputs()], reference, micro_batch=micro_batch)
    assert report["relative_loss"] == pytest.approx(float(relative_loss.detach()), abs=1e-5)
    for k, p in net.named_parameters():
        if k in expected:
            torch.testing.assert_close(p.grad, expected[k], atol=1e-5, rtol=1e-4)


def test_matched_sampling_and_rng_restore(modules):
    helper, _ = modules

    class Sampler:
        def sample(self, rng, count):
            return [rng.randrange(50) for _ in range(count)]

    a, b = random.Random(32), random.Random(32)
    arguments = (Sampler(), Sampler(), [[1, 2], [3, 4]], [5, 6])
    for _ in range(10):
        assert helper.sample_step(a, *arguments) == helper.sample_step(b, *arguments)
    recovered = random.Random()
    recovered.setstate(a.getstate())
    assert helper.sample_step(a, *arguments) == helper.sample_step(recovered, *arguments)


@pytest.mark.parametrize("corruption", [None, "cc", "moves", "duplicate", "good", "hash"])
def test_policy_loader_resource_and_integrity_checks(modules, tmp_path, corruption):
    helper, _ = modules
    request, _ = inputs()
    instance, state, actions = request
    row = {"state": asdict(state), "actions": actions, "positive_actions": [actions[0]]}
    if corruption == "cc":
        row["state"]["used_cc"] = -1
    if corruption == "moves":
        row["state"]["used_moves"] = -1
    if corruption == "good":
        row["positive_actions"] *= 2
    data = {"condition": 0, "instance": asdict(instance), "positives": [row], "preferences": []}
    if corruption == "duplicate":
        data["positives"].append(row)
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(data))
    manifest = {
        "status": "completed",
        "conditions": [{"file": path.name, "sha256": helper.sha(path)}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    if corruption == "hash":
        path.write_text("{}")
    if corruption:
        with pytest.raises(ValueError):
            helper.load_policy_targets(tmp_path)
    else:
        positives, preferences = helper.load_policy_targets(tmp_path)
        assert len(positives) == len(positives[0]) == 1
        loaded, indices = positives[0][0]
        assert asdict(loaded[0]) == asdict(instance)
        assert loaded[1:] == request[1:]
        assert indices == (0,)
        assert not preferences


def test_recovery_publishes_checkpoint_hash_with_step_and_survives_second_crash(modules):
    _, runner = modules
    manifest = {"completed_steps": 16, "progress_sha256": "old"}
    progress = {"steps": 32, "contract_sha256": "contract", "architecture": {}}
    assert runner.validate_progress(progress, manifest, "contract", {}, "new") == 32
    assert manifest["progress_sha256"] == "new"
    assert runner.validate_progress(progress, manifest, "contract", {}, "new") == 32
    with pytest.raises(ValueError, match="hash"):
        runner.validate_progress(progress, manifest, "contract", {}, "tampered")
    with pytest.raises(ValueError, match="contract/step"):
        runner.validate_progress({**progress, "steps": 64}, manifest, "contract", {}, "new")


def test_nonfinite_optimizer_rejected(modules):
    _, runner = modules
    p = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([p])
    p.sum().backward()
    optimizer.step()
    runner.finite_optimizer(optimizer)
    optimizer.state[p]["exp_avg"].fill_(float("nan"))
    with pytest.raises(ValueError, match="optimizer"):
        runner.finite_optimizer(optimizer)
