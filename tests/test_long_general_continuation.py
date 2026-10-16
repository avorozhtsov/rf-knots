import importlib
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("continue_long_general_pair")


def test_restore_full_adam_and_rng(module, monkeypatch):
    monkeypatch.setattr(torch.mps, "set_rng_state", lambda x: None)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: torch.tensor([1], dtype=torch.uint8))
    net, objective = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW([*net.parameters(), *objective.parameters()], lr=1e-4)
    objective(net(torch.ones(1, 2))).sum().backward()
    opt.step()
    rng, long_rng = random.Random(4), random.Random(5)
    payload = module.cpu_tree(
        dict(
            model=net.state_dict(),
            objective=objective.state_dict(),
            optimizer=opt.state_dict(),
            rng=rng.getstate(),
            long_rng=long_rng.getstate(),
            torch_rng=torch.get_rng_state(),
            mps_rng=torch.mps.get_rng_state(),
        )
    )
    restored_net, restored_objective = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
    restored_opt = torch.optim.AdamW([*restored_net.parameters(), *restored_objective.parameters()])
    module.restore(payload, restored_net, restored_objective, restored_opt, rng, long_rng)
    assert module.equal_tree(restored_opt.state_dict(), payload["optimizer"])
    assert not module.equal_tree(payload, {**payload, "rng": random.Random(6).getstate()})


def test_checkpoint_contract_and_hash(module, tmp_path):
    payload = dict(
        contract_sha256="contract",
        arm="long",
        steps=2112,
        config=module.base.CONFIG,
        continuation_config=module.CONFIG,
        relative_counts=[132],
        relative_groups=1,
    )
    entry = module.base.save_checkpoint(tmp_path, payload)
    assert module.load_checkpoint(tmp_path, entry, "contract", "long")["steps"] == 2112
    for changed, contract, arm in (
        (entry, "other", "long"),
        (entry, "contract", "control"),
        ({**entry, "file": "../escape.pt"}, "contract", "long"),
        ({**entry, "steps": 2176}, "contract", "long"),
    ):
        with pytest.raises(ValueError):
            module.load_checkpoint(tmp_path, changed, contract, arm)


def test_time_budget_and_parent_configuration_not_mutated(module):
    assert module.CONFIG["minimum_training_seconds"] == 72 * 3600
    assert module.CONFIG["maximum_training_seconds"] == 96 * 3600
    assert module.CONFIG["round_steps"] % module.CONFIG["checkpoint_interval"] == 0
    assert module.base.CONFIG["steps"] == 2048
    assert not module.CONFIG["test_selection"] and not module.CONFIG["optimizer_reset"]


def test_block_interrupted_resume_matches_uninterrupted(module, monkeypatch, tmp_path):
    # CPU toy tests the NEW orchestration, not native raster/MPS scientific learning.
    class CPUModel(torch.nn.Linear):
        def to(self, *args, **kwargs):
            return self

    net, objective = CPUModel(1, 1), CPUModel(1, 1)
    monkeypatch.setattr(module.base, "with_resource_units", lambda n, features: n)
    monkeypatch.setattr(module.base, "MetricEmbeddingObjective", lambda: CPUModel(1, 1))
    monkeypatch.setattr(torch.mps, "set_rng_state", lambda x: None)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: torch.tensor([1], dtype=torch.uint8))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: None)
    preferences = [(None, torch.tensor([0.5]))]
    reference = module.base.digest([[0.5]])
    monkeypatch.setattr(module.base, "reference_preferences", lambda *a, **kw: preferences)
    monkeypatch.setattr(module.base, "check_pins", lambda pins: None)
    monkeypatch.setattr(module.base, "count_batch", lambda counters, draw, rel: None)
    parent_root = tmp_path / "parent"
    (parent_root / "control").mkdir(parents=True)
    opt = torch.optim.AdamW(
        [*net.parameters(), *objective.parameters()],
        lr=module.base.CONFIG["lr"],
        weight_decay=module.base.CONFIG["weight_decay"],
    )
    payload = dict(
        model=net.state_dict(),
        objective=objective.state_dict(),
        optimizer=opt.state_dict(),
        config=module.base.CONFIG,
        arm="control",
        steps=2048,
        architecture={},
        rng=random.Random(3).getstate(),
        long_rng=random.Random(4).getstate(),
        torch_rng=torch.get_rng_state(),
        mps_rng=torch.mps.get_rng_state(),
        relative_groups=1,
        relative_counts=[128],
        shared_draw_digest="start",
        selected_draw_digest="start",
        size_counters={},
        contract_sha256="parent",
    )
    entry = module.base.save_checkpoint(parent_root / "control", payload)
    parent = dict(
        contract_sha256="parent", control=dict(checkpoint=entry, reference_sha256=reference)
    )
    calls = [0]

    def draw(arm, rng, long_rng, *args):
        calls[0] += 1
        value, shadow = rng.random(), long_rng.random()
        return SimpleNamespace(
            metrics=[value],
            positives=[],
            relative=([], []),
            group_index=0,
            shared_digest=str(value),
            selected_digest=str(shadow),
        )

    def optimize(n, o, optimizer, rows, positives, relative):
        optimizer.zero_grad()
        loss = o(n(torch.tensor([[rows[0]]]))).square().sum()
        loss.backward()
        optimizer.step()
        return dict(metric_loss=loss.item(), positive_loss=0.0)

    monkeypatch.setattr(module.base, "paired_draw", draw)
    monkeypatch.setattr(module.base, "optimize", optimize)
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    whole.mkdir()
    resumed.mkdir()
    args = (parent_root, parent, net, {}, (None,) * 6, {}, 2176, "new", {})
    full = module.train_block("control", whole, *args)
    calls[0] = 0

    def interrupted(*a):
        if calls[0] == 64:
            raise RuntimeError("simulated interruption")
        return draw(*a)

    monkeypatch.setattr(module.base, "paired_draw", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        module.train_block("control", resumed, *args)
    assert module.base.read(resumed / "control" / "manifest.json")["completed_steps"] == 2112
    monkeypatch.setattr(module.base, "paired_draw", draw)
    result = module.train_block("control", resumed, *args)
    a = module.load_checkpoint(whole / "control", full["checkpoint"], "new", "control")
    b = module.load_checkpoint(resumed / "control", result["checkpoint"], "new", "control")
    assert module.equal_tree(a, b)
