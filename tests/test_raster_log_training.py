import importlib
import random
from pathlib import Path

import pytest
import torch

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE


@pytest.fixture
def helper(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("raster_log_training")


def test_full_group_schedule_and_counters(helper):
    due = [i + 1 for i in range(helper.CONFIG["steps"]) if helper.relative_due(i)]
    assert due == list(range(16, 2049, 16))
    helper.validate_relative_counts([32, 32, 32, 32], 4, 2048)
    for counts in ([129, -1, 0, 0], [128.0, 0, 0, 0], [True, 127, 0, 0], [128], [0] * 4):
        with pytest.raises(ValueError):
            helper.validate_relative_counts(counts, 4, 2048)
    with pytest.raises(ValueError):
        helper.relative_due(-1)


def test_complete_group_draws_are_model_independent_and_resumable(helper):
    class Sampler:
        def sample(self, rng, count):
            return [
                helper.pair_row(
                    rng.randrange(100), "test", ((1, 1, 1), 2), ((-1, -1, -1), 2), 2, 2, 6
                )
                for _ in range(count)
            ]

    instance = M.Instance("fixture", (1, 1, 1), 2, 1)
    state = M.SearchState(instance.word, 2, 0, 0)
    request = (instance, state, (instance.spec.encode(CROSSING_CHANGE, 0),))
    positives = [[(request, (0,))]]
    groups = [[(request, 0, (-1.0) ** i) for i in range(n)] for n in (3, 19)]
    refs = [(g, torch.full((len(g),), -1.0)) for g in groups]
    other = [(g, torch.full((len(g),), -5.0)) for g in groups]
    a, b = random.Random(97), random.Random(97)
    sampler = Sampler()
    for _ in range(32):
        x = helper.draw_batch(a, sampler, sampler, sampler, positives, refs)
        y = helper.draw_batch(b, sampler, sampler, sampler, positives, other)
        assert len(x[0]) == 32 and len(x[1]) == 8
        assert x[2] is refs[x[4]]  # No row subsampling or coefficient rescaling.
        assert x[3:] == y[3:] and a.getstate() == b.getstate()
    restored = random.Random()
    restored.setstate(a.getstate())
    args = (sampler, sampler, sampler, positives, refs)
    assert helper.draw_batch(a, *args)[3:] == helper.draw_batch(restored, *args)[3:]


@pytest.mark.parametrize("with_relative", [False, True])
def test_optimizer_composes_losses_and_scheduled_weight(helper, monkeypatch, with_relative):
    trainer = importlib.import_module("train_raster_log_pair")
    net = torch.nn.Linear(1, 1, bias=False)
    objective = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        net.weight.fill_(0.1)
        objective.weight.fill_(0.1)
    calls = []

    def metric(model, obj, rows, microbatch):
        loss = model.weight.sum() + obj.weight.sum()
        loss.backward()
        return float(loss.detach())

    def positive(model, rows, relative, *, microbatch, positive_weight):
        assert relative is None and positive_weight == 0.1
        (positive_weight * model.weight.sum()).backward()
        return {"positive_loss": 0.1, "relative_loss": 0.0}

    def relative(model, group, *, microbatch, weight):
        calls.append((group, weight))
        (weight * model.weight.sum()).backward()
        return {"relative_loss": 0.1}

    monkeypatch.setattr(trainer, "backward_metric", metric)
    monkeypatch.setattr(trainer, "policy_backward", positive)
    monkeypatch.setattr(trainer, "relative_backward", relative)
    optimizer = torch.optim.SGD([*net.parameters(), *objective.parameters()], lr=0.01)
    group = object() if with_relative else None
    result = trainer.optimize(net, objective, optimizer, [], [], group)
    assert calls == ([(group, 0.16)] if with_relative else [])
    expected_norm = ((1.26 if with_relative else 1.1) ** 2 + 1) ** 0.5
    assert result["gradient_norm_before_clip"] == pytest.approx(expected_norm)
    assert net.weight.item() < 0.1 and objective.weight.item() < 0.1
    assert all(torch.isfinite(p).all() for p in net.parameters())
