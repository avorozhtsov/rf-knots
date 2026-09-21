import copy
import importlib
import random
from pathlib import Path

import pytest
import torch
from test_mcts_joint_training import inputs, network

from rf_knots.embedding_metric import MetricEmbeddingObjective


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    torch.set_num_threads(1)
    return (
        importlib.import_module("graph_metric_sampling"),
        importlib.import_module("train_graph_metric_mixture"),
    )


def test_graph_sampler_balances_endpoint_groups_and_excludes_trivial_zero(modules):
    sample, train = modules
    entries = []
    for i, (lo, hi) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (1, 1))):
        row = train.pair_row(i, "test", ((1, 1, 1), 2), ((1, -1, 1), 2), lo, hi, hi)
        a, b = ((3, 2), (1, 0)) if i != 4 else ((1, 0), (3, -2))
        entries.append((row, a, b))
    sampler = sample.GraphMetricSampler(entries)
    assert sampler.support["exact-1"] == {"groups": 1, "records": 2}
    assert "exact-0" not in sampler.support
    rng, expected = random.Random(8), random.Random(8)
    rows = sampler.sample(rng, 500)
    for _ in range(4 * 500):
        expected.random()
    assert rng.getstate() == expected.getstate()
    assert {r.spec.index for r in rows} == {1, 2, 3, 4}
    assert any(r.left_word == (1, -1, 1) for r in rows)
    assert all(r.left_word in ((1, 1, 1), (1, -1, 1)) for r in rows)
    with pytest.raises(ValueError, match="three fixed classes"):
        sample.GraphMetricSampler(entries[:2])


def test_paired_draws_hold_policy_and_rng_identical_while_replacing_only_metric_source(modules):
    sample, _ = modules

    class Sampler:
        def __init__(self, name):
            self.name = name

        def sample(self, rng, count):
            return [(self.name, rng.random()) for _ in range(count)]

    pools = (Sampler("old"), Sampler("suffix"), Sampler("graph"), [[1, 2], [3, 4]], [5, 6])
    a, b = random.Random(74), random.Random(74)
    for _ in range(12):
        graph, pg, rg = sample.paired_draws(a, *pools, "graph")
        control, pc, rc = sample.paired_draws(b, *pools, "control")
        assert len(graph) == len(control) == 127
        assert len(pg) == len(pc) == 32 and pg == pc and rg == rc
        assert graph[:31] == control[:31]
        assert graph[31:63] == control[95:127]
        assert all(x[0] == "graph" for x in graph[63:])
        assert a.getstate() == b.getstate()
    recovered = random.Random()
    recovered.setstate(a.getstate())
    assert sample.paired_draws(a, *pools, "graph") == sample.paired_draws(
        recovered, *pools, "graph"
    )


def test_actual_optimizer_step_bulk_equivalence_and_frozen_value(modules):
    _, train = modules
    net = network()
    initial = copy.deepcopy(net.state_dict())
    objective = MetricEmbeddingObjective(cc_rank_weight=0.1, witness_priority=False)
    config = {**train.CONFIG, "metric_batch": 1, "policy_batch": 1}
    row = train.pair_row(0, "mirror", ((1, 1, 1), 2), ((-1, -1, -1), 2), 2, 2, 6)
    parameters = [p for p in [*net.parameters(), *objective.parameters()] if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    original_objective = copy.deepcopy(objective.state_dict())
    reference = train.optimize_step(net, objective, optimizer, [row], [inputs()], None, config)
    expected = copy.deepcopy(net.state_dict())
    expected_objective = copy.deepcopy(objective.state_dict())
    net.load_state_dict(initial)
    objective.load_state_dict(original_objective)
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    with train.use_adjacency(train.bulk_crossing_adjacency):
        actual = train.optimize_step(net, objective, optimizer, [row], [inputs()], None, config)
    assert reference == actual
    for name, value in net.state_dict().items():
        assert torch.equal(value, expected[name])
        if name.startswith("value."):
            assert torch.equal(value, initial[name])
    assert any(
        not torch.equal(v, initial[k]) for k, v in expected.items() if k.startswith("encoder.")
    )
    assert any(not torch.equal(v, initial[k]) for k, v in expected.items() if k.startswith("head."))
    for name, value in objective.state_dict().items():
        assert torch.equal(value, expected_objective[name])
    train.finite_state(net, objective)
    train.finite_optimizer(optimizer)
    with pytest.raises(ValueError, match="draw coverage"):
        train.optimize_step(net, objective, optimizer, [], [inputs()], None, config)
