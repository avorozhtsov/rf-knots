import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.cost_mcts_rollout import BatchedPolicyContinuation, run_continuation_cost_searches
from rf_knots.embedding_metric import MetricEmbeddingObjective


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("pretrained_raster_policy")


@pytest.fixture
def net(bridge):
    class Block(torch.nn.Module):
        wrap_strands = False

        def forward(self, state, active):
            return state * active

    graph = importlib.import_module("pretrained_raster_graph")
    tiles = importlib.import_module("pretrained_raster_tiles")
    torch.manual_seed(92)
    teacher = SimpleNamespace(
        variant="axial",
        recurrent_steps=0,
        max_strands=12,
        input=torch.nn.Conv2d(4, 8, 1, bias=False),
        blocks=torch.nn.ModuleList([Block()]),
    )
    return graph.RasterGraphPolicy(tiles.PretrainedRasterTiles(teacher), width=16).train()


def requests():
    result = []
    for index, word in enumerate(((1, 1, 1), (1, -1, 1, 1, 1), (-1, -1, -1))):
        instance = M.Instance(str(index), word, 2, 1)
        result.append(
            (
                instance,
                M.SearchState(word, 2, 0, 0),
                tuple(instance.spec.encode(CROSSING_CHANGE, p) for p in range(len(word))),
            )
        )
    return result


def test_prior_order_head_context_counts_and_no_value(bridge, net):
    rows = requests()
    expected = bridge.policy_log_probabilities(net, rows, heads=[0, 1, 2]).exp().detach().numpy()
    evaluator = bridge.RasterPolicyEvaluator(net, microbatch=1)
    results = evaluator.evaluate(rows, heads=[0, 1, 2])
    assert net.training
    assert evaluator.calls == evaluator.forwards == 3
    assert evaluator.action_scores == 11
    for index, result in enumerate(results):
        np.testing.assert_allclose(
            result.priors, expected[index, : len(rows[index][2])], rtol=1e-5, atol=1e-6
        )
        with pytest.raises(RuntimeError, match="no value"):
            _ = result.value
    assert evaluator.evaluate([]) == []


def test_illegal_resources_and_nonfinite_do_not_fall_back_to_uniform(bridge, net):
    evaluator = bridge.RasterPolicyEvaluator(net)
    instance, state, actions = requests()[0]
    spent = M.SearchState(state.word, state.strands, 1, 1)
    with pytest.raises(ValueError, match="illegal"):
        evaluator.evaluate([(instance, spent, actions)])
    assert net.training
    with pytest.raises(ValueError, match="resource"):
        evaluator.evaluate([(instance, state, actions + actions)])
    with pytest.raises(ValueError, match="integer head"):
        evaluator.evaluate([(instance, state, actions)], heads=[0.5])
    with torch.no_grad():
        next(net.head.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        evaluator.evaluate([(instance, state, actions)])
    assert net.training


@pytest.mark.parametrize("microbatch", [1, 2])
def test_policy_accumulation_preserves_set_mean_and_relative_sum(bridge, net, microbatch):
    positives = [(row, (0, 1)) for row in requests()]
    group = [
        (row, 0, coefficient)
        for row, coefficient in zip(requests(), (1.0, -0.5, -0.5), strict=True)
    ]
    relative = bridge.reference_preferences(net, [group], microbatch=2)[0]
    other = copy.deepcopy(net)
    whole = bridge.policy_backward(net, positives, relative, microbatch=3)
    split = bridge.policy_backward(other, positives, relative, microbatch=microbatch)
    # Batched GEMMs and normalization have small float32 roundoff differences.
    assert split == pytest.approx(whole, abs=5e-6)
    for a, b in zip(net.parameters(), other.parameters(), strict=True):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=2e-6)


def test_raw_distance_is_not_normalized_or_cached_across_parameter_changes(bridge, net):
    objective = MetricEmbeddingObjective()
    heuristic = bridge.RawKnotDistance(net, objective, microbatch=1)
    states = [M.SearchState((1,), 2, 0, 0), M.SearchState((1, -1, 1), 2, 0, 0)]
    first = heuristic(states)
    assert heuristic.evaluations == 3  # separate raw diagrams, plus unknot anchor
    assert net.training
    with torch.no_grad():
        objective.cc_scale_raw.add_(1)
    second = heuristic(states)
    assert heuristic.evaluations == 6
    assert all(b > a >= 0 for a, b in zip(first, second, strict=True))
    assert heuristic([]) == []


def test_terminal_cost_search_never_reads_missing_value(bridge, net):
    instance = M.Instance("one", (1,), 2, 0, cap_moves=2, growth_len=0, growth_strands=0)
    policy = bridge.RasterPolicyEvaluator(net, microbatch=1)
    tree_prior = bridge.RasterDistancePrior(bridge.RawKnotDistance(net, MetricEmbeddingObjective()))
    trees = run_continuation_cost_searches(
        [instance],
        tree_prior,
        BatchedPolicyContinuation(policy, horizon=2),
        simulations=2,
        seed=92,
    )
    assert trees[0].best_witness is not None
    trees[0].best_witness.verify()
    assert trees[0].best_cost == 1
