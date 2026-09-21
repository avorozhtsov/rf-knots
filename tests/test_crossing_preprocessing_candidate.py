import importlib
from pathlib import Path

import pytest
import torch

from rf_knots import strand_graph_encoder as graph


@pytest.fixture
def candidate(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    torch.set_num_threads(1)
    return importlib.import_module("benchmark_crossing_preprocessing")


def test_bulk_integer_adjacency_is_exact_including_padding_and_large_lanes(candidate):
    rng = torch.Generator().manual_seed(813)
    for batch in (0, 1, 7):
        for padded in (0, 1, 17, 257):
            words = torch.randint(-100, 101, (batch, padded), generator=rng)
            lengths = torch.randint(0, padded + 1, (batch,), generator=rng)
            if batch and padded:
                words[0, 0] = 2**60
            expected = graph.crossing_adjacency(words, lengths)
            actual = candidate.bulk_crossing_adjacency(words, lengths)
            assert torch.equal(actual, expected)
            assert actual.dtype == expected.dtype and actual.device == expected.device


def test_context_restores_original_even_on_failure(candidate):
    original = graph.crossing_adjacency
    with pytest.raises(RuntimeError):
        with candidate.use_adjacency(candidate.bulk_crossing_adjacency):
            assert graph.crossing_adjacency is candidate.bulk_crossing_adjacency
            raise RuntimeError("deliberate")
    assert graph.crossing_adjacency is original


def test_full_encoder_outputs_and_gradients_are_unchanged(candidate):
    torch.manual_seed(182)
    net = graph.StrandGraphBraidEncoder(width=16, depth=2, embedding_dim=8, normalized_knot=True)
    words = torch.tensor([[1, -1, 1, 0, 0], [2, 1, -2, 1, 2]])
    lengths, strands = torch.tensor([3, 5]), torch.tensor([2, 3])

    def run():
        net.zero_grad(set_to_none=True)
        result = net.encode_sites(words, lengths, strands)
        sum(
            value.square().sum() for value in result.values() if value.is_floating_point()
        ).backward()
        gradients = {
            name: p.grad.clone() for name, p in net.named_parameters() if p.grad is not None
        }
        return result, gradients

    before, gradients = run()
    with candidate.use_adjacency(candidate.bulk_crossing_adjacency):
        after, new_gradients = run()
    assert before.keys() == after.keys() and gradients.keys() == new_gradients.keys()
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    for name in gradients:
        torch.testing.assert_close(gradients[name], new_gradients[name], rtol=0, atol=0)


def test_policy_continuation_search_is_identical_on_engineering_task(candidate):
    evaluation = importlib.import_module("evaluate_joint_cost_mcts")
    from rf_knots import mcts_loop as search
    from rf_knots.embedding_metric import MetricEmbeddingObjective

    torch.manual_seed(508)
    architecture = {
        "width": 16,
        "depth": 2,
        "embedding_dim": 8,
        "budget_dim": 5,
        "encoder_kind": "strand_graph",
        "normalized_knot": True,
        "context_rounds": -1,
        "window_size": 8,
        "hierarchy_levels": 2,
    }
    net = search.SitePolicyValueNet(**architecture)
    checkpoint = {
        "architecture": architecture,
        "model": net.state_dict(),
        "objective": MetricEmbeddingObjective(
            cc_rank_weight=0.1, witness_priority=False
        ).state_dict(),
    }
    instance = search.Instance("engineering-parity", (1, -1, 1), 2, 0, cap_moves=8, cc_weight=10)
    roots = [{"condition": 0, "panel_index": 0, "instance": instance.to_dict()}]
    config = {**evaluation.CONFIG, "simulations": 3, "horizon": 8}
    before = evaluation.run_batch(roots, checkpoint, 0, config)
    with candidate.use_adjacency(candidate.bulk_crossing_adjacency):
        after = evaluation.run_batch(roots, checkpoint, 0, config)
    assert before == after
    assert before[1]["episodes"] > 0
    assert before[1]["policy_neural_evaluations"] > 0
