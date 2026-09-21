import pytest

from rf_knots.embedding_cost_search import greedy_cost_search
from rf_knots.mcts_loop import Instance


def test_solved_empty_and_budget_zero():
    result = greedy_cost_search(
        Instance("empty", (), 1, 0), lambda states: [0] * len(states), expansions=0, cc_weight=1000
    )
    assert result.witness is not None and result.witness.moves == 0
    result = greedy_cost_search(
        Instance("t", (1, 1, 1), 2, 1), lambda states: [0] * len(states), expansions=0, cc_weight=10
    )
    assert result.witness is None


@pytest.mark.parametrize("weight", [10, 1000])
@pytest.mark.parametrize("layout", [False, True])
def test_replay_cost_and_no_learned_pruning(weight, layout):
    instance = Instance("t", (1, 1, 1), 2, 1, cap_moves=8, growth_len=2, growth_strands=1)
    # All positive generators have nonzero writhe; prefer a changed diagram.
    result = greedy_cost_search(
        instance,
        lambda states: [abs(sum(s.word)) for s in states],
        expansions=40,
        cc_weight=weight,
        layout_tiebreak=layout,
    )
    assert result.witness is not None
    result.witness.verify()
    assert result.witness.crossing_changes == 1
    assert result.witness.moves == 3
    costs = [r["L10"] if weight == 10 else r["L1000"] for r in result.incumbents]
    assert all(b < a for a, b in zip(costs, costs[1:], strict=False))
    assert result.expansions > result.incumbents[0]["expansion"]


def test_bad_heuristic_is_rejected():
    with pytest.raises(ValueError, match="heuristic"):
        greedy_cost_search(
            Instance("t", (1, 1, 1), 2, 1),
            lambda states: [float("nan")] * len(states),
            expansions=1,
            cc_weight=10,
        )


def test_cached_normalized_heuristic_matches_regular_encoder():
    import importlib.util
    from pathlib import Path

    import torch

    from rf_knots.embedding_metric import MetricEmbeddingObjective
    from rf_knots.mcts_loop import SearchState, SitePolicyValueNet

    spec = importlib.util.spec_from_file_location(
        "cost_evaluator", Path(__file__).parents[1] / "scripts/evaluate_embedding_cost_search.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = SitePolicyValueNet(
        width=16, depth=2, embedding_dim=8, encoder_kind="strand_graph", normalized_knot=True
    ).eval()
    objective = MetricEmbeddingObjective()
    heuristic = module.KnotDistanceHeuristic(net, objective)
    # Same trefoil after stabilization and inverse-pair insertion.
    states = [SearchState((1, 1, 1), 2, 0, 0), SearchState((1, 1, 1, 2, 2, -2), 3, 0, 0)]
    scores = heuristic(states)
    assert scores[0] == scores[1]
    assert heuristic.evaluations == 2  # one anchor + one normalized trefoil
    assert heuristic(states) == scores
    assert heuristic.evaluations == 2
    with torch.no_grad():
        z = net.encoder(torch.tensor([[1, 1, 1]]), torch.tensor([3]), torch.tensor([2]))["knot"]
        a = net.encoder(
            torch.zeros((1, 1), dtype=torch.long), torch.tensor([0]), torch.tensor([1])
        )["knot"]
        expected = float(objective.scale("cc") * objective.cc_distance((z * a).sum(-1)))
    assert scores[0] == pytest.approx(expected, abs=1e-6)
    # Unit-vector dot products can round just above one; zero distance must
    # remain finite/nonnegative instead of aborting the search.
    epsilon = torch.finfo(heuristic.anchor.dtype).eps
    vector = torch.zeros_like(heuristic.anchor)
    vector[0] = 1 + epsilon
    heuristic.anchor = vector
    heuristic.embeddings[((), 1)] = vector
    heuristic.scores.clear()
    assert heuristic([SearchState((), 1, 0, 0)]) == [0.0]


def test_witness_feasible_caps_include_temporary_growth():
    import importlib.util
    from pathlib import Path

    from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, INSERT, REDUCE, ActionSpec
    from rf_knots.evidence import UnknotWitness

    module_spec = importlib.util.spec_from_file_location(
        "cost_evaluator_caps",
        Path(__file__).parents[1] / "scripts/evaluate_embedding_cost_search.py",
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    spec = ActionSpec(16, 5)
    actions = [spec.encode(INSERT, 0, 1, 1)] * 2 + [spec.encode(REDUCE, 0)] * 2
    actions += [spec.encode(CROSSING_CHANGE, 0), spec.encode(REDUCE, 0), spec.encode(DESTABILIZE)]
    witness = UnknotWitness.from_actions((1, 1, 1), 2, spec, actions)
    _, fits = module.query_instance(0, witness)
    assert not fits
    expanded, fits = module.query_instance(0, witness, witness_caps=True)
    assert fits and expanded.spec.max_len == 7
    assert expanded.cap_crossing_changes == 1
