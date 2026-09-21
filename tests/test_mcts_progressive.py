import numpy as np
import pytest

from rf_knots import mcts_loop as M
from rf_knots import reference


def instance():
    return M.Instance("trefoil", (1, 1, 1), 2, 1, cap_moves=8)


def expand(tree):
    pending = tree.select_leaf()
    assert pending is not None
    tree.prepare_pending(pending)
    n = len(pending.actions)
    tree.apply(pending, M.Evaluation(np.full(n, 1 / max(n, 1)), 0.5))
    return pending


def test_widen_preserves_edge_visits_values_children_and_dead_flags():
    tree = M.Tree(instance(), seed=0, widening_batch=2)
    expand(tree)
    root = tree.root
    old_actions = root.actions
    root.visits[:] = [3, 4]
    root.total[:] = [1.25, 2.5]
    root.dead[0] = True
    child = M.Node(root.children_states[1], root, 1)
    root.children[1] = child
    root.count = 9
    pending = expand(tree)
    assert pending.widening
    assert root.actions[:2] == old_actions
    np.testing.assert_equal(root.visits[:2], [3, 4])
    np.testing.assert_equal(root.total[:2], [1.25, 2.5])
    assert root.dead[0] and root.children[1] is child
    assert root.catalogue_cursor == 4
    assert tree.simulations == 2 and tree.expansions == 2


def test_dead_prefix_does_not_prune_unopened_actions():
    problem = M.Instance("no-cc", (1, 1, 1), 2, 0, cap_moves=8)
    tree = M.Tree(problem, seed=0, widening_batch=1)
    expand(tree)
    while tree.root.catalogue_cursor < len(tree.root.catalogue):
        tree.root.dead[:] = True
        assert expand(tree).widening
    expected = {a for a, _, _ in reference.successors(problem.spec, problem.word, 2, False)}
    assert set(tree.root.actions) == expected
    assert len(tree.root.actions) == len(expected)
    tree.root.dead[:] = True
    assert tree.select_leaf() is None


def test_progressive_search_replays_without_full_successor_enumeration(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("eager successor enumeration")

    monkeypatch.setattr(reference, "successors", forbidden)
    results = M.run_searches(
        [instance()], M.UniformEvaluator(), simulations=256, seeds=(0,), widening_batch=4
    )
    assert results[0].solved
    results[0].witness.verify()
    assert results[0].witness.crossing_changes == 1


def test_invalid_widening_modes_fail_explicitly():
    with pytest.raises(ValueError, match="progressive"):
        M.Tree(instance(), seed=0, widening_batch=4, root_selection="gumbel")
    with pytest.raises(ValueError, match="progressive"):
        M.Tree(instance(), seed=0, widening_batch=-1)


def test_no_evaluation_round_is_not_an_early_stop(monkeypatch):
    original = M.Tree

    class OneBackupOnlyRound(original):
        def select_leaf(self):
            if self.simulations == 2:
                self.simulations += 1
                return None
            return super().select_leaf()

    monkeypatch.setattr(M, "Tree", OneBackupOnlyRound)
    problem = M.Instance("budget", (1, 1, 1), 2, 0, cap_moves=4)
    result = M.run_searches([problem], M.UniformEvaluator(), simulations=8, widening_batch=1)[0]
    assert result.simulations == 8
    assert result.expansions > 2


def test_factorized_provider_opens_candidates_and_keeps_catalogue_fallback():
    net = M.SitePolicyValueNet(
        width=16,
        depth=2,
        embedding_dim=8,
        encoder_kind="strand_graph",
        budget_dim=5,
        factorized_actions=True,
    ).eval()
    provider = M.FactorizedProposalProvider(net, top_positions=2, top_generators=2)
    tree = M.Tree(instance(), seed=0, widening_batch=2, proposal_provider=provider)
    expand(tree)
    assert tree.root.actions
    assert tree.root.catalogue is not None
    # The exact catalogue remains attached even when factorized proposals open
    # the first edges, so later widening can recover any proposal miss.
    for _ in range(3):
        expand(tree)
    assert tree.root.catalogue_cursor > 0
