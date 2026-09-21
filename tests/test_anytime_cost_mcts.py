import numpy as np
import pytest

from rf_knots import mcts_loop as M
from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, INSERT, REDUCE
from rf_knots.anytime_cost_mcts import AnytimeCostTree, RootCostUtility, run_anytime_cost_searches


def problem(**kwargs):
    return M.Instance("trefoil", (1, 1, 1), 2, 1, cap_moves=8, **kwargs)


@pytest.mark.parametrize("weight", [10, 1000])
def test_fixed_utility_orders_costs_and_separates_success(weight):
    u = RootCostUtility(weight, 2, 80)
    assert 0.5 < u.solved_return(2, 80) < u.solved_return(2, 79) <= 1
    assert u.solved_return(0, 0) == 1
    with pytest.raises(ValueError, match="caps"):
        u.solved_return(3, 1)
    with pytest.raises(ValueError):
        RootCostUtility(float("nan"), 1, 8)


def attach_path(tree, actions):
    node = tree.root
    for action in actions:
        s = node.state
        word, strands = reference.apply(tree.spec, s.word, s.strands, action)
        node.actions = (*node.actions, action)
        state = M.SearchState(word, strands,
                              s.used_cc + int(tree.spec.decode(action)[0] == CROSSING_CHANGE),
                              s.used_moves + 1)
        node.children_states = (*node.children_states, state)
        child = M.Node(state, node, len(node.actions) - 1)
        node.children.append(child)
        node.dead = np.append(node.dead, False)
        node = child
    return node


def test_later_cheaper_verified_witness_replaces_first_without_stopping():
    t = AnytimeCostTree(problem(), seed=0)
    s = t.spec
    short = [s.encode(CROSSING_CHANGE, 0), s.encode(REDUCE, 0), s.encode(DESTABILIZE)]
    long = [s.encode(INSERT, 0, 1, 1), s.encode(REDUCE, 0)] + short
    r_long = t._record_solution(attach_path(t, long))
    assert not t.finished and t.best_witness.moves == 5
    r_short = t._record_solution(attach_path(t, short))
    assert not t.finished and t.best_witness.moves == 3 and r_short > r_long
    assert [r["cost"] for r in t.incumbents] == [1005, 1003]
    t._record_solution(attach_path(t, long))
    assert len(t.incumbents) == 2 and t.best_witness.moves == 3
    t.best_witness.verify()


def test_real_search_continues_past_first_solution_and_ignores_probability_head():
    trees = [run_anytime_cost_searches([problem()], M.UniformEvaluator(value=v),
                                      simulations=256)[0] for v in (0, 1, float("nan"))]
    for t in trees:
        assert t.best_witness is not None
        t.best_witness.verify()
        assert t.simulations > t.incumbents[0]["simulation"]
        assert t.simulations <= 256
    assert trees[0].solved_path == trees[1].solved_path == trees[2].solved_path
    np.testing.assert_equal(trees[0].root.total, trees[2].root.total)


def test_zero_budget_and_initially_solved_are_distinct():
    t = run_anytime_cost_searches([problem()], M.UniformEvaluator(), simulations=0)[0]
    assert t.best_witness is None and not t.exhausted and t.simulations == 0
    empty = M.Instance("empty", (), 1, 0)
    t = run_anytime_cost_searches([empty], M.UniformEvaluator(), simulations=0)[0]
    assert t.best_witness.moves == 0 and t.solved_path == () and t.best_cost == 0


def test_widening_never_rebacks_old_terminal_edge():
    t = AnytimeCostTree(M.Instance("one", (1,), 2, 0, cap_moves=5), seed=0, widening_batch=1)
    for _ in range(50):
        p = t.select_leaf()
        if p is None:
            continue
        t.prepare_pending(p)
        t.apply(p, None)
    root = t.root
    terminal = [i for i, s in enumerate(root.children_states) if s.key == M.SOLVED]
    assert terminal
    assert all(root.visits[i] == 1 and root.dead[i] for i in terminal)


def test_lazy_search_does_not_eagerly_enumerate_successors(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("eager enumeration")
    monkeypatch.setattr(reference, "successors", forbidden)
    t = run_anytime_cost_searches([problem()], M.UniformEvaluator(), simulations=256)[0]
    assert t.best_witness is not None


def test_invalid_prior_fails_instead_of_silent_repair():
    t = AnytimeCostTree(problem(), seed=0)
    p = t.select_leaf()
    t.prepare_pending(p)
    with pytest.raises(ValueError, match="priors"):
        t.apply(p, M.Evaluation(np.full(len(p.actions), -1), 0.5))


def test_evaluator_coverage_is_checked():
    class Empty(M.Evaluator):
        def evaluate(self, requests):
            return []
    with pytest.raises(ValueError, match="count"):
        run_anytime_cost_searches([problem()], Empty(), simulations=1)


def test_exhaustion_is_not_inferred_from_dead_prefix():
    t = AnytimeCostTree(problem(), seed=0, widening_batch=1)
    p = t.select_leaf()
    t.prepare_pending(p)
    t.apply(p, None)
    t.root.dead[:] = True
    assert t.select_leaf().widening
    assert not t.finished and not t.exhausted
