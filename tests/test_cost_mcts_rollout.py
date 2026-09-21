from dataclasses import replace

import numpy as np
import pytest

from rf_knots import mcts_loop as M
from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, REDUCE
from rf_knots.cost_mcts_rollout import (
    BatchedPolicyContinuation,
    ContinuationCostTree,
    advance,
    run_continuation_cost_searches,
)


class SimplePolicy(M.Evaluator):
    """Handwritten engineering control, not a learned-policy performance test."""

    def evaluate(self, requests):
        values = []
        order = {REDUCE: 0, DESTABILIZE: 1, CROSSING_CHANGE: 2}
        for instance, _, actions in requests:
            chosen = min(
                range(len(actions)),
                key=lambda j: (order.get(instance.spec.decode(actions[j])[0], 3), actions[j]),
            )
            prior = np.zeros(len(actions))
            prior[chosen] = 1
            values.append(M.Evaluation(prior, float("nan")))
        return values


def problem(**kwargs):
    return M.Instance("trefoil", (1, 1, 1), 2, 1, cap_moves=12, **kwargs)


def start(instance):
    return M.SearchState(instance.word, instance.strands, 0, 0)


def test_batched_full_trajectories_preserve_resources_and_raw_coordinates():
    instances = [problem(), M.Instance("long", (1, -1) * 40 + (1,), 2, 0, cap_moves=42)]
    policy = BatchedPolicyContinuation(SimplePolicy(), horizon=50)
    results = policy.run([(i, start(i)) for i in instances], seeds=[17, 18])
    assert [len(r.actions) for r in results] == [3, 41]
    assert policy.state_evaluations == 44
    for instance, result in zip(instances, results, strict=True):
        assert result.stop_reason == "solved" and result.end.key == M.SOLVED
        state = result.start
        for action in result.actions:
            state = advance(instance, state, action)
        assert state == result.end
        assert all(p == 1 for p in result.probabilities)


@pytest.mark.parametrize(
    "horizon,cap,reason", [(0, 12, "horizon"), (2, 12, "horizon"), (12, 2, "move_cap")]
)
def test_truncated_rollout_is_not_a_dead_state_label(horizon, cap, reason):
    instance = replace(problem(), cap_moves=cap)
    r = BatchedPolicyContinuation(SimplePolicy(), horizon=horizon).run(
        [(instance, start(instance))], seeds=[3]
    )[0]
    assert r.stop_reason == reason and r.end.key != M.SOLVED
    assert len(r.actions) == min(horizon, cap)


def test_empty_root_solved_even_at_zero_horizon():
    i = M.Instance("empty", (), 1, 0, cap_moves=0)
    r = BatchedPolicyContinuation(SimplePolicy(), horizon=0).run([(i, start(i))], seeds=[0])[0]
    assert r.stop_reason == "solved" and not r.actions


def test_rollout_uses_remaining_not_reset_cc_and_move_budgets():
    i = replace(problem(), cap_moves=3)
    s = M.SearchState((-1, 1, 1), 2, 1, 1)
    r = BatchedPolicyContinuation(SimplePolicy(), horizon=20).run([(i, s)], seeds=[0])[0]
    assert r.stop_reason == "solved" and r.end.used_cc == 1 and r.end.used_moves == 3
    assert all(i.spec.decode(a)[0] != CROSSING_CHANGE for a in r.actions)


def test_real_mcts_continuation_reaches_beyond_tree_depth_and_keeps_searching():
    i = M.Instance("long", (1, -1) * 40 + (1,), 2, 0, cap_moves=44)
    t = run_continuation_cost_searches(
        [i],
        M.UniformEvaluator(),
        BatchedPolicyContinuation(SimplePolicy(), horizon=44),
        simulations=3,
        seed=17,
    )[0]
    assert t.best_witness is not None and t.best_witness.moves == 41
    t.best_witness.verify()
    assert t.simulations == 3 > t.incumbents[0]["simulation"]
    assert t.maximum_tree_depth < t.maximum_observed_depth
    assert t.maximum_observed_depth >= 41


def leaf_pending(t):
    for _ in range(40):
        p = t.select_leaf()
        assert p is not None
        t.prepare_pending(p)
        if p.path and not p.widening:
            return p
        t.apply(p, None)
    raise AssertionError("never reached a child")


def test_successful_continuation_backs_up_once_and_does_not_exhaust_leaf():
    t = ContinuationCostTree(problem(), seed=0)
    p = leaf_pending(t)
    r = BatchedPolicyContinuation(SimplePolicy(), horizon=12).run(
        [(t.instance, p.state)], seeds=[0]
    )[0]
    assert r.stop_reason == "solved"
    before = [(node.visits[index], node.total[index]) for node, index in p.path]
    t.apply(p, None, r)
    reward = t.utility.solved_return(r.end.used_cc, r.end.used_moves)
    for (node, index), (visits, total) in zip(p.path, before, strict=True):
        assert node.visits[index] == visits + 1
        assert node.total[index] == pytest.approx(total + reward)
        assert not node.dead[index]
    assert not t.finished
    t.best_witness.verify()
    assert t.best_witness.moves == r.end.used_moves
    assert t.best_witness.moves > len(r.actions)  # Count the tree prefix too.
    assert t._continuation_backup is None


@pytest.mark.parametrize("fault", ["start", "end", "label", "probability", "coverage"])
def test_invalid_continuation_rejected_before_tree_mutation(fault):
    t = ContinuationCostTree(problem(), seed=0)
    p = t.select_leaf()
    t.prepare_pending(p)
    r = BatchedPolicyContinuation(SimplePolicy(), horizon=12).run(
        [(t.instance, p.state)], seeds=[0]
    )[0]
    changes = {
        "start": {"start": replace(r.start, used_moves=1)},
        "end": {"end": replace(r.end, used_cc=0)},
        "label": {"stop_reason": "horizon"},
        "probability": {"probabilities": (float("nan"),) + r.probabilities[1:]},
        "coverage": {"probabilities": ()},
    }
    with pytest.raises(ValueError):
        t.apply(p, None, replace(r, **changes[fault]))
    assert t.simulations == 0 and not t.root.expanded and t.best_witness is None


def test_repeatable_sampling_and_no_eager_cartesian_enumeration(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("eager successors")

    monkeypatch.setattr(reference, "successors", forbidden)
    i = problem()
    policy = BatchedPolicyContinuation(M.UniformEvaluator(), horizon=10)
    a = policy.run([(i, start(i))], seeds=[391])
    b = policy.run([(i, start(i))], seeds=[391])
    assert a == b
    r = a[0]
    keys = {r.start.key}
    state = r.start
    for action in r.actions:
        state = advance(i, state, action)
        assert state.key not in keys
        keys.add(state.key)


def test_invalid_evaluator_coverage_and_priors_fail_closed():
    class Bad(M.Evaluator):
        def __init__(self, empty):
            self.empty = empty

        def evaluate(self, requests):
            return [] if self.empty else [M.Evaluation(np.array([float("nan")]), 0)]

    i = problem()
    for empty in (True, False):
        with pytest.raises(ValueError):
            BatchedPolicyContinuation(Bad(empty), horizon=1).run([(i, start(i))], seeds=[0])
    with pytest.raises(ValueError):
        BatchedPolicyContinuation(SimplePolicy(), horizon=-1)
    with pytest.raises(ValueError):
        BatchedPolicyContinuation(SimplePolicy(), horizon=1).run([(i, start(i))], seeds=[])


def test_zero_budget_runner_never_calls_rollout():
    policy = BatchedPolicyContinuation(SimplePolicy(), horizon=10)
    t = run_continuation_cost_searches([problem()], M.UniformEvaluator(), policy, simulations=0)[0]
    assert t.simulations == 0 and t.best_witness is None and not t.continuations
    assert policy.state_evaluations == 0
