"""Small exact-state diamonds and rejected fake root forks."""

from rf_knots import mcts_loop as M
from rf_knots.actions import DESTABILIZE, REDUCE
from rf_knots.evidence import UnknotWitness
from rf_knots.cost_mcts_rollout import (
    BatchedPolicyContinuation, run_continuation_cost_searches,
)

from reference_guided_bridge_v1 import (
    ReferenceCorridorPrior, completed_bridges, reference_actions_states,
)


def diamond():
    instance = M.Instance("diamond", (1, -1, 1, 1, -1), 2, 0, 5)
    spec = instance.spec
    reference = UnknotWitness.from_actions(
        instance.word, instance.strands, spec,
        (spec.encode(REDUCE, 0), spec.encode(REDUCE, 1),
         spec.encode(DESTABILIZE)))
    return instance, *reference_actions_states(instance, reference)


def episode(*actions):
    return {"tree_prefix": [], "continuation": {"actions": list(actions)}}


def test_nontrivial_exact_state_bridge_is_fully_replayed():
    instance, reference, states = diamond()
    other = (instance.spec.encode(REDUCE, 3), instance.spec.encode(REDUCE, 0))
    assert states[1].key != M.SearchState((1, -1, 1), 2, 0, 1).key
    counts, found = completed_bridges(instance, reference, states, episode(*other))
    assert counts["completed_root_fork"] == 1
    assert counts["equal_cost"] == 1
    assert counts.get("strict_cost_pair", 0) == 0
    assert len(found) == 1
    UnknotWitness.from_dict(found[0]["witness"]).verify()
    assert found[0]["reference_index"] == 2


def test_same_successor_is_not_a_root_fork():
    instance, reference, states = diamond()
    # Different reduction site but the same exact first successor as reference.
    same = (instance.spec.encode(REDUCE, 1), instance.spec.encode(REDUCE, 1))
    counts, found = completed_bridges(instance, reference, states, episode(*same))
    assert not found
    assert counts.get("strict_cost_pair", 0) == 0


def test_reference_prior_suppresses_but_does_not_remove_first_action():
    instance, reference, states = diamond()
    prior = ReferenceCorridorPrior(instance, reference, states)
    root = M.SearchState(instance.word, instance.strands, 0, 0)
    actions = (reference[0], instance.spec.encode(REDUCE, 3))
    result = prior.evaluate([(instance, root, actions)])[0]
    assert 0 < result.priors[0] < result.priors[1] < 1


def test_guided_prior_runs_with_existing_cost_mcts_kernel():
    instance, reference, states = diamond()
    prior = ReferenceCorridorPrior(instance, reference, states)
    policy = BatchedPolicyContinuation(
        M.UniformEvaluator(0.0), horizon=instance.cap_moves,
        insertion_candidates=1)
    tree = run_continuation_cost_searches(
        [instance], prior, policy, simulations=3, seed=19,
        widening_batch=2, c_puct=1.5)[0]
    assert tree.simulations == 3
    assert prior.action_scores > 0
