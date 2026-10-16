"""Experimental exact-reference-corridor prior and verified root bridge extraction.

This is a TRAIN-only search component. It emits no training labels by itself.
"""

import math
from collections import Counter

import numpy as np

from rf_knots import mcts_loop as M
from rf_knots.cost_mcts_rollout import advance
from rf_knots.evidence import UnknotWitness


def reference_actions_states(instance, witness):
    """Re-encode a verified semantic reference in the *actual* search spec."""
    witness.verify()
    if (witness.start.word, witness.start.strands) != (instance.word, instance.strands):
        raise ValueError("reference start differs from TRAIN root")
    actions = tuple(step.action.to_flat(instance.spec) for step in witness.steps)
    states = [M.SearchState(instance.word, instance.strands, 0, 0)]
    for action in actions:
        states.append(advance(instance, states[-1], action))
    if states[-1].key != M.SOLVED:
        raise ValueError("reference did not terminate in search envelope")
    return actions, states


def raw_distance(state, target):
    """Cheap deterministic guidance only; never a value or solve probability."""
    return (4 * abs(state.strands - target.strands)
            + abs(len(state.word) - len(target.word))
            + sum(a != b for a, b in zip(state.word, target.word)))


class ReferenceCorridorPrior(M.Evaluator):
    """Guide MCTS expansions toward future exact reference states.

    The known first action is suppressed at the root so the experiment probes
    different legal first actions; it is not removed from the action pool.
    """

    def __init__(self, instance, reference_actions, reference_states,
                 *, temperature=2.0, suppress_first=6.0):
        if temperature <= 0 or suppress_first < 0 or not math.isfinite(temperature):
            raise ValueError("invalid reference prior")
        self.instance = instance
        self.first = reference_actions[0]
        self.targets = tuple(reference_states[i]
                             for i in range(2, min(len(reference_states), 34), 2))
        if not self.targets:
            raise ValueError("reference too short for nontrivial join")
        self.temperature = temperature
        self.suppress_first = suppress_first
        self.requests = 0
        self.action_scores = 0

    def evaluate(self, requests):
        out = []
        for instance, state, actions in requests:
            if instance is not self.instance:
                raise ValueError("corridor prior handles one pinned instance")
            logits = []
            for action in actions:
                child = advance(instance, state, action)
                distance = min(raw_distance(child, target) for target in self.targets)
                logit = -distance / self.temperature
                if state.used_moves == 0 and action == self.first:
                    logit -= self.suppress_first
                logits.append(logit)
            if logits:
                shifted = np.asarray(logits, dtype=np.float64) - max(logits)
                prior = np.exp(np.maximum(shifted, -690.0))
                prior /= prior.sum()
            else:
                prior = np.zeros(0, dtype=np.float64)
            out.append(M.Evaluation(prior, 0.0))
            self.action_scores += len(actions)
        self.requests += len(requests)
        return out


def completed_bridges(instance, reference_actions, reference_states, episode):
    """Return replay-verified, non-loop root forks found within one MCTS episode."""
    actions = tuple(episode["tree_prefix"] + episode["continuation"]["actions"])
    state = M.SearchState(instance.word, instance.strands, 0, 0)
    visited = {state.key}
    by_key = {}
    for index in range(2, len(reference_states)):
        by_key.setdefault(reference_states[index].key, []).append(index)
    counts = Counter()
    found = []
    first_successor_key = None
    for depth, action in enumerate(actions, 1):
        state = advance(instance, state, action)
        if depth == 1:
            first_successor_key = state.key
        if state.key in visited:
            counts["repeated_prefix_state"] += 1
            break
        visited.add(state.key)
        for index in by_key.get(state.key, ()):
            counts["exact_state"] += 1
            if actions[0] == reference_actions[0]:
                counts["same_first_action"] += 1
                continue
            if first_successor_key == reference_states[1].key:
                counts["same_first_successor"] += 1
                continue
            suffix = reference_actions[index:]
            if (state.used_cc + reference_states[-1].used_cc
                    - reference_states[index].used_cc > instance.cap_crossing_changes
                    or state.used_moves + len(suffix) > instance.cap_moves):
                counts["resource_cap"] += 1
                continue
            try:
                witness = UnknotWitness.from_actions(
                    instance.word, instance.strands, instance.spec,
                    actions[:depth] + suffix)
                witness.verify()
            except ValueError:
                counts["invalid_splice"] += 1
                continue
            if (witness.crossing_changes > instance.cap_crossing_changes
                    or witness.moves > instance.cap_moves):
                counts["resource_cap"] += 1
                continue
            cost = instance.cc_weight * witness.crossing_changes + witness.moves
            reference_cost = (instance.cc_weight * reference_states[-1].used_cc
                              + reference_states[-1].used_moves)
            counts["completed_root_fork"] += 1
            if cost == reference_cost:
                counts["equal_cost"] += 1
            else:
                counts["strict_cost_pair"] += 1
            found.append({"prefix_depth": depth, "reference_index": index,
                          "cost": cost, "reference_cost": reference_cost,
                          "strict_cost_pair": cost != reference_cost,
                          "witness": witness.to_dict()})
    return dict(counts), found
