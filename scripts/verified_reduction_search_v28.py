"""Separate v28 search adapter; every automatic move remains an elementary move.

Only REDUCE and DESTABILIZE are automatic. No normalization teleport, free moves,
CC change, learned pruning, or change to the frozen v2 comparison is permitted.
"""
import signal
import time
from dataclasses import asdict

import numpy as np
import torch

from rf_knots import reference, mcts_loop as M
from rf_knots.actions import REDUCE, DESTABILIZE
from rf_knots.cost_mcts_rollout import advance, ContinuationCostTree, BatchedPolicyContinuation
from rf_knots.site_proposals import proposal_candidates
from pretrained_raster_policy import (
    PriorOnlyResult, RawKnotDistance, RasterDistancePrior, RasterPolicyEvaluator,
)


def automatic_action(instance, state, actions):
    """Choose a replay-verified strictly shrinking zero-CC action, or None."""
    for kind in (REDUCE, DESTABILIZE):
        for action in sorted(actions):
            if instance.spec.decode(action)[0] != kind:
                continue
            if not reference.is_legal(instance.spec, state.word, state.strands, action, False):
                raise ValueError('automatic action was not independently legal')
            after = advance(instance, state, action)
            if not (after.used_cc == state.used_cc
                    and after.used_moves == state.used_moves + 1
                    and len(after.word) < len(state.word)
                    and after.strands <= state.strands):
                raise ValueError('automatic action did not strictly shrink/account resources')
            return action
    return None


def reduce_trace(instance, state):
    """Diagnostic/replay helper, not a free search-state preprocessing step."""
    actions = []
    while state.key != M.SOLVED and state.used_moves < instance.cap_moves:
        candidates = proposal_candidates(instance, state, 4)
        action = automatic_action(instance, state, candidates)
        if action is None:
            break
        actions.append(action)
        state = advance(instance, state, action)
    return state, actions


class ReductionEvaluator:
    """Bypass neural inference on mechanical moves, retaining canonical replay."""
    batched = True

    def __init__(self, fallback):
        self.fallback = fallback
        self.automatic_requests = 0
        self.neural_requests = 0

    def evaluate(self, requests):
        results = [None] * len(requests)
        pending, indices = [], []
        for i, (instance, state, actions) in enumerate(requests):
            action = automatic_action(instance, state, actions)
            if action is None:
                indices.append(i)
                pending.append((instance, state, actions))
            else:
                p = np.zeros(len(actions), dtype=np.float64)
                p[actions.index(action)] = 1.
                results[i] = PriorOnlyResult(p)
                self.automatic_requests += 1
        if pending:
            evaluated = self.fallback.evaluate(pending)
            if len(evaluated) != len(pending):
                raise ValueError('fallback coverage differs')
            self.neural_requests += len(pending)
            for i, result in zip(indices, evaluated, strict=True):
                results[i] = result
        return results


def search(instance, net, objective, seed, *, cpu_seconds=20., simulations=16):
    """Same canonical tree/continuation, with separately versioned auto policies.

This is a heuristic change, NOT a claim to preserve shortest-move optimality.
An interrupted simulation is discarded; completed incumbents survive.
"""
    from audit_cost_mcts_continuations import audit_episode
    if cpu_seconds <= 0 or simulations < 1:
        raise ValueError('positive resource budget required')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    distance = RawKnotDistance(net, objective, microbatch=2)
    policy = RasterPolicyEvaluator(net, microbatch=2)
    tree_prior = ReductionEvaluator(RasterDistancePrior(distance, temperature=1.))
    rollout_prior = ReductionEvaluator(policy)
    tree = ContinuationCostTree(instance, seed=seed * 9973, budget=simulations,
                                widening_batch=4, c_puct=1.5)
    rollout = BatchedPolicyContinuation(rollout_prior, horizon=instance.cap_moves,
                                       insertion_candidates=4)
    completed, incumbent, actual = [], None, 0
    stop, censored = 'simulation_ceiling', False
    cpu, wall = time.process_time(), time.monotonic()

    def timeout(*_):
        raise TimeoutError('v28 search CPU ceiling')

    previous = signal.signal(signal.SIGPROF, timeout)
    signal.setitimer(signal.ITIMER_PROF, cpu_seconds)
    try:
        with torch.no_grad():
            for iteration in range(simulations):
                pending = tree.select_leaf()
                if pending is None:
                    if tree.finished:
                        stop = 'tree_finished'
                        break
                    continue
                tree.prepare_pending(pending)
                values = (tree_prior.evaluate([(instance, pending.state, pending.actions)])
                          if pending.state.key != M.SOLVED and pending.actions else [])
                continuation = None
                if not pending.widening and pending.state.key != M.SOLVED:
                    continuation = rollout.run([(instance, pending.state)],
                        seeds=[tree.seed * (simulations + 1) + iteration])[0]
                tree.apply(pending, values[0] if values else None, continuation)
                completed = list(tree.continuations)
                incumbent, actual = tree.best_witness, tree.simulations
    except TimeoutError:
        stop, censored = 'CPU_budget', True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGPROF, previous)
    search_cpu, search_wall = time.process_time() - cpu, time.monotonic() - wall
    config = dict(horizon=instance.cap_moves, insertion_candidates=4)
    episodes = []
    for simulation, prefix, continuation in completed:
        # JSON-compatible data; normal auditor replays every elementary action.
        import json
        episode = json.loads(json.dumps(dict(simulation=simulation, tree_prefix=list(prefix),
                                             continuation=asdict(continuation))))
        audit_episode(instance, episode, config)
        episodes.append(episode)
    if incumbent:
        incumbent.verify()
    return dict(solved=incumbent is not None,
        witness=incumbent.to_dict() if incumbent else None,
        CC=incumbent.crossing_changes if incumbent else None,
        moves=incumbent.moves if incumbent else None,
        episodes=episodes, actual_simulations=actual, cpu_seconds=search_cpu,
        wall_seconds=search_wall, stop_reason=stop, cpu_censored=censored,
        automatic_requests=tree_prior.automatic_requests + rollout_prior.automatic_requests,
        neural_requests=tree_prior.neural_requests + rollout_prior.neural_requests,
        neural_evaluations=policy.calls + distance.evaluations)
