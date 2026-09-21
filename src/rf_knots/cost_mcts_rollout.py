"""Budgeted policy continuations for cost MCTS, with replayed root-relative returns.

Rollouts are observations under one frozen policy and resource contract, not
proofs of solvability/unsolvability. No value-head predictions or training labels
are consumed here. The frozen terminal-only cost tree remains unchanged.
"""

from dataclasses import dataclass

import numpy as np

from rf_knots import mcts_loop as M
from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.anytime_cost_mcts import AnytimeCostTree
from rf_knots.evidence import UnknotWitness
from rf_knots.site_proposals import proposal_candidates


@dataclass(frozen=True)
class PolicyContinuation:
    start: M.SearchState
    end: M.SearchState
    actions: tuple[int, ...]
    probabilities: tuple[float, ...]
    stop_reason: str
    scored_actions: int


def advance(instance, state, action):
    if state.used_moves >= instance.cap_moves or not reference.is_legal(
        instance.spec,
        state.word,
        state.strands,
        action,
        state.used_cc < instance.cap_crossing_changes,
    ):
        raise ValueError("continuation action illegal or outside remaining resources")
    word, strands = reference.apply(instance.spec, state.word, state.strands, action)
    return M.SearchState(
        word,
        strands,
        state.used_cc + int(instance.spec.decode(action)[0] == CROSSING_CHANGE),
        state.used_moves + 1,
    )


class BatchedPolicyContinuation:
    """Sample full policy over O(L) ordinary actions plus bounded insertions.

    Exact repeated states are omitted within each continuation (nonnegative costs).
    The insertion pool is incomplete: its exhaustion is NOT a dead-state proof.
    Every action is applied in the root ActionSpec and on the raw current word.
    """

    def __init__(self, evaluator, *, horizon, insertion_candidates=4):
        if horizon < 0 or insertion_candidates < 1:
            raise ValueError("invalid continuation horizon or insertion pool")
        self.evaluator = evaluator
        self.horizon = horizon
        self.insertion_candidates = insertion_candidates
        self.state_evaluations = 0
        self.action_scores = 0

    def run(self, requests, *, seeds):
        if len(requests) != len(seeds):
            raise ValueError("one continuation seed required per request")
        states = [state for _, state in requests]
        for instance, state in requests:
            if not (
                0 <= state.used_cc <= instance.cap_crossing_changes
                and 0 <= state.used_moves <= instance.cap_moves
            ):
                raise ValueError("continuation starts outside resources")
        rngs = [np.random.default_rng(seed) for seed in seeds]
        visited = [{state.key} for state in states]
        paths, probabilities = [[] for _ in requests], [[] for _ in requests]
        reasons, counts = [None for _ in requests], [0 for _ in requests]
        for step in range(self.horizon + 1):
            active, evaluations = [], []
            for i, (instance, _) in enumerate(requests):
                if reasons[i] is not None:
                    continue
                state = states[i]
                if state.key == M.SOLVED:
                    reasons[i] = "solved"
                elif state.used_moves >= instance.cap_moves:
                    reasons[i] = "move_cap"
                elif step >= self.horizon:
                    reasons[i] = "horizon"
                else:
                    candidates = proposal_candidates(instance, state, self.insertion_candidates)
                    pairs = [(a, advance(instance, state, a)) for a in candidates]
                    pairs = [(a, s) for a, s in pairs if s.key not in visited[i]]
                    if not pairs:
                        reasons[i] = "candidate_pool_exhausted"
                        continue
                    active.append((i, pairs))
                    evaluations.append((instance, state, tuple(a for a, _ in pairs)))
            if not active:
                break
            results = self.evaluator.evaluate(evaluations)
            if len(results) != len(active):
                raise ValueError("continuation evaluator result count differs")
            for (i, pairs), evaluation in zip(active, results, strict=True):
                prior = np.asarray(evaluation.priors, dtype=np.float64)
                if (
                    prior.shape != (len(pairs),)
                    or not np.isfinite(prior).all()
                    or (prior < 0).any()
                    or not 0 < prior.sum() < np.inf
                ):
                    raise ValueError("invalid continuation priors")
                prior = prior / prior.sum()
                chosen = int(rngs[i].choice(len(pairs), p=prior))
                action, child = pairs[chosen]
                paths[i].append(action)
                probabilities[i].append(float(prior[chosen]))
                states[i] = child
                visited[i].add(child.key)
                counts[i] += len(pairs)
            self.state_evaluations += len(active)
            self.action_scores += sum(len(pairs) for _, pairs in active)
        return [
            PolicyContinuation(start, end, tuple(path), tuple(probs), reason, count)
            for (_, start), end, path, probs, reason, count in zip(
                requests, states, paths, probabilities, reasons, counts, strict=True
            )
        ]


class ContinuationCostTree(AnytimeCostTree):
    """One MCTS observation per expansion, even when a rollout also succeeds.

    The continuation return participates in the SAME ancestor backup as immediate
    terminal children. It does not add a second visit or mark a nonterminal edge
    dead. A rollout never enters the tree's transposition table.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._continuation_backup = None
        self.continuations = []
        self.maximum_tree_depth = 0
        self.maximum_observed_depth = 0

    def _backup(self, path, value):
        context = self._continuation_backup
        if context is not None and path is context[0]:
            value = max(value, context[1])
        super()._backup(path, value)

    def apply(self, pending, evaluation, continuation=None):
        witness, reward = None, 0.0
        if continuation is not None:
            if continuation.start != pending.state:
                raise ValueError("continuation does not start at the selected leaf")
            if len(continuation.actions) != len(continuation.probabilities):
                raise ValueError("continuation probability coverage differs")
            if any(not 0 < p <= 1 for p in continuation.probabilities):
                raise ValueError("invalid chosen-action probability")
            state = pending.state
            for action in continuation.actions:
                state = advance(self.instance, state, action)
            if state != continuation.end:
                raise ValueError("continuation endpoint failed replay")
            if (state.key == M.SOLVED) != (continuation.stop_reason == "solved"):
                raise ValueError("continuation terminal label differs from replay")
            if state.key == M.SOLVED:
                prefix = tuple(node.actions[index] for node, index in pending.path)
                witness = UnknotWitness.from_actions(
                    self.instance.word,
                    self.instance.strands,
                    self.spec,
                    prefix + continuation.actions,
                )
                witness.verify()
                if witness.moves != state.used_moves or witness.crossing_changes != state.used_cc:
                    raise ValueError("continuation resource counters differ from root replay")
                reward = self.utility.solved_return(witness.crossing_changes, witness.moves)
        # The base implementation backs up once per ancestor, including when it
        # discovers several terminal children during this same expansion.
        self._continuation_backup = (pending.path, reward)
        try:
            super().apply(pending, evaluation)
        finally:
            self._continuation_backup = None
        self.maximum_tree_depth = max(self.maximum_tree_depth, pending.state.used_moves)
        observed = max([pending.state.used_moves] + [s.used_moves for s in pending.children])
        if continuation is not None:
            self.maximum_observed_depth = max(
                self.maximum_observed_depth, observed, continuation.end.used_moves
            )
            self.continuations.append(
                (
                    self.simulations,
                    tuple(node.actions[index] for node, index in pending.path),
                    continuation,
                )
            )
        else:
            self.maximum_observed_depth = max(self.maximum_observed_depth, observed)
        if witness is not None:
            cost = self.utility.cost(witness.crossing_changes, witness.moves)
            if cost < self.best_cost:
                self.best_cost, self.best_witness = cost, witness
                self.solved_path = tuple(step.action.to_flat(self.spec) for step in witness.steps)
                self.incumbents.append(
                    {
                        "simulation": self.simulations,
                        "cost": cost,
                        "cc": witness.crossing_changes,
                        "moves": witness.moves,
                        "source": "policy_continuation",
                    }
                )


def run_continuation_cost_searches(
    instances, evaluator, continuation_policy, *, simulations, seed=0, widening_batch=4, c_puct=1.5
):
    if simulations < 0:
        raise ValueError("negative simulation budget")
    trees = [
        ContinuationCostTree(
            i,
            seed=seed * 9973 + n,
            budget=simulations,
            widening_batch=widening_batch,
            c_puct=c_puct,
        )
        for n, i in enumerate(instances)
    ]
    for iteration in range(simulations):
        pending = [p for t in trees if (p := t.select_leaf()) is not None]
        if not pending:
            if all(t.finished or t.simulations >= simulations for t in trees):
                break
            continue
        for p in pending:
            p.tree.prepare_pending(p)
        requested = [p for p in pending if p.state.key != M.SOLVED and p.actions]
        values = evaluator.evaluate([(p.tree.instance, p.state, p.actions) for p in requested])
        if len(values) != len(requested):
            raise ValueError("tree evaluator result count differs")
        by_id = {id(p): v for p, v in zip(requested, values, strict=True)}
        # Widening only changes the action pool; don't repeatedly roll out the
        # same node just because more candidates are being opened there.
        starts = [p for p in pending if not p.widening and p.state.key != M.SOLVED]
        results = continuation_policy.run(
            [(p.tree.instance, p.state) for p in starts],
            seeds=[p.tree.seed * (simulations + 1) + iteration for p in starts],
        )
        if len(results) != len(starts):
            raise ValueError("continuation result count differs")
        continuations = {id(p): r for p, r in zip(starts, results, strict=True)}
        for p in pending:
            p.tree.apply(p, by_id.get(id(p)), continuations.get(id(p)))
    return trees
