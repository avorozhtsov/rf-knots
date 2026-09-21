"""Experimental PUCT with verified incumbents and root-relative cost returns.

Separate from the legacy first-success Tree. Nonterminal leaf values are NOT
used: existing success-probability heads do not predict this utility. Zero
backup means no solution observed in this simulation, not an RL negative label.
"""

import math
from dataclasses import dataclass

import numpy as np

from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness


@dataclass(frozen=True)
class RootCostUtility:
    cc_weight: float
    cc_cap: int
    move_cap: int

    def __post_init__(self):
        if (not math.isfinite(self.cc_weight) or self.cc_weight <= 0
                or self.cc_cap < 0 or self.move_cap < 0):
            raise ValueError("require positive finite weight and nonnegative caps")

    def cost(self, cc, moves):
        if not (0 <= cc <= self.cc_cap and 0 <= moves <= self.move_cap):
            raise ValueError("solution exceeds root resource caps")
        return self.cc_weight * cc + moves

    def solved_return(self, cc, moves):
        # Fixed per root, never re-normalized using the current incumbent.
        # Every feasible success has utility > 1/2; smaller L is strictly better.
        maximum = self.cc_weight * self.cc_cap + self.move_cap
        return 1.0 - self.cost(cc, moves) / (2.0 * (maximum + 1.0))


class AnytimeCostTree(M.Tree):
    """Progressively widened PUCT; consume budget beyond the first witness.

No learned-bound pruning, no training targets and no Gumbel mode. Priors may
come from an evaluator, but Evaluation.value is deliberately ignored.
"""

    def __init__(self, instance, *, seed, budget=1024, widening_batch=4,
                 c_puct=1.5, dirichlet_fraction=0.0, proposal_provider=None):
        if widening_batch <= 0:
            raise ValueError("anytime cost tree requires progressive widening")
        super().__init__(instance, seed=seed, budget=budget,
                         widening_batch=widening_batch, c_puct=c_puct,
                         dirichlet_fraction=dirichlet_fraction,
                         proposal_provider=proposal_provider)
        self.utility = RootCostUtility(instance.cc_weight,
                                       instance.cap_crossing_changes, instance.cap_moves)
        self.incumbents = []
        self.best_cost = math.inf
        self.best_witness = None
        self.exhausted = False
        if self.root.state.key == M.SOLVED:
            self._record_solution(self.root)
            self.finished = True

    def _unopened(self, node):
        return ((node.catalogue is not None and node.catalogue_cursor < len(node.catalogue))
                or (self.proposal_provider is not None and not node.proposal_exhausted))

    def select_leaf(self):
        if self.simulations >= self.budget:
            return None
        if (self.root.expanded and self.root.dead.all()
                and not self._unopened(self.root)):
            self.exhausted = True
            self.finished = True
            return None
        return super().select_leaf()

    def _record_solution(self, node):
        actions = []
        current = node
        while current.parent is not None:
            actions.append(current.parent.actions[current.parent_index])
            current = current.parent
        path = tuple(reversed(actions))
        witness = UnknotWitness.from_actions(
            self.instance.word, self.instance.strands, self.spec, path)
        witness.verify()
        cost = self.utility.cost(witness.crossing_changes, witness.moves)
        if cost < self.best_cost:
            self.best_cost, self.best_witness, self.solved_path = cost, witness, path
            self.incumbents.append({"simulation": self.simulations, "cost": cost,
                                    "cc": witness.crossing_changes, "moves": witness.moves})
        # This exact terminal edge is exhausted, not the whole tree.
        if node.parent is not None:
            node.parent.dead[node.parent_index] = True
        return self.utility.solved_return(witness.crossing_changes, witness.moves)

    def apply(self, pending, evaluation):
        node, index, state = pending.node, pending.child_index, pending.state
        if index >= 0:
            child = M.Node(state, node, index)
            node.children[index] = child
            self.nodes += 1
        else:
            child = node
        self.expansions += 1
        self.simulations += 1
        M.remember_cost(self.seen, state)
        if state.key == M.SOLVED:
            child.expanded = True
            self._backup(pending.path, self._record_solution(child))
            return

        old = len(child.actions) if pending.widening else 0
        child.actions, child.children_states = pending.actions, pending.children
        count = len(child.actions)
        child.children = (child.children if pending.widening else []) + [None] * (count - old)
        child.catalogue, child.catalogue_cursor = pending.catalogue, pending.catalogue_cursor
        child.proposal_exhausted = pending.proposal_exhausted
        if evaluation is None:
            prior = np.ones(count)
        else:
            prior = np.asarray(evaluation.priors, dtype=np.float64)
            if (prior.shape != (count,) or not np.isfinite(prior).all()
                    or (prior < 0).any() or (count and prior.sum() <= 0)):
                raise ValueError("invalid action priors")
        child.prior = prior / prior.sum() if count else np.zeros(0)
        child.visits = np.concatenate((child.visits[:old], np.zeros(count - old)))
        child.total = np.concatenate((child.total[:old], np.zeros(count - old)))
        child.dead = np.concatenate((child.dead[:old], np.zeros(count - old, dtype=bool)))
        child.expanded = True
        if count == 0 and child.parent is not None and not self._unopened(child):
            child.parent.dead[child.parent_index] = True
        if child is self.root and self.dirichlet_fraction > 0 and count:
            noise = self.rng.dirichlet(np.full(count, self.dirichlet_alpha))
            child.prior = ((1 - self.dirichlet_fraction) * child.prior
                           + self.dirichlet_fraction * noise)

        found = []
        # A widening update must not repeatedly back up old terminal edges.
        for position in range(old, count):
            candidate = pending.children[position]
            if candidate.key != M.SOLVED:
                continue
            terminal = M.Node(candidate, child, position)
            terminal.expanded = True
            child.children[position] = terminal
            self.nodes += 1
            reward = self._record_solution(terminal)
            # One local edge observation each; only one backup per ancestor.
            self._backup([(child, position)], reward)
            found.append(reward)
        self._backup(pending.path, max(found, default=0.0))


def run_anytime_cost_searches(instances, evaluator, *, simulations, seed=0,
                             widening_batch=4, c_puct=1.5, proposal_provider=None):
    """Batched prior inference, returning trees with replayed incumbents.

The evaluator's success-probability values are ignored. No per-state negative
labels or visit-based training samples are produced by this function.
"""
    if simulations < 0:
        raise ValueError("negative simulation budget")
    trees = [AnytimeCostTree(i, seed=seed * 9973 + n, budget=simulations,
                            widening_batch=widening_batch, c_puct=c_puct,
                            proposal_provider=proposal_provider)
             for n, i in enumerate(instances)]
    for _ in range(simulations):
        pending = [p for t in trees if (p := t.select_leaf()) is not None]
        if not pending:
            if all(t.finished or t.simulations >= simulations for t in trees):
                break
            continue
        if callable(getattr(proposal_provider, "prepare_batch", None)):
            requests = [r for p in pending if (r := p.tree.proposal_request(p)) is not None]
            proposal_provider.prepare_batch(requests)
        for p in pending:
            p.tree.prepare_pending(p)
        requested = [p for p in pending if p.state.key != M.SOLVED and p.actions]
        values = evaluator.evaluate([(p.tree.instance, p.state, p.actions) for p in requested])
        if len(values) != len(requested):
            raise ValueError("evaluator result count differs")
        by_id = {id(p): v for p, v in zip(requested, values, strict=True)}
        for p in pending:
            p.tree.apply(p, by_id.get(id(p)))
    return trees
