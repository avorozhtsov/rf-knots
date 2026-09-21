"""Anytime greedy search measuring a knot-distance channel directly.

This is NOT MCTS and not an optimality claim. Learned remaining u orders the
frontier; actual L_lambda selects incumbents. Only replay, resource caps, Pareto
dominance and an admissible move-count bound prune states, never a learned score.
"""

import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rf_knots.evidence import UnknotWitness
from rf_knots.mcts_loop import SOLVED, Instance, SearchState, remember_cost, successors_of


@dataclass
class CostSearchResult:
    witness: UnknotWitness | None
    expansions: int
    generated: int
    incumbents: list[dict]
    exhausted: bool


def greedy_cost_search(
    instance: Instance,
    heuristic: Callable[[Sequence[SearchState]], Sequence[float]],
    *,
    expansions: int,
    cc_weight: int,
    layout_tiebreak: bool = False,
) -> CostSearchResult:
    """Keep searching after the first success, returning the least actual cost.

    The exact reference moves delete at most two letters in one step. Hence
    ceil(length/2) is a valid remaining semantic-move lower bound (including the
    final destabilisation); it is not a claim that such a route exists.
    """
    if expansions < 0 or cc_weight <= 0:
        raise ValueError("invalid cost search budget")
    start = SearchState(instance.word, instance.strands, 0, 0)
    if start.key == SOLVED:
        return CostSearchResult(
            UnknotWitness.from_actions((), 1, instance.spec, ()), 0, 0, [], True
        )
    seen = {start.key: [(0, 0)]}
    frontier = [(0.0, 0, 0, 0, start, ())]
    counter, expanded, generated = 0, 0, 0
    best_cost, best = math.inf, None
    history = []
    while frontier and expanded < expansions:
        _, _, _, _, state, path = heapq.heappop(frontier)
        if (state.used_cc, state.used_moves) not in seen[state.key]:
            continue
        if cc_weight * state.used_cc + state.used_moves + (len(state.word) + 1) // 2 >= best_cost:
            continue
        expanded += 1
        children, paths = [], []
        for action, child in successors_of(state, instance, instance.spec):
            generated += 1
            cost = cc_weight * child.used_cc + child.used_moves
            if cost + (len(child.word) + 1) // 2 >= best_cost:
                continue
            child_path = path + (action,)
            if child.key == SOLVED:
                witness = UnknotWitness.from_actions(
                    instance.word, instance.strands, instance.spec, child_path
                )
                witness.verify()
                best, best_cost = witness, cost
                history.append(
                    {
                        "expansion": expanded,
                        "cc": witness.crossing_changes,
                        "moves": witness.moves,
                        "L10": 10 * witness.crossing_changes + witness.moves,
                        "L1000": 1000 * witness.crossing_changes + witness.moves,
                    }
                )
                continue
            if remember_cost(seen, child):
                children.append(child)
                paths.append(child_path)
        scores = list(heuristic(children))
        if len(scores) != len(children) or any(not math.isfinite(s) or s < 0 for s in scores):
            raise ValueError("heuristic must give a finite nonnegative score for every child")
        for score, child, child_path in zip(scores, children, paths, strict=True):
            counter += 1
            cheap_cost = cc_weight * child.used_cc + child.used_moves + (len(child.word) + 1) // 2
            # Knot invariants cannot guide zero-CC simplification on a plateau.
            # This optional layout signal is only a tie-break, never a bound.
            layout = len(child.word) if layout_tiebreak else 0
            heapq.heappush(frontier, (float(score), layout, cheap_cost, counter, child, child_path))
    return CostSearchResult(best, expanded, generated, history, not frontier)
