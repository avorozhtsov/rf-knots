"""Embedding distances order opened legal actions; never label solvability."""

import math

import numpy as np

from rf_knots import mcts_loop as M
from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE


class EmbeddingDistancePrior(M.Evaluator):
    """P(a|s) proportional to exp(-d(child(a), unknot)/temperature).

    The heuristic may cache normalized global views, but actions are applied
    to the actual raw state using its own ActionSpec. Only supplied opened
    actions are scored. Returned value is zero; it is not p(solve).
    """

    def __init__(self, heuristic, *, temperature=1.0):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        self.heuristic = heuristic
        self.temperature = temperature
        self.requests = 0
        self.action_scores = 0

    def evaluate(self, requests):
        children, sizes = [], []
        for instance, state, actions in requests:
            sizes.append(len(actions))
            for action in actions:
                if state.used_moves >= instance.cap_moves or not reference.is_legal(
                    instance.spec, state.word, state.strands, action,
                    state.used_cc < instance.cap_crossing_changes,
                ):
                    raise ValueError("prior requested for illegal or out-of-budget action")
                word, strands = reference.apply(instance.spec, state.word, state.strands, action)
                children.append(M.SearchState(
                    word, strands,
                    state.used_cc + int(instance.spec.decode(action)[0] == CROSSING_CHANGE),
                    state.used_moves + 1,
                ))
        distances = np.asarray(self.heuristic(children) if children else [], dtype=np.float64)
        if (distances.shape != (len(children),) or not np.isfinite(distances).all()
                or (distances < 0).any()):
            raise ValueError("distance heuristic must return finite nonnegative scores")
        out, cursor = [], 0
        for count in sizes:
            scores = distances[cursor:cursor + count]
            if count:
                # Clipping only guards floating-point underflow; no edge gets
                # exactly zero prior due to a large learned distance gap.
                with np.errstate(over="ignore"):
                    logits = -(scores - scores.min()) / self.temperature
                prior = np.exp(np.maximum(logits, -690.0))
                prior /= prior.sum()
            else:
                prior = np.zeros(0)
            out.append(M.Evaluation(prior, 0.0))
            cursor += count
        self.requests += len(requests)
        self.action_scores += len(children)
        return out
