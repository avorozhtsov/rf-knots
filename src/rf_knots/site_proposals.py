"""Whole-word proposals from an existing site head; no new trained parameters.

Score all O(L) ordinary actions plus O(k) arithmetic insertion samples, without
materialising successor words or allocating the insertion Cartesian product.
The tree retains its exact exhaustive catalogue as a fallback.
"""

from rf_knots.lazy_actions import LazyActionCatalogue
from rf_knots.mcts_loop import NetworkEvaluator, SOLVED


def proposal_candidates(instance, state, limit):
    if limit < 1 or state.key == SOLVED or state.used_moves >= instance.cap_moves:
        return ()
    catalogue = LazyActionCatalogue(instance.spec, state.word, state.strands,
                                    state.used_cc < instance.cap_crossing_changes)
    actions = list(catalogue.ordinary)
    count = min(limit, catalogue.insertions)
    common = min(len(catalogue.ordinary), catalogue.insertions)
    for j in range(count):
        insertion = j * catalogue.insertions // count
        index = 2 * insertion + 1 if insertion < common else insertion + common
        actions.append(catalogue.action_at(index))
    return tuple(actions)


class SiteProposalProvider:
    """Batch-scored, per-round bounded ranking cache for frozen site policies."""

    def __init__(self, net, device="cpu", *, micro_batch=64):
        self.evaluator = NetworkEvaluator(net, device, micro_batch=micro_batch)
        self.prepared = {}
        self.logical_requests = 0
        self.action_scores = 0

    @staticmethod
    def key(instance, state, limit):
        return (state.word, state.strands, state.used_moves, state.used_cc,
                instance.cap_moves, instance.cap_crossing_changes,
                instance.spec.max_len, instance.spec.max_strands, instance.cc_weight, limit)

    def prepare_batch(self, requests):
        # No cross-round state/embedding cache, keeping memory bounded by the
        # current cohort. Identical requests across seeds share one inference.
        self.prepared = {}
        self.logical_requests += len(requests)
        pending, keys, limits, seen = [], [], [], set()
        for instance, state, limit in requests:
            key = self.key(instance, state, limit)
            if key in seen:
                continue
            seen.add(key)
            actions = proposal_candidates(instance, state, limit)
            if not actions:
                self.prepared[key] = ()
                continue
            pending.append((instance, state, actions))
            keys.append(key)
            limits.append(limit)
            self.action_scores += len(actions)
        for key, request, limit, evaluation in zip(
            keys, pending, limits, self.evaluator.evaluate(pending), strict=True
        ):
            actions = request[2]
            order = sorted(range(len(actions)), key=lambda i: (-evaluation.priors[i], actions[i]))
            self.prepared[key] = tuple(actions[i] for i in order[:limit])

    def __call__(self, instance, state, limit):
        key = self.key(instance, state, limit)
        if key not in self.prepared:
            self.prepare_batch([(instance, state, limit)])
        return self.prepared[key]

    def statistics(self):
        return {"state_evaluations": self.evaluator.calls,
                "forward_batches": self.evaluator.forwards,
                "action_scores": self.action_scores,
                "logical_requests": self.logical_requests}
