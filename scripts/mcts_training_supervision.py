"""Success-only supervision from verified alternatives; no MCTS-failure labels.

Callers must gate path ownership independently before passing witnesses here.
Reference-pool probabilities are NOT behaviour probabilities of MCTS paths.
"""

import math
from collections import defaultdict
from fractions import Fraction

from rf_knots import mcts_loop as M
from rf_knots.cost_mcts_rollout import advance
from rf_knots.site_proposals import proposal_candidates


def verified_success_traces(instance, witnesses):
    """Deduplicate exact paths; retain different successful orders and budgets."""
    if not math.isfinite(instance.cc_weight) or instance.cc_weight <= 0:
        raise ValueError("require positive finite CC weight")
    paths = {}
    for witness in witnesses:
        witness.verify()
        if witness.start.word != instance.word or witness.start.strands != instance.strands:
            raise ValueError("successful trajectory has a different root")
        if witness.cyclic_band_generators != instance.spec.cyclic_band_generators:
            raise ValueError("successful trajectory uses a different action alphabet")
        actions = tuple(s.action.to_flat(instance.spec) for s in witness.steps)
        state = M.SearchState(instance.word, instance.strands, 0, 0)
        rows = []
        for action in actions:
            if state.key == M.SOLVED:
                raise ValueError("trajectory continued past success")
            rows.append((state, action))
            state = advance(instance, state, action)
        if state.key != M.SOLVED:
            raise ValueError("successful trajectory did not solve the root")
        if (state.used_cc, state.used_moves) != (witness.crossing_changes, witness.moves):
            raise ValueError("successful trajectory cost differs from replay")
        paths[actions] = {
            "rows": tuple(rows),
            "cc": state.used_cc,
            "moves": state.used_moves,
            "cost": instance.cc_weight * state.used_cc + state.used_moves,
        }
    return list(paths.values())


def success_supervision(instance, witnesses, *, insertion_candidates=4):
    """Set-valued positive targets plus exactly cancelled success preferences.

    For each state, all observed next actions achieving the best KNOWN suffix cost
    are positive alternatives. An unobserved action is not certified inferior.
    Preference terms compare only complete successes of different full-root cost;
    no censored episode enters this function. Shared state/action occurrences use
    the SAME history-free sparse candidate pool before any clipped optimization.
    """
    traces = verified_success_traces(instance, witnesses)
    if not traces:
        raise ValueError("at least one verified successful trajectory required")
    observed = defaultdict(set)
    suffix_costs = defaultdict(dict)
    for trace in traces:
        for state, action in trace["rows"]:
            observed[state].add(action)
            remaining = (
                instance.cc_weight * (trace["cc"] - state.used_cc)
                + trace["moves"]
                - state.used_moves
            )
            old = suffix_costs[state].get(action, float("inf"))
            suffix_costs[state][action] = min(old, remaining)
    pools = {
        state: tuple(
            sorted(set(proposal_candidates(instance, state, insertion_candidates)) | actions)
        )
        for state, actions in observed.items()
    }
    positives = []
    for state, costs in suffix_costs.items():
        best = min(costs.values())
        good = tuple(a for a in pools[state] if costs.get(a) == best)
        positives.append(
            {
                "state": state,
                "actions": pools[state],
                "positive_actions": good,
                "known_suffix_cost": best,
            }
        )
    coefficients = defaultdict(Fraction)
    best_cost = min(t["cost"] for t in traces)
    winners = sum(t["cost"] == best_cost for t in traces)
    losers = len(traces) - winners
    if losers:
        for trace in traces:
            coefficient = (
                Fraction(1, winners) if trace["cost"] == best_cost else Fraction(-1, losers)
            )
            for state, action in trace["rows"]:
                coefficients[state, action] += coefficient
    preferences = [
        {
            "state": state,
            "action": action,
            "actions": pools[state],
            "coefficient_numerator": coefficient.numerator,
            "coefficient_denominator": coefficient.denominator,
        }
        for (state, action), coefficient in coefficients.items()
        if coefficient
    ]
    return {
        "unique_success_paths": len(traces),
        "distinct_success_costs": len({t["cost"] for t in traces}),
        "positives": positives,
        "preferences": preferences,
    }


def set_valued_policy_loss(log_probabilities, positive_mask):
    """-log mass of the full positive set, with no preferred witness ordering."""
    import torch

    if (
        log_probabilities.shape != positive_mask.shape
        or log_probabilities.ndim != 2
        or positive_mask.dtype != torch.bool
        or not positive_mask.any(dim=1).all()
    ):
        raise ValueError("invalid set-valued policy targets")
    selected = log_probabilities.masked_fill(~positive_mask, -torch.inf)
    loss = -torch.logsumexp(selected, dim=1).mean()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite set-valued policy loss")
    return loss
