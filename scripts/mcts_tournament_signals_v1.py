"""Verified-success tournament signals for idea (3), independent of training.

The public entry point accepts complete UnknotWitness objects and replays
them through the original instance. Censored searches are never inputs.
"""

import random
from collections import defaultdict

from mcts_training_supervision import verified_success_traces

from rf_knots.site_proposals import proposal_candidates

CONTROL = "A"
VARIANTS = tuple("BCDEFGHI")


def _trace_key(trace):
    return tuple(action for _, action in trace["rows"])


def _add(coefficients, state, action, value):
    coefficients[(state, action)] += value


def _first_fork(winner, loser):
    for (positive_state, positive_action), (negative_state, negative_action) in zip(
        winner["rows"], loser["rows"], strict=False
    ):
        if positive_state != negative_state or positive_action != negative_action:
            if positive_state != negative_state:
                raise ValueError("paths separated without a shared-prefix action")
            return ((positive_state, positive_action), (negative_state, negative_action))
    raise ValueError("strictly different costs require a divergent action")


def _pair_terms(winner, loser, mode):
    if mode == "first":
        positive, negative = _first_fork(winner, loser)
        return [(positive, negative)]
    if mode == "suffix":
        # Return separately because suffix coefficients are summed over all
        # actions, including those at states not reached by the other path.
        return None
    if mode == "shared_context":
        negative_at = defaultdict(set)
        for state, action in loser["rows"]:
            negative_at[state].add(action)
        pairs = []
        for state, action in winner["rows"]:
            for negative_action in negative_at.get(state, ()):
                if action != negative_action:
                    pairs.append(((state, action), (state, negative_action)))
        if not pairs:
            # Every two complete root paths have a first fork. This catches
            # an inconsistent state/resource identity.
            raise ValueError("no branch disagreement between unequal paths")
        return pairs
    raise ValueError("unknown tournament comparison mode")


def _compare(winners, losers, weights, mode):
    if len(losers) != len(weights) or not winners or not losers:
        raise ValueError("require positives and weighted strict losers")
    if abs(sum(weights) - 1.0) > 1e-9 or any(w <= 0 for w in weights):
        raise ValueError("loser weights must be positive and total one")
    coefficients = defaultdict(float)
    for winner in winners:
        for loser, loser_weight in zip(losers, weights, strict=True):
            pair_weight = loser_weight / len(winners)
            if mode == "suffix":
                for state, action in winner["rows"]:
                    _add(coefficients, state, action, pair_weight)
                for state, action in loser["rows"]:
                    _add(coefficients, state, action, -pair_weight)
            else:
                for positive, negative in _pair_terms(winner, loser, mode):
                    _add(coefficients, *positive, pair_weight)
                    _add(coefficients, *negative, -pair_weight)
    return {key: value for key, value in coefficients.items() if abs(value) > 1e-12}


def _groups(traces, arm, *, seed, k, near_margin, gap_scale, gap_cap):
    if arm not in (CONTROL, *VARIANTS):
        raise ValueError("unknown arm")
    if k < 1 or near_margin < 0 or gap_scale <= 0 or gap_cap <= 0:
        raise ValueError("invalid tournament hyperparameters")
    if arm == CONTROL:
        return []
    traces = sorted(traces, key=lambda row: (row["cost"], _trace_key(row)))
    if len(traces) < 2:
        return []
    best_cost = traces[0]["cost"]
    winners = traces[:1]
    if arm == "F":
        winners = [row for row in traces if row["cost"] <= best_cost + near_margin][:3]
    worst_positive = max(row["cost"] for row in winners)
    eligible_losers = [row for row in traces if row["cost"] > worst_positive]
    if not eligible_losers:
        return []
    if arm == "B":
        losers = [random.Random(seed).choice(eligible_losers)]
    else:
        losers = eligible_losers[:k]
    if arm == "I":
        raw = [
            min((row["cost"] - best_cost) / gap_scale, gap_cap)
            for row in losers
        ]
        weights = [value / sum(raw) for value in raw]
    else:
        weights = [1.0 / len(losers)] * len(losers)
    mode = (
        "first"
        if arm in ("B", "C")
        else "suffix"
        if arm == "D"
        else "shared_context"
    )
    result = [
        {
            "objective": "lambda_cost",
            "coefficients": _compare(winners, losers, weights, mode),
            "positive_paths": len(winners),
            "negative_paths": len(losers),
            "mode": mode,
        }
    ]
    if arm == "G":
        cc_order = sorted(traces, key=lambda row: (row["cc"], row["cost"], _trace_key(row)))
        cc_best = cc_order[0]["cc"]
        cc_losers = [row for row in cc_order if row["cc"] > cc_best][:k]
        if cc_losers:
            result.append(
                {
                    "objective": "lexicographic_cc_extra",
                    "coefficients": _compare(
                        cc_order[:1],
                        cc_losers,
                        [1.0 / len(cc_losers)] * len(cc_losers),
                        "shared_context",
                    ),
                    "positive_paths": 1,
                    "negative_paths": len(cc_losers),
                    "mode": "shared_context",
                }
            )
    return result


def build_tournament_signals(
    instance,
    witnesses,
    arm,
    *,
    seed,
    k,
    near_margin,
    gap_scale,
    gap_cap,
    insertion_candidates=4,
):
    """Replay complete witnesses and return cancelled, legal policy terms.

    Ownership/exclusion must be checked by the caller before providing a
    witness. The returned groups remain separate by objective.
    """
    traces = verified_success_traces(instance, witnesses)
    groups = _groups(
        traces,
        arm,
        seed=seed,
        k=k,
        near_margin=near_margin,
        gap_scale=gap_scale,
        gap_cap=gap_cap,
    )
    for group in groups:
        pools = {}
        for state, action in group["coefficients"]:
            pools.setdefault(state, set()).add(action)
        requests = []
        for (state, action), coefficient in sorted(
            group["coefficients"].items(),
            key=lambda item: (item[0][0].used_moves, item[0][0].used_cc, item[0][1]),
        ):
            actions = tuple(
                sorted(
                    set(proposal_candidates(instance, state, insertion_candidates))
                    | pools[state]
                )
            )
            requests.append(((instance, state, actions), actions.index(action), coefficient))
        group["terms"] = requests
        del group["coefficients"]
    return groups
