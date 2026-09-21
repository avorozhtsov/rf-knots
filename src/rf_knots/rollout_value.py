"""Resource-aware rollout value contract; no exhausted-MCTS negative labels.

The behavioural policy and proposal rule are frozen in each data artifact.
Value labels are complete finite-horizon episode returns, not proof of
unsolvability. A policy can fail despite a replayed witness fitting all caps.
"""

import hashlib
import json
import random
from collections import defaultdict
from fractions import Fraction

import torch

from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.lazy_actions import LazyActionCatalogue
from rf_knots.mcts_loop import SOLVED, SearchState, SitePolicyValueNet


def expanded_resource_model(checkpoint):
    """Expand INPUT conditioning only, preserving the old function at init.

    No output residual/adapter is introduced. New input columns initially zero;
    every weight, including the shared encoder and both heads, remains trainable.
    """
    old_dim = checkpoint["architecture"].get("budget_dim", 2)
    if old_dim == 5:
        net = SitePolicyValueNet(**checkpoint["architecture"])
        net.load_state_dict(checkpoint["model"], strict=True)
        return net, checkpoint["architecture"]
    if old_dim != 2:
        raise ValueError("migration requires a two- or five-budget checkpoint")
    architecture = {**checkpoint["architecture"], "budget_dim": 5}
    net = SitePolicyValueNet(**architecture)
    old, new = checkpoint["model"], net.state_dict()
    prefix = 3 * architecture["embedding_dim"]
    expanded = {"head.context_film.0.weight": 3, "value.0.weight": 2}
    for key, value in old.items():
        if key not in expanded:
            new[key].copy_(value)
            continue
        new[key].zero_()
        new[key][:, :prefix].copy_(value[:, :prefix])
        blocks = expanded[key]
        for block in range(blocks):
            new[key][:, prefix + block * 5 : prefix + block * 5 + 2].copy_(
                value[:, prefix + block * 2 : prefix + block * 2 + 2]
            )
        new[key][:, prefix + blocks * 5 :].copy_(value[:, prefix + blocks * 2 :])
    net.load_state_dict(new, strict=True)
    return net, architecture


def rollout_candidates(instance, state, insertions=16):
    """All O(L) ordinary actions plus a fixed state-hashed insertion subset.

    Frozen behavioural rollout only. This restricted proposal set is NOT the
    complete-action progressive MCTS catalogue. Identical state/resource context
    has identical proposals, which also permits exact shared-prefix cancellation.
    """
    if state.key == SOLVED or state.used_moves >= instance.cap_moves:
        return ()
    cat = LazyActionCatalogue(
        instance.spec, state.word, state.strands, state.used_cc < instance.cap_crossing_changes
    )
    payload = [
        state.word,
        state.strands,
        instance.spec.max_len,
        instance.spec.max_strands,
        instance.cap_moves - state.used_moves,
        instance.cap_crossing_changes - state.used_cc,
        instance.cc_weight,
    ]
    rng = random.Random(hashlib.sha256(json.dumps(payload).encode()).digest())
    # The first 2*ordinary_count positions contain all ordinary entries; sample
    # from the insertion space via its direct index rather than constructing it.
    chosen = list(cat.ordinary)
    common = min(len(cat.ordinary), cat.insertions)
    for i in sorted(rng.sample(range(cat.insertions), min(insertions, cat.insertions))):
        index = 2 * i + 1 if i < common else i + common
        chosen.append(cat.action_at(index))
    return tuple(chosen)


def verify_episode(episode, instance):
    state = SearchState(instance.word, instance.strands, 0, 0)
    for action in episode["actions"]:
        if action not in rollout_candidates(instance, state):
            raise ValueError("action outside frozen rollout proposal set")
        if not reference.is_legal(
            instance.spec,
            state.word,
            state.strands,
            action,
            state.used_cc < instance.cap_crossing_changes,
        ):
            raise ValueError("illegal rollout action")
        word, n = reference.apply(instance.spec, state.word, state.strands, action)
        state = SearchState(
            word,
            n,
            state.used_cc + int(instance.spec.decode(action)[0] == CROSSING_CHANGE),
            state.used_moves + 1,
        )
    success = state.key == SOLVED
    if success != episode["success"]:
        raise ValueError("rollout return differs from replay")
    if (
        not success
        and state.used_moves < instance.cap_moves
        and rollout_candidates(instance, state)
    ):
        raise ValueError("incomplete rollout is not a terminal failure label")
    if state.used_cc != episode["cc"] or state.used_moves != episode["moves"]:
        raise ValueError("rollout counts differ from replay")
    return state


def episode_states(episode, instance):
    state = SearchState(instance.word, instance.strands, 0, 0)
    for action in episode["actions"]:
        yield state, action
        word, n = reference.apply(instance.spec, state.word, state.strands, action)
        state = SearchState(
            word,
            n,
            state.used_cc + int(instance.spec.decode(action)[0] == CROSSING_CHANGE),
            state.used_moves + 1,
        )


def relative_terms(episodes, instance):
    """Exact shared-prefix cancellation BEFORE any clipped policy surrogate.

    Best ties share +1, strictly worse complete outcomes share -1. If all fail
    or all tie there is no ranking target. Never divide by trajectory length.
    Each group must use one root, resource contract, lambda and behaviour version.
    Fractions avoid nonzero float residue from +1 and nine copies of -1/9.
    """
    if not episodes or any(not e["eligible"] for e in episodes):
        return []
    for episode in episodes:
        verify_episode(episode, instance)
    costs = [
        instance.cc_weight * e["cc"] + e["moves"] if e["success"] else float("inf")
        for e in episodes
    ]
    best = min(costs)
    if best == float("inf") or all(c == best for c in costs):
        return []
    winners = sum(c == best for c in costs)
    losers = len(costs) - winners
    coefficients = defaultdict(Fraction)
    for episode, cost in zip(episodes, costs, strict=True):
        weight = Fraction(1, winners) if cost == best else Fraction(-1, losers)
        for state, action in episode_states(episode, instance):
            coefficients[state, action] += weight
    return [
        (state, action, float(weight)) for (state, action), weight in coefficients.items() if weight
    ]


def clipped_relative_loss(log_prob, old_log_prob, coefficients, clip=0.2):
    """PPO-style guard for fixed-data relative updates, not vanilla PPO/GRPO."""
    ratio = torch.exp((log_prob - old_log_prob).clamp(-20, 20))
    return -torch.minimum(
        ratio * coefficients, ratio.clamp(1 - clip, 1 + clip) * coefficients
    ).sum()
