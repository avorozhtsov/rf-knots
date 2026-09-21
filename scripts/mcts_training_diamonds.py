"""Exact two-step alternatives to one recorded witness order, not learned labels."""

from rf_knots.actions import BRAID, COMMUTE, CROSSING_CHANGE, REDUCE
from rf_knots.cost_mcts_rollout import advance
from rf_knots.embedding_replay_corpus import invariants, owner

WIDTH = {CROSSING_CHANGE: 1, COMMUTE: 2, BRAID: 3, REDUCE: 2}


def swapped_local_actions(instance, state, first, second):
    """Reindex disjoint non-wrapping local supports; all claims require replay.

    REDUCE deletes two letters, so the second recorded position belongs to the
    shorter word. Positionless and seam-crossing actions are conservatively skipped.
    This constructs at most one candidate, never an O(L*N) or O(L**2) action scan.
    """
    spec = instance.spec
    a, p, ga, sa = spec.decode(first)
    b, q, gb, sb = spec.decode(second)
    if a not in WIDTH or b not in WIDTH:
        return None
    length = len(state.word)
    after_length = length - (2 if a == REDUCE else 0)
    if not (0 <= p <= length - WIDTH[a] and 0 <= q <= after_length - WIDTH[b]):
        return None
    original_q = q + 2 if a == REDUCE and q >= p else q
    if not (p + WIDTH[a] <= original_q or original_q + WIDTH[b] <= p):
        return None
    shifted_p = p - 2 if b == REDUCE and original_q < p else p
    return spec.encode(b, original_q, gb, sb), spec.encode(a, shifted_p, ga, sa)


def verified_diamond(instance, state, first, second, start_fp, end_fp):
    """Both orders replay to the same raw state/counters; alternative stays TRAIN.

    If either step is zero-CC, the alternative intermediate is isotopic to one
    known endpoint. For two CC steps its fingerprint must be computed explicitly.
    Caller supplies endpoint fingerprints from an independently replayed TRAIN path.
    """
    if owner(start_fp) != "train" or owner(end_fp) != "train":
        raise ValueError("diamond endpoint is not TRAIN-owned")
    swapped = swapped_local_actions(instance, state, first, second)
    if swapped is None:
        return None
    original_middle = advance(instance, state, first)
    original_end = advance(instance, original_middle, second)
    try:
        alternate_middle = advance(instance, state, swapped[0])
        alternate_end = advance(instance, alternate_middle, swapped[1])
    except ValueError:
        return None
    if alternate_end != original_end:
        return None
    first_cc = instance.spec.decode(swapped[0])[0] == CROSSING_CHANGE
    second_cc = instance.spec.decode(swapped[1])[0] == CROSSING_CHANGE
    if not first_cc:
        fp, provenance = start_fp, "zero-CC first step"
    elif not second_cc:
        fp, provenance = end_fp, "zero-CC second step"
    else:
        fp = invariants(alternate_middle.word, alternate_middle.strands)
        provenance = "exact determinant/signature after crossing change"
    if owner(fp) != "train":
        return None
    return {
        "actions": swapped,
        "intermediate_fingerprint": tuple(fp),
        "fingerprint_reason": provenance,
        "end": original_end,
    }


def reorder_reduce_prefix(instance, states, actions, offset, span, fingerprint):
    """Move a later independent REDUCE first; preserve the entire prefix endpoint."""
    if span < 2 or offset < 0 or offset + span > len(actions):
        raise ValueError("invalid reduction prefix range")
    work = list(actions[offset : offset + span])
    if any(instance.spec.decode(a)[0] != REDUCE for a in work):
        return None
    # First j actions are unchanged while bubbling right-to-left, so cached
    # source states before those actions remain exact throughout this loop.
    for j in range(span - 2, -1, -1):
        result = verified_diamond(
            instance, states[offset + j], work[j], work[j + 1], fingerprint, fingerprint
        )
        if result is None:
            return None
        work[j : j + 2] = result["actions"]
    state = states[offset]
    for action in work:
        state = advance(instance, state, action)
    if state != states[offset + span]:
        raise ValueError("reordered reduction prefix changed its endpoint or cost")
    return tuple(work)
