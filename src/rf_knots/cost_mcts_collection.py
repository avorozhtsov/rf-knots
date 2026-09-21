"""TRAIN-only selection and full-path ownership, separate from learning targets."""

import random
from collections import defaultdict

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.cost_mcts_rollout import advance
from rf_knots.embedding_replay_corpus import invariants, owner


def choose_distinct_training_roots(records, count, seed):
    if count < 1:
        raise ValueError("positive number of root groups required")
    groups = defaultdict(list)
    for i, record in enumerate(records):
        if record["split"] != "train":
            raise ValueError("root selection requires TRAIN-only records")
        det, sigma = record["left_fingerprint"]
        if owner((det, sigma)) != "train":
            raise ValueError("non-TRAIN fingerprint in supplied TRAIN records")
        groups[det, abs(sigma)].append(i)
    if len(groups) < count:
        raise ValueError("insufficient distinct TRAIN fingerprint groups")
    rng = random.Random(seed)
    chosen = rng.sample(sorted(groups), count)
    return [rng.choice(groups[key]) for key in chosen]


def annotate_path_ownership(instance, prefix, continuation, heldout_groups):
    """Exact CC endpoints, including the MCTS prefix; never relabel a failure."""
    state = M.SearchState(instance.word, instance.strands, 0, 0)
    groups = {invariants(state.word, state.strands)}
    for action in (*prefix, *continuation.actions):
        state = advance(instance, state, action)
        if instance.spec.decode(action)[0] == CROSSING_CHANGE:
            groups.add(invariants(state.word, state.strands))
    if state != continuation.end:
        raise ValueError("full path endpoint differs from continuation replay")
    excluded = {
        fp for fp in groups if owner(fp) != "train" or (fp[0], abs(fp[1])) in heldout_groups
    }
    return {
        "eligible": not excluded,
        "path_fingerprints": sorted(groups),
        "excluded_fingerprints": sorted(excluded),
    }
