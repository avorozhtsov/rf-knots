"""Compact replay-derived short-u supervision from full TRAIN witnesses.

A suffix certifies an upper bound, not a shortest route. Signed signature gives
the lower bound. Zero-CC moves preserve it, so only CC endpoints need a new exact
invariant computation. Parent hashes and suffix indices retain the evidence.
"""

import hashlib
import json
from collections import Counter

from rf_knots.embedding_normalization import fast_normalize_global_view
from rf_knots.embedding_replay_corpus import canonical, invariants, owner, verify_record


def category(view):
    return str(view["lower"]) if view["lower"] == view["upper"] else "interval"


def collect_u_views(records, *, intermediate: bool, maximum_cc: int = 2):
    unique = {}
    parent_hashes = []
    candidate_count = 0
    for parent, record in enumerate(records):
        if record["split"] != "train":
            raise ValueError("suffix supervision accepts TRAIN records only")
        witness = verify_record(record)
        parent_hashes.append(
            hashlib.sha256(
                json.dumps(record["witness"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        remaining_cc = witness.crossing_changes
        fp = tuple(record["left_fingerprint"])
        states = [witness.start] + [s.after for s in witness.steps]
        for depth, state in enumerate(states[:-1]):
            if depth and witness.steps[depth - 1].action.kind == "CROSSING_CHANGE":
                remaining_cc -= 1
                fp = invariants(state.word, state.strands)
            if not intermediate and depth:
                break
            if owner(fp) != "train":
                raise ValueError("heldout fingerprint in suffix supervision")
            if remaining_cc > maximum_cc:
                continue
            lo = abs(fp[1]) // 2
            if lo > remaining_cc:
                raise ValueError("suffix signature contradicts replayed upper bound")
            view = {
                "parent_index": parent,
                "suffix_index": depth,
                "word": list(state.word),
                "strands": state.strands,
                "fingerprint": list(fp),
                "lower": lo,
                "upper": remaining_cc,
                "remaining_moves": witness.moves - depth,
            }
            candidate_count += 1
            key = canonical(state.word, state.strands)
            old = unique.get(key)
            if old and old["lower"] != lo:
                raise ValueError("same cyclic representation has inconsistent lower bounds")
            # Choose the tightest demonstrated u upper bound, then the shortest
            # witness among those. This is metric evidence, not policy imitation.
            if old is None or (view["upper"], view["remaining_moves"]) < (
                old["upper"],
                old["remaining_moves"],
            ):
                unique[key] = view
    views = list(unique.values())
    normalized = {fast_normalize_global_view(tuple(v["word"]), v["strands"]) for v in views}
    return {
        "schema": "rf-knots-suffix-u-views-v1",
        "intermediate": intermediate,
        "parent_witness_sha256": parent_hashes,
        "views": views,
        "summary": {
            "candidate_states": candidate_count,
            "unique_cyclic_views": len(views),
            "unique_normalized_views": len(normalized),
            "normalized_empty_views": sum(
                not fast_normalize_global_view(tuple(v["word"]), v["strands"])[0] for v in views
            ),
            "categories": dict(Counter(category(v) for v in views)),
            "fingerprint_groups": len(
                {(v["fingerprint"][0], abs(v["fingerprint"][1])) for v in views}
            ),
        },
    }


def verify_u_view(view, records, parent_hashes):
    """Independent, direct suffix replay; useful for auditing sampled index rows."""
    record = records[view["parent_index"]]
    expected = hashlib.sha256(
        json.dumps(record["witness"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected != parent_hashes[view["parent_index"]]:
        raise ValueError("parent witness hash mismatch")
    witness = verify_record(record)
    depth = view["suffix_index"]
    if not 0 <= depth < witness.moves:
        raise ValueError("invalid suffix index")
    state = witness.start if depth == 0 else witness.steps[depth - 1].after
    fp = invariants(state.word, state.strands)
    upper = sum(s.action.kind == "CROSSING_CHANGE" for s in witness.steps[depth:])
    if (
        list(state.word),
        state.strands,
        list(fp),
        abs(fp[1]) // 2,
        upper,
        witness.moves - depth,
    ) != (
        view["word"],
        view["strands"],
        view["fingerprint"],
        view["lower"],
        view["upper"],
        view["remaining_moves"],
    ):
        raise ValueError("suffix view differs from replay or invariant")
    if record["split"] != "train" or owner(fp) != "train":
        raise ValueError("non-train suffix")
