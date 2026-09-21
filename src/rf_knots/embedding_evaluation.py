"""Evaluation-only pair validation; does not relax training split ownership."""

from rf_knots.embedding_replay_corpus import invariants, owner, verify_record


def verify_evaluation_pair(record):
    if record["split"] != "test":
        raise ValueError("evaluation pair must belong to test")
    witness = verify_record(record, check_invariants=False)
    pair = record["interval"]
    left = invariants(witness.start.word, witness.start.strands)
    endpoint = pair["right"]
    right = invariants(tuple(endpoint["word"]), endpoint["strands"])
    if list(left) != record["left_fingerprint"] or list(right) != record["right_fingerprint"]:
        raise ValueError("evaluation fingerprints differ from recomputation")
    if owner(left) != "test" or owner(right) == "quarantine":
        raise ValueError("evaluation pair violates reserved ownership")
    lower = abs(left[1] - right[1]) // 2
    if pair["lower"] != lower or lower > pair["upper"]:
        raise ValueError("invalid evaluation signed-signature bound")
    return witness
