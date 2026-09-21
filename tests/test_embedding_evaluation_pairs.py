from copy import deepcopy

import pytest

from rf_knots.actions import ActionSpec, CROSSING_CHANGE, REDUCE, DESTABILIZE
from rf_knots.embedding_evaluation import verify_evaluation_pair
from rf_knots.embedding_replay_corpus import invariants, lift, verify_record
from rf_knots.evidence import UnknotWitness


def record(prefix):
    spec = ActionSpec(24, 6)
    w = UnknotWitness.from_actions((1,)*5, 2, spec, [
        spec.encode(CROSSING_CHANGE, 0), spec.encode(REDUCE, 0),
        spec.encode(CROSSING_CHANGE, 0), spec.encode(REDUCE, 0), spec.encode(DESTABILIZE)])
    right = w.steps[prefix-1].after if prefix else w.start
    a,b = invariants(w.start.word, 2), invariants(right.word,right.strands)
    return {"split":"test","witness":w.to_dict(),"pair_prefix_length":prefix,
            "left_fingerprint":list(a),"right_fingerprint":list(b),
            "interval":{"left":w.start.to_dict(),"right":right.to_dict(),
                        "lower":abs(a[1]-b[1])//2,
                        "upper":sum(s.action.kind=="CROSSING_CHANGE" for s in w.steps[:prefix]),
                        "witness_semantic_length":prefix}}


@pytest.mark.parametrize("prefix,expected", [(0,0),(2,1),(5,2)])
def test_exact_pair_distance_and_prefix(prefix, expected):
    r = record(prefix)
    w = verify_evaluation_pair(r)
    assert r["interval"]["lower"] == r["interval"]["upper"] == expected
    large, added = lift(w, 9, 60)
    r["witness"] = large.to_dict()
    r["pair_prefix_length"] += added
    r["interval"]["witness_semantic_length"] += added
    r["interval"]["left"] = large.start.to_dict()
    verify_evaluation_pair(r)


def test_eval_permission_does_not_relax_training_validator():
    r = record(0)
    verify_evaluation_pair(r)
    with pytest.raises(ValueError, match="reference endpoint"):
        verify_record(r)
    r["split"] = "train"
    with pytest.raises(ValueError, match="must belong to test"):
        verify_evaluation_pair(r)


def test_forged_signed_bound_and_endpoint_are_rejected():
    original = record(2)
    for field in ("lower", "upper"):
        r=deepcopy(original)
        r["interval"][field]=0
        with pytest.raises(ValueError):
            verify_evaluation_pair(r)
    r=deepcopy(original)
    r["right_fingerprint"][1] *= -1
    with pytest.raises(ValueError,match="fingerprints"):
        verify_evaluation_pair(r)
