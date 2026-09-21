from copy import deepcopy

import pytest

pytest.importorskip("spherogram")

from rf_knots import embedding_replay_corpus as C  # noqa: E402
from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, REDUCE, ActionSpec  # noqa: E402
from rf_knots.evidence import UnknotWitness  # noqa: E402


def witness():
    spec = ActionSpec(16, 5)
    return UnknotWitness.from_actions(
        (1, 1, 1),
        2,
        spec,
        [spec.encode(CROSSING_CHANGE, 0), spec.encode(REDUCE, 0), spec.encode(DESTABILIZE)],
    )


def test_mirror_changes_signed_invariant_not_split():
    w = witness()
    mirrored = C.mirror(w)
    assert C.invariants(w.start.word, 2) == (3, -2)
    assert C.invariants(mirrored.start.word, 2) == (3, 2)
    assert C.owner((3, -2)) == C.owner((3, 2))
    assert C.mirror(mirrored) == w
    for determinant in range(1, 30, 2):
        assert C.owner((determinant, 4)) == C.owner((determinant, -4))


def test_lift_is_replayable_and_invariant_preserving():
    w, prefix = C.lift(witness(), 9, 50)
    assert len(w.start.word) >= 50 and w.start.strands == 9
    assert w.crossing_changes == 1
    assert w.steps[prefix - 1].after == witness().start
    assert C.invariants(w.start.word, w.start.strands) == (3, -2)
    assert all(C.owner(f) == "train" for f in C.path_fingerprints(w))


def test_record_recomputes_bounds_and_replays():
    w = witness()
    row = {
        "witness": w.to_dict(),
        "pair_prefix_length": w.moves,
        "left_fingerprint": [3, -2],
        "right_fingerprint": [1, 0],
        "split": "train",
        "interval": {
            "left": w.start.to_dict(),
            "right": w.steps[-1].after.to_dict(),
            "lower": 1,
            "upper": 1,
            "witness_semantic_length": w.moves,
        },
    }
    assert C.verify_record(row) == w
    for change in ("lower", "upper"):
        bad = deepcopy(row)
        bad["interval"][change] = 0
        with pytest.raises(ValueError):
            C.verify_record(bad)
    bad = deepcopy(row)
    bad["split"] = "validation"
    with pytest.raises(ValueError, match="wrong split"):
        C.verify_record(bad)


def test_link_is_not_silently_treated_as_a_knot():
    with pytest.raises(ValueError, match="knots"):
        C.invariants((1, 1), 2)


def test_target_quarantine_and_test_bucket_are_distinct_from_train():
    assert C.owner((49, 0)) == "quarantine"
    assert {C.owner((det, 2)) for det in range(1, 100, 2)} >= {"train", "validation", "test"}
