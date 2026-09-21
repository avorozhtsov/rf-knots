from copy import deepcopy

import pytest

from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, REDUCE, ActionSpec
from rf_knots.embedding_replay_corpus import invariants, lift
from rf_knots.embedding_suffix_supervision import collect_u_views, verify_u_view
from rf_knots.evidence import UnknotWitness


def source(extra_flips=False):
    spec = ActionSpec(24, 8)
    actions = [spec.encode(CROSSING_CHANGE, 0), spec.encode(REDUCE, 0), spec.encode(DESTABILIZE)]
    if extra_flips:
        actions = [spec.encode(CROSSING_CHANGE, 0)] * 2 + actions
    witness = UnknotWitness.from_actions((1, 1, 1), 2, spec, actions)
    return row(witness)


def row(w):
    return {
        "split": "train",
        "witness": w.to_dict(),
        "pair_prefix_length": w.moves,
        "left_fingerprint": list(invariants(w.start.word, w.start.strands)),
        "right_fingerprint": [1, 0],
        "interval": {
            "left": w.start.to_dict(),
            "right": w.steps[-1].after.to_dict(),
            "lower": 1,
            "upper": w.crossing_changes,
            "witness_semantic_length": w.moves,
        },
    }


def test_dense_suffix_bounds_match_independent_replay():
    w, _ = lift(UnknotWitness.from_dict(source()["witness"]), 6, 20)
    records = [row(w)]
    index = collect_u_views(records, intermediate=True)
    assert len(index["views"]) > 3
    for view in index["views"]:
        verify_u_view(view, records, index["parent_witness_sha256"])
        assert view["lower"] <= view["upper"]
    assert {v["upper"] for v in index["views"]} == {0, 1}


def test_root_only_and_tightest_evidence_dedup():
    records = [source(True), source()]
    index = collect_u_views(records, intermediate=False, maximum_cc=4)
    assert len(index["views"]) == 1
    assert index["views"][0]["upper"] == 1
    assert index["views"][0]["parent_index"] == 1
    assert index["summary"]["unique_normalized_views"] == 1


def test_index_and_parent_tamper_fail_closed():
    records = [source()]
    index = collect_u_views(records, intermediate=True)
    view = deepcopy(index["views"][0])
    view["upper"] += 1
    with pytest.raises(ValueError, match="differs"):
        verify_u_view(view, records, index["parent_witness_sha256"])
    with pytest.raises(ValueError, match="hash"):
        verify_u_view(index["views"][0], records, ["bad"])
    bad = deepcopy(records)
    bad[0]["split"] = "validation"
    with pytest.raises(ValueError, match="TRAIN"):
        collect_u_views(bad, intermediate=True)
