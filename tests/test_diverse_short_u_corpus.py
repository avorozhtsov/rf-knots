import importlib.util
from pathlib import Path

import pytest

from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_replay_corpus import verify_record


@pytest.fixture
def builder():
    path = Path(__file__).parents[1] / "scripts/build_diverse_short_u_corpus.py"
    spec = importlib.util.spec_from_file_location("diverse_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_cc_path_yields_valid_anchor_general_pair_and_suffix(builder):
    witness = conjugate_witness(20261017, 16, max_strands=17, min_depth=8, max_depth=48)
    rows = builder.derive({"index": 16, "witness": witness.to_dict()})
    assert [r["derivation"] for r in rows] == ["root-anchor", "first-CC-pair", "post-first-CC-anchor"]
    assert [r["interval"]["upper"] for r in rows] == [2, 1, 1]
    assert rows[0]["interval"]["lower"] == 0
    assert rows[0]["determinant_difference_lower_bound"] == 1
    for row in rows:
        verify_record(row)
    assert rows[1]["interval"]["right"] == rows[2]["interval"]["left"]
    assert rows[1]["pair_prefix_length"] + rows[2]["pair_prefix_length"] == rows[0]["pair_prefix_length"]
    assert len({builder.pair_key(r) for r in rows}) == 3


def test_raw_pair_key_is_ordered_and_does_not_fold_mirrors(builder):
    row = {"interval": {"left": {"word": [1, 1, 1], "strands": 2}, "right": {"word": [], "strands": 1}}}
    mirrored = {"interval": {"left": {"word": [-1, -1, -1], "strands": 2}, "right": row["interval"]["right"]}}
    swapped = {"interval": {"left": row["interval"]["right"], "right": row["interval"]["left"]}}
    assert len({builder.pair_key(r) for r in (row, mirrored, swapped)}) == 3
