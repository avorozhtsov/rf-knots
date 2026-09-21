import importlib.util
from pathlib import Path

import pytest

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, REDUCE
from rf_knots.evidence import UnknotWitness


@pytest.fixture
def builder(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/build_mcts_training_supervision.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("mcts_metric_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trefoil_witness():
    i = M.Instance("trefoil", (1, 1, 1), 2, 1, cap_moves=5)
    s = i.spec
    w = UnknotWitness.from_actions(
        i.word,
        i.strands,
        s,
        (s.encode(CROSSING_CHANGE, 0), s.encode(REDUCE, 0), s.encode(DESTABILIZE)),
    )
    return i, w


def test_suffix_bounds_have_explicit_unknot_endpoint_and_provenance(builder):
    _, w = trefoil_witness()
    views = builder.short_u_views(w, "verified-witness")
    assert [(v["lower"], v["upper"]) for v in views] == [(1, 1), (0, 0), (0, 0)]
    assert [v["suffix_index"] for v in views] == [0, 1, 2]
    assert all(v["right_word"] == [] and v["right_strands"] == 1 for v in views)
    assert all(v["witness_id"] == "verified-witness" for v in views)


def test_reference_is_positive_but_an_ineligible_found_path_is_not_imported(builder):
    i, w = trefoil_witness()
    root = {"reference_witness": w.to_dict()}
    row = {
        "condition": 0,
        "episodes": [{"eligible": False, "continuation": {"stop_reason": "solved"}}],
        "witness": w.to_dict(),
    }
    gate = {"condition": 0, "best_witness_train_owned": False}
    assert builder.condition_witnesses(i, root, row, gate, set()) == [w]
    with pytest.raises(ValueError, match="heldout"):
        builder.condition_witnesses(i, root, row, gate, {(3, 2)})


def test_metric_lower_bound_contradiction_is_not_clamped(builder, monkeypatch):
    _, w = trefoil_witness()
    monkeypatch.setattr(builder, "invariants", lambda *args: (3, 10))
    monkeypatch.setattr(builder, "owner", lambda fp: "train")
    with pytest.raises(ValueError, match="lower bound"):
        builder.short_u_views(w, "bad-invariant")
