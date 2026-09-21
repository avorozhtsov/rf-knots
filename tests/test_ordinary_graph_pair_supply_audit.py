import importlib
from pathlib import Path

import pytest


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("audit_ordinary_graph_pair_supply")


def test_trace_bounds_recompute_labels_and_reject_free_mirror(audit):
    from rf_knots.embedding_replay_corpus import invariants, owner

    states = [{"word": [1, 1, 1], "strands": 2}, {"word": [-1, 1, 1], "strands": 2}]
    fps = [list(invariants(tuple(s["word"]), s["strands"])) for s in states]
    row = {
        "states": states,
        "cc_positions": [0],
        "fingerprints": fps,
        "owners": [owner(tuple(f)) for f in fps],
        "all_owner_train": True,
        "lower": 1,
        "upper": 1,
    }
    assert audit.trace_bounds(row) == {(1, 0), (3, 2)}
    row["lower"] = 0
    with pytest.raises(ValueError, match="labels/partition"):
        audit.trace_bounds(row)
    row["lower"] = 1
    row["states"][1]["word"] = [-1, -1, -1]
    with pytest.raises(ValueError, match="literal CC"):
        audit.trace_bounds(row)


def test_excludes_heldout_query_mirror_groups_but_not_shared_reference(audit):
    panel = {"validation": [{"left_fingerprint": [5, -2], "right_fingerprint": [1, 0]}]}
    assert audit.heldout_queries([panel]) == {(5, 2)}
    with pytest.raises(ValueError, match="schema"):
        audit.heldout_queries([{}])
