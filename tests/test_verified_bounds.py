"""The ratchet is only worth having if a bad claim cannot survive a read."""

from __future__ import annotations

import json

import pytest

from rf_knots import verified_bounds as vb
from rf_knots.actions import ActionSpec
from rf_knots.evidence import UnknotWitness


def _trefoil_witness() -> UnknotWitness:
    """`sigma_1^3` unknotted by one crossing change, then reduced away."""
    from rf_knots.unknot_search import search

    report = search((1, 1, 1), 2, max_crossing_changes=1, node_budget=2000)
    assert report.witness is not None
    return report.witness


def test_the_key_matches_the_legacy_one_exactly():
    """Both logs must key the same way or they cannot be folded together."""
    from pathlib import Path

    legacy = Path(__file__).resolve().parents[2] / "pgx-mcts-bench"
    del legacy  # the neighbour repository is not importable from the test suite
    assert vb.knot_id((1, 1, 1), 2) == "b2:1,1,1"
    assert vb.knot_id((), 1) == "b1:e"
    assert vb.knot_id((1, 0, -2), 3) == "b3:1,-2"


def test_a_witness_round_trips_and_verifies(tmp_path):
    witness = _trefoil_witness()
    record = vb.from_witness(witness, agent="test", lower_bound=1)
    path = tmp_path / "bounds.jsonl"
    vb.claim(path, record)
    records, rejected = vb.best(path)
    assert not rejected
    stored = records["b2:1,1,1"]
    assert stored.verified
    assert stored.crossing_changes == 1
    assert stored.exact  # meets |sigma|/2 = 1, so u = 1 is determined


def test_a_tampered_witness_is_rejected_on_read_not_silently_demoted(tmp_path):
    """The property the old log could not have: a false claim cannot survive."""
    witness = _trefoil_witness()
    path = tmp_path / "bounds.jsonl"
    vb.claim(path, vb.from_witness(witness, agent="honest", lower_bound=1))

    row = json.loads(path.read_text().splitlines()[0])
    row["witness"]["steps"] = row["witness"]["steps"][:-1]  # drop the last move
    row["crossing_changes"] = 0
    row["agent"] = "liar"
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")

    records, rejected = vb.best(path)
    assert len(rejected) == 1
    assert "unknot" in rejected[0] or "mismatch" in rejected[0]
    # The honest record still stands; the cheaper false one never entered.
    assert records["b2:1,1,1"].agent == "honest"
    assert records["b2:1,1,1"].crossing_changes == 1


def test_an_illegal_action_in_a_witness_is_refused_at_construction():
    spec = ActionSpec(16, 4)
    with pytest.raises(ValueError, match="illegal"):
        UnknotWitness.from_actions((1, 1, 1), 2, spec, [spec.num_actions - 1] * 3)


def test_a_verified_record_displaces_an_unverified_one_at_equal_cost(tmp_path):
    path = tmp_path / "bounds.jsonl"
    vb.claim(path, vb.Record("b2:1,1,1", 1, 3, "legacy", None))
    records, _ = vb.best(path)
    assert not records["b2:1,1,1"].verified

    vb.claim(path, vb.from_witness(_trefoil_witness(), agent="searcher", lower_bound=1))
    records, _ = vb.best(path)
    assert records["b2:1,1,1"].verified
    assert records["b2:1,1,1"].agent == "searcher"


def test_a_cheaper_unverified_record_still_wins_on_cost(tmp_path):
    """Cost first, evidence only as a tie-break: an upper bound is an upper bound."""
    path = tmp_path / "bounds.jsonl"
    vb.claim(path, vb.from_witness(_trefoil_witness(), agent="searcher", lower_bound=1))
    vb.claim(path, vb.Record("b2:1,1,1", 0, 1, "impossible-but-cheaper", None))
    records, _ = vb.best(path)
    assert records["b2:1,1,1"].crossing_changes == 0
    assert not records["b2:1,1,1"].verified


def test_a_torn_line_is_reported_rather_than_fatal(tmp_path):
    path = tmp_path / "bounds.jsonl"
    vb.claim(path, vb.from_witness(_trefoil_witness(), agent="ok", lower_bound=1))
    with path.open("a") as handle:
        handle.write('{"knot": "b2:1,1,1", "crossing_ch\n')
    records, rejected = vb.best(path)
    assert records["b2:1,1,1"].agent == "ok"
    assert len(rejected) == 1 and "JSON" in rejected[0]


def test_the_report_counts_verified_and_exact_records(tmp_path):
    path = tmp_path / "bounds.jsonl"
    vb.claim(path, vb.from_witness(_trefoil_witness(), agent="searcher", lower_bound=1))
    vb.claim(path, vb.Record("b3:1,2,1", 4, 9, "legacy", None))
    text = vb.report(path)
    assert "2 knots, 1 with a replayable witness, 1 with `u` determined exactly." in text
