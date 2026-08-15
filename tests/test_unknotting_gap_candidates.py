"""The open-gap catalogue is reproducible and keeps targets separate from evidence."""

import json
from pathlib import Path

from rf_knots.evidence import braid_instance_id

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "benchmarks/knotinfo-unknotting-gap-candidates-20260814.json"


def test_gap_catalogue_counts_and_provenance():
    payload = json.loads(CATALOGUE.read_text())

    assert payload["schema"] == "rf-knots-unknotting-gap-candidates-v1"
    assert payload["source"]["sha256"] == (
        "1829b7056eefc653f77a42acc9e471df4020e64fe93a8b6a211ca3d49fe86e7b"
    )
    assert payload["summary"] == {
        "candidate_count": 482,
        "interval_counts": {"[1,3]": 409, "[1,4]": 6, "[2,4]": 67},
        "knotinfo_reference_rows": 5,
        "replayable_external_witnesses": 0,
        "representation_status_counts": {
            "available": 51,
            "outside_local_crossing_table": 426,
            "unsupported_local_strand_cap": 5,
        },
    }


def test_available_rows_have_exact_stable_representation_ids():
    rows = json.loads(CATALOGUE.read_text())["candidates"]
    available = [row for row in rows if row["representation_status"] == "available"]

    assert len(available) == 51
    assert {tuple(row["knotinfo_interval_at_snapshot"]) for row in available} == {
        (1, 3),
        (2, 4),
    }
    for row in available:
        rep = row["stored_representation"]
        assert row["external_witness_status"] == (
            "no_replayable_actions_in_knotinfo_snapshot"
        )
        assert rep["instance_id"] == braid_instance_id(rep["word"], rep["strands"])
        assert row["strict_upper_bound_target"] == row["knotinfo_interval_at_snapshot"][1] - 1


def test_initial_low_capacity_targets_are_present():
    rows = {
        row["canonical_name"]: row
        for row in json.loads(CATALOGUE.read_text())["candidates"]
    }

    assert rows["12a_815"]["knotinfo_interval_at_snapshot"] == [2, 4]
    assert rows["12a_815"]["stored_representation"]["strands"] == 3
    assert rows["12n_140"]["knotinfo_interval_at_snapshot"] == [1, 3]
    assert rows["12n_140"]["stored_representation"]["word_length"] == 14
