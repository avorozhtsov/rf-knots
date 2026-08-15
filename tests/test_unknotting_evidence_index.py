from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from rf_knots.actions import DESTABILIZE, ActionSpec


def _module():
    path = Path(__file__).parents[1] / "scripts" / "build_unknotting_evidence_index.py"
    spec = importlib.util.spec_from_file_location("build_unknotting_evidence_index", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_index_replays_event_witness_and_keeps_knotinfo_fail_closed(tmp_path: Path):
    run = tmp_path / "results" / "run"
    (run / "events").mkdir(parents=True)
    (run / "bank.json").write_text(
        json.dumps(
            {
                "schema": "test-bank",
                "rows": [{"id": "test-knot", "word": [1], "strands": 2}],
            }
        )
    )
    action = ActionSpec(32, 2).encode(DESTABILIZE)
    evaluation = {
        "best_witness": {
            "crossing_changes": 0,
            "semantic_moves": 1,
            "semantic_actions": [action],
        }
    }
    (run / "events" / "000.json").write_text(
        json.dumps(
            {
                "selected": "test-knot",
                "scientists": {
                    "ours-a": {"evaluation": {"1000.0": evaluation}},
                    "ours-b": {"evaluation": {"1000.0": evaluation}},
                },
            }
        )
    )

    payload = _module().build(tmp_path / "results")

    assert payload["scope"] == {
        "event_files_joined": 1,
        "knots_with_verified_native_evidence": 1,
        "replay_errors": 0,
        "knotinfo_replayable_evidence": 0,
    }
    row = payload["knots"]["test-knot"]
    assert row["scientists"]["ours-a"]["replay_verified"] is True
    assert row["scientists"]["ours-a"]["l10"] == 1
    assert row["knotinfo-shortest-evidence"]["rankable_by_l10"] is False
