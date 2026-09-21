import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, REDUCE


@pytest.fixture
def auditor(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/audit_cost_mcts_training.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("training_collection_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_fingerprint_endpoints_not_one_label_for_whole_path(auditor):
    i = M.Instance("trefoil", (1, 1, 1), 2, 1, cap_moves=3)
    actions = [
        i.spec.encode(CROSSING_CHANGE, 0),
        i.spec.encode(REDUCE, 0),
        i.spec.encode(DESTABILIZE),
    ]
    groups = auditor.path_groups(i, actions)
    assert {(d, abs(s)) for d, s in groups} == {(3, 2), (1, 0)}


def test_audit_rechecks_cc_and_move_caps(auditor):
    i = M.Instance("trefoil", (1, 1, 1), 2, 1, cap_moves=3)
    crossing = i.spec.encode(CROSSING_CHANGE, 0)
    with pytest.raises(ValueError, match="out-of-budget"):
        auditor.path_groups(replace(i, cap_crossing_changes=0), [crossing])
    with pytest.raises(ValueError, match="out-of-budget"):
        auditor.path_groups(replace(i, cap_moves=0), [crossing])


def test_zero_cc_steps_do_not_need_distinct_fingerprint_work(auditor, monkeypatch):
    i = M.Instance("reduce", (1, -1, 1), 2, 0, cap_moves=3)
    calls = []

    def invariant(word, strands):
        calls.append((word, strands))
        return (1, 0)

    monkeypatch.setattr(auditor, "invariants", invariant)
    result = auditor.path_groups(i, [i.spec.encode(REDUCE, 0), i.spec.encode(DESTABILIZE)])
    assert result == {(1, 0)} and len(calls) == 1


def test_incomplete_data_not_accepted_as_training_gate(auditor, tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema": "rf-knots-cost-mcts-trajectories-v1", "status": "collecting"})
    )
    monkeypatch.setattr(
        "sys.argv", ["audit", "--data", str(tmp_path), "--out", str(tmp_path / "audit.json")]
    )
    with pytest.raises(ValueError, match="completed"):
        auditor.main()
    assert not (tmp_path / "audit.json").exists()
