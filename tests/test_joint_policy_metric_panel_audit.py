import importlib
from pathlib import Path

import pytest

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, REDUCE
from rf_knots.evidence import UnknotWitness


@pytest.fixture
def auditor(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("audit_joint_policy_metric_panel")


def test_provenance_conflicts_cannot_be_overwritten(auditor):
    pins = {"corpus": "first"}
    auditor.merge_pins(pins, {"corpus": "first", "actor": "second"})
    assert pins == {"corpus": "first", "actor": "second"}
    with pytest.raises(ValueError, match="conflicting provenance"):
        auditor.merge_pins(pins, {"corpus": "different"})


def test_strata_are_disjoint_and_cover_both_size_axes(auditor):
    assert auditor.size_stratum((1,) * 208, 17) == "within"
    assert auditor.size_stratum((1,) * 209, 17) == "length_only"
    assert auditor.size_stratum((1,) * 208, 18) == "strands_only"
    assert auditor.size_stratum((1,) * 209, 18) == "both"


def test_rollout_extension_schema_not_confused_with_base_certificate(auditor):
    reports = {
        "metric": {"train_path_groups": [[1, 0], [3, 2]]},
        "rollout": {"rollout_path_groups": [[3, 2]]},
        "mcts": {"eligible_path_groups": [[1, 0]]},
    }
    assert auditor.historical_groups(reports) == ({(1, 0), (3, 2)}, {(3, 2)}, {(1, 0)})
    reports["rollout"] = {"optimizer_path_groups": [[3, 2]]}
    with pytest.raises(ValueError, match="rollout/rollout_path_groups"):
        auditor.historical_groups(reports)
    reports["rollout"] = {"rollout_path_groups": 82}
    with pytest.raises(ValueError, match="group list"):
        auditor.historical_groups(reports)


def test_replayed_states_keep_resource_context_and_recompute_cc_endpoints(auditor, monkeypatch):
    instance = M.Instance("t", (1, 1, 1), 2, 1, cap_moves=10)
    spec = instance.spec
    witness = UnknotWitness.from_actions(
        instance.word,
        instance.strands,
        spec,
        [spec.encode(CROSSING_CHANGE, 0), spec.encode(REDUCE, 0), spec.encode(DESTABILIZE)],
    )
    monkeypatch.setattr(auditor, "owner", lambda fp: "train")
    states = auditor.witness_states(witness)
    assert list(states) == [
        M.SearchState((1, 1, 1), 2, 0, 0),
        M.SearchState((-1, 1, 1), 2, 1, 1),
        M.SearchState((1,), 2, 1, 2),
        M.SearchState((), 1, 1, 3),
    ]
    fingerprints = list(states.values())
    assert fingerprints[0][0] == 3
    assert fingerprints[1:] == [(1, 0)] * 3
    monkeypatch.setattr(auditor, "owner", lambda fp: "test")
    with pytest.raises(ValueError, match="non-TRAIN"):
        auditor.witness_states(witness)
