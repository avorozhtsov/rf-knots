import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("evaluate_common_greedy_policies")


def test_greedy_ties_are_by_action_id_and_nonfinite_rejected(runner):
    assert runner.choose([3, 3, 2], [8, 2, 1]) == 2
    with pytest.raises(ValueError):
        runner.choose([np.nan], [1])
    with pytest.raises(ValueError):
        runner.choose([], [])


def test_controller_transitions_must_not_change_braid(runner):
    spec = runner.ActionSpec(48, 12)
    before = ((1, 1, 1), 2)
    assert runner.replay_step(spec, before, None, before, 0) == 0
    with pytest.raises(ValueError):
        runner.replay_step(spec, before, None, ((1,), 2), 0)


def test_cc_cap_and_replay_mismatch_fail_closed(runner):
    spec = runner.ActionSpec(48, 12)
    before = ((1, 1, 1), 2)
    action = spec.encode(runner.CROSSING_CHANGE, 0)
    after = ((-1, 1, 1), 2)
    assert runner.replay_step(spec, before, action, after, 0) == 1
    with pytest.raises(ValueError):
        runner.replay_step(spec, before, action, after, 2)
    with pytest.raises(ValueError):
        runner.replay_step(spec, before, action, before, 0)


def test_independent_audit_replays_success_and_rejects_false_claims(runner):
    auditor = importlib.import_module("audit_common_greedy_policies")
    spec = runner.ActionSpec(48, 12)
    from rf_knots.actions import DESTABILIZE, REDUCE

    actions = [spec.encode(runner.CROSSING_CHANGE, 0), spec.encode(REDUCE, 0),
               spec.encode(DESTABILIZE)]
    witness = runner.UnknotWitness.from_actions((1, 1, 1), 2, spec, actions)
    before, transcript = witness.start, []
    for step in witness.steps:
        transcript.append({
            "before": before.to_dict(), "after": step.after.to_dict(),
            "semantic_action": step.action.to_flat(spec), "native_action": None,
            "head_before": None, "legal_scores": 1,
        })
        before = step.after
    row = {"transcript": transcript, "solved": True, "final": before.to_dict(),
           "used_cc": 1, "semantic_moves": 3, "neural_samples": 3, "forward_batches": 3,
           "controller_only_decisions": 0, "repeated_semantic_states": 0,
           "scored_actions": 3, "witness": witness.to_dict(), "cost": 13,
           "cc_weight": 10, "stop_reason": "solved"}
    root = {"witness": witness.to_dict()}
    assert auditor.audit_row(row, root)["solved"]
    for key, value in (("cost", 12), ("neural_samples", 2), ("used_cc", 0), ("solved", False)):
        wrong = deepcopy(row)
        wrong[key] = value
        with pytest.raises(ValueError):
            auditor.audit_row(wrong, root)
