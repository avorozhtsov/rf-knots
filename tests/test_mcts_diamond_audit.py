import importlib.util
from pathlib import Path

import pytest

from rf_knots import mcts_loop as M
from rf_knots.actions import DESTABILIZE, INSERT, REDUCE
from rf_knots.cost_mcts_rollout import advance
from rf_knots.evidence import SemanticAction, UnknotWitness


@pytest.fixture
def audit(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/audit_mcts_training_diamonds.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("diamond_certificate_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case():
    i = M.Instance("many-pairs", (1, -1) * 3 + (1,), 2, 0, cap_moves=8)
    actions = [i.spec.encode(REDUCE, 0)] * 3 + [i.spec.encode(DESTABILIZE)]
    w = UnknotWitness.from_actions(i.word, i.strands, i.spec, actions)
    states = [M.SearchState(i.word, i.strands, 0, 0)]
    for action in actions:
        states.append(advance(i, states[-1], action))
    cert = {
        "suffix_index": 0,
        "span": 1,
        "method": "same-successor",
        "first_action": SemanticAction.from_flat(i.spec, i.spec.encode(REDUCE, 2)).to_dict(),
    }
    return i, w, actions, states, [(1, 0)] * len(states), cert


def test_same_successor_and_reordered_prefix_have_replayed_proofs(audit):
    i, w, actions, states, fps, cert = case()
    assert audit.verify_certificate(i, cert, w, states, actions, fps, 4, set()) == (
        states[0],
        i.spec.encode(REDUCE, 2),
    )
    cert.update(
        method="independent-reduce-prefix",
        span=3,
        first_action=SemanticAction.from_flat(i.spec, i.spec.encode(REDUCE, 4)).to_dict(),
    )
    assert audit.verify_certificate(i, cert, w, states, actions, fps, 4, set())[1] == i.spec.encode(
        REDUCE, 4
    )


@pytest.mark.parametrize("fault", ["cost", "endpoint", "first", "heldout"])
def test_false_certificate_does_not_justify_a_positive_target(audit, fault):
    i, w, actions, states, fps, cert = case()
    cost, heldout = 4, set()
    if fault == "cost":
        cost = 3
    elif fault == "endpoint":
        cert["first_action"] = SemanticAction.from_flat(
            i.spec, i.spec.encode(INSERT, 0, 1, 1)
        ).to_dict()
    elif fault == "first":
        cert.update(method="independent-reduce-prefix", span=3)  # True first must be REDUCE 4.
    elif fault == "heldout":
        heldout = {(1, 0)}
    with pytest.raises(AssertionError):
        audit.verify_certificate(i, cert, w, states, actions, fps, cost, heldout)
