import importlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, PASS, REDUCE, ActionSpec


@dataclass
class Transition:
    state: object
    legal_actions: object
    terminated: bool


class FixedToyGame:
    """Exact reference braid dynamics behind the native wrapper interface."""

    def __init__(self):
        self.spec = ActionSpec(48, 12)
        self.base = SimpleNamespace(spec=self.spec, underlying_action=lambda a, h, n: a)

    def view(self, word, n, cc):
        raw = SimpleNamespace(_word=np.array(word), _n=n, _crossing_changes=cc)
        state = SimpleNamespace(raw=raw, base_state=SimpleNamespace(head=0))
        legal = np.array(
            [reference.is_legal(self.spec, word, n, a, True) for a in range(self.spec.num_actions)]
        )
        return Transition(state, legal, (word, n) == ((), 1))

    def reset(self, seed):
        return self.view((1, 1, 1), 2, 0)

    def unwrap(self, state):
        return state.raw

    def step(self, state, action):
        raw = state.raw
        word, n = reference.apply(self.spec, tuple(raw._word), raw._n, action)
        cc = raw._crossing_changes + int(self.spec.decode(action)[0] == CROSSING_CHANGE)
        return self.view(word, n, cc)


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("evaluate_common_search_policies")


def test_virtual_unselected_solution_is_retained_across_epochs(runner):
    base = FixedToyGame()
    game = runner.NativeWitnessSearch(base, ((1, 1, 1), 2), 10)
    root = game.reset(0)
    a = game.step(root.state, base.spec.encode(CROSSING_CHANGE, 0))
    b = game.step(a.state, base.spec.encode(REDUCE, 0))
    c = game.step(b.state, base.spec.encode(DESTABILIZE))
    assert c.terminated
    assert game.best["cost"] == 13
    assert len(game.best["native_actions"]) == 3
    assert game.best["discovery_transition"] == 3
    # The selected controller path may still be at the unsolved original root.
    game.begin(root)
    assert game.best["cost"] == 13
    assert tuple(root.state.raw._word) == (1, 1, 1)
    assert len(game.paths) == 1
    with pytest.raises(KeyError):
        game.step(a.state, base.spec.encode(REDUCE, 0))


def test_virtual_cc_mask_and_pass_exclusion_apply_at_every_depth(runner):
    base = FixedToyGame()
    game = runner.NativeWitnessSearch(base, ((1, 1, 1), 2), 1000)
    t = game.reset(0)
    action = base.spec.encode(CROSSING_CHANGE, 0)
    assert not t.legal_actions[base.spec.encode(PASS)]
    t = game.step(t.state, action)
    assert t.legal_actions[action]
    t = game.step(t.state, action)
    assert not t.legal_actions[action]
    with pytest.raises(ValueError, match="masked action"):
        game.step(t.state, action)


def test_search_counter_rejects_nonfinite_value_even_with_finite_logits(runner):
    counter = runner.SearchForwardCounter()
    with pytest.raises(ValueError, match="MCTS value"):
        counter(None, (torch.zeros(1, 2),), (torch.zeros(1, 2), torch.tensor([float("nan")])))


def test_rf_search_audit_rejects_false_cost_and_counters(runner):
    auditor = importlib.import_module("audit_common_search_policies")
    spec = ActionSpec(48, 12)
    actions = [spec.encode(CROSSING_CHANGE, 0), spec.encode(REDUCE, 0), spec.encode(DESTABILIZE)]
    witness = runner.UnknotWitness.from_actions((1, 1, 1), 2, spec, actions)
    instance = runner.M.Instance(
        "toy",
        (1, 1, 1),
        2,
        cap_crossing_changes=2,
        cap_moves=128,
        growth_len=45,
        growth_strands=10,
        cc_weight=10,
    )
    row = {
        "cc_weight": 10,
        "roots": [{"instance": instance.to_dict()}],
        "rf_config": {"simulations": 8, "insertion_candidates": 1056, "horizon": 128},
        "search": {
            "simulations": 3,
            "episodes": [],
            "solved": True,
            "witness": witness.to_dict(),
            "L10": 13,
            "L1000": 1003,
            "incumbents": [{"cc": 1, "moves": 3, "cost": 13}],
        },
        "counts": {
            "continuation_moves": 0,
            "policy_neural_evaluations": 0,
            "policy_action_scores": 0,
            "episodes": 0,
        },
        "solved": True,
    }
    assert auditor.audit_rf(row, witness) == 13
    wrong = deepcopy(row)
    wrong["search"]["L1000"] = 1002
    with pytest.raises(ValueError, match="cost differs"):
        auditor.audit_rf(wrong, witness)
    wrong = deepcopy(row)
    wrong["counts"]["policy_neural_evaluations"] = 1
    with pytest.raises(ValueError, match="counters differ"):
        auditor.audit_rf(wrong, witness)
