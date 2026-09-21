import importlib.util
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, INSERT, REDUCE
from rf_knots.evidence import UnknotWitness


@pytest.fixture
def supervision():
    path = Path(__file__).parents[1] / "scripts/mcts_training_supervision.py"
    spec = importlib.util.spec_from_file_location("mcts_success_targets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def instance():
    return M.Instance("trefoil", (1, 1, 1), 2, 1, cap_moves=32)


def witness(i, loops=0, flip=0):
    s = i.spec
    actions = [s.encode(CROSSING_CHANGE, flip)]
    actions += [s.encode(INSERT, 0, 1, 1), s.encode(REDUCE, 0)] * loops
    actions += [s.encode(REDUCE, 0 if flip == 0 else 1), s.encode(DESTABILIZE)]
    return UnknotWitness.from_actions(i.word, i.strands, s, actions)


def test_one_winner_nine_losers_cancel_shared_prefix_exactly(supervision):
    i = instance()
    targets = supervision.success_supervision(i, [witness(i, loops=n) for n in range(10)])
    assert targets["unique_success_paths"] == 10
    assert targets["distinct_success_costs"] == 10
    assert not any(r["state"].used_moves == 0 for r in targets["preferences"])
    first = [r for r in targets["preferences"] if r["state"].used_moves == 1]
    coefficients = [
        Fraction(r["coefficient_numerator"], r["coefficient_denominator"]) for r in first
    ]
    assert sorted(coefficients) == [Fraction(-1), Fraction(1)]


def test_equal_successes_are_all_positive_without_artificial_ranking(supervision):
    i = instance()
    targets = supervision.success_supervision(i, [witness(i), witness(i, flip=2)])
    assert not targets["preferences"]
    root = next(r for r in targets["positives"] if r["state"].used_moves == 0)
    assert set(root["positive_actions"]) == {
        i.spec.encode(CROSSING_CHANGE, 0),
        i.spec.encode(CROSSING_CHANGE, 2),
    }
    assert root["known_suffix_cost"] == 1003


def test_duplicate_witnesses_do_not_reweight_preference(supervision):
    i = instance()
    a, b = witness(i), witness(i, loops=1)
    assert supervision.success_supervision(i, [a, b]) == supervision.success_supervision(
        i, [a, a, b]
    )


def test_root_resource_and_lambda_contract(supervision):
    i = instance()
    w = witness(i)
    with pytest.raises(ValueError, match="different root"):
        supervision.success_supervision(replace(i, word=(-1, -1, -1)), [w])
    with pytest.raises(ValueError, match="resources"):
        supervision.success_supervision(replace(i, cap_moves=2), [w])
    for weight in (10, 1000):
        targets = supervision.success_supervision(replace(i, cc_weight=weight), [w])
        root = next(r for r in targets["positives"] if r["state"].used_moves == 0)
        assert root["known_suffix_cost"] == weight + 3


def test_preferences_and_positives_share_exact_same_action_pool(supervision):
    i = instance()
    result = supervision.success_supervision(i, [witness(i), witness(i, loops=2)])
    pools = {r["state"]: r["actions"] for r in result["positives"]}
    for row in result["preferences"]:
        assert row["actions"] == pools[row["state"]]
        assert row["action"] in row["actions"]


def test_loss_rewards_total_positive_mass_not_one_arbitrary_witness(supervision):
    mask = torch.tensor([[True, True, False]])
    a = torch.tensor([[0.4, 0.4, 0.2]], requires_grad=True)
    b = torch.tensor([[0.7, 0.1, 0.2]])
    loss = supervision.set_valued_policy_loss(a.log(), mask)
    torch.testing.assert_close(loss, supervision.set_valued_policy_loss(b.log(), mask))
    loss.backward()
    assert torch.isfinite(a.grad).all()
    with pytest.raises(ValueError):
        supervision.set_valued_policy_loss(a.log(), torch.zeros_like(mask))


def test_cost_weight_changes_ranking_without_mixing_resource_conditions(supervision):
    i = M.Instance("unknot", (1, -1, 1), 2, 1, cap_moves=20)
    s = i.spec
    long = UnknotWitness.from_actions(
        i.word,
        i.strands,
        s,
        [s.encode(INSERT, 0, 1, 1), s.encode(REDUCE, 0)] * 6
        + [s.encode(REDUCE, 0), s.encode(DESTABILIZE)],
    )
    short = UnknotWitness.from_actions(
        i.word,
        i.strands,
        s,
        [s.encode(CROSSING_CHANGE, 0), s.encode(REDUCE, 1), s.encode(DESTABILIZE)],
    )
    for weight, preferred in ((10, CROSSING_CHANGE), (1000, INSERT)):
        targets = supervision.success_supervision(replace(i, cc_weight=weight), [long, short])
        root = next(r for r in targets["positives"] if r["state"].used_moves == 0)
        assert [s.decode(a)[0] for a in root["positive_actions"]] == [preferred]
    with pytest.raises(ValueError, match="finite"):
        supervision.success_supervision(replace(i, cc_weight=float("nan")), [long])
