import importlib.util
from pathlib import Path

import pytest

from rf_knots import mcts_loop as M
from rf_knots.actions import COMMUTE, CROSSING_CHANGE, REDUCE
from rf_knots.cost_mcts_rollout import advance


@pytest.fixture
def diamond():
    path = Path(__file__).parents[1] / "scripts/mcts_training_diamonds.py"
    spec = importlib.util.spec_from_file_location("verified_local_diamonds", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_reductions_are_reindexed_not_blindly_swapped(diamond):
    i = M.Instance("unknot", (1, -1, 1, -1, 1), 2, 0)
    state = M.SearchState(i.word, i.strands, 0, 0)
    first = second = i.spec.encode(REDUCE, 0)
    d = diamond.verified_diamond(i, state, first, second, (1, 0), (1, 0))
    assert d is not None
    assert [i.spec.decode(a)[1] for a in d["actions"]] == [2, 0]
    assert d["end"].word == (1,)
    assert d["end"].used_cc == 0 and d["end"].used_moves == 2


@pytest.mark.parametrize("flip_first", [False, True])
def test_reduction_and_disjoint_cc_keep_exact_cost_and_endpoint(diamond, flip_first):
    i = M.Instance("trefoil", (1, -1, 1, 1, 1), 2, 1)
    state = M.SearchState(i.word, i.strands, 0, 0)
    a, b = (
        (i.spec.encode(CROSSING_CHANGE, 4), i.spec.encode(REDUCE, 0))
        if flip_first
        else (i.spec.encode(REDUCE, 0), i.spec.encode(CROSSING_CHANGE, 2))
    )
    d = diamond.verified_diamond(i, state, a, b, (3, 2), (1, 0))
    assert d is not None
    assert d["end"] == advance(i, advance(i, state, a), b)
    assert d["end"].used_cc == 1
    assert d["fingerprint_reason"].startswith("zero-CC")


def test_two_cc_alternative_requires_its_own_ownership_check(diamond, monkeypatch):
    i = M.Instance("trefoil", (1, 1, 1), 2, 2)
    state = M.SearchState(i.word, i.strands, 0, 0)
    first, second = i.spec.encode(CROSSING_CHANGE, 0), i.spec.encode(CROSSING_CHANGE, 2)
    calls = []

    def invariant(word, strands):
        calls.append((word, strands))
        return 999, 0

    monkeypatch.setattr(diamond, "invariants", invariant)
    monkeypatch.setattr(diamond, "owner", lambda fp: "test" if fp == (999, 0) else "train")
    assert diamond.verified_diamond(i, state, first, second, (3, 2), (1, 0)) is None
    assert calls == [((1, 1, -1), 2)]


def test_overlapping_or_seam_supports_are_not_claimed_independent(diamond):
    i = M.Instance("example", (1, 2, -1, 2, 1), 3, 1)
    state = M.SearchState(i.word, i.strands, 0, 0)
    assert (
        diamond.swapped_local_actions(
            i, state, i.spec.encode(CROSSING_CHANGE, 0), i.spec.encode(COMMUTE, 0)
        )
        is None
    )
    assert (
        diamond.swapped_local_actions(
            i, state, i.spec.encode(REDUCE, 4), i.spec.encode(CROSSING_CHANGE, 1)
        )
        is None
    )


def test_reduction_straddling_deleted_gap_is_not_a_disjoint_move(diamond):
    i = M.Instance("gap", (1, 1, -1, -1, 1), 2, 0)
    state = M.SearchState(i.word, i.strands, 0, 0)
    assert (
        diamond.swapped_local_actions(i, state, i.spec.encode(REDUCE, 1), i.spec.encode(REDUCE, 0))
        is None
    )


def test_entire_independent_reduction_block_can_supply_first_actions(diamond):
    i = M.Instance("many-pairs", (1, -1) * 6 + (1,), 2, 0)
    actions = [i.spec.encode(REDUCE, 0)] * 6
    states = [M.SearchState(i.word, i.strands, 0, 0)]
    for action in actions:
        states.append(advance(i, states[-1], action))
    alternatives = [
        diamond.reorder_reduce_prefix(i, states, actions, 0, span, (1, 0)) for span in range(2, 7)
    ]
    assert [i.spec.decode(a[0])[1] for a in alternatives] == [2, 4, 6, 8, 10]
    assert all(len(a) == span for a, span in zip(alternatives, range(2, 7), strict=True))
