from dataclasses import replace

import pytest

from rf_knots import cost_mcts_collection as C
from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.cost_mcts_rollout import PolicyContinuation, advance
from rf_knots.embedding_replay_corpus import owner


def records():
    groups = [(d, 0) for d in range(1, 81, 2) if owner((d, 0)) == "train"]
    return [{"split": "train", "left_fingerprint": list(fp)} for fp in groups for _ in range(3)]


def test_model_blind_selection_distinct_groups_and_repeatable():
    rows = records()
    chosen = C.choose_distinct_training_roots(rows, 12, 20261022)
    assert chosen == C.choose_distinct_training_roots(rows, 12, 20261022)
    assert len({tuple(rows[i]["left_fingerprint"]) for i in chosen}) == 12
    assert chosen != C.choose_distinct_training_roots(rows, 12, 20261023)
    with pytest.raises(ValueError, match="insufficient"):
        C.choose_distinct_training_roots(rows, len(rows), 0)


def test_mirrors_share_one_selection_group_and_nontrain_rejected():
    rows = [
        {"split": "train", "left_fingerprint": [3, 2]},
        {"split": "train", "left_fingerprint": [3, -2]},
    ]
    with pytest.raises(ValueError, match="insufficient"):
        C.choose_distinct_training_roots(rows, 2, 0)
    rows[0]["split"] = "validation"
    with pytest.raises(ValueError, match="TRAIN"):
        C.choose_distinct_training_roots(rows, 1, 0)


def case():
    instance = M.Instance("trefoil", (1, 1, 1), 2, 2, cap_moves=5)
    state = M.SearchState(instance.word, instance.strands, 0, 0)
    action = instance.spec.encode(CROSSING_CHANGE, 0)
    end = advance(instance, state, action)
    return instance, state, action, end


@pytest.mark.parametrize("in_prefix", [False, True])
def test_crossing_in_prefix_or_rollout_can_exclude_whole_path(monkeypatch, in_prefix):
    instance, state, action, end = case()
    blocked = next((d, 0) for d in range(5, 101, 2) if owner((d, 0)) == "test")
    calls = []

    def fake(word, strands):
        calls.append((word, strands))
        return (3, 2) if word == state.word else blocked

    monkeypatch.setattr(C, "invariants", fake)
    continuation = PolicyContinuation(
        end if in_prefix else state,
        end,
        () if in_prefix else (action,),
        () if in_prefix else (1.0,),
        "horizon",
        1,
    )
    result = C.annotate_path_ownership(
        instance, (action,) if in_prefix else (), continuation, set()
    )
    assert not result["eligible"] and result["excluded_fingerprints"] == [blocked]
    assert len(calls) == 2


def test_explicit_heldout_guard_and_false_endpoint(monkeypatch):
    instance, state, action, end = case()
    monkeypatch.setattr(C, "invariants", lambda *args: (3, -2))
    continuation = PolicyContinuation(state, end, (action,), (1.0,), "horizon", 1)
    assert C.annotate_path_ownership(instance, (), continuation, set())["eligible"]
    assert not C.annotate_path_ownership(instance, (), continuation, {(3, 2)})["eligible"]
    with pytest.raises(ValueError, match="endpoint"):
        C.annotate_path_ownership(instance, (), replace(continuation, end=state), set())
