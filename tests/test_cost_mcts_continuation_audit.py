import importlib.util
from dataclasses import asdict
from pathlib import Path

import pytest

from rf_knots import mcts_loop as M
from rf_knots.cost_mcts_rollout import BatchedPolicyContinuation


@pytest.fixture
def auditor():
    spec = importlib.util.spec_from_file_location(
        "continuation_audit", Path(__file__).parents[1] / "scripts/audit_cost_mcts_continuations.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode_case():
    instance = M.Instance("trefoil", (1, 1, 1), 2, 1, cap_moves=10)
    state = M.SearchState(instance.word, instance.strands, 0, 0)
    result = BatchedPolicyContinuation(M.UniformEvaluator(), horizon=2).run(
        [(instance, state)], seeds=[14]
    )[0]
    assert result.stop_reason == "horizon"
    # JSON-style data are what the independent auditor actually consumes.
    import json

    episode = json.loads(json.dumps({"tree_prefix": [], "continuation": asdict(result)}))
    return instance, episode, {"horizon": 2, "insertion_candidates": 4}


def test_censored_trajectory_is_replayed_too(auditor):
    instance, episode, config = episode_case()
    steps, scores, depth = auditor.audit_episode(instance, episode, config)
    assert steps == depth == 2 and scores > steps


@pytest.mark.parametrize(
    "fault",
    ["endpoint", "start", "label", "score_count", "probability", "coverage", "action", "horizon"],
)
def test_corrupted_censored_trace_is_not_accepted(auditor, fault):
    instance, episode, config = episode_case()
    result = episode["continuation"]
    if fault == "endpoint":
        result["end"]["used_moves"] += 1
    elif fault == "start":
        result["start"]["used_cc"] += 1
    elif fault == "label":
        result["stop_reason"] = "solved"
    elif fault == "score_count":
        result["scored_actions"] += 1
    elif fault == "probability":
        result["probabilities"][0] = float("nan")
    elif fault == "coverage":
        result["probabilities"].pop()
    elif fault == "action":
        result["actions"][0] = -1
    elif fault == "horizon":
        config["horizon"] = 1
    with pytest.raises((AssertionError, ValueError)):
        auditor.audit_episode(instance, episode, config)
