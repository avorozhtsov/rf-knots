import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from rf_knots import mcts_loop as M


@pytest.fixture
def evaluate():
    path = Path(__file__).parents[1] / "scripts/evaluate_rollout_value.py"
    spec = importlib.util.spec_from_file_location("rollout_value_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "status,purpose", [("collecting", "evaluation-only"), ("completed", "training")]
)
def test_incomplete_or_train_data_rejected(evaluate, tmp_path, monkeypatch, status, purpose):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": status, "purpose": purpose}))
    monkeypatch.setattr(
        "sys.argv", ["evaluate", "--data", str(tmp_path), "--output", str(tmp_path / "result.json")]
    )
    with pytest.raises(ValueError, match="evaluation-only"):
        evaluate.main()


def fixture_data(path, seeds=(0, 1), claimed_fp=(5, 4)):
    architecture = dict(
        width=16, depth=2, embedding_dim=8, encoder_kind="strand_graph", budget_dim=5
    )
    net = M.SitePolicyValueNet(**architecture)
    torch.save({"architecture": architecture, "model": net.state_dict()}, path / "actor.pt")
    problem = M.Instance("test", (-1,) * 5, 2, 2, cap_moves=0)
    roots = [{"instance": problem.to_dict(), "fingerprint": list(claimed_fp), "panel_index": 0}]
    (path / "roots.json").write_text(json.dumps(roots))
    episodes = [
        {
            "group": 0,
            "seed": seed,
            "eligible": False,
            "actions": [],
            "success": False,
            "cc": 0,
            "moves": 0,
        }
        for seed in seeds
    ]
    (path / "episodes.json").write_text(json.dumps(episodes))

    def digest(name):
        return hashlib.sha256((path / name).read_bytes()).hexdigest()

    manifest = {
        "status": "completed",
        "purpose": "evaluation-only",
        "actor_sha256": digest("actor.pt"),
        "roots_sha256": digest("roots.json"),
        "config": {"rollouts": 2},
        "shards": [{"file": "episodes.json", "sha256": digest("episodes.json")}],
    }
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_evaluation_replays_and_keeps_failed_trials(evaluate, tmp_path, monkeypatch):
    fixture_data(tmp_path)
    monkeypatch.setattr(
        "sys.argv", ["evaluate", "--data", str(tmp_path), "--output", str(tmp_path / "result.json")]
    )
    evaluate.main()
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["summary"]["trials"] == 2 and result["summary"]["successes"] == 0
    assert result["constant_half"]["brier"] == 0.25


@pytest.mark.parametrize(
    "seeds,fp,match",
    [((0,), (5, 4), "trials"), ((0, 0), (5, 4), "trials"), ((0, 1), (3, 2), "heldout")],
)
def test_missing_duplicate_seeds_or_false_identity_rejected(
    evaluate, tmp_path, monkeypatch, seeds, fp, match
):
    fixture_data(tmp_path, seeds, fp)
    monkeypatch.setattr(
        "sys.argv", ["evaluate", "--data", str(tmp_path), "--output", str(tmp_path / "result.json")]
    )
    with pytest.raises(ValueError, match=match):
        evaluate.main()
