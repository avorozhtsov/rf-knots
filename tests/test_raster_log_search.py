import importlib
import json
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("evaluate_raster_log_search")


def test_exact_original_resource_contract_is_reused(module):
    old = importlib.import_module("evaluate_graph_mixture_cost_mcts")
    assert module.CONFIG is old.CONFIG
    assert module.build_roots is old.build_roots
    assert module.batch_config is old.batch_config
    assert module.CONFIG["simulations"] == 8 and module.CONFIG["seed"] == 20261034


def fixture_state(module, directory):
    roots = [{"condition": i} for i in range(128)]
    (directory / "roots.json").write_text(json.dumps(roots))
    model, audit = directory / "model.pt", directory / "pair-audit.json"
    pins = {str(model): "fixed-model"}
    state = {
        "schema": "rf-knots-raster-log-cost-mcts-v1",
        "status": "evaluating",
        "arm": "q254",
        "config": deepcopy(module.CONFIG),
        "inputs": pins,
        "model_checkpoint": str(model),
        "checkpoint_sha256": "fixed-model",
        "training_pair_audit": str(audit),
        "value_head_used": False,
        "next_condition": 0,
        "shards": [],
        "roots_sha256": module.sha(directory / "roots.json"),
    }
    return state, pins, roots, model, audit


def test_resume_validates_empty_and_committed_prefix(module, tmp_path):
    state, pins, roots, model, audit = fixture_state(module, tmp_path)
    module.validate_resume(state, "q254", pins, roots, tmp_path, model, audit)
    shard = tmp_path / "conditions-00000.json"
    shard.write_text(json.dumps(roots[:8]))
    state.update(next_condition=8, shards=[{"file": shard.name, "sha256": module.sha(shard)}])
    module.validate_resume(state, "q254", pins, roots, tmp_path, model, audit)
    shard.write_text(json.dumps(roots[1:9]))
    with pytest.raises(ValueError, match="shard changed"):
        module.validate_resume(state, "q254", pins, roots, tmp_path, model, audit)


@pytest.mark.parametrize(
    "key,value",
    [
        ("arm", "random"),
        ("checkpoint_sha256", "other"),
        ("value_head_used", True),
        ("next_condition", 7),
        ("next_condition", 136),
        ("schema", "old"),
        ("model_checkpoint", "elsewhere.pt"),
        ("training_pair_audit", "another.json"),
        ("status", "failed"),
    ],
)
def test_resume_rejects_changed_contract(module, tmp_path, key, value):
    state, pins, roots, model, audit = fixture_state(module, tmp_path)
    state[key] = value
    with pytest.raises(ValueError, match="contract"):
        module.validate_resume(state, "q254", pins, roots, tmp_path, model, audit)


def test_partial_manifest_cannot_be_marked_completed(module, tmp_path):
    state, pins, roots, model, audit = fixture_state(module, tmp_path)
    state["status"] = "completed"
    with pytest.raises(ValueError, match="coverage"):
        module.validate_resume(state, "q254", pins, roots, tmp_path, model, audit)
