import importlib
from pathlib import Path

import pytest


@pytest.fixture
def auditor(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("audit_graph_metric_training_pair")


@pytest.mark.parametrize("mode", ["graph", "control"])
def test_both_arms_require_policy_training_and_exact_sample_contract(auditor, monkeypatch, mode):
    calls = []
    monkeypatch.setattr(
        auditor,
        "inspect_checkpoint",
        lambda checkpoint, initial, **kw: calls.append(kw) or {"ok": True},
    )
    config = {**auditor.CONFIG, "mode": mode}
    manifest = {
        "status": "completed",
        "config": config,
        "completed_steps": 4096,
        "initial_checkpoint_sha256": auditor.INITIAL_SHA,
        "value_head_trained": False,
        "inputs": {},
        "sources": {},
        "metric_sampled_rows": 524288,
        "policy_sampled_states": 131072,
        "graph_sampled_rows": 262144 if mode == "graph" else 0,
    }
    contract = auditor.digest({"config": config, "inputs": {}, "sources": {}})
    manifest["contract_sha256"] = contract
    checkpoint = {"contract_sha256": contract}
    assert auditor.inspect_arm(mode, manifest, checkpoint, {}) == {"ok": True}
    assert calls == [{"mode": "joint", "steps": 4096}]
    manifest["policy_sampled_states"] = 0
    with pytest.raises(ValueError, match="sample accounting"):
        auditor.inspect_arm(mode, manifest, checkpoint, {})
    manifest["policy_sampled_states"] = 131072
    checkpoint["contract_sha256"] = "wrong"
    with pytest.raises(ValueError, match="contract"):
        auditor.inspect_arm(mode, manifest, checkpoint, {})
