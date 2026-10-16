import json
from types import SimpleNamespace

import pytest
import run_continuous_train_mcts_v1 as stream
from build_mcts_training_supervision import atomic, sha


def test_task_pass_covers_each_root_both_costs_and_independent_seeds():
    n = 8
    first = [stream.task(i, n) for i in range(n * 4)]
    assert len(set(first)) == 32
    for root in range(n):
        assert {(w, seed) for r, w, seed in first if r == root} == {
            (0, 20271420), (0, 20271421), (1, 20271420), (1, 20271421)}
    second = [stream.task(i, n) for i in range(n * 4, n * 8)]
    assert {seed for _, _, seed in second} == {20271422, 20271423}


def saved_batch(tmp_path):
    state = {"next_task": 32, "contract": {"fixed": True},
             "shards": [{"file": f"attempt-{i:06d}.json", "sha256": "x"}
                        for i in range(32)],
             "batches": {"0": {"file": "checkpoint.pt", "sha256": "frozen"}},
             "audits": []}
    batch = tmp_path / "batch-000000"
    batch.mkdir()
    manifest = {"schema": "rf-knots-continuous-MCTS-batch-v1",
                "contract": state["contract"], "actor": state["batches"]["0"],
                "batch_id": 0, "first_task": 0, "shards": state["shards"]}
    atomic(batch / "manifest.json", manifest)
    audit = batch / "audit"
    audit.mkdir()
    atomic(audit / "report.json", {"partial": False, "conditions": 32,
                                   "batch_manifest_sha256": sha(batch / "manifest.json")})
    return state, batch


def test_resume_reconciles_completed_audit_without_duplicate_process(tmp_path, monkeypatch):
    state, batch = saved_batch(tmp_path)
    monkeypatch.setattr(stream.subprocess, "run", lambda *a, **k: pytest.fail("duplicate audit"))
    stream.audit_pending(SimpleNamespace(out=tmp_path), state, tmp_path / "manifest.json")
    assert len(state["audits"]) == 1
    assert state["audits"][0]["sha256"] == sha(batch / "audit" / "report.json")
    stream.audit_pending(SimpleNamespace(out=tmp_path), state, tmp_path / "manifest.json")
    assert len(state["audits"]) == 1


def test_resume_rejects_changed_immutable_batch(tmp_path):
    state, batch = saved_batch(tmp_path)
    changed = json.loads((batch / "manifest.json").read_text())
    changed["actor"]["sha256"] = "changed"
    atomic(batch / "manifest.json", changed)
    with pytest.raises(ValueError, match="immutable batch manifest differs"):
        stream.audit_pending(SimpleNamespace(out=tmp_path), state, tmp_path / "manifest.json")
