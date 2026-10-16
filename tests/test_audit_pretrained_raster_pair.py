import copy
import importlib
import json
import shutil
from pathlib import Path

import pytest
import torch


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("audit_pretrained_raster_pair")


def fixture_files(audit, tmp_path):
    state = {
        key: torch.zeros(2)
        for key in ("encoder.tiles.weight", "encoder.heads.weight", "head.weight")
    }
    architecture = {"fixture": True}
    initial = {"model": state}
    config = {**audit.CONFIG, "arm": "q254"}
    contract = audit.digest({"config": config, "inputs": {}, "sources": {}})
    final = {
        "model": {k: v + 1 for k, v in state.items()},
        "objective": {"scale": torch.ones(1)},
        "architecture": architecture,
        "steps": audit.CONFIG["steps"],
        "contract_sha256": contract,
        "draw_digest": "same-data",
        "optimizer": {
            "state": {0: {"step": torch.tensor(audit.CONFIG["steps"]), "exp_avg": torch.zeros(2)}},
            "param_groups": [
                {"lr": audit.CONFIG["lr"], "weight_decay": audit.CONFIG["weight_decay"]}
            ],
        },
    }
    torch.save(initial, tmp_path / "initial.pt")
    manifest = {
        "status": "completed",
        "completed_steps": audit.CONFIG["steps"],
        "config": config,
        "architecture": architecture,
        "inputs": {},
        "sources": {},
        "contract_sha256": contract,
        "draw_digest": "same-data",
        "initial_checkpoint_sha256": audit.sha(tmp_path / "initial.pt"),
        "metric_sampled_rows": 32 * audit.CONFIG["steps"],
        "positive_sampled_states": 8 * audit.CONFIG["steps"],
        "relative_sampled_rows": 8 * audit.CONFIG["steps"],
    }
    persist(audit, tmp_path, manifest, final)
    return state, architecture, manifest, final


def persist(audit, path, manifest, final):
    torch.save(final, path / "progress.pt")
    shutil.copy2(path / "progress.pt", path / "checkpoint.pt")
    manifest["progress_sha256"] = audit.sha(path / "progress.pt")
    manifest["checkpoint_sha256"] = audit.sha(path / "checkpoint.pt")
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_training_audit_accepts_valid_fixture_and_rejects_counter_tampering(audit, tmp_path):
    state, architecture, manifest, final = fixture_files(audit, tmp_path)
    parent = {"inputs": {}, "sources": {}}
    _, result = audit.audit_arm(tmp_path, "q254", state, architecture, parent)
    assert result["completed_steps"] == audit.CONFIG["steps"]
    manifest["metric_sampled_rows"] -= 1
    persist(audit, tmp_path, manifest, final)
    with pytest.raises(ValueError, match="accounting"):
        audit.audit_arm(tmp_path, "q254", state, architecture, parent)


def test_training_audit_rejects_nonfinite_even_with_consistent_file_hashes(audit, tmp_path):
    state, architecture, manifest, final = fixture_files(audit, tmp_path)
    final = copy.deepcopy(final)
    final["optimizer"]["state"][0]["exp_avg"][0] = float("nan")
    persist(audit, tmp_path, manifest, final)
    with pytest.raises(ValueError, match="nonfinite optimizer"):
        audit.audit_arm(tmp_path, "q254", state, architecture, {"inputs": {}, "sources": {}})


def test_incomplete_arm_cannot_be_promoted(audit, tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "training"}))
    with pytest.raises(ValueError, match="incomplete"):
        audit.audit_arm(tmp_path, "q254", {}, {}, {})
