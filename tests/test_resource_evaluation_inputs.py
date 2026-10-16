import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("resource_evaluation_inputs")


def test_checkpoint_requires_feature_arm_stage_and_pair_binding(module, tmp_path):
    run = tmp_path / "run"
    path = run / "units" / "step.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fixture, not loaded as torch")
    entry = dict(file="step.pt", sha256=module.sha(path), steps=0)
    audit = dict(
        status="passed",
        arms={"units": {"initial": entry}, "raw": {}},
        initial_tensors_identical=True,
        data_rng_aligned=True,
        inputs={str(path): module.sha(path)},
    )
    assert module.model_entry(run, audit, "units", "initial") == (entry, path)
    for changed in (
        {**audit, "status": "pending"},
        {**audit, "data_rng_aligned": False},
        {**audit, "inputs": {}},
    ):
        with pytest.raises(ValueError):
            module.model_entry(run, changed, "units", "initial")
    for arm, stage in (("q254", "initial"), ("units", "progress")):
        with pytest.raises(ValueError):
            module.model_entry(run, audit, arm, stage)
    entry["steps"] = 16
    with pytest.raises(ValueError, match="stage/file"):
        module.model_entry(run, audit, "units", "initial")


@pytest.mark.parametrize("kind", ["short-u", "general"])
def test_incomplete_or_inadequate_panel_never_scores(module, tmp_path, monkeypatch, kind):
    monkeypatch.setitem(module.PANEL_ROOTS, kind, tmp_path)
    state = tmp_path / ("manifest.json" if kind == "short-u" else "progress.json")
    for status, ready in (("building", True), ("completed", False)):
        state.write_text(json.dumps(dict(status=status, evaluation_ready=ready)))
        with pytest.raises(ValueError, match="not completed and adequate"):
            module.load_panel(kind)
