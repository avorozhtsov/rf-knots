import importlib
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("run_graph_mixture_evaluation")


def arm(c, mode, *, completed=False):
    return {
        "status": "completed" if completed else "training",
        "config": {**c.CONFIG, "mode": mode},
        "initial_checkpoint_sha256": c.INITIAL_SHA,
        "completed_steps": c.CONFIG["steps"] if completed else 80,
    }


def test_serial_control_can_be_queued_but_not_without_live_launcher(controller):
    c = controller
    pair = {"status": "training"}
    manifests = {"graph": arm(c, "graph"), "control": None}
    workers = {"graph": [102], "control": []}
    rows = c.inspect_training(pair, manifests, [101], workers)
    assert [r["phase"] for r in rows] == ["worker", "queued-or-transition"]
    with pytest.raises(RuntimeError, match="no live launcher"):
        c.inspect_training(pair, manifests, [], workers)
    for duplicate in ({"graph": [102, 103], "control": []}, {"graph": [102], "control": [103]}):
        with pytest.raises(RuntimeError):
            c.inspect_training(pair, manifests, [101], duplicate)


def test_finalization_and_declared_steps_are_both_required(controller):
    c = controller
    manifests = {a: arm(c, a, completed=True) for a in ("graph", "control")}
    no_workers = {"graph": [], "control": []}
    assert c.inspect_training({"status": "completed"}, manifests, [], no_workers) == []
    assert (
        c.inspect_training({"status": "training"}, manifests, [101], no_workers)[0]["kind"]
        == "training-finalization"
    )
    with pytest.raises(RuntimeError, match="did not finalize"):
        c.inspect_training({"status": "training"}, manifests, [], no_workers)
    bad = deepcopy(manifests)
    bad["graph"]["completed_steps"] -= 1
    with pytest.raises(ValueError, match="declared steps"):
        c.inspect_training({"status": "completed"}, bad, [], no_workers)


def test_failed_or_changed_training_is_not_restart_permission(controller):
    c = controller
    manifests = {"graph": arm(c, "graph"), "control": None}
    workers = {"graph": [102], "control": []}
    with pytest.raises(RuntimeError, match="reported failure"):
        c.inspect_training({"status": "needs_attention"}, manifests, [101], workers)
    manifests["graph"]["config"]["lr"] *= 2
    with pytest.raises(ValueError, match="contract differs"):
        c.inspect_training({"status": "training"}, manifests, [101], workers)


def test_panel_wait_requires_live_command_and_completed_supply(controller):
    progress = {"status": "building", "next_index": 100, "groups": 3, "contract": {"seed": 1}}
    assert controller.panel_gate(progress, None, [20])[0]["attempts"] == 100
    with pytest.raises(RuntimeError, match="no live builder"):
        controller.panel_gate(progress, None, [])
    progress["status"] = "completed"
    panel = {"complete": False, "construction_complete": True, "contract": progress["contract"]}
    with pytest.raises(RuntimeError, match="inadequate"):
        controller.panel_gate(progress, panel, [])
    panel["complete"] = True
    assert controller.panel_gate(progress, panel, []) == []
    panel["contract"] = {"seed": 2}
    with pytest.raises(ValueError, match="contract differs"):
        controller.panel_gate(progress, panel, [])


def test_job_matrix_is_audits_then_final_models_with_no_empty_general_test(controller, tmp_path):
    planned = controller.jobs(tmp_path)
    assert len(planned) == 14
    assert [j["name"] for j in planned[:3]] == [
        "short-u-base-audit",
        "short-u-full-audit",
        "training-pair-audit",
    ]
    assert [j["gate"] for j in planned[:3]] == ["panel", "panel", "training"]
    assert sum(j["name"].startswith("metric-") for j in planned) == 3
    assert sum(j["name"].startswith("mcts-audit-") for j in planned) == 4
    assert all("gordian" not in j["name"] for j in planned)
    assert all("train_graph_metric_mixture.py" not in j["command"][2] for j in planned)
    assert all("build_constructive_short_u_panel.py" not in j["command"][2] for j in planned)


def test_live_detection_accepts_only_exact_command_including_resume(controller):
    argv = controller.command(
        "evaluate_graph_mixture_cost_mcts", "--arm", "graph", "--out", "sample"
    )
    listing = [
        (20, argv, ""),
        (21, argv + ["--resume"], ""),
        (22, argv[:-1] + ["other"], ""),
        (23, ["bash", *argv[1:]], ""),
    ]
    assert controller.live_commands(argv, listing) == [20, 21]
