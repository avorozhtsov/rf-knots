import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("run_joint_policy_metric_evaluation")


@pytest.fixture
def gates(controller, tmp_path, monkeypatch):
    runs = {name: tmp_path / name for name in ("parent", "joint", "metric-control")}
    panels = {name: tmp_path / name / "panel.json" for name in ("short-u", "gordian")}
    monkeypatch.setattr(controller, "RUNS", runs)
    monkeypatch.setattr(controller, "PANELS", panels)
    for arm in ("joint", "metric-control"):
        directory = runs[arm]
        directory.mkdir()
        (directory / "checkpoint.pt").write_bytes(b"test fixture, not a torch model")
        manifest = {
            "status": "completed",
            "completed_steps": 1024,
            "config": {**controller.EXPECTED_CONFIG, "mode": arm},
            "initial_checkpoint_sha256": controller.ACTOR_SHA,
            "checkpoint_sha256": controller.sha(directory / "checkpoint.pt"),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest))
    for path in panels.values():
        path.parent.mkdir()
        path.write_text(json.dumps({"complete": True}))
        path.with_name("independence-audit.json").write_text(
            json.dumps({"status": "passed", "inputs": {str(path): controller.sha(path)}})
        )
    return runs, panels


def process(pid, argv):
    return pid, argv, " ".join(argv)


def test_frozen_matrix_contains_all_models_panels_and_search_audits(controller, tmp_path):
    jobs = controller.jobs(tmp_path)
    assert len(jobs) == 15 and len({j["output"] for j in jobs}) == 15
    assert jobs[0]["name"] == "training-pair-audit"
    assert [j["name"] for j in jobs[7:]] == [
        name
        for arm in ("parent", "joint", "metric-control", "uniform")
        for name in (f"mcts-{arm}", f"mcts-audit-{arm}")
    ]
    assert sum(j["command"][2].endswith("evaluate_strand_graph_panel.py") for j in jobs) == 6
    assert all("train_mcts_policy_metric.py" not in " ".join(j["command"]) for j in jobs)


def test_completed_gates_pin_files_and_do_not_require_old_pids(controller, gates):
    waiting, pins = controller.gate_snapshot([])
    assert waiting == [] and len(pins) == 6
    controller.check_pins(pins)


def test_incomplete_training_requires_exact_live_worker_not_stale_manifest(controller, gates):
    runs, _ = gates
    p = runs["metric-control"] / "manifest.json"
    m = controller.read(p)
    m.update(status="training", completed_steps=128, pid=123)
    p.write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="stopped before completion"):
        controller.gate_snapshot([])
    argv = controller.command(
        "train_mcts_policy_metric", "--mode", "metric-control", "--out", runs["metric-control"]
    )
    waiting, _ = controller.gate_snapshot([process(456, argv)])
    assert waiting == [{"kind": "training", "arm": "metric-control", "pid": 456, "steps": 128}]
    with pytest.raises(RuntimeError, match="duplicate training"):
        controller.gate_snapshot([process(456, argv), process(457, argv + ["--resume"])])


def test_incomplete_audit_requires_exact_live_verifier(controller, gates):
    _, panels = gates
    panel = panels["gordian"]
    output = panel.with_name("independence-audit.json")
    output.unlink()
    with pytest.raises(RuntimeError, match="no live verifier"):
        controller.gate_snapshot([])
    argv = controller.command("audit_joint_policy_metric_panel", "--panel", panel, "--out", output)
    waiting, _ = controller.gate_snapshot([process(456, argv)])
    assert waiting == [{"kind": "data-audit", "panel": "gordian", "pid": 456}]


def test_explicitly_resumed_mcts_worker_is_adoptable(controller, tmp_path):
    job = next(j for j in controller.jobs(tmp_path) if j["name"] == "mcts-joint")
    assert controller.job_workers(job, [process(21, job["command"] + ["--resume"])]) == [21]
    assert controller.job_workers(job, [process(21, job["command"] + ["--different-option"])]) == []


def test_corrupt_completed_checkpoint_fails_gate(controller, gates):
    runs, _ = gates
    (runs["joint"] / "checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint hash"):
        controller.gate_snapshot([])


def test_empty_success_report_does_not_satisfy_job(controller, tmp_path):
    path = tmp_path / "fake.json"
    path.write_text(json.dumps({"status": "completed"}))
    with pytest.raises(ValueError, match="provenance"):
        controller.verify_job({"name": "training-pair-audit", "output": str(path)})


@pytest.mark.parametrize("fault", [None, "nan", "label", "coverage", "group"])
def test_metric_output_identity_coverage_and_finiteness(controller, tmp_path, fault):
    run = tmp_path / "model"
    run.mkdir()
    (run / "checkpoint.pt").write_bytes(b"test model identity only")
    source = tmp_path / "source.py"
    source.write_text("# test source identity")
    panel = tmp_path / "panel.json"
    record = {
        "left_fingerprint": [3, -2],
        "right_fingerprint": [1, 0],
        "interval": {"lower": 0, "upper": 1},
    }
    panel.write_text(json.dumps({"validation": [record], "size_shift": [record]}))
    prediction = {
        "panel_index": 0,
        "left_fingerprint": [3, -2],
        "right_fingerprint": [1, 0],
        "fingerprint_group": "3:2",
        "lower": 1,
        "upper": 1,
        "prediction": 1.2,
        "violation": 0.2,
    }
    if fault == "nan":
        prediction["prediction"] = float("nan")
    elif fault == "label":
        prediction["lower"] = 0
    elif fault == "group":
        prediction["fingerprint_group"] = "wrong"
    report = {
        "status": "completed",
        "run": str(run),
        "checkpoint_sha256": controller.sha(run / "checkpoint.pt"),
        "panel_sha256": controller.sha(panel),
        "initial": False,
        "bounds": "determinant",
        "normalization": "none",
        "code_sha256": {str(source): controller.sha(source)},
        **{
            view: {"count": 1, "predictions": [prediction]} for view in ("validation", "size_shift")
        },
    }
    if fault == "coverage":
        report["size_shift"]["count"] = 2
    output = tmp_path / "metric.json"
    output.write_text(json.dumps(report))
    job = {
        "name": "metric-parent-short-u",
        "output": str(output),
        "command": controller.command(
            "evaluate_strand_graph_panel", "--run", run, "--panel", panel
        ),
    }
    if fault:
        with pytest.raises(ValueError):
            controller.verify_job(job)
    else:
        assert controller.verify_job(job)["sha256"] == controller.sha(output)
