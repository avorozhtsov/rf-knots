import importlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("run_long_general_evaluation_v2")


def test_exact_frozen18_job_matrix(module, tmp_path):
    jobs = module.jobs(tmp_path / "out", tmp_path / "train", tmp_path / "pgx")
    assert len(jobs) == len({j["name"] for j in jobs}) == len({j["output"] for j in jobs}) == 18
    assert Counter(j["kind"] for j in jobs) == {
        "training": 1,
        "cache": 1,
        "metric": 6,
        "search": 5,
        "search-audit": 5,
    }
    searches = [{k: j[k] for k in ("arm", "stage", "prior")} for j in jobs if j["kind"] == "search"]
    assert searches == module.SEARCH_MATRIX
    for j in jobs:
        assert "--out" in j["command"] and "--resume" not in j["command"]
        assert "--seed" not in j["command"] and "--simulations" not in j["command"]


def worker_fixture(module, monkeypatch, tmp_path, kind="search"):
    (tmp_path / "logs").mkdir()
    output = tmp_path / "result.json"
    job = dict(name="fixture", kind=kind, output=str(output), command=["python", "-u", "worker.py"])
    stored = dict(name=job["name"], output=str(output), sha256="verified")
    monkeypatch.setattr(module, "verify_job", lambda *a: stored)
    monkeypatch.setattr(module, "processes", lambda: [])
    return job, stored, output


def test_completed_artifact_prevents_dispatch(module, monkeypatch, tmp_path):
    job, stored, output = worker_fixture(module, monkeypatch, tmp_path)
    output.write_text(json.dumps(dict(status="completed")))
    monkeypatch.setattr(module, "live_commands", lambda *a: [])
    monkeypatch.setattr(
        module.subprocess, "Popen", lambda *a, **k: pytest.fail("duplicate dispatch")
    )
    assert module.await_job(job, tmp_path, {}, tmp_path / "state.json", tmp_path / "run") == stored


def test_live_worker_is_adopted_until_actual_exit(module, monkeypatch, tmp_path):
    job, stored, _ = worker_fixture(module, monkeypatch, tmp_path)
    listings = iter(([4321], [4321], []))
    waits = []
    monkeypatch.setattr(module, "live_commands", lambda *a: next(listings))
    monkeypatch.setattr(module.time, "sleep", lambda s: waits.append(s))
    monkeypatch.setattr(
        module.subprocess, "Popen", lambda *a, **k: pytest.fail("duplicate dispatch")
    )
    state = {}
    assert (
        module.await_job(job, tmp_path, state, tmp_path / "state.json", tmp_path / "run") == stored
    )
    assert waits == [30] and state["current"]["pid"] == 4321


@pytest.mark.parametrize(
    "kind,exists,expected",
    [("search", False, False), ("search", True, True), ("metric", False, False)],
)
def test_new_or_resumed_worker_command(module, monkeypatch, tmp_path, kind, exists, expected):
    job, stored, output = worker_fixture(module, monkeypatch, tmp_path, kind)
    if exists:
        output.write_text(json.dumps(dict(status="evaluating")))
    monkeypatch.setattr(module, "live_commands", lambda *a: [])
    calls = []

    def launch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(pid=1234, poll=lambda: 0)

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    assert module.await_job(job, tmp_path, {}, tmp_path / "state.json", tmp_path / "run") == stored
    assert len(calls) == 1 and (calls[0][-1] == "--resume") == expected


@pytest.mark.parametrize("problem", ["duplicate", "partial", "failed"])
def test_unsafe_dispatch_stops_without_retry_or_overwrite(module, monkeypatch, tmp_path, problem):
    job, _, output = worker_fixture(module, monkeypatch, tmp_path, "metric")
    if problem == "partial":
        output.write_text(json.dumps(dict(status="partial")))
    monkeypatch.setattr(
        module, "live_commands", lambda *a: [1, 2] if problem == "duplicate" else []
    )
    calls = []

    def launch(*a, **kw):
        calls.append(1)
        return SimpleNamespace(pid=1234, poll=lambda: 7)

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError):
        module.await_job(job, tmp_path, {}, tmp_path / "state.json", tmp_path / "run")
    assert len(calls) == int(problem == "failed")
    if problem == "partial":
        assert json.loads(output.read_text()) == {"status": "partial"}


def test_controller_interruption_resume_and_completed_bytes(module, monkeypatch, tmp_path):
    """Dispatch lifecycle fixture, not an actual model evaluation."""
    run, out, pgx = (tmp_path / name for name in ("training", "evaluation", "pgx"))
    run.mkdir()
    (run / "manifest.json").write_text("{}")
    monkeypatch.setattr(
        module, "ready", lambda _: dict(contract={"inputs": {}}, contract_sha256="training")
    )
    planned = module.jobs(out, run, pgx)

    def verify(job, *_):
        return dict(name=job["name"], output=job["output"], sha256=module.sha(Path(job["output"])))

    monkeypatch.setattr(module, "verify_job", verify)
    seen = []

    def finish(job, *args):
        seen.append(job["name"])
        if len(seen) == 4:
            raise RuntimeError("fixture child interruption")
        path = Path(job["output"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(status="fixture-only", name=job["name"])))
        return verify(job)

    monkeypatch.setattr(module, "await_job", finish)
    summary = importlib.import_module("summarize_long_general_experiment")
    monkeypatch.setattr(
        summary, "summarize_verified", lambda _: dict(status="completed", scope="dispatch fixture")
    )
    argv = ["controller", "--run", str(run), "--out", str(out), "--pgx-root", str(pgx)]
    monkeypatch.setattr(module.sys, "argv", argv)
    with pytest.raises(RuntimeError, match="fixture child"):
        module.main()
    state = module.read(out / "manifest.json")
    assert state["status"] == "needs_attention" and len(state["completed"]) == 3
    monkeypatch.setattr(module.sys, "argv", argv + ["--resume"])
    module.main()
    assert seen[4:] == [j["name"] for j in planned[3:]]
    state = module.read(out / "manifest.json")
    assert state["status"] == "completed" and len(state["completed"]) == 18
    before = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    module.main()
    assert before == {
        str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()
    }
    state["completed"].pop()
    (out / "manifest.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="full matrix"):
        module.main()


@pytest.mark.parametrize("mutation", [None, "arm", "checkpoint", "positive", "sizes", "source"])
def test_metric_child_verified_against_frozen_records(module, monkeypatch, tmp_path, mutation):
    import torch
    from test_long_general_summary import metric_fixture

    result, records = metric_fixture()
    run = tmp_path / "run"
    run.mkdir()
    panel = tmp_path / "corpus.json"
    panel.write_text(json.dumps({"records": {"test": records}}))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not loaded: identity fixture only")
    source = Path("scripts/evaluate_long_general_distances.py").resolve()
    pair = tmp_path / "training-pair-audit.json"
    pair.write_text(
        json.dumps(
            dict(arms={a: {"size_counters": {"metric_pairs": 65536}} for a in ("control", "long")})
        )
    )
    (run / "manifest.json").write_text(
        json.dumps(
            dict(
                contract_sha256="training",
                contract={
                    "evaluation": dict(long_corpus=str(panel), long_corpus_sha256=module.sha(panel))
                },
            )
        )
    )
    report = dict(
        schema="rf-knots-long-general-distances-v1",
        status="completed",
        arm="long",
        stage="final",
        panel=str(panel),
        panel_sha256=module.sha(panel),
        checkpoint_sha256=module.sha(checkpoint),
        training_contract_sha256="training",
        weights_unchanged=True,
        optimizer_steps=0,
        device="cpu",
        threads=1,
        microbatch=1,
        torch_version=torch.__version__,
        observed_training_sizes={a: {"metric_pairs": 65536} for a in ("control", "long")},
        results={"test": result},
        inputs={str(p): module.sha(p) for p in (source, pair, panel, checkpoint)},
    )
    if mutation == "arm":
        report["arm"] = "control"
    elif mutation == "checkpoint":
        report["checkpoint_sha256"] = "changed"
    elif mutation == "positive":
        report["results"]["test"]["certified_positive"]["rows"] += 1
    elif mutation == "sizes":
        report["observed_training_sizes"]["long"]["metric_pairs"] += 1
    elif mutation == "source":
        report["inputs"].pop(str(source))
    path = tmp_path / "metric.json"
    path.write_text(json.dumps(report))
    job = dict(
        name="metric-long-final-general",
        kind="metric",
        arm="long",
        stage="final",
        panel="general",
        output=str(path),
        command=["python", "-u", str(source)],
    )
    monkeypatch.setattr(module, "model_entry", lambda *a: ({}, checkpoint))
    if mutation:
        with pytest.raises(ValueError):
            module.verify_job(job, tmp_path, run)
    else:
        assert module.verify_job(job, tmp_path, run)["sha256"] == module.sha(path)


@pytest.mark.parametrize("mutation", [None, "hash", "config", "incomplete"])
def test_ready_uses_producer_hash_codec(module, monkeypatch, tmp_path, mutation):
    evaluation = {"long_corpus": "long/corpus.json", "short_u_panel": "short/panel.json"}
    contract = {"config": module.CONFIG, "inputs": {}, "evaluation": evaluation}
    parent = dict(
        schema=module.SCHEMA,
        contract=contract,
        contract_sha256=module.training_digest(contract),
        status="completed",
        completed_arms=module.CONFIG["arms"],
        current_arm=None,
    )
    assert module.digest(contract) != parent["contract_sha256"]
    if mutation == "hash":
        parent["contract_sha256"] = module.digest(contract)
    elif mutation == "config":
        parent["contract"]["config"] = {**module.CONFIG, "steps": 1}
        parent["contract_sha256"] = module.training_digest(parent["contract"])
    elif mutation == "incomplete":
        parent["status"] = "training"
    monkeypatch.setattr(module, "check_pins", lambda _: None)
    monkeypatch.setattr(module, "evaluation_contract", lambda *a: (evaluation, {}))
    (tmp_path / "manifest.json").write_text(json.dumps(parent))
    if mutation is None:
        assert module.ready(tmp_path) == parent
    else:
        with pytest.raises((ValueError, RuntimeError)):
            module.ready(tmp_path)
