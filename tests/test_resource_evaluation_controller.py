import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return (
        importlib.import_module("run_resource_evaluation"),
        importlib.import_module("summarize_resource_experiment"),
    )


def test_matrix_covers_initial_final_and_both_final_uniform_controls(modules):
    controller, _ = modules
    jobs = controller.jobs(Path("out"), Path("run"), Path("pgx"))
    assert len(jobs) == len({j["name"] for j in jobs}) == len({j["output"] for j in jobs}) == 24
    searches = [j for j in jobs if j["kind"] == "search"]
    assert [(j["arm"], j["stage"], j["prior"]) for j in searches] == controller.search_matrix()
    assert len([j for j in jobs if j["kind"] == "metric"]) == 8
    assert len([j for j in jobs if j["kind"] == "search-audit"]) == 6
    for job in searches:
        following = jobs[jobs.index(job) + 1]
        assert following["kind"] == "search-audit"
        assert following["output"] == str(Path(job["output"]).parent / "replay-audit.json")


def test_partial_matrix_cannot_publish_summary(modules):
    controller, summary = modules
    jobs = controller.jobs(Path("out"), Path("run"), Path("pgx"))
    completed = [{"name": j["name"], "output": j["output"]} for j in jobs]
    state = dict(status="scored", jobs=jobs, completed=completed)
    summary.require_complete_matrix(state, jobs)
    for bad in (
        {**state, "status": "evaluating"},
        {**state, "completed": completed[:-1]},
        {**state, "completed": completed[::-1]},
        {**state, "jobs": jobs[:-1]},
    ):
        with pytest.raises(ValueError, match="complete24-job"):
            summary.require_complete_matrix(bad, jobs)


def test_metric_deltas_keep_group_and_record_weighting_distinct(modules):
    _, summary = modules
    baseline = [
        dict(panel_index=i, fingerprint_group=g, lower=1, upper=1, violation=1.0)
        for i, g in enumerate(("a", "a", "b"))
    ]
    candidate = [
        {**r, "violation": error} for r, error in zip(baseline, (0.0, 0.0, 2.0), strict=True)
    ]
    result = summary.compare_metrics(baseline, candidate)
    assert result["mean_group_error_delta"] == 0
    assert result["mean_record_error_delta"] == pytest.approx(-1 / 3)


def test_unfinished_training_blocks_controller_without_mutation(modules, monkeypatch):
    controller, _ = modules
    parent = dict(config=controller.TRAIN_CONFIG, status="training", completed_arms=[])
    monkeypatch.setattr(controller, "read", lambda _: parent)
    with pytest.raises(RuntimeError, match="does not alter or restart"):
        controller.ready(Path("run"))
    assert parent["status"] == "training"


def metric_fixture():
    scorer = importlib.import_module("evaluate_raster_log_distances")
    records = [
        dict(
            left_fingerprint=[3, -2],
            right_fingerprint=[1, 0],
            interval=dict(lower=0, upper=1),
        )
    ]
    rows = [dict(lower=1, upper=1, prediction=0.5, violation=0.5)]
    grouped = scorer.group_report(rows, records)
    return records, dict(
        predictions=rows,
        statistics=scorer.metric_stats(rows),
        by_fingerprint=grouped,
        ranking=scorer.interval_ranking(rows),
        constant_baselines={str(c): scorer.constant_baseline(rows, c) for c in (0, 1, 2)},
        encoded_endpoints=2,
    )


def test_metric_consumer_checks_actual_scorer_schema_without_mutation(modules):
    _, summary = modules
    records, report = metric_fixture()
    before = copy.deepcopy(report)
    result = summary.validate_metric_report(report, records)
    assert result["constants"]["1"]["mean_interval_violation"] == 0
    assert report == before


@pytest.mark.parametrize(
    "field", ["statistics", "ranking", "by_fingerprint", "constant_baselines", "encoded_endpoints"]
)
def test_metric_consumer_rejects_inconsistent_redundant_fields(modules, field):
    _, summary = modules
    records, report = metric_fixture()
    report[field] = 0 if field == "encoded_endpoints" else {}
    with pytest.raises(ValueError, match="differ"):
        summary.validate_metric_report(report, records)


def test_wrong_checkpoint_cannot_be_adopted_as_resource_metric(modules, tmp_path):
    controller, _ = modules
    pair = dict(arms={"units": {"final": {"sha256": "expected"}}})
    (tmp_path / "training-pair-audit.json").write_text(json.dumps(pair))
    output = tmp_path / "metric.json"
    output.write_text(json.dumps(dict(status="completed", inputs={}, checkpoint_sha256="other")))
    job = dict(kind="metric", arm="units", stage="final", output=str(output))
    with pytest.raises(ValueError, match="another checkpoint"):
        controller.verify_job(job, tmp_path)


def test_existing_general_consumer_can_be_adopted(modules):
    controller, _ = modules
    output = controller.PANEL_ROOTS["general"] / "consumer-audit.json"
    if not output.exists():
        pytest.skip("optional completed local panel is not present")
    job = dict(kind="panel", panel="general", output=str(output), name="input-general")
    result = controller.verify_job(job, Path("unused-evaluation-root"))
    assert result["sha256"] == controller.sha(output)


@pytest.fixture
def driver(modules, monkeypatch, tmp_path):
    """Exercise the real controller driver with isolated fake child computation.

    This tests lifecycle behavior, not scientific model scoring or replay.
    No production training files/processes are accessed by these fixtures.
    """
    controller, summary = modules
    out = tmp_path / "evaluation"
    output = out / "child.json"
    job = dict(
        name="fixture-search",
        command=[controller.sys.executable, "-u", "scripts/evaluate_resource_search.py"],
        output=str(output),
        kind="search",
    )
    parent = dict(inputs={}, sources={}, contract_sha256="fixture-training-contract")
    monkeypatch.setattr(controller, "ready", lambda _: parent)
    monkeypatch.setattr(controller, "jobs", lambda *_: [job])
    monkeypatch.setattr(controller, "processes", lambda: [])
    monkeypatch.setattr(controller, "live_commands", lambda *_: [])
    monkeypatch.setattr(summary, "summarize_verified", lambda _: dict(status="fixture-only"))
    calls = []

    def verify(job, _):
        assert controller.read(output)["status"] == "completed"
        return dict(name=job["name"], output=job["output"], sha256=controller.sha(output))

    def popen(argv, **_):
        calls.append(argv)
        controller.atomic(output, dict(status="completed"))
        return SimpleNamespace(pid=12345, poll=lambda: 0)

    monkeypatch.setattr(controller, "verify_job", verify)
    monkeypatch.setattr(controller.subprocess, "Popen", popen)

    def run(resume=False):
        monkeypatch.setattr(
            controller.sys,
            "argv",
            [
                "controller",
                "--run", str(tmp_path / "training"),
                "--pgx-root", str(tmp_path / "pgx"),
                "--out", str(out),
                *(["--resume"] if resume else []),
            ],
        )
        controller.main()

    return SimpleNamespace(
        controller=controller, out=out, output=output, job=job, calls=calls, run=run
    )


def test_driver_completion_and_completed_resume_launch_once(driver):
    driver.run()
    state = driver.controller.read(driver.out / "manifest.json")
    assert state["status"] == "completed"
    assert state["summary_sha256"] == driver.controller.sha(driver.out / "summary.json")
    driver.run(resume=True)
    assert driver.calls == [driver.job["command"]]


def test_driver_reuses_completed_uncommitted_child(driver):
    driver.run()
    manifest = driver.out / "manifest.json"
    state = driver.controller.read(manifest)
    state.update(status="evaluating", completed=[], current=dict(name=driver.job["name"], pid=42))
    driver.controller.atomic(manifest, state)
    driver.run(resume=True)
    assert len(driver.calls) == 1
    assert driver.controller.read(manifest)["status"] == "completed"


def test_driver_adopts_live_worker_without_launching(driver, monkeypatch):
    observations = iter(([76543], []))

    def live(*_):
        result = next(observations)
        if not result:
            driver.controller.atomic(driver.output, dict(status="completed"))
        return result

    monkeypatch.setattr(driver.controller, "live_commands", live)
    driver.run()
    assert driver.calls == []
    assert driver.controller.read(driver.out / "manifest.json")["status"] == "completed"


def test_driver_duplicate_workers_preserved_and_reported(driver, monkeypatch):
    monkeypatch.setattr(driver.controller, "live_commands", lambda *_: [1, 2])
    with pytest.raises(RuntimeError, match="duplicate matching"):
        driver.run()
    assert driver.calls == []
    assert driver.controller.read(driver.out / "manifest.json")["status"] == "needs_attention"


@pytest.mark.parametrize("kind", ["search", "metric"])
def test_driver_partial_failure_resume_only_for_search(driver, monkeypatch, kind):
    driver.job["kind"] = kind
    original_popen = driver.controller.subprocess.Popen

    def failed(argv, **_):
        driver.controller.atomic(driver.output, dict(status="evaluating", saved_progress=8))
        return SimpleNamespace(pid=12345, poll=lambda: 7)

    monkeypatch.setattr(driver.controller.subprocess, "Popen", failed)
    with pytest.raises(RuntimeError, match="child failed 7"):
        driver.run()
    partial = driver.output.read_bytes()
    monkeypatch.setattr(driver.controller.subprocess, "Popen", original_popen)
    if kind == "search":
        driver.run(resume=True)
        assert driver.calls == [driver.job["command"] + ["--resume"]]
    else:
        with pytest.raises(RuntimeError, match="partial nonresumable"):
            driver.run(resume=True)
        assert driver.calls == []
        assert driver.output.read_bytes() == partial
