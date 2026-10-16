import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def sequence(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("run_raster_log_evaluation")


def test_full_fixed_job_matrix(sequence, tmp_path):
    planned = sequence.jobs(tmp_path / "out", tmp_path / "train", tmp_path / "pgx")
    assert len(planned) == len({j["name"] for j in planned}) == 13
    assert len({j["output"] for j in planned}) == 13
    assert planned[0]["kind"] == "training"
    assert sum(j["kind"] == "metric" for j in planned) == 8
    assert [j["kind"] for j in planned[-4:]] == ["search", "search-audit"] * 2
    metrics = {(j["arm"], j["state"], j["panel"]) for j in planned if j["kind"] == "metric"}
    assert metrics == {
        (a, s, p)
        for a in ("q254", "random")
        for s in ("initial", "final")
        for p in ("short-u", "general")
    }
    assert not any("train_raster_log_pair" in " ".join(j["command"]) for j in planned)


def waiting_fixture(sequence, path):
    pgx = Path("/canonical/pgx")
    parent = {
        "contract_sha256": "fixed",
        "config": sequence.CONFIG,
        "status": "running",
        "pid": 123,
        "completed_arms": [],
    }
    (path / "manifest.json").write_text(json.dumps(parent))
    (path / "q254").mkdir()
    (path / "q254" / "manifest.json").write_text(
        json.dumps({"status": "training", "phase": "optimizer", "completed_steps": 16})
    )
    cmd = sequence.command("run_raster_log_pair", "--pgx-root", pgx, "--out", path)
    listing = [(123, cmd, " ".join(cmd))]
    return pgx, parent, listing


def test_wait_requires_real_unique_launcher_and_preserves_queued_arm(sequence, tmp_path):
    pgx, _, listing = waiting_fixture(sequence, tmp_path)
    ready, observations = sequence.training_snapshot(tmp_path, pgx, "fixed", listing)
    assert not ready and observations[0]["steps"] == 16
    assert observations[1]["phase"] == "queued"
    for bad in ([], listing + [(124, listing[0][1], listing[0][2])]):
        with pytest.raises(RuntimeError, match="unique live"):
            sequence.training_snapshot(tmp_path, pgx, "fixed", bad)


def test_wait_rejects_failed_or_changed_training(sequence, tmp_path):
    pgx, parent, listing = waiting_fixture(sequence, tmp_path)
    with pytest.raises(ValueError, match="contract"):
        sequence.training_snapshot(tmp_path, pgx, "another", listing)
    parent["status"] = "failed"
    (tmp_path / "manifest.json").write_text(json.dumps(parent))
    with pytest.raises(RuntimeError, match="training failed"):
        sequence.training_snapshot(tmp_path, pgx, "fixed", listing)


def test_completed_training_requires_both_final_arms(sequence, tmp_path):
    pgx, parent, _ = waiting_fixture(sequence, tmp_path)
    parent.update(status="completed", completed_arms=["q254", "random"])
    (tmp_path / "manifest.json").write_text(json.dumps(parent))
    with pytest.raises(ValueError, match="both final"):
        sequence.training_snapshot(tmp_path, pgx, "fixed", [])
    (tmp_path / "random").mkdir()
    for arm in ("q254", "random"):
        (tmp_path / arm / "manifest.json").write_text(
            json.dumps({"status": "completed", "completed_steps": sequence.CONFIG["steps"]})
        )
    assert sequence.training_snapshot(tmp_path, pgx, "fixed", []) == (True, [])


def test_summary_refuses_incomplete_matrix(sequence, tmp_path):
    summary = importlib.import_module("summarize_raster_log_experiment")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "training_run": "train",
                "pgx_root": "pgx",
                "status": "waiting_for_training",
                "jobs": [],
                "completed": [],
            }
        )
    )
    with pytest.raises(ValueError, match="complete13"):
        summary.summarize_verified(tmp_path)


def test_metric_labels_and_constants_are_recomputed(sequence):
    summary = importlib.import_module("summarize_raster_log_experiment")
    records = [
        {
            "left_fingerprint": [3, -2],
            "right_fingerprint": [1, 0],
            "interval": {"lower": 0, "upper": 1},
        }
    ]
    predictions = [
        {
            "panel_index": 0,
            "fingerprint_group": "3:2",
            "left_fingerprint": [3, -2],
            "right_fingerprint": [1, 0],
            "lower": 1,
            "upper": 1,
            "prediction": 0.5,
            "violation": 0.5,
        }
    ]
    result = summary.validate_predictions(predictions, records)
    assert result["mean_interval_violation"] == 0.5
    assert result["constants"]["1"]["mean_interval_violation"] == 0
    predictions[0]["lower"] = 0
    with pytest.raises(ValueError, match="identity"):
        summary.validate_predictions(predictions, records)


def test_wrong_checkpoint_cannot_be_adopted_as_completed_metric(sequence, tmp_path):
    pair = {"arms": {"q254": {"initial_sha256": "initial", "checkpoint_sha256": "final"}}}
    (tmp_path / "training-pair-audit.json").write_text(json.dumps(pair))
    output = tmp_path / "metric.json"
    output.write_text(
        json.dumps(
            {"status": "completed", "inputs": {}, "arm": "q254", "checkpoint_sha256": "other"}
        )
    )
    job = {"kind": "metric", "arm": "q254", "state": "initial", "output": str(output)}
    with pytest.raises(ValueError, match="model differs"):
        sequence.verify_job(job, tmp_path)
