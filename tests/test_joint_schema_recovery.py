import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def recovery(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("recover_joint_evaluation_schema")


def test_source_amendment_is_exact_and_no_other_change_is_accepted(recovery):
    before = "prefix\n" + recovery.OLD + "suffix\n"
    after = "prefix\n" + recovery.NEW + "suffix\n"
    recovery.accepted_amendment(before, after)
    for bad in (after + "# changed\n", before, after.replace("False", "True")):
        with pytest.raises(ValueError, match="only"):
            recovery.accepted_amendment(before, bad)


@pytest.mark.parametrize("status", ["evaluating", "completed", "waiting_for_gates"])
def test_no_recovery_of_live_or_successful_sequence(recovery, tmp_path, status):
    with pytest.raises(ValueError, match="exact seven-job"):
        recovery.validate_failed_state({"status": status}, tmp_path)


def test_existing_search_outputs_are_never_discarded(recovery, tmp_path):
    jobs = recovery.sequence.jobs(tmp_path)
    state = {
        "status": "attention_required",
        "error": "evaluation job failed: mcts-parent",
        "jobs": jobs,
        "completed": [{"name": j["name"], "output": j["output"]} for j in jobs[:7]],
        "current": {"name": "mcts-parent"},
    }
    (tmp_path / "mcts").mkdir()
    with pytest.raises(ValueError, match="artifacts exist"):
        recovery.validate_failed_state(state, tmp_path)


def test_normal_sequence_has_no_recovery_pins(recovery):
    assert recovery.verified_recovery({}) == {}


def test_recovery_copies_exact_prefix_and_retains_failed_lineage(recovery, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    previous, target = Path("old"), Path("new")
    original_code = "prefix\n" + recovery.OLD + "suffix\n"
    corrected_code = original_code.replace(recovery.OLD, recovery.NEW)
    source = Path(recovery.SOURCE)
    source.parent.mkdir()
    source.write_text(corrected_code)
    snapshot = previous / "source_snapshot" / recovery.SOURCE
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(original_code)
    (previous / "logs").mkdir()
    (previous / "logs" / "mcts-parent.log").write_text(
        "ValueError: search query is not an explicit-unknot task\n"
    )
    jobs = recovery.sequence.jobs(previous)

    def checked(job):
        return {
            "name": job["name"],
            "output": job["output"],
            "sha256": recovery.sha(Path(job["output"])),
        }

    for i, job in enumerate(jobs[:7]):
        path = Path(job["output"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fixture": i}))
    monkeypatch.setattr(recovery.sequence, "verify_job", checked)
    monkeypatch.setattr(recovery.sequence, "processes", lambda: [])
    state = {
        "status": "attention_required",
        "error": "evaluation job failed: mcts-parent",
        "jobs": jobs,
        "completed": list(map(checked, jobs[:7])),
        "inputs": {recovery.SOURCE: recovery.sha(snapshot)},
        "gate_pins": {},
        "current": {"name": "mcts-parent"},
    }
    manifest = previous / "manifest.json"
    manifest.write_text(json.dumps(state))
    initial_manifest_bytes = manifest.read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["recovery", "--previous", str(manifest), "--out", str(target)]
    )
    recovery.main()
    prepared = recovery.sequence.read(target / "manifest.json")
    assert manifest.read_bytes() == initial_manifest_bytes
    assert prepared["status"] == "prepared_recovery"
    assert len(prepared["completed"]) == 7
    assert (target / "source_snapshot" / recovery.SOURCE).read_text() == corrected_code
    assert snapshot.read_text() == original_code
    assert recovery.verified_recovery(prepared)[str(manifest)] == recovery.sha(manifest)
    Path(prepared["completed"][3]["output"]).write_text("corrupt")
    with pytest.raises(ValueError, match="changed"):
        recovery.verified_recovery(prepared)
