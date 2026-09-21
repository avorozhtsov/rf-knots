"""Prepare a new lineage after the exact pre-search explicit-unknot schema bug.

Copies the seven unchanged completed outputs, never edits the failed lineage,
and never starts/stops a worker. Only one exact source amendment is accepted.
Run the existing evaluation controller with --resume on the new output next.
"""

import argparse
import hashlib
import shutil
from pathlib import Path

import run_joint_policy_metric_evaluation as sequence
from audit_joint_policy_metric_panel import check_pins
from build_mcts_training_supervision import atomic, sha

SOURCE = "scripts/evaluate_joint_cost_mcts.py"
OLD = '        if row["interval"]["right"] != {"word": [], "strands": 1}:\n'
NEW = (
    '        if row["interval"]["right"] not in (\n'
    '            {"word": [], "strands": 1},\n'
    '            {"word": [], "strands": 1, "cyclic_band_generators": False},\n'
    "        ):\n"
)


def accepted_amendment(before, after):
    if before.count(OLD) != 1 or after != before.replace(OLD, NEW):
        raise ValueError("only the explicit false-band schema amendment is allowed")


def validate_failed_state(state, directory):
    planned = sequence.jobs(directory)
    if (
        state["status"] != "attention_required"
        or state.get("error") != "evaluation job failed: mcts-parent"
        or state["jobs"] != planned
        or len(state["completed"]) != 7
        or state.get("current", {}).get("name") != "mcts-parent"
    ):
        raise ValueError("require exact seven-job prefix and failed first search, not live work")
    sequence.check_completed_prefix(state["completed"], planned)
    if (directory / "mcts").exists():
        raise ValueError(
            "search artifacts exist; this narrow pre-search recovery cannot adopt them"
        )
    failure_log = directory / "logs" / "mcts-parent.log"
    if (
        not failure_log.read_text()
        .rstrip()
        .endswith("ValueError: search query is not an explicit-unknot task")
    ):
        raise ValueError("first search failed for a different reason")
    for job, completed in zip(planned[:7], state["completed"], strict=True):
        if sequence.verify_job(job) != completed:
            raise ValueError("completed prefix changed")


def verified_recovery(state):
    """Return immutable reuse pins; final consumers must retain these as evidence."""
    recovery = state.get("recovery")
    if recovery is None:
        return {}
    if recovery["schema"] != "rf-knots-explicit-unknot-schema-recovery-v1":
        raise ValueError("unknown evaluation recovery")
    previous = Path(recovery["previous_manifest"])
    pins = {
        str(previous): recovery["previous_manifest_sha256"],
        str(previous.parent / "logs" / "mcts-parent.log"): recovery["failure_log_sha256"],
    }
    check_pins(pins)
    old = sequence.read(previous)
    validate_failed_state(old, previous.parent)
    before_path = previous.parent / "source_snapshot" / SOURCE
    before = before_path.read_text()
    accepted_amendment(before, Path(SOURCE).read_text())
    if sha(before_path) != old["inputs"][SOURCE]:
        raise ValueError("original source snapshot differs from frozen source")
    expected = {**old["inputs"], SOURCE: sha(Path(SOURCE))}
    if state["inputs"] != expected or state["gate_pins"] != old["gate_pins"]:
        raise ValueError("recovery changed more than the permitted source")
    if recovery["source_change"] != {
        "path": SOURCE,
        "before": old["inputs"][SOURCE],
        "after": expected[SOURCE],
    }:
        raise ValueError("source amendment receipt differs")
    for original, copied in zip(old["completed"], state["completed"][:7], strict=True):
        if original["name"] != copied["name"] or original["sha256"] != copied["sha256"]:
            raise ValueError("reused output changed")
        pins[original["output"]] = original["sha256"]
        pins[copied["output"]] = copied["sha256"]
    pins[str(before_path)] = old["inputs"][SOURCE]
    check_pins(pins)
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    old = sequence.read(args.previous)
    validate_failed_state(old, args.previous.parent)
    listing = sequence.processes()
    if any(sequence.job_workers(job, listing) for job in old["jobs"]):
        raise ValueError("old evaluation still has a live worker")
    old_controller = sequence.command(
        "run_joint_policy_metric_evaluation", "--out", args.previous.parent
    )
    if sequence.matching_children(
        {"command": old_controller}, listing
    ) or sequence.matching_children({"command": old_controller + ["--resume"]}, listing):
        raise ValueError("old controller is still live")
    before_path = args.previous.parent / "source_snapshot" / SOURCE
    before, after = before_path.read_text(), Path(SOURCE).read_text()
    accepted_amendment(before, after)
    if hashlib.sha256(before.encode()).hexdigest() != old["inputs"][SOURCE]:
        raise ValueError("original source bytes differ")
    pins = {**old["inputs"], SOURCE: sha(Path(SOURCE))}
    check_pins({**pins, **old["gate_pins"]})
    state = {
        "status": "prepared_recovery",
        "inputs": pins,
        "gate_pins": old["gate_pins"],
        "jobs": sequence.jobs(args.out),
        "completed": [],
        "recovery": {
            "schema": "rf-knots-explicit-unknot-schema-recovery-v1",
            "previous_manifest": str(args.previous),
            "previous_manifest_sha256": sha(args.previous),
            "failure_log_sha256": sha(args.previous.parent / "logs" / "mcts-parent.log"),
            "source_change": {
                "path": SOURCE,
                "before": old["inputs"][SOURCE],
                "after": pins[SOURCE],
            },
            "reason": "accept explicit ordinary one-braid with cyclic_band_generators=false",
        },
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "logs").mkdir()
    for job, original in zip(state["jobs"][:7], old["completed"], strict=True):
        target = Path(job["output"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original["output"], target)
        state["completed"].append(sequence.verify_job(job))
    shutil.copytree(args.previous.parent / "source_snapshot", args.out / "source_snapshot")
    shutil.copy2(SOURCE, args.out / "source_snapshot" / SOURCE)
    verified_recovery(state)
    atomic(args.out / "manifest.json", state)
    print("Prepared recovery with seven byte-identical outputs; no worker launched.")


if __name__ == "__main__":
    main()
