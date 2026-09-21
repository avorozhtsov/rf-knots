"""Serial protocol-58/60 evaluation after live training and data-audit gates.

Never starts/stops training or data audits. Waits only on verified exact-command
workers. Completed jobs are hash-checked; a lost partial worker needs attention.
"""

import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from audit_joint_training_pair import ACTOR_SHA, EXPECTED_CONFIG
from build_mcts_training_supervision import atomic, sha
from evaluate_joint_cost_mcts import RUNS
from run_clean_paired_evaluation import check_completed_prefix, matching_children, processes

PANELS = {
    name: Path(f"artifacts/mcts-policy-metric-fresh-{name}-test-v1/panel.json")
    for name in ("short-u", "gordian")
}


def read(path):
    return json.loads(path.read_text())


def command(script, *args):
    return [sys.executable, "-u", f"scripts/{script}.py", *map(str, args)]


def jobs(out):
    result = []

    def add(name, script, arguments, output):
        result.append({"name": name, "command": command(script, *arguments), "output": str(output)})

    output = out / "training-pair-audit.json"
    add("training-pair-audit", "audit_joint_training_pair", ["--out", output], output)
    for arm, run in RUNS.items():
        for panel, path in PANELS.items():
            name = f"metric-{arm}-{panel}"
            output = out / "metric" / f"{name}.json"
            add(
                name,
                "evaluate_strand_graph_panel",
                ["--run", run, "--panel", path, "--bounds", "determinant", "--output", output],
                output,
            )
    for arm in (*RUNS, "uniform"):
        directory = out / "mcts" / arm
        add(
            f"mcts-{arm}",
            "evaluate_joint_cost_mcts",
            ["--arm", arm, "--out", directory],
            directory / "manifest.json",
        )
        output = directory / "replay-audit.json"
        add(
            f"mcts-audit-{arm}",
            "audit_joint_cost_mcts",
            ["--run", directory, "--out", output],
            output,
        )
    return result


def training_workers(arm, listing):
    args = command("train_mcts_policy_metric", "--mode", arm, "--out", RUNS[arm])
    return matching_children({"command": args}, listing) + matching_children(
        {"command": args + ["--resume"]}, listing
    )


def job_workers(job, listing):
    found = matching_children(job, listing)
    if job["command"][2] == "scripts/evaluate_joint_cost_mcts.py":
        found += matching_children({"command": job["command"] + ["--resume"]}, listing)
    return found


def gate_snapshot(listing):
    """A missing unfinished dependency is not an invitation to launch a replacement."""
    observations, completed_pins = [], {}
    for arm in ("joint", "metric-control"):
        path = RUNS[arm] / "manifest.json"
        manifest = read(path) if path.exists() else None
        live = training_workers(arm, listing)
        if len(live) > 1:
            raise RuntimeError(f"duplicate training workers: {arm}")
        if manifest and manifest["status"] == "completed":
            if manifest["config"] != {**EXPECTED_CONFIG, "mode": arm}:
                raise ValueError("completed training protocol differs")
            if manifest["initial_checkpoint_sha256"] != ACTOR_SHA:
                raise ValueError("completed training initial state differs")
            checkpoint = RUNS[arm] / "checkpoint.pt"
            if sha(checkpoint) != manifest["checkpoint_sha256"]:
                raise ValueError("completed training checkpoint hash differs")
            completed_pins.update({str(path): sha(path), str(checkpoint): sha(checkpoint)})
        else:
            if not live:
                raise RuntimeError(
                    f"training stopped before completion: {arm}; inspect/resume explicitly"
                )
            observations.append(
                {
                    "kind": "training",
                    "arm": arm,
                    "pid": live[0],
                    "steps": manifest.get("completed_steps") if manifest else None,
                }
            )
    for name, panel in PANELS.items():
        path = panel.parent / "independence-audit.json"
        args = command("audit_joint_policy_metric_panel", "--panel", panel, "--out", path)
        live = matching_children({"command": args}, listing)
        if len(live) > 1:
            raise RuntimeError(f"duplicate independence audits: {name}")
        if path.exists():
            audit = read(path)
            if audit["status"] != "passed" or audit["inputs"].get(str(panel)) != sha(panel):
                raise ValueError("independence audit mismatch")
            completed_pins[str(path)] = sha(path)
        else:
            if not live:
                raise RuntimeError(f"independence audit missing and no live verifier: {name}")
            observations.append({"kind": "data-audit", "panel": name, "pid": live[0]})
    return observations, completed_pins


def verify_job(job):
    output = Path(job["output"])
    report = read(output)
    if report.get("status") not in ("passed", "completed"):
        raise ValueError("job output is not final")
    provenance = report.get("inputs", report.get("code_sha256", {}))
    if not provenance:
        raise ValueError("job output lacks provenance")
    check_pins(provenance)
    if job["name"].startswith("metric-"):
        argv = job["command"]
        run, panel = Path(argv[argv.index("--run") + 1]), Path(argv[argv.index("--panel") + 1])
        if (
            report["run"] != str(run)
            or report["checkpoint_sha256"] != sha(run / "checkpoint.pt")
            or report["panel_sha256"] != sha(panel)
            or report["initial"]
            or report["bounds"] != "determinant"
            or report["normalization"] != "none"
        ):
            raise ValueError("metric job model/panel/contract differs")
        records = read(panel)
        for view in ("validation", "size_shift"):
            predictions = report[view]["predictions"]
            if len(predictions) != len(records[view]) or report[view]["count"] != len(predictions):
                raise ValueError("metric query coverage differs")
            for index, (prediction, record) in enumerate(
                zip(predictions, records[view], strict=True)
            ):
                pair = record["interval"]
                lo = max(
                    pair["lower"],
                    int(record["left_fingerprint"][0] != record["right_fingerprint"][0]),
                )
                hi = pair["upper"]
                if (
                    prediction["panel_index"] != index
                    or prediction["lower"] != lo
                    or prediction["upper"] != hi
                    or not 0 <= lo <= hi
                    or prediction["left_fingerprint"] != record["left_fingerprint"]
                    or prediction["right_fingerprint"] != record["right_fingerprint"]
                    or prediction["fingerprint_group"]
                    != f"{record['left_fingerprint'][0]}:{abs(record['left_fingerprint'][1])}"
                ):
                    raise ValueError("metric labels or query identities differ")
                value, violation = prediction["prediction"], prediction["violation"]
                if (
                    not math.isfinite(value)
                    or not math.isfinite(violation)
                    or not math.isclose(violation, max(lo - value, value - hi, 0), abs_tol=1e-8)
                ):
                    raise ValueError("invalid metric prediction/interval violation")
    return {"name": job["name"], "output": str(output), "sha256": sha(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    planned = jobs(args.out)
    sources = {Path(j["command"][2]) for j in planned} | {
        Path(__file__).resolve(),
        Path("scripts/run_clean_paired_evaluation.py"),
        Path("scripts/audit_joint_policy_metric_panel.py"),
        Path("scripts/build_mcts_training_supervision.py"),
        Path("scripts/burau_integer_check.py"),
        Path("scripts/check_cost_mcts_continuations.py"),
        Path("scripts/audit_cost_mcts_continuations.py"),
        Path("scripts/evaluate_embedding_cost_search.py"),
        Path("scripts/run_strand_graph_pilot.py"),
        Path("scripts/mcts_joint_training.py"),
        Path("scripts/train_mcts_policy_metric.py"),
        Path("scripts/mcts_training_supervision.py"),
        Path("scripts/balanced_metric_sampling.py"),
        Path("research/57-joint-policy-metric-training-protocol.md"),
        Path("research/58-prospective-joint-policy-metric-test.md"),
        Path("research/60-prospective-joint-general-gordian-test.md"),
        *Path("src/rf_knots").glob("*.py"),
    }
    stable = {
        str(p): sha(p)
        for p in [
            *sources,
            *PANELS.values(),
            RUNS["parent"] / "manifest.json",
            RUNS["parent"] / "checkpoint.pt",
        ]
    }
    if stable[str(RUNS["parent"] / "checkpoint.pt")] != ACTOR_SHA:
        raise ValueError("parent differs from fixed initial actor")
    if any(not read(panel)["complete"] for panel in PANELS.values()):
        raise ValueError("incomplete prospective panel")
    if args.out.exists() != args.resume:
        raise ValueError("existing output requires resume; new output cannot resume")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.out / "manifest.json"
    if args.resume:
        state = read(path)
        if state["inputs"] != stable or state["jobs"] != planned:
            raise ValueError("controller inputs/jobs changed")
        check_completed_prefix(state["completed"], planned)
        for stored, job in zip(state["completed"], planned[: len(state["completed"])], strict=True):
            if stored != verify_job(job):
                raise ValueError("completed job changed")
        if state["status"] == "completed":
            if len(state["completed"]) != len(planned):
                raise ValueError("incomplete job matrix marked completed")
            print("already completed")
            return
    else:
        state = {"status": "waiting_for_gates", "inputs": stable, "jobs": planned, "completed": []}
        (args.out / "logs").mkdir()
        for p in sources:
            relative = p.resolve().relative_to(Path.cwd())
            destination = args.out / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, destination)
    state["pid"] = os.getpid()
    atomic(path, state)
    try:
        while True:
            check_pins(stable)
            observations, gate_pins = gate_snapshot(processes())
            if not observations:
                break
            if state.get("waiting") != observations:
                state.update(status="waiting_for_gates", waiting=observations)
                atomic(path, state)
            time.sleep(15)
        if state.get("gate_pins", gate_pins) != gate_pins:
            raise ValueError("previous final training/audit pins changed")
        state["gate_pins"] = gate_pins
        for panel in PANELS.values():
            check_pins(read(panel.parent / "independence-audit.json")["inputs"])
        all_pins = dict(stable)
        merge_pins(all_pins, gate_pins)
        atomic(path, state)
        for job in planned[len(state["completed"]) :]:
            check_pins(all_pins)
            live = job_workers(job, processes())
            output = Path(job["output"])
            prior = state.get("current", {}).get("name") == job["name"]
            if len(live) > 1 or (output.exists() and not prior):
                raise RuntimeError("duplicate writer or unexpected existing output")
            if prior and not live and not output.exists():
                raise RuntimeError(
                    "partial job lost its worker; preserve it and inspect/resume explicitly"
                )
            child = None
            if not live and not output.exists():
                with (args.out / "logs" / f"{job['name']}.log").open("xb") as log:
                    child = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT)
                live = [child.pid]
            state.update(status="evaluating", current={**job, "pid": live[0] if live else None})
            atomic(path, state)
            if child is not None:
                if child.wait() != 0:
                    raise RuntimeError(f"evaluation job failed: {job['name']}")
            else:
                while job_workers(job, processes()):
                    time.sleep(15)
            check_pins(all_pins)
            state["completed"].append(verify_job(job))
            state.pop("current", None)
            atomic(path, state)
            print(
                json.dumps(
                    {
                        "completed": job["name"],
                        "jobs": len(state["completed"]),
                        "total": len(planned),
                    }
                ),
                flush=True,
            )
        state.update(status="completed", waiting=[])
        atomic(path, state)
    except Exception as error:
        state.update(status="attention_required", error=str(error))
        atomic(path, state)
        raise


if __name__ == "__main__":
    main()
