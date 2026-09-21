"""Serial protocol70 gates/evaluation; never launch or restart training/data workers."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, sha
from evaluate_graph_mixture_cost_mcts import PANEL, RUNS
from graph_evaluation_partition import partition_inputs
from run_clean_paired_evaluation import check_completed_prefix, matching_children, processes
from run_joint_policy_metric_evaluation import verify_job as verify_common_job
from train_graph_metric_mixture import CONFIG, INITIAL_SHA

PAIR = Path("artifacts/graph-mixture-training-pair-v1")
GENERAL = Path("artifacts/graph-mixture-fresh-gordian-test-v1")


def read(path):
    return json.loads(path.read_text())


def command(script, *args):
    return [sys.executable, "-u", f"scripts/{script}.py", *map(str, args)]


def jobs(out):
    planned = []

    def add(name, script, args, output, gate):
        planned.append(
            {"name": name, "command": command(script, *args), "output": str(output), "gate": gate}
        )

    base = PANEL.parent / "base-lineage-audit.json"
    add(
        "short-u-base-audit",
        "audit_joint_policy_metric_panel",
        ["--panel", PANEL, "--out", base],
        base,
        "panel",
    )
    audit = PANEL.parent / "independence-audit.json"
    add(
        "short-u-full-audit",
        "audit_graph_mixture_short_u_panel",
        ["--panel", PANEL, "--base-audit", base, "--out", audit],
        audit,
        "panel",
    )
    output = out / "training-pair-audit.json"
    add(
        "training-pair-audit",
        "audit_graph_metric_training_pair",
        ["--out", output],
        output,
        "training",
    )
    for arm, run in RUNS.items():
        output = out / "metric" / f"{arm}-short-u.json"
        add(
            f"metric-{arm}-short-u",
            "evaluate_strand_graph_panel",
            ["--run", run, "--panel", PANEL, "--bounds", "determinant", "--output", output],
            output,
            "training",
        )
    for arm in (*RUNS, "uniform"):
        directory = out / "mcts" / arm
        add(
            f"mcts-{arm}",
            "evaluate_graph_mixture_cost_mcts",
            ["--arm", arm, "--out", directory],
            directory / "manifest.json",
            "training",
        )
        output = directory / "replay-audit.json"
        add(
            f"mcts-audit-{arm}",
            "audit_graph_mixture_cost_mcts",
            ["--run", directory, "--out", output],
            output,
            "training",
        )
    return planned


def live_commands(argv, listing):
    return sorted(
        set(
            matching_children({"command": argv}, listing)
            + matching_children({"command": argv + ["--resume"]}, listing)
        )
    )


def inspect_training(pair, manifests, live_parent, live_arms):
    """Pure state check, including the queued serial control and transition windows."""
    if len(live_parent) > 1 or any(len(v) > 1 for v in live_arms.values()):
        raise RuntimeError("duplicate training processes")
    if sum(map(len, live_arms.values())) > 1:
        raise RuntimeError("both serial training arms are live")
    if pair["status"] == "needs_attention":
        raise RuntimeError("training launcher reported failure; preserve and inspect")
    observed = []
    for arm in ("graph", "control"):
        m = manifests[arm]
        if m is not None:
            if (
                m["config"] != {**CONFIG, "mode": arm}
                or m["initial_checkpoint_sha256"] != INITIAL_SHA
            ):
                raise ValueError("training arm contract differs")
            if m["status"] == "completed":
                if m["completed_steps"] != CONFIG["steps"]:
                    raise ValueError("training completion lacks declared steps")
                continue
        if not live_parent:
            raise RuntimeError("unfinished serial training has no live launcher; inspect recovery")
        if m is None and arm == "graph":
            raise RuntimeError("expected already-started graph arm is missing")
        observed.append(
            {
                "kind": "training",
                "arm": arm,
                "steps": m.get("completed_steps") if m else None,
                "pids": live_arms[arm],
                "launcher": live_parent[0],
                "phase": "worker" if live_arms[arm] else "queued-or-transition",
            }
        )
    if pair["status"] == "completed" and observed:
        raise ValueError("completed launcher has incomplete arms")
    if not observed and pair["status"] != "completed":
        if not live_parent:
            raise RuntimeError("arms completed but serial launcher did not finalize")
        observed.append({"kind": "training-finalization", "launcher": live_parent[0]})
    return observed


def panel_gate(progress, panel, live):
    if len(live) > 1:
        raise RuntimeError("duplicate constructive panel builders")
    if progress["status"] == "completed":
        if panel is None or not panel["complete"] or not panel["construction_complete"]:
            raise RuntimeError("fixed constructive panel supply is incomplete/inadequate")
        if panel["contract"] != progress["contract"]:
            raise ValueError("panel construction contract differs from progress")
        return []
    if not live:
        raise RuntimeError("unfinished panel has no live builder; inspect recovery")
    return [
        {
            "kind": "panel-builder",
            "pid": live[0],
            "attempts": progress["next_index"],
            "groups": progress["groups"],
        }
    ]


def builder_command(progress):
    c = progress["contract"]
    panels = sorted(Path(p) for p in c["inputs"] if Path(p).name == "panel.json")
    if len(panels) != 15 or {
        k: c[k] for k in ("seed", "attempts", "max_strands", "min_depth", "max_depth")
    } != {"seed": 20261032, "attempts": 10000, "max_strands": 33, "min_depth": 24, "max_depth": 96}:
        raise ValueError("constructive builder protocol differs")
    return command(
        "build_constructive_short_u_panel",
        "--out",
        PANEL.parent,
        "--seed",
        c["seed"],
        "--attempts",
        c["attempts"],
        "--max-strands",
        c["max_strands"],
        "--min-depth",
        c["min_depth"],
        "--max-depth",
        c["max_depth"],
        "--protocol",
        "research/70-graph-mixture-prospective-evaluation.md",
        *[item for p in panels for item in ("--exclude-panel", p)],
    )


def gate_snapshot(required, frozen_pair_contract):
    listing = processes()
    pair = read(PAIR / "manifest.json")
    if pair["contract"] != frozen_pair_contract:
        raise ValueError("serial training contract changed")
    manifests = {
        a: read(RUNS[a] / "manifest.json") if (RUNS[a] / "manifest.json").exists() else None
        for a in ("graph", "control")
    }
    parent = live_commands(command("run_graph_metric_training_pair", "--out", PAIR), listing)
    workers = {
        a: live_commands(
            command("train_graph_metric_mixture", "--mode", a, "--out", RUNS[a]), listing
        )
        for a in manifests
    }
    training = inspect_training(pair, manifests, parent, workers)
    progress = read(PANEL.parent / "progress.json")
    panel = read(PANEL) if PANEL.exists() else None
    data = panel_gate(progress, panel, live_commands(builder_command(progress), listing))
    observations = data + training
    ready = not data and (required == "panel" or not training)
    pins = {}
    if ready:
        pins.update(
            {
                str(PANEL): sha(PANEL),
                str(PANEL.parent / "progress.json"): sha(PANEL.parent / "progress.json"),
            }
        )
        if required == "training":
            for arm, m in manifests.items():
                path = RUNS[arm] / "checkpoint.pt"
                if sha(path) != m["checkpoint_sha256"]:
                    raise ValueError("final training checkpoint SHA differs")
                pins.update(
                    {
                        str(path): sha(path),
                        str(RUNS[arm] / "manifest.json"): sha(RUNS[arm] / "manifest.json"),
                    }
                )
            pins[str(PAIR / "manifest.json")] = sha(PAIR / "manifest.json")
    return ready, observations, pins


def verify_job(job):
    record = verify_common_job(job)
    result = read(Path(job["output"]))
    if job["name"].startswith("short-u-"):
        if result["inputs"].get(str(PANEL)) != sha(PANEL):
            raise ValueError("panel audit covers a different panel")
        if job["name"] == "short-u-full-audit" and result.get("evaluation_ready") is not True:
            raise ValueError("panel has not passed evaluation gate")
    if (
        job["name"] == "training-pair-audit"
        and result.get("same_initial_checkpoint") != INITIAL_SHA
    ):
        raise ValueError("training audit initial checkpoint differs")
    if job["name"].startswith("mcts-"):
        arm = job["name"].removeprefix("mcts-audit-").removeprefix("mcts-")
        if result["arm"] != arm:
            raise ValueError("search arm differs")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("new evaluation output required, or explicit resume")
    planned = jobs(args.out)
    stable, _, _, _ = partition_inputs()
    pair_contract = read(PAIR / "manifest.json")["contract"]
    if pair_contract["config"] != CONFIG or pair_contract["initial_sha256"] != INITIAL_SHA:
        raise ValueError("pair protocol differs")
    merge_pins(stable, pair_contract["inputs"])
    merge_pins(stable, pair_contract["sources"])
    launcher = Path("scripts/run_graph_metric_training_pair.py")
    merge_pins(stable, {str(launcher): pair_contract["launcher_sha256"]})
    progress = read(PANEL.parent / "progress.json")
    builder_command(progress)
    merge_pins(stable, progress["contract"]["inputs"])
    general = read(GENERAL / "independence-audit.json")
    if (
        general["status"] != "passed"
        or general["evaluation_ready"] is not False
        or general["selected_pairs"] != 0
    ):
        raise ValueError("expected audited unavailable general-distance supply")
    merge_pins(stable, general["inputs"])
    sources = {Path(j["command"][2]) for j in planned} | {
        Path(__file__).resolve(),
        Path("scripts/run_joint_policy_metric_evaluation.py"),
        Path("scripts/run_clean_paired_evaluation.py"),
        Path("scripts/audit_joint_cost_mcts.py"),
        Path("scripts/evaluate_joint_cost_mcts.py"),
        Path("scripts/audit_cost_mcts_continuations.py"),
        Path("scripts/check_cost_mcts_continuations.py"),
        Path("scripts/evaluate_embedding_cost_search.py"),
        Path("scripts/audit_joint_training_pair.py"),
        Path("scripts/evaluate_strand_graph_panel.py"),
        GENERAL / "independence-audit.json",
    }
    merge_pins(stable, {str(p): sha(p) for p in sources})
    check_pins(stable)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.out / "manifest.json"
    if args.resume:
        state = read(path)
        if (
            state["inputs"] != stable
            or state["jobs"] != planned
            or state["pair_contract"] != pair_contract
        ):
            raise ValueError("evaluation resume contract differs")
        check_completed_prefix(state["completed"], planned)
        for saved, job in zip(state["completed"], planned[: len(state["completed"])], strict=True):
            if saved != verify_job(job):
                raise ValueError("completed evaluation prefix changed")
        check_pins(state.get("completed_gate_inputs", {}))
        if state["status"] == "completed":
            if len(state["completed"]) != len(planned):
                raise ValueError("completed evaluation lacks job coverage")
            print("already completed, all outputs verified")
            return
    else:
        state = {
            "inputs": stable,
            "jobs": planned,
            "pair_contract": pair_contract,
            "completed": [],
            "completed_gate_inputs": {},
            "current": None,
            "general_distance": {
                "status": "unavailable",
                "reason": "zero eligible fresh TEST queries",
                "audit": str(GENERAL / "independence-audit.json"),
            },
        }
        snapshots = args.out / "source_snapshot"
        snapshots.mkdir()
        for index, name in enumerate(sorted(stable)):
            source = Path(name)
            if source.suffix in {".py", ".md"}:
                shutil.copy2(source, snapshots / f"{index:04d}-{source.name}")
    (args.out / "logs").mkdir(exist_ok=True)
    state["pid"] = os.getpid()
    try:
        for job in planned[len(state["completed"]) :]:
            while True:
                ready, observations, pins = gate_snapshot(job["gate"], pair_contract)
                check_pins(state["completed_gate_inputs"])
                if ready:
                    merge_pins(state["completed_gate_inputs"], pins)
                    break
                state.update(status="waiting_for_gates", waiting=observations)
                atomic(path, state)
                time.sleep(15)
            check_pins(stable)
            live = live_commands(job["command"], processes())
            if len(live) > 1:
                raise RuntimeError("duplicate evaluation workers")
            output = Path(job["output"])
            child = None
            if output.exists() and not live:
                # Adopt only a hash-verified final artifact; never replace partial output.
                record = verify_job(job)
            else:
                prior = state.get("current")
                if not live and prior and prior["name"] == job["name"]:
                    raise RuntimeError(
                        "partial evaluation worker disappeared; inspect explicit recovery"
                    )
                if not live:
                    with (args.out / "logs" / f"{job['name']}.log").open("a") as log:
                        child = subprocess.Popen(
                            job["command"], stdout=log, stderr=subprocess.STDOUT
                        )
                    live = [child.pid]
                state.update(
                    status="evaluating",
                    waiting=[],
                    current={"name": job["name"], "pid": live[0], "command": job["command"]},
                )
                atomic(path, state)
                print(json.dumps({"status": "evaluating", **state["current"]}), flush=True)
                while True:
                    if child is not None:
                        code = child.poll()
                        if code is not None:
                            if code != 0:
                                raise RuntimeError(
                                    f"evaluation worker exited {code}: {job['name']}"
                                )
                            break
                    elif not live_commands(job["command"], processes()):
                        break
                    time.sleep(10)
                record = verify_job(job)
            state["completed"].append(record)
            state.update(current=None, status="between_jobs")
            atomic(path, state)
            print(
                json.dumps(
                    {
                        "completed": len(state["completed"]),
                        "total": len(planned),
                        "job": job["name"],
                    }
                ),
                flush=True,
            )
        check_pins(stable)
        check_pins(state["completed_gate_inputs"])
        state.update(status="completed", current=None)
        atomic(path, state)
    except Exception as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        atomic(path, state)
        raise


if __name__ == "__main__":
    main()
