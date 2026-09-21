"""Wait on actual protocol74/69 auditors, then score three immutable final models."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, sha
from evaluate_graph_mixture_cost_mcts import PANEL as SHORT_U_PANEL
from evaluate_pure_braid_distance_panel import PAIR_AUDIT, PANEL, RUNS, summaries
from run_clean_paired_evaluation import check_completed_prefix, processes
from run_graph_mixture_evaluation import command, live_commands
from summarize_joint_policy_metric_evaluation import paired_metric
from train_graph_metric_mixture import INITIAL_SHA

BASE_CONTROLLER = Path("artifacts/graph-mixture-evaluation-v1")
DATA_AUDIT = PANEL.parent / "independence-audit.json"


def read(path):
    return json.loads(path.read_text())


def jobs(directory):
    return [
        {
            "name": arm,
            "command": command(
                "evaluate_pure_braid_distance_panel",
                "--arm",
                arm,
                "--out",
                directory / f"{arm}.json",
            ),
            "output": str(directory / f"{arm}.json"),
        }
        for arm in RUNS
    ]


def data_gate(audit, live, panel_sha):
    if len(live) > 1:
        raise RuntimeError("duplicate pure-braid auditors")
    if audit is not None:
        if (
            audit["status"] != "passed"
            or audit["evaluation_ready"] is not True
            or audit["inputs"].get(str(PANEL)) != panel_sha
        ):
            raise ValueError("independent data audit failed, inadequate, or different")
        return []
    if not live:
        raise RuntimeError("data audit missing and no live auditor; inspect recovery")
    return [{"kind": "pure-braid-audit", "pid": live[0]}]


def pair_gate(audit, controller, live):
    if len(live) > 1:
        raise RuntimeError("duplicate base evaluation controllers")
    if audit is not None:
        if (
            audit["status"] != "passed"
            or audit["same_initial_checkpoint"] != INITIAL_SHA
            or set(audit["arms"]) != {"graph", "control"}
        ):
            raise ValueError("final training-pair audit differs")
        return []
    if controller["status"] == "needs_attention":
        raise RuntimeError("base controller needs attention; do not restart dependencies")
    if not live:
        raise RuntimeError("training-pair audit missing and base controller is not live")
    if controller["status"] == "completed":
        raise ValueError("completed base controller lacks training-pair audit")
    return [{"kind": "training-pair-audit-queued-or-running", "controller_pid": live[0]}]


def gate_snapshot(panel_sha):
    listing = processes()
    data = read(DATA_AUDIT) if DATA_AUDIT.exists() else None
    pair = read(PAIR_AUDIT) if PAIR_AUDIT.exists() else None
    waiting = data_gate(
        data,
        live_commands(
            command("audit_pure_braid_distance_panel", "--panel", PANEL, "--out", DATA_AUDIT),
            listing,
        ),
        panel_sha,
    )
    waiting += pair_gate(
        pair,
        read(BASE_CONTROLLER / "manifest.json"),
        live_commands(command("run_graph_mixture_evaluation", "--out", BASE_CONTROLLER), listing),
    )
    pins = {}
    if not waiting:
        for path, value in ((DATA_AUDIT, data), (PAIR_AUDIT, pair)):
            merge_pins(pins, value["inputs"])
            merge_pins(pins, {str(path): sha(path)})
        # Pair-audit publication follows both short-u audits in the base queue.
        short = read(SHORT_U_PANEL.parent / "independence-audit.json")
        if short["status"] != "passed" or short["inputs"].get(str(SHORT_U_PANEL)) != sha(
            SHORT_U_PANEL
        ):
            raise ValueError("short-u overlap report lacks a completed independent panel")
        merge_pins(pins, short["inputs"])
        merge_pins(
            pins,
            {
                str(SHORT_U_PANEL.parent / "independence-audit.json"): sha(
                    SHORT_U_PANEL.parent / "independence-audit.json"
                )
            },
        )
    return waiting, pins


def verify_report(job, panel, pair):
    path = Path(job["output"])
    report = read(path)
    arm = job["name"]
    checkpoint = INITIAL_SHA if arm == "initial" else pair["arms"][arm]["checkpoint_sha256"]
    if (
        report["status"] != "completed"
        or report["arm"] != arm
        or report["run"] != str(RUNS[arm])
        or report["checkpoint_sha256"] != checkpoint
        or report["panel_sha256"] != sha(PANEL)
        or report["inputs"].get(str(RUNS[arm] / "checkpoint.pt")) != checkpoint
        or report["inputs"].get(str(PAIR_AUDIT)) != sha(PAIR_AUDIT)
        or report["inputs"].get(str(DATA_AUDIT)) != sha(DATA_AUDIT)
        or report["normalization"] != "none beyond checkpoint architecture"
        or report["calibration"] != "frozen checkpoint objective"
    ):
        raise ValueError("metric report identity/model/calibration differs")
    check_pins(report["inputs"])
    if report["summary"] != summaries(report["validation"]["predictions"], panel["validation"]):
        raise ValueError("stored metric summary differs from raw predictions")
    return {"name": arm, "output": str(path), "sha256": sha(path)}, report


def compare(reports, short_u):
    predicates = {
        "all": lambda r: True,
        "positive_lower_bound": lambda r: r["lower"] >= 1,
        "exact_zero_coordinate": lambda r: r["lower"] == r["upper"] == 0,
        "exact_one": lambda r: r["lower"] == r["upper"] == 1,
        "exact_two": lambda r: r["lower"] == r["upper"] == 2,
    }
    comparisons = {}
    for baseline, candidate in (("initial", "graph"), ("initial", "control"), ("control", "graph")):
        a, b = (
            reports[baseline]["validation"]["predictions"],
            reports[candidate]["validation"]["predictions"],
        )
        comparisons[f"{candidate}-vs-{baseline}"] = {
            name: paired_metric([r for r in a if predicate(r)], [r for r in b if predicate(r)])
            for name, predicate in predicates.items()
        }
    groups = {
        tuple((r["left_fingerprint"][0], abs(r["left_fingerprint"][1])))
        for r in reports["initial"]["validation"]["predictions"]
    }
    short_groups = {
        (r["left_fingerprint"][0], abs(r["left_fingerprint"][1])) for r in short_u["validation"]
    }
    return {
        "arms": {arm: report["summary"] for arm, report in reports.items()},
        "comparisons": comparisons,
        "query_groups_also_in_simultaneous_short_u_panel": sorted(groups & short_groups),
        "scope": "Three fixed final checkpoints, one training seed and one synthetic family. "
        "No automatic promotion. Short-u and general-distance scores are not pooled.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("new output required, or explicit resume")
    panel = read(PANEL)
    if not panel["complete"] or not panel["construction_complete"]:
        raise ValueError("require completed adequate supply before waiting on audits")
    planned = jobs(args.out)
    pins = dict(panel["contract"]["inputs"])
    sources = [
        Path(__file__).resolve(),
        Path("scripts/evaluate_pure_braid_distance_panel.py"),
        Path("scripts/run_graph_mixture_evaluation.py"),
        Path("scripts/run_clean_paired_evaluation.py"),
        Path("scripts/summarize_joint_policy_metric_evaluation.py"),
        Path("scripts/summarize_clean_paired_evaluation.py"),
        Path("scripts/audit_pure_braid_distance_panel.py"),
        Path("scripts/evaluate_strand_graph_panel.py"),
        Path("scripts/evaluate_graph_mixture_cost_mcts.py"),
        Path("scripts/audit_graph_mixture_short_u_panel.py"),
    ]
    merge_pins(pins, {str(p): sha(p) for p in [PANEL, *sources]})
    check_pins(pins)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mp = args.out / "manifest.json"
    if args.resume:
        state = read(mp)
        if state["inputs"] != pins or state["jobs"] != planned:
            raise ValueError("evaluation resume protocol differs")
        check_completed_prefix(state["completed"], planned)
        check_pins(state.get("gate_inputs", {}))
    else:
        state = {
            "status": "waiting_for_audits",
            "inputs": pins,
            "jobs": planned,
            "completed": [],
            "gate_inputs": {},
            "current": None,
        }
        snapshots = args.out / "source_snapshot"
        snapshots.mkdir()
        for index, name in enumerate(sorted(pins)):
            p = Path(name)
            if p.suffix in {".py", ".md"}:
                shutil.copy2(p, snapshots / f"{index:04d}-{p.name}")
    (args.out / "logs").mkdir(exist_ok=True)
    state["pid"] = os.getpid()
    try:
        while True:
            waiting, gate_pins = gate_snapshot(pins[str(PANEL)])
            if not waiting:
                merge_pins(state["gate_inputs"], gate_pins)
                check_pins(state["gate_inputs"])
                break
            state.update(status="waiting_for_audits", waiting=waiting)
            atomic(mp, state)
            time.sleep(15)
        pair = read(PAIR_AUDIT)
        reports = {}
        for saved, job in zip(state["completed"], planned[: len(state["completed"])], strict=True):
            actual, report = verify_report(job, panel, pair)
            if saved != actual:
                raise ValueError("committed metric report changed")
            reports[job["name"]] = report
        if state["status"] == "completed":
            if len(reports) != 3 or sha(args.out / "comparison.json") != state["comparison_sha256"]:
                raise ValueError("completed comparison coverage/SHA differs")
            print("already completed, all final metrics verified")
            return
        for job in planned[len(state["completed"]) :]:
            check_pins(pins)
            check_pins(state["gate_inputs"])
            live = live_commands(job["command"], processes())
            if len(live) > 1:
                raise RuntimeError("duplicate pure-braid metric workers")
            child = None
            if not Path(job["output"]).exists() or live:
                if not live and state.get("current") and state["current"]["name"] == job["name"]:
                    raise RuntimeError("partial metric worker missing; inspect before recovery")
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
                atomic(mp, state)
                while True:
                    if child is not None:
                        code = child.poll()
                        if code is not None:
                            if code != 0:
                                raise RuntimeError(f"metric worker exited {code}")
                            break
                    elif not live_commands(job["command"], processes()):
                        break
                    time.sleep(10)
            record, report = verify_report(job, panel, pair)
            state["completed"].append(record)
            reports[job["name"]] = report
            state.update(status="between_jobs", current=None)
            atomic(mp, state)
        comparison = compare(reports, read(SHORT_U_PANEL))
        all_pins = dict(pins)
        merge_pins(all_pins, state["gate_inputs"])
        merge_pins(all_pins, {r["output"]: r["sha256"] for r in state["completed"]})
        check_pins(all_pins)
        output = args.out / "comparison.json"
        if output.exists():
            if read(output) != {
                "status": "completed",
                "inputs": all_pins,
                **json.loads(json.dumps(comparison)),
            }:
                raise ValueError("uncommitted comparison differs; preserving it")
        else:
            atomic(output, {"status": "completed", "inputs": all_pins, **comparison})
        state.update(status="completed", current=None, comparison_sha256=sha(output))
        atomic(mp, state)
        print(
            json.dumps({"status": "completed", "comparison_sha256": state["comparison_sha256"]}),
            flush=True,
        )
    except Exception as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        atomic(mp, state)
        raise


if __name__ == "__main__":
    main()
