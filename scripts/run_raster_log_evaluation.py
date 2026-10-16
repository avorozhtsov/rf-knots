"""Wait on actual97 training, then run the complete immutable metric/search matrix."""

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
from evaluate_raster_log_distances import PANELS
from raster_log_training import CONFIG, digest
from run_clean_paired_evaluation import check_completed_prefix, processes
from run_graph_mixture_evaluation import live_commands


def read(path):
    return json.loads(path.read_text())


def command(script, *args):
    return [sys.executable, "-u", f"scripts/{script}.py", *map(str, args)]


def jobs(out, run, pgx):
    planned = []

    def add(name, script, arguments, output, **identity):
        planned.append(
            {
                "name": name,
                "command": command(script, *arguments),
                "output": str(output),
                **identity,
            }
        )

    audit = out / "training-pair-audit.json"
    add(
        "training-pair-audit",
        "audit_raster_log_pair",
        ["--run", run, "--pgx-root", pgx, "--out", audit],
        audit,
        kind="training",
    )
    for arm in ("q254", "random"):
        for state in ("initial", "final"):
            for panel in PANELS:
                output = out / "metric" / f"{arm}-{state}-{panel}.json"
                add(
                    f"metric-{arm}-{state}-{panel}",
                    "evaluate_raster_log_distances",
                    [
                        "--run",
                        run,
                        "--pair-audit",
                        audit,
                        "--arm",
                        arm,
                        "--state",
                        state,
                        "--panel",
                        panel,
                        "--pgx-root",
                        pgx,
                        "--out",
                        output,
                    ],
                    output,
                    kind="metric",
                    arm=arm,
                    state=state,
                    panel=panel,
                )
    for arm in ("q254", "random"):
        directory = out / "mcts" / arm
        add(
            f"search-{arm}",
            "evaluate_raster_log_search",
            [
                "--run",
                run,
                "--pair-audit",
                audit,
                "--arm",
                arm,
                "--pgx-root",
                pgx,
                "--out",
                directory,
            ],
            directory / "manifest.json",
            kind="search",
            arm=arm,
        )
        output = directory / "replay-audit.json"
        add(
            f"search-audit-{arm}",
            "audit_raster_log_search",
            ["--run", directory, "--out", output],
            output,
            kind="search-audit",
            arm=arm,
        )
    return planned


def verify_job(job, out):
    path = Path(job["output"])
    report = read(path)
    expected_status = "passed" if job["kind"] in ("training", "search-audit") else "completed"
    if report["status"] != expected_status:
        raise ValueError("evaluation output is not final")
    check_pins(report["inputs"])
    if job["kind"] == "training":
        if set(report["arms"]) != {"q254", "random"} or report["data_rng_aligned"] is not True:
            raise ValueError("training audit coverage differs")
    else:
        pair = read(out / "training-pair-audit.json")
        if report["arm"] != job["arm"]:
            raise ValueError("evaluation arm differs")
        initial = job.get("state") == "initial"
        expected = pair["arms"][job["arm"]]["initial_sha256" if initial else "checkpoint_sha256"]
        if report["checkpoint_sha256"] != expected:
            raise ValueError("evaluation model differs from audited pair")
        if job["kind"] == "metric":
            if (report["state"], report["panel"]) != (job["state"], job["panel"]):
                raise ValueError("metric state/panel identity differs")
            counts = (
                {"validation": 64, "size_shift": 64}
                if job["panel"] == "short-u"
                else {"validation": 126}
            )
            if set(report["results"]) != set(counts) or any(
                len(report["results"][s]["predictions"]) != n for s, n in counts.items()
            ):
                raise ValueError("metric query coverage differs")
            if report["panel_sha256"] != sha(PANELS[job["panel"]]):
                raise ValueError("metric panel hash differs")
        elif job["kind"] == "search":
            from evaluate_raster_log_search import validate_resume

            validate_resume(
                report,
                job["arm"],
                report["inputs"],
                read(path.parent / "roots.json"),
                path.parent,
                Path(report["model_checkpoint"]),
                out / "training-pair-audit.json",
            )
            if report["next_condition"] != 128:
                raise ValueError("full128 search coverage required")
        elif report["counts_replayed"]["conditions"] != 128:
            raise ValueError("incomplete search replay")
    return {"name": job["name"], "output": str(path), "sha256": sha(path)}


def training_snapshot(run, pgx, expected_contract, listing):
    parent = read(run / "manifest.json")
    if parent["contract_sha256"] != expected_contract or parent["config"] != CONFIG:
        raise ValueError("training contract changed")
    if parent["status"] == "failed":
        raise RuntimeError("training failed; preserve artifacts and inspect")
    manifests = [
        read(run / a / "manifest.json") if (run / a / "manifest.json").exists() else None
        for a in ("q254", "random")
    ]
    if parent["status"] == "completed":
        if parent["completed_arms"] != ["q254", "random"] or any(
            not m or m["status"] != "completed" or m["completed_steps"] != CONFIG["steps"]
            for m in manifests
        ):
            raise ValueError("training completion lacks both final arms")
        return True, []
    launch = live_commands(command("run_raster_log_pair", "--pgx-root", pgx, "--out", run), listing)
    if len(launch) != 1 or launch[0] != parent["pid"]:
        raise RuntimeError("unfinished training lacks its unique live launcher")
    workers = []
    for arm in ("q254", "random"):
        workers += live_commands(
            command("train_raster_log_pair", "--arm", arm, "--pgx-root", pgx, "--out", run / arm),
            listing,
        )
    if len(workers) > 1:
        raise RuntimeError("duplicate serial training workers")
    return False, [
        {
            "arm": a,
            "steps": m.get("completed_steps") if m else None,
            "phase": m.get("phase") if m else "queued",
        }
        for a, m in zip(("q254", "random"), manifests, strict=True)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("new output required or explicit controller resume")
    parent = read(args.run / "manifest.json")
    stable = {**parent["inputs"], **parent["sources"]}
    planned = jobs(args.out, args.run, args.pgx_root)
    paths = {Path(j["command"][2]) for j in planned} | {
        Path(__file__),
        Path("scripts/raster_cost_search.py"),
        Path("scripts/summarize_raster_log_experiment.py"),
        Path("scripts/run_graph_mixture_evaluation.py"),
        Path("scripts/run_clean_paired_evaluation.py"),
        Path("scripts/evaluate_graph_mixture_cost_mcts.py"),
        Path("scripts/evaluate_joint_cost_mcts.py"),
        Path("scripts/evaluate_strand_graph_panel.py"),
        Path("scripts/audit_graph_mixture_cost_mcts.py"),
        Path("scripts/audit_joint_cost_mcts.py"),
        Path("scripts/audit_cost_mcts_continuations.py"),
        Path("scripts/audit_semantic_cost_floor.py"),
        Path("scripts/summarize_graph_mixture_experiment.py"),
        Path("scripts/summarize_joint_policy_metric_evaluation.py"),
        Path("scripts/summarize_clean_paired_evaluation.py"),
        Path("scripts/burau_integer_check.py"),
        Path("tests/test_raster_cost_search.py"),
        Path("tests/test_raster_log_search.py"),
        Path("tests/test_audit_raster_log_pair.py"),
        Path("tests/test_raster_log_distances.py"),
        Path("tests/test_raster_log_evaluation.py"),
        Path("research/99-raster-log-evaluation-execution.md"),
    }
    merge_pins(stable, {str(p): sha(p) for p in paths})
    check_pins(stable)
    contract = digest(
        {"training_contract": parent["contract_sha256"], "inputs": stable, "jobs": planned}
    )
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.out / "manifest.json"
    state = (
        read(path)
        if args.resume
        else {
            "inputs": stable,
            "jobs": planned,
            "completed": [],
            "current": None,
            "training_run": str(args.run),
            "pgx_root": str(args.pgx_root),
            "training_contract": parent["contract_sha256"],
            "contract_sha256": contract,
        }
    )
    if (
        state["contract_sha256"] != contract
        or state["inputs"] != stable
        or state["jobs"] != planned
    ):
        raise ValueError("controller resume contract changed")
    check_completed_prefix(state["completed"], planned)
    for job, record in zip(planned, state["completed"], strict=False):
        if verify_job(job, args.out) != record:
            raise ValueError("completed child report changed")
    if state.get("status") == "completed":
        if sha(args.out / "summary.json") != state["summary_sha256"]:
            raise ValueError("final summary changed")
        print("already completed; outputs verified")
        return
    if not args.resume:
        snapshot = args.out / "source_snapshot"
        snapshot.mkdir()
        for i, p in enumerate(sorted(paths)):
            shutil.copy2(p, snapshot / f"{i:03d}-{p.name}")
    (args.out / "logs").mkdir(exist_ok=True)
    (args.out / "metric").mkdir(exist_ok=True)
    state["pid"] = os.getpid()
    try:
        while True:
            ready, observations = training_snapshot(
                args.run, args.pgx_root, state["training_contract"], processes()
            )
            if ready:
                break
            state.update(status="waiting_for_training", training=observations)
            atomic(path, state)
            time.sleep(15)
        check_pins(stable)
        for job in planned[len(state["completed"]) :]:
            check_pins(stable)
            live = live_commands(job["command"], processes())
            if len(live) > 1:
                raise RuntimeError("duplicate evaluation workers")
            output = Path(job["output"])
            child = None
            if (
                output.exists()
                and not live
                and read(output).get("status") in ("completed", "passed")
            ):
                record = verify_job(job, args.out)
            else:
                if not live:
                    argv = job["command"].copy()
                    if output.exists():
                        if job["kind"] != "search":
                            raise RuntimeError("partial non-resumable output; preserve and inspect")
                        argv.append("--resume")
                    with (args.out / "logs" / f"{job['name']}.log").open("a") as log:
                        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                    live = [child.pid]
                state.update(status="evaluating", current={"name": job["name"], "pid": live[0]})
                atomic(path, state)
                print(json.dumps({"status": "evaluating", **state["current"]}), flush=True)
                while True:
                    if child is not None:
                        code = child.poll()
                        if code is not None:
                            if code:
                                raise RuntimeError(f"child exited {code}: {job['name']}")
                            break
                    elif not live_commands(job["command"], processes()):
                        break
                    time.sleep(10)
                record = verify_job(job, args.out)
            state["completed"].append(record)
            state.update(current=None, status="between_jobs")
            atomic(path, state)
        state.update(status="scored", current=None)
        atomic(path, state)
        from summarize_raster_log_experiment import summarize_verified

        result = summarize_verified(args.out)
        summary_path = args.out / "summary.json"
        if summary_path.exists() and read(summary_path) != result:
            raise ValueError("existing final summary differs")
        if not summary_path.exists():
            atomic(summary_path, result)
        check_pins(stable)
        state.update(status="completed", summary_sha256=sha(summary_path))
        atomic(path, state)
    except Exception as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        atomic(path, state)
        raise


if __name__ == "__main__":
    main()
