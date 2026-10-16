"""Serial139 evaluation controller; never starts, stops or modifies training."""

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
from build_mcts_training_supervision import atomic, digest, sha
from evaluate_resource_search import CONFIG as SEARCH_CONFIG
from evaluate_resource_search import validate_resume
from resource_evaluation_inputs import PANEL_ROOTS
from run_clean_paired_evaluation import check_completed_prefix, processes
from run_graph_mixture_evaluation import live_commands
from train_raster_resource_pair import CONFIG as TRAIN_CONFIG


def read(path):
    return json.loads(Path(path).read_text())


def panel_path(kind):
    root = PANEL_ROOTS[kind]
    return root / "generator" / "panel.json" if kind == "short-u" else root / "panel.json"


def jobs(out, run, pgx):
    result = []

    def add(name, script, args, output, kind, **identity):
        result.append(
            dict(
                name=name,
                command=[sys.executable, "-u", f"scripts/{script}.py", *map(str, args)],
                output=str(output),
                kind=kind,
                **identity,
            )
        )

    pair, cache = out / "training-pair-audit.json", out / "cache-parity.json"
    add(
        "training-pair-audit",
        "audit_resource_training_pair",
        ["--run", run, "--pgx-root", pgx, "--out", pair],
        pair,
        "training",
    )
    for kind in PANEL_ROOTS:
        output = PANEL_ROOTS[kind] / "consumer-audit.json"
        add(
            f"input-{kind}",
            "audit_resource_test_panel",
            ["--panel", kind, "--out", output],
            output,
            "panel",
            panel=kind,
        )
    add(
        "cache-parity",
        "check_resource_search_cache",
        ["--run", run, "--pair-audit", pair, "--pgx-root", pgx, "--out", cache],
        cache,
        "cache",
    )
    for arm in ("units", "raw"):
        for stage in ("initial", "final"):
            for kind in PANEL_ROOTS:
                label = f"{arm}-{stage}-{kind}"
                output = out / "metric" / f"{label}.json"
                add(
                    f"metric-{label}",
                    "evaluate_resource_distances",
                    [
                        "--run",
                        run,
                        "--pair-audit",
                        pair,
                        "--arm",
                        arm,
                        "--stage",
                        stage,
                        "--panel",
                        kind,
                        "--pgx-root",
                        pgx,
                        "--out",
                        output,
                    ],
                    output,
                    "metric",
                    arm=arm,
                    stage=stage,
                    panel=kind,
                )
    for arm, stage, prior in search_matrix():
        label = f"{arm}-{stage}-{prior}"
        directory = out / "mcts" / label
        add(
            f"search-{label}",
            "evaluate_resource_search",
            [
                "--run",
                run,
                "--pair-audit",
                pair,
                "--cache-gate",
                cache,
                "--arm",
                arm,
                "--stage",
                stage,
                "--prior",
                prior,
                "--pgx-root",
                pgx,
                "--out",
                directory,
            ],
            directory / "manifest.json",
            "search",
            arm=arm,
            stage=stage,
            prior=prior,
        )
        output = directory / "replay-audit.json"
        add(
            f"search-audit-{label}",
            "audit_resource_search",
            ["--search", directory, "--out", output],
            output,
            "search-audit",
            arm=arm,
            stage=stage,
            prior=prior,
        )
    if len(result) != 24 or len({j["output"] for j in result}) != 24:
        raise ValueError("incomplete or duplicate24-job matrix")
    return result


def search_matrix():
    return [(a, s, "distance") for a in ("units", "raw") for s in ("initial", "final")] + [
        (a, "final", "uniform") for a in ("units", "raw")
    ]


def verify_job(job, out):
    path = Path(job["output"])
    report = read(path)
    kind = job["kind"]
    expected = "completed" if kind in ("metric", "search") else "passed"
    if report["status"] != expected:
        raise ValueError("child report is not final")
    pins = report["contract"]["inputs"] if kind == "search" else report["inputs"]
    check_pins(pins)
    if kind == "training":
        if (
            set(report["arms"]) != {"units", "raw"}
            or report.get("data_rng_aligned") is not True
            or report.get("initial_tensors_identical") is not True
            or any(r["steps"] != 2048 for r in report["arms"].values())
        ):
            raise ValueError("incomplete or unmatched training audit")
    elif kind == "panel":
        panel = panel_path(job["panel"])
        if (
            report["evaluation_ready"] is not True
            or report["panel"] != str(panel)
            or report["panel_sha256"] != sha(panel)
            or report["selected_pairs"] != len(read(panel)["validation"])
            or report["model_forwards"] != 0
            or report["optimizer_steps"] != 0
        ):
            raise ValueError("panel audit binding/coverage differs")
    else:
        pair = read(out / "training-pair-audit.json")
        if kind == "cache":
            if set(report["models"]) != {
                f"{a}-{s}" for a in ("units", "raw") for s in ("initial", "final")
            }:
                raise ValueError("cache model coverage differs")
            for label, row in report["models"].items():
                arm, stage = label.split("-")
                if (
                    row["status"] != "passed"
                    or row["weights_unchanged"] is not True
                    or row["checkpoint_sha256"] != pair["arms"][arm][stage]["sha256"]
                ):
                    raise ValueError("cache checkpoint differs")
        else:
            expected_sha = pair["arms"][job["arm"]][job["stage"]]["sha256"]
            context = report["contract"] if kind == "search" else report
            if context["checkpoint_sha256"] != expected_sha:
                raise ValueError("child used another checkpoint")
            if kind == "metric":
                panel = panel_path(job["panel"])
                data = read(panel)
                views = (
                    ("validation", "size_shift") if job["panel"] == "short-u" else ("validation",)
                )
                if (
                    context["arm"] != job["arm"]
                    or context["stage"] != job["stage"]
                    or context["panel"] != str(panel)
                    or pins.get(str(panel)) != sha(panel)
                    or report["weights_unchanged"] is not True
                    or set(report["results"]) != set(views)
                    or any(len(report["results"][v]["predictions"]) != len(data[v]) for v in views)
                ):
                    raise ValueError("metric model/panel/coverage differs")
            elif kind == "search":
                if (
                    context["config"] != SEARCH_CONFIG
                    or any(context[k] != job[k] for k in ("arm", "stage", "prior"))
                    or context["pair_audit"] != str(out / "training-pair-audit.json")
                    or context["cache_gate"] != str(out / "cache-parity.json")
                    or report["next_condition"] != 128
                ):
                    raise ValueError("search model/resources/coverage differs")
                validate_resume(report, context, read(path.parent / "roots.json"), path.parent)
            else:
                label = "-".join(job[k] for k in ("arm", "stage", "prior"))
                manifest = path.parent / "manifest.json"
                if (
                    report["label"] != label
                    or report["counts_replayed"]["conditions"] != 128
                    or pins.get(str(manifest)) != sha(manifest)
                ):
                    raise ValueError("search audit binding/coverage differs")
    return dict(name=job["name"], output=str(path), sha256=sha(path))


def ready(run):
    parent = read(run / "manifest.json")
    if parent["config"] != TRAIN_CONFIG:
        raise ValueError("training config differs")
    if parent["status"] != "completed" or parent["completed_arms"] != ["units", "raw"]:
        raise RuntimeError("training is not complete; controller does not alter or restart it")
    for kind, root in PANEL_ROOTS.items():
        status = read(root / ("manifest.json" if kind == "short-u" else "progress.json"))
        if status["status"] != "completed" or status.get("evaluation_ready") is not True:
            raise RuntimeError(f"{kind} panel not complete/adequate; preserve it and inspect")
    return parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("existing controller requires explicit resume")
    parent = ready(args.run)
    planned = jobs(args.out, args.run, args.pgx_root)
    pins = {**parent["inputs"], **parent["sources"]}
    paths = {Path(j["command"][2]) for j in planned} | {
        Path(__file__),
        Path("scripts/summarize_resource_experiment.py"),
        Path("scripts/resource_evaluation_inputs.py"),
        Path("scripts/resource_cost_search.py"),
        Path("scripts/run_clean_paired_evaluation.py"),
        Path("scripts/run_graph_mixture_evaluation.py"),
        Path("scripts/evaluate_raster_log_distances.py"),
        Path("scripts/evaluate_strand_graph_panel.py"),
        Path("scripts/evaluate_graph_mixture_cost_mcts.py"),
        Path("scripts/evaluate_joint_cost_mcts.py"),
        Path("scripts/audit_joint_cost_mcts.py"),
        Path("scripts/audit_cost_mcts_continuations.py"),
        Path("scripts/audit_semantic_cost_floor.py"),
        Path("scripts/probe_raster_tile_cache.py"),
        Path("scripts/probe_raster_cache_search.py"),
        Path("scripts/probe_raster_cache_full_network.py"),
        Path("scripts/summarize_raster_log_experiment.py"),
        Path("scripts/summarize_joint_policy_metric_evaluation.py"),
        Path("scripts/summarize_clean_paired_evaluation.py"),
        Path("scripts/summarize_graph_mixture_experiment.py"),
        Path("tests/test_resource_evaluation_controller.py"),
        Path("research/139-resource-units-evaluation-protocol.md"),
    }
    merge_pins(pins, {str(p): sha(p) for p in paths})
    check_pins(pins)
    contract = digest(dict(training_contract=parent["contract_sha256"], inputs=pins, jobs=planned))
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.out / "manifest.json"
    state = (
        read(path)
        if args.resume
        else dict(
            contract_sha256=contract,
            inputs=pins,
            jobs=planned,
            completed=[],
            current=None,
            training_run=str(args.run),
            pgx_root=str(args.pgx_root),
            training_contract=parent["contract_sha256"],
        )
    )
    if state["contract_sha256"] != contract or state["jobs"] != planned or state["inputs"] != pins:
        raise ValueError("controller resume contract differs")
    check_completed_prefix(state["completed"], planned)
    for job, stored in zip(planned, state["completed"], strict=False):
        if verify_job(job, args.out) != stored:
            raise ValueError("committed child changed")
    if state.get("status") == "completed":
        if sha(args.out / "summary.json") != state["summary_sha256"]:
            raise ValueError("completed summary changed")
        return
    if not args.resume:
        snapshot = args.out / "source_snapshot"
        snapshot.mkdir()
        for i, source in enumerate(sorted(paths)):
            shutil.copy2(source, snapshot / f"{i:03d}-{source.name}")
    (args.out / "logs").mkdir(exist_ok=True)
    (args.out / "metric").mkdir(exist_ok=True)
    state.update(pid=os.getpid(), status="prepared")
    atomic(path, state)
    try:
        for job in planned[len(state["completed"]) :]:
            check_pins(pins)
            live = live_commands(job["command"], processes())
            if len(live) > 1:
                raise RuntimeError("duplicate matching evaluation workers")
            output = Path(job["output"])
            if (
                output.exists()
                and not live
                and read(output).get("status") in ("passed", "completed")
            ):
                stored = verify_job(job, args.out)
            else:
                child = None
                if not live:
                    argv = job["command"].copy()
                    if output.exists():
                        if job["kind"] != "search":
                            raise RuntimeError("partial nonresumable output; preserving it")
                        argv.append("--resume")
                    with (args.out / "logs" / f"{job['name']}.log").open("a") as log:
                        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                    live = [child.pid]
                state.update(status="evaluating", current=dict(name=job["name"], pid=live[0]))
                atomic(path, state)
                print(json.dumps(state["current"]), flush=True)
                while True:
                    if child is not None:
                        code = child.poll()
                        if code is not None:
                            if code:
                                raise RuntimeError(f"child failed {code}: {job['name']}")
                            break
                    elif not live_commands(job["command"], processes()):
                        break
                    time.sleep(30)
                stored = verify_job(job, args.out)
            state["completed"].append(stored)
            state.update(status="between_jobs", current=None)
            atomic(path, state)
        state.update(status="scored", current=None)
        atomic(path, state)
        from summarize_resource_experiment import summarize_verified

        summary = summarize_verified(args.out)
        summary_path = args.out / "summary.json"
        if summary_path.exists() and read(summary_path) != summary:
            raise ValueError("existing summary differs; preserved")
        if not summary_path.exists():
            atomic(summary_path, summary)
        check_pins(pins)
        state.update(status="completed", summary_sha256=sha(summary_path))
        atomic(path, state)
    except Exception as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        atomic(path, state)
        raise


if __name__ == "__main__":
    main()
