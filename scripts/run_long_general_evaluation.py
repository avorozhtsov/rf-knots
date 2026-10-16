"""Serial18-job149 evaluation; does not start, stop, or change training."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, digest, sha
from check_long_general_search_cache import MODELS
from evaluate_graph_mixture_cost_mcts import build_roots
from evaluate_long_general_search import require_cache_gate, validate_resume
from long_general_evaluation_inputs import load_inputs, model_entry
from run_clean_paired_evaluation import check_completed_prefix, processes
from run_graph_mixture_evaluation import live_commands
from train_long_general_pair import (
    CONFIG,
    SCHEMA,
    SEARCH_CONFIG,
    SEARCH_MATRIX,
    evaluation_contract,
)


def read(path):
    return json.loads(Path(path).read_text())


def panel_views(parent, kind):
    evaluation = parent["contract"]["evaluation"]
    if kind == "general":
        path = Path(evaluation["long_corpus"])
        data = {"test": read(path)["records"]["test"]}
    elif kind == "short-u":
        path = Path(evaluation["short_u_panel"])
        panel = read(path)
        data = {key: panel[key] for key in ("validation", "size_shift")}
    else:
        raise ValueError("unknown149 panel")
    expected = evaluation["long_corpus_sha256" if kind == "general" else "short_u_panel_sha256"]
    if sha(path) != expected:
        raise ValueError("frozen149 panel changed")
    return path, data


def jobs(out, run, pgx):
    result = []

    def add(name, script, arguments, output, kind, **identity):
        result.append(
            dict(
                name=name,
                command=[sys.executable, "-u", f"scripts/{script}.py", *map(str, arguments)],
                output=str(output),
                kind=kind,
                **identity,
            )
        )

    pair, cache = out / "training-pair-audit.json", out / "cache-parity.json"
    common = ["--run", run, "--pair-audit", pair, "--pgx-root", pgx]
    add(
        "training-pair-audit",
        "audit_long_general_training_pair",
        ["--run", run, "--pgx-root", pgx, "--out", pair],
        pair,
        "training",
    )
    add(
        "cache-parity", "check_long_general_search_cache", [*common, "--out", cache], cache, "cache"
    )
    for arm, stage in MODELS:
        for panel in ("general", "short-u"):
            label = f"{arm}-{stage}-{panel}"
            output = out / "metric" / f"{label}.json"
            add(
                f"metric-{label}",
                "evaluate_long_general_distances",
                [*common, "--arm", arm, "--stage", stage, "--panel", panel, "--out", output],
                output,
                "metric",
                arm=arm,
                stage=stage,
                panel=panel,
            )
    for row in SEARCH_MATRIX:
        label = "-".join(row[k] for k in ("arm", "stage", "prior"))
        directory = out / "mcts" / label
        selection = [item for k, v in row.items() for item in ("--" + k, v)]
        add(
            f"search-{label}",
            "evaluate_long_general_search",
            [*common, "--cache-gate", cache, *selection, "--out", directory],
            directory / "manifest.json",
            "search",
            **row,
        )
        output = directory / "replay-audit.json"
        add(
            f"search-audit-{label}",
            "audit_long_general_search",
            ["--search", directory, "--out", output],
            output,
            "search-audit",
            **row,
        )
    if len(result) != 18 or len({r["output"] for r in result}) != 18:
        raise ValueError("expected18 distinct149 evaluation jobs")
    return result


def ready(run):
    parent = read(run / "manifest.json")
    if (
        parent["schema"] != SCHEMA
        or parent["contract"]["config"] != CONFIG
        or digest(parent["contract"]) != parent["contract_sha256"]
    ):
        raise ValueError("149 training contract differs")
    if (
        parent["status"] != "completed"
        or parent["completed_arms"] != CONFIG["arms"]
        or parent["current_arm"] is not None
    ):
        raise RuntimeError("149 training not complete; controller cannot alter/restart it")
    check_pins(parent["contract"]["inputs"])
    evaluation = parent["contract"]["evaluation"]
    actual, _ = evaluation_contract(
        Path(evaluation["long_corpus"]).parent, Path(evaluation["short_u_panel"]).parent
    )
    if actual != evaluation:
        raise ValueError("frozen149 evaluation matrix differs")
    return parent


def verify_job(job, out, run):
    path = Path(job["output"])
    report = read(path)
    kind = job["kind"]
    if report["status"] != ("completed" if kind in ("metric", "search") else "passed"):
        raise ValueError("149 child report is not final")
    pins = report["contract"]["inputs"] if kind == "search" else report["inputs"]
    source = Path(job["command"][2]).resolve()
    if pins.get(str(source)) != sha(source):
        raise ValueError("child source binding differs")
    check_pins(pins)
    if kind == "training":
        load_inputs(run, path)
    else:
        parent = read(run / "manifest.json")
        pair_path = out / "training-pair-audit.json"
        pair = read(pair_path)
        if pins.get(str(pair_path)) != sha(pair_path):
            raise ValueError("child not bound to the paired audit")
        if kind == "cache":
            for arm, stage in MODELS:
                _, checkpoint = model_entry(run, pair, arm, stage)
                require_cache_gate(path, checkpoint, arm, stage, {}, parent["contract_sha256"])
        else:
            _, checkpoint = model_entry(run, pair, job["arm"], job["stage"])
            context = report["contract"] if kind == "search" else report
            if (
                context["checkpoint_sha256"] != sha(checkpoint)
                or context["training_contract_sha256"] != parent["contract_sha256"]
            ):
                raise ValueError("child model/training identity differs")
            if kind == "metric":
                from summarize_long_general_experiment import validate_metric_result

                panel, views = panel_views(parent, job["panel"])
                if (
                    report["schema"] != "rf-knots-long-general-distances-v1"
                    or any(report[k] != job[k] for k in ("arm", "stage"))
                    or report["panel"] != str(panel)
                    or report["panel_sha256"] != sha(panel)
                    or report["weights_unchanged"] is not True
                    or report["optimizer_steps"] != 0
                    or (report["device"], report["threads"], report["microbatch"]) != ("cpu", 1, 1)
                    or report["torch_version"] != torch.__version__
                    or set(report["results"]) != set(views)
                    or report["observed_training_sizes"]
                    != {a: r["size_counters"] for a, r in pair["arms"].items()}
                ):
                    raise ValueError("149 metric identity/config/coverage differs")
                if job["stage"] == "initial":
                    identity = report["shared_initial_identity"]
                    if not identity or any(
                        identity[k] is not True
                        for k in ("exact_tensors", "encoder_equal", "policy_equal")
                    ):
                        raise ValueError("metric shared initial identity absent")
                for view, rows in views.items():
                    validate_metric_result(report["results"][view], rows)
            elif kind == "search":
                panel, _ = panel_views(parent, "short-u")
                if (
                    context["config"] != SEARCH_CONFIG
                    or any(context[k] != job[k] for k in ("arm", "stage", "prior"))
                    or context["pair_audit"] != str(pair_path)
                    or context["panel"] != str(panel)
                    or context["cache_gate"] != str(out / "cache-parity.json")
                    or report["next_condition"] != 128
                ):
                    raise ValueError("149 search identity/resources/coverage differs")
                validate_resume(report, context, build_roots(read(panel)), path.parent)
            elif kind == "search-audit":
                label = "-".join(job[k] for k in ("arm", "stage", "prior"))
                manifest = path.parent / "manifest.json"
                if (
                    report["schema"] != "rf-knots-long-general-search-audit-v1"
                    or report["label"] != label
                    or report["search"] != str(path.parent)
                    or report["counts_replayed"]["conditions"] != 128
                    or pins.get(str(manifest)) != sha(manifest)
                ):
                    raise ValueError("149 replay audit binding/coverage differs")
            else:
                raise ValueError("unknown149 job kind")
    return dict(name=job["name"], output=str(path), sha256=sha(path))


def await_job(job, out, state, state_path, run):
    live = live_commands(job["command"], processes())
    if len(live) > 1:
        raise RuntimeError("duplicate matching149 evaluation workers; preserve and inspect")
    output = Path(job["output"])
    if output.exists() and not live and read(output).get("status") in ("passed", "completed"):
        return verify_job(job, out, run)
    child = None
    if not live:
        argv = job["command"].copy()
        if output.exists():
            if job["kind"] != "search":
                raise RuntimeError("partial finite-job output; preserve and inspect")
            argv.append("--resume")
        with (out / "logs" / f"{job['name']}.log").open("a") as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        live = [child.pid]
    state.update(status="evaluating", current=dict(name=job["name"], pid=live[0]))
    atomic(state_path, state)
    print(json.dumps(state["current"]), flush=True)
    while True:
        if child is not None:
            code = child.poll()
            if code is not None:
                if code:
                    raise RuntimeError(f"149 child failed {code}: {job['name']}")
                break
        elif not live_commands(job["command"], processes()):
            break
        time.sleep(30)
    return verify_job(job, out, run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "pgx-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("new controller output required, or explicit resume")
    parent = ready(args.run)
    planned = jobs(args.out, args.run, args.pgx_root)
    pins = dict(parent["contract"]["inputs"])
    paths = {Path(j["command"][2]).resolve() for j in planned} | {
        Path(name).resolve()
        for name in (
            "scripts/run_long_general_evaluation.py",
            "scripts/summarize_long_general_experiment.py",
            "scripts/long_general_evaluation_inputs.py",
            "scripts/run_clean_paired_evaluation.py",
            "scripts/run_graph_mixture_evaluation.py",
            "scripts/summarize_resource_experiment.py",
            "scripts/summarize_joint_policy_metric_evaluation.py",
            "scripts/summarize_raster_log_experiment.py",
            "scripts/summarize_graph_mixture_experiment.py",
            "scripts/audit_semantic_cost_floor.py",
            "tests/test_long_general_evaluation_controller.py",
            "tests/test_long_general_summary.py",
            "research/151-long-mixture-evaluation-contract.md",
        )
    }
    merge_pins(pins, {str(p): sha(p) for p in paths | {args.run / "manifest.json"}})
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
            schema="rf-knots-long-general-evaluation-v1",
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
    if (
        state["schema"] != "rf-knots-long-general-evaluation-v1"
        or state["contract_sha256"] != contract
        or state["jobs"] != planned
        or state["inputs"] != pins
        or state["training_run"] != str(args.run)
        or state["pgx_root"] != str(args.pgx_root)
        or state["training_contract"] != parent["contract_sha256"]
    ):
        raise ValueError("149 controller resume contract differs")
    check_completed_prefix(state["completed"], planned)
    for job, stored in zip(planned, state["completed"], strict=False):
        if verify_job(job, args.out, args.run) != stored:
            raise ValueError("committed149 child changed")
    snapshot = args.out / "source_snapshot"
    if not args.resume:
        snapshot.mkdir()
        for i, source in enumerate(sorted(paths)):
            shutil.copy2(source, snapshot / f"{i:03d}-{source.name}")
    for i, source in enumerate(sorted(paths)):
        if sha(snapshot / f"{i:03d}-{source.name}") != pins[str(source)]:
            raise ValueError("149 controller source snapshot differs")
    if state.get("status") == "completed":
        if len(state["completed"]) != len(planned) or state["current"] is not None:
            raise ValueError("completed149 controller lacks the full matrix")
        if sha(args.out / "summary.json") != state["summary_sha256"]:
            raise ValueError("completed149 summary changed")
        return
    for name in ("logs", "metric"):
        (args.out / name).mkdir(exist_ok=True)
    state.update(pid=os.getpid(), status="prepared")
    atomic(path, state)
    try:
        for job in planned[len(state["completed"]) :]:
            check_pins(pins)
            stored = await_job(job, args.out, state, path, args.run)
            state["completed"].append(stored)
            state.update(status="between_jobs", current=None)
            atomic(path, state)
        state.update(status="scored", current=None)
        atomic(path, state)
        from summarize_long_general_experiment import summarize_verified

        summary = summarize_verified(args.out)
        target = args.out / "summary.json"
        if target.exists() and read(target) != summary:
            raise ValueError("existing149 summary differs; preserving it")
        if not target.exists():
            atomic(target, summary)
        check_pins(pins)
        state.update(status="completed", summary_sha256=sha(target))
        atomic(path, state)
    except Exception as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        atomic(path, state)
        raise


if __name__ == "__main__":
    main()
