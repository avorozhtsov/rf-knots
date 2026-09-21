"""Serial control training and final-only evaluation; research/42,44,45."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from run_clean_paired_evaluation import read, sha, save, processes, matching_children, check_completed_prefix

BASE = Path("artifacts")
RUNS = {"parent": BASE / "embedding-multisize-clean-joint-v1",
        "candidate": BASE / "embedding-diverse-metric-joint-v1",
        "control": BASE / "embedding-diverse-metric-control-v1"}
DATA = BASE / "embedding-clean-paired-rollouts-v1"
CORPORA = {"candidate": BASE / "embedding-diverse-short-u-corpus-v1/corpus.json",
           "control": BASE / "embedding-multisize-clean-corpus-v1/corpus.json"}
PANELS = {"supplement": BASE / "embedding-common-block-gordian-supplement-v1/panel.json",
          "short-u": BASE / "embedding-diverse-metric-short-u-test-v1/panel.json"}


def training_command(model):
    return [sys.executable, "-u", "scripts/train_rollout_value.py", "--data", str(DATA),
            "--metric-corpus", str(CORPORA[model]), "--out", str(RUNS[model]),
            "--value-weight", "1", "--steps", "4096", "--seed", "20261018",
            "--policy-micro-batch", "64", "--min-relative-groups", "32",
            "--min-long-relative-groups", "8", "--long-root-length", "96",
            "--metric-bounds", "determinant", "--metric-sampling", "group-balanced"]


def evaluation_jobs(out):
    jobs = []
    def add(name, script, args, output):
        jobs.append({"name": name, "command": [sys.executable, "-u", f"scripts/{script}.py", *map(str, args)],
                     "output": str(output)})
    for name, run in RUNS.items():
        for panel, bounds in (("supplement", "original"), ("short-u", "original"), ("short-u", "determinant")):
            label = f"metric-{name}-{panel}-{bounds}"
            output = out / "metric" / f"{label}.json"
            add(label, "evaluate_strand_graph_panel", ["--run", run, "--panel", PANELS[panel],
                "--bounds", bounds, "--output", output], output)
    for name in (*RUNS, "length"):
        target = out / "cost" / name
        add(f"cost-{name}", "evaluate_embedding_cost_search", ["--run", RUNS.get(name, RUNS["parent"]),
            "--panel", PANELS["short-u"], "--out", target, "--mode", "length" if name == "length" else "learned",
            "--expansions", 128, "--witness-caps", "--all-queries", "--layout-tiebreak"], target / "report.json")
    for baseline, candidate in (("parent", "candidate"), ("parent", "control"), ("control", "candidate"), ("length", "candidate")):
        name = f"cost-{candidate}-vs-{baseline}"
        output = out / "comparisons" / f"{name}.json"
        add(name, "compare_embedding_cost_search", ["--baseline", out / "cost" / baseline / "report.json",
            "--candidate", out / "cost" / candidate / "report.json", "--panel", PANELS["short-u"],
            "--output", output], output)
    return jobs


def training_workers(model, listing):
    command = training_command(model)
    return matching_children({"command": command}, listing) + matching_children({"command": command + ["--resume"]}, listing)


def check_final_manifest(name, m):
    cfg = m["config"]
    expected = {"steps": 4096, "seed": 20261018, "value_weight": 1, "data": str(DATA),
                "metric_sampling": "group-balanced", "metric_bounds": "determinant",
                "metric_corpus": str(CORPORA[name]), "out": str(RUNS[name]),
                "policy_micro_batch": 64, "min_relative_groups": 32, "min_long_relative_groups": 8,
                "long_root_length": 96, "batch_size": 64, "lr": 2.5e-5,
                "policy_weight": 0.1, "clip": 0.2, "train_seeds": list(range(6)),
                "window_size": 0, "hierarchy_levels": 0}
    if (m["status"] != "completed" or m["completed_steps"] != 4096
            or any(cfg.get(k) != v for k, v in expected.items())):
        raise ValueError("final training contract differs")
    if m["initial_checkpoint_sha256"] != "b2cd2502dace3b8a0daddcc7424151c6ef87578f63e3394467aa437f1c6e64c3":
        raise ValueError("initial tensors do not match pinned parent actor")
    if m["relative_groups"] < 32 or m["long_relative_groups"] < 8:
        raise ValueError("relative supervision gate not met")


def check_pair(manifests):
    stripped = []
    for name, m in zip(("candidate", "control"), manifests, strict=True):
        cfg = m["config"]
        check_final_manifest(name, m)
        stripped.append({k: v for k, v in cfg.items() if k not in ("metric_corpus", "out")})
    if stripped[0] != stripped[1]:
        raise ValueError("training configurations differ beyond metric corpus/output")
    for key in ("sources", "architecture", "initial_checkpoint_sha256", "value_rows", "relative_groups", "long_relative_groups"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"training {key} differ")
    inputs = [{k: v for k, v in m["inputs"].items() if k != m["config"]["metric_corpus"]} for m in manifests]
    if inputs[0] != inputs[1]:
        raise ValueError("shared training inputs differ")


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from tensors(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from tensors(v)


def verify_pins(pins):
    for name, digest in pins.items():
        if sha(Path(name)) != digest:
            raise ValueError(f"pinned input changed: {name}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    planned = evaluation_jobs(args.out)
    source = {Path(__file__).resolve().relative_to(Path.cwd()), Path("scripts/run_clean_paired_evaluation.py"),
              Path("scripts/train_rollout_value.py"), Path("scripts/balanced_metric_sampling.py"),
              Path("scripts/run_strand_graph_pilot.py"), *Path("src/rf_knots").rglob("*.py"),
              *(Path(j["command"][2]) for j in planned)}
    paths = source | {Path(f"research/{name}") for name in (
        "42-diverse-metric-training-protocol.md", "44-common-block-gordian-supplement.md", "45-diverse-metric-sequence.md")}
    paths |= {*PANELS.values(), *CORPORA.values(), DATA / "manifest.json", DATA / "actor.pt", DATA / "roots.json",
              RUNS["parent"] / "checkpoint.pt", BASE / "embedding-diverse-metric-gordian-test-v1/panel.json"}
    pins = {str(path): sha(path) for path in sorted(paths)}
    args.out.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.out / "writer.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.out / "manifest.json"
    state = read(path) if args.resume else {"status": "waiting_candidate", "inputs": pins,
        "jobs": planned, "completed": [], "pid": os.getpid(), "protocol_amendment": "research/44; original general panel inadequate"}
    if state["inputs"] != pins or state["jobs"] != planned:
        raise ValueError("sequence resume contract differs")
    check_completed_prefix(state["completed"], planned)
    for row in state["completed"]:
        if sha(Path(row["output"])) != row["sha256"]:
            raise ValueError("completed output changed")
    if state["status"] == "completed":
        verify_pins({**state["audit_pins"], **{str(RUNS[n] / "checkpoint.pt"): h for n, h in state["final_checkpoints"].items()}})
        print("already completed"); return
    if not args.resume:
        for filename in source:
            target = args.out / "source_snapshot" / filename
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(filename, target)
        (args.out / "logs").mkdir()
    state["pid"] = os.getpid(); save(path, state)
    try:
        for model in ("candidate", "control"):
            job = {"name": f"train-{model}", "command": training_command(model)}
            owned = None
            verify_pins(pins)
            while True:
                verify_pins({str(p): pins[str(p)] for p in source})
                manifest = RUNS[model] / "manifest.json"
                preparation = RUNS[model] / "preparation.json"
                m = read(manifest) if manifest.exists() else read(preparation) if preparation.exists() else None
                live = training_workers(model, processes())
                if owned is not None and owned.poll() not in (None, 0):
                    raise RuntimeError("control training failed; inspect its log and checkpoint")
                if len(live) > 1:
                    raise RuntimeError("duplicate training workers")
                if m and m["status"] == "completed" and not live:
                    check_final_manifest(model, m)
                    verify_pins({**m["sources"], **m["inputs"]})
                    if sha(RUNS[model] / "checkpoint.pt") != m["checkpoint_sha256"]:
                        raise ValueError("completed worker checkpoint differs")
                    break
                if not live:
                    if model == "candidate" or RUNS[model].exists() or state.get("control_started"):
                        raise RuntimeError("partial/missing training worker; preserve it and explicitly resume, never restart from scratch")
                    with (args.out / "logs/train-control.log").open("xb") as log:
                        owned = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT)
                    state["control_started"] = True
                    live = [owned.pid]
                observed = {"model": model, "pid": live[0], "status": m.get("status") if m else "starting",
                            "completed_steps": m.get("completed_steps") if m else None,
                            "preparation_phase": m.get("phase") if m else None,
                            "prepared_episodes": m.get("verified") if m else None}
                if state.get("training") != observed:
                    state.update(status=f"waiting_{model}", training=observed); save(path, state)
                time.sleep(15)
        manifests = [read(RUNS[n] / "manifest.json") for n in ("candidate", "control")]
        check_pair(manifests)
        checkpoints = []
        for name, m in zip(("candidate", "control"), manifests, strict=True):
            verify_pins({**m["sources"], **m["inputs"]})
            if sha(RUNS[name] / "checkpoint.pt") != m["checkpoint_sha256"] or sha(RUNS[name] / "initial.pt") != m["initial_checkpoint_sha256"]:
                raise ValueError("training checkpoint hash mismatch")
            c = torch.load(RUNS[name] / "checkpoint.pt", map_location="cpu", weights_only=False)
            if c["steps"] != 4096 or any(not torch.isfinite(t).all() for t in tensors({k: c[k] for k in ("model", "objective", "optimizer")})):
                raise ValueError("invalid final checkpoint tensors")
            checkpoints.append(c)
        if checkpoints[0]["rng"] != checkpoints[1]["rng"] or not torch.equal(checkpoints[0]["torch_rng"], checkpoints[1]["torch_rng"]):
            raise ValueError("paired training RNG streams differ")
        finals = {name: sha(run / "checkpoint.pt") for name, run in RUNS.items()}
        if state.get("final_checkpoints", finals) != finals:
            raise ValueError("previous final model pins changed")
        state["final_checkpoints"] = finals
        audits = {}
        for panel in PANELS.values():
            if not read(panel)["complete"]:
                raise ValueError("inadequate evaluation panel")
            for filename in ("metric-independence-audit.json", "rollout-independence-audit.json"):
                audit_path = panel.parent / filename
                report = read(audit_path)
                if report["status"] != "passed" or report["inputs"][str(panel)] != sha(panel):
                    raise ValueError("missing or mismatched independence certificate")
                verify_pins(report["inputs"])
                audits[str(audit_path)] = sha(audit_path)
        if state.get("audit_pins", audits) != audits:
            raise ValueError("previous audit pins changed")
        state["audit_pins"] = audits; save(path, state)
        for job in planned[len(state["completed"]):]:
            verify_pins({**pins, **audits, **{str(RUNS[n] / "checkpoint.pt"): h for n, h in finals.items()}})
            output = Path(job["output"])
            live = matching_children(job, processes())
            prior = state.get("current", {}).get("name") == job["name"]
            if len(live) > 1 or (not live and output.exists() and not prior):
                raise RuntimeError("duplicate or unexpected evaluation output")
            if prior and not live and not output.exists():
                raise RuntimeError("lost partial evaluator; no automatic restart")
            child = None
            if not live and not output.exists():
                with (args.out / "logs" / f"{job['name']}.log").open("xb") as log:
                    child = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT)
                live = [child.pid]
            state.update(status="evaluating", current={**job, "pid": live[0] if live else None}); save(path, state)
            if child is not None:
                if child.wait() != 0:
                    raise RuntimeError(f"evaluation failed: {job['name']}")
            else:
                while matching_children(job, processes()): time.sleep(15)
            if read(output).get("status") not in ("passed", "completed"):
                raise ValueError("evaluation output not final")
            verify_pins({**pins, **audits, **{str(RUNS[n] / "checkpoint.pt"): h for n, h in finals.items()}})
            state["completed"].append({"name": job["name"], "output": str(output), "sha256": sha(output)})
            state.pop("current", None); save(path, state)
            print(json.dumps({"completed": job["name"], "jobs": len(state["completed"]), "total": len(planned)}), flush=True)
        state["status"] = "completed"; save(path, state)
    except Exception as error:
        state.update(status="attention_required", error=str(error)); save(path, state); raise


if __name__ == "__main__":
    main()
