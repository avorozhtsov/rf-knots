"""Resumable TRAIN-only short positive-seed R2 MCTS F-yield pilot."""

import argparse
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path

import torch

from bootstrap_cost_value_teacher_v1 import load_actor
from build_mcts_training_supervision import atomic, digest, sha
from resource_cost_search import run_batch
from rf_knots import mcts_loop as M

ROSTER = Path("artifacts/f-short-positive-roster-v1/roster.json")
ROSTER_SHA = "53d6a26828acb966747d17699a3b74f5ab13e0121fa055edbbaa2f57fd242f42"
AUDIT = Path("artifacts/f-short-positive-roster-audit-v1/report.json")
AUDIT_SHA = "86257d258ca2f121e924d4343fe39f67410aa51b762bf435aae6af9ae23cc61e"
ACTOR = Path("artifacts/additive-f-r2-stage-v2/step-152964.pt")
ACTOR_SHA = "12f4a3675dd6880203a2968431d3c7bdd44a45b586e92937a25373facc32803e"
PROTOCOL = Path("research/340-short-positive-r2-mcts-yield-pilot-v1.md")
SEEDS = (20271340, 20271341)
CONFIG = {"insertion_candidates": 4, "widening_batch": 4,
          "c_puct": 1.5, "temperature": 1.0, "simulations": 64}
TASK_CPU_CEILING = 120
TOTAL_CPU_CEILING = 2400


def read(path):
    return json.loads(path.read_text())


def load_policy():
    net, objective = load_actor()
    saved = torch.load(ACTOR, map_location="cpu", weights_only=False)
    if saved["step"] != 152964:
        raise ValueError("frozen R2 checkpoint step differs")
    net.encoder.graph_rounds = 2
    net.load_state_dict(saved["model"], strict=True)
    objective.load_state_dict(saved["objective"], strict=True)
    net.eval()
    objective.eval()
    return net, objective


def cpu_timeout(_signum, _frame):
    raise TimeoutError("per-condition CPU ceiling")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.stop_after < 0 or args.out.exists() != args.resume:
        raise ValueError("output/resume mismatch")
    if not (args.pgx_root.resolve() / "src" / "pgx_mcts_bench").exists():
        raise ValueError("canonical pgx-mcts-bench missing")
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    torch.set_num_threads(1)
    if (sha(ROSTER) != ROSTER_SHA or sha(AUDIT) != AUDIT_SHA
            or sha(ACTOR) != ACTOR_SHA):
        raise ValueError("frozen roster/audit/actor SHA differs")
    roster = read(ROSTER)
    roots = roster["stages"][0][:4] + roster["stages"][1][:4]
    if (len(roots) != 8
            or len({tuple(root["query_group"]) for root in roots}) != 8):
        raise ValueError("fixed balanced root panel differs")
    tasks = [(r, w, seed) for r in range(8)
             for w in range(2) for seed in SEEDS]
    pins = (ROSTER, AUDIT, ACTOR, PROTOCOL, Path(__file__),
            Path("scripts/bootstrap_cost_value_teacher_v1.py"),
            Path("scripts/resource_cost_search.py"),
            Path("src/rf_knots/cost_mcts_rollout.py"))
    inputs = {str(path): sha(path) for path in pins}
    contract = {
        "schema": "rf-knots-short-positive-R2-F-yield-pilot-v1",
        "role": "TRAIN-only search collection, not optimizer training",
        "root_generator_indices": [root["generator_index"] for root in roots],
        "search_seeds": list(SEEDS), "config": CONFIG,
        "task_cpu_ceiling": TASK_CPU_CEILING,
        "total_cpu_ceiling": TOTAL_CPU_CEILING,
        "gradient_boundary": "root only; no intermediate targets",
        "inputs": inputs,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out / "manifest.json"
        if args.resume:
            state = read(manifest_path)
            if (state["contract"] != contract
                    or state["contract_sha256"] != digest(contract)
                    or state["next_task"] != len(state["shards"])):
                raise ValueError("resume contract/shard state differs")
            for position, entry in enumerate(state["shards"]):
                if (entry["file"] != f"attempt-{position:04d}.json"
                        or sha(args.out / entry["file"]) != entry["sha256"]):
                    raise ValueError("committed shard SHA differs")
        else:
            state = {"contract": contract,
                     "contract_sha256": digest(contract),
                     "status": "preparing", "next_task": 0,
                     "cpu_seconds": 0.0, "shards": []}
            atomic(manifest_path, state)
        if state["next_task"] == len(tasks):
            state["status"] = "completed"
            atomic(manifest_path, state)
            return
        if state["cpu_seconds"] >= TOTAL_CPU_CEILING:
            state["status"] = "cpu_ceiling"
            atomic(manifest_path, state)
            return
        net, objective = load_policy()
        state.update(status="collecting", pid=os.getpid())
        atomic(manifest_path, state)
        end = min(len(tasks), state["next_task"] +
                  (args.stop_after or len(tasks)))
        for position in range(state["next_task"], end):
            if state["cpu_seconds"] >= TOTAL_CPU_CEILING:
                state["status"] = "cpu_ceiling"
                break
            if any(sha(Path(path)) != expected
                   for path, expected in inputs.items()):
                raise ValueError("pinned input changed")
            root_index, weight_index, seed = tasks[position]
            root = roots[root_index]
            instance_dict = root["instances"][weight_index]
            instance = M.Instance(**{**instance_dict,
                                     "word": tuple(instance_dict["word"])})
            config = {**CONFIG, "seed": seed,
                      "horizon": instance.cap_moves}
            started_wall = time.monotonic()
            started_cpu = time.process_time()
            previous_handler = signal.signal(signal.SIGPROF, cpu_timeout)
            signal.setitimer(signal.ITIMER_PROF, TASK_CPU_CEILING)
            try:
                rows, counts = run_batch(
                    [{"condition": root_index,
                      "panel_index": root["generator_index"],
                      "instance": instance_dict}],
                    net, objective, 0, config, prior="distance", cached=True)
                row = rows[0]
                censored = False
            except TimeoutError:
                row, counts, censored = None, {}, True
            finally:
                signal.setitimer(signal.ITIMER_PROF, 0)
                signal.signal(signal.SIGPROF, previous_handler)
            cpu_seconds = time.process_time() - started_cpu
            shard = args.out / f"attempt-{position:04d}.json"
            atomic(shard, {
                "position": position, "root_index": root_index,
                "generator_index": root["generator_index"],
                "weight": instance.cc_weight, "seed": seed,
                "cpu_censored": censored, "row": row, "counts": counts,
                "cpu_seconds": cpu_seconds,
                "wall_seconds": time.monotonic() - started_wall,
            })
            state["shards"].append({"file": shard.name, "sha256": sha(shard)})
            state["next_task"] = position + 1
            state["cpu_seconds"] += cpu_seconds
            atomic(manifest_path, state)
            print(json.dumps({"next_task": state["next_task"],
                              "weight": instance.cc_weight,
                              "cpu_censored": censored,
                              "cpu_seconds": cpu_seconds}), flush=True)
        if state["next_task"] == len(tasks):
            state["status"] = "completed"
        elif state["cpu_seconds"] >= TOTAL_CPU_CEILING:
            state["status"] = "cpu_ceiling"
        atomic(manifest_path, state)


if __name__ == "__main__":
    main()
