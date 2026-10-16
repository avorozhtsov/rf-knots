"""Resumable eight-condition TRAIN-only root-only MCTS yield micro-pilot."""

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import torch

from bootstrap_cost_value_teacher_v1 import load_actor
from build_mcts_training_supervision import atomic, digest, sha
from resource_cost_search import run_batch
from rf_knots import mcts_loop as M

ROSTER = Path("artifacts/f-root-only-roster-v1/roster.json")
ROSTER_SHA = "3c4a8eb62a63d9f4e22a4314aa94434164bb5a4c57fa70282783b9c602e23a81"
AUDIT = Path("artifacts/f-root-only-roster-audit-v1/report.json")
AUDIT_SHA = "755d48594e69c76c16041fce7279a258532b82e9ebc65254d64666d47511dd66"
ACTOR = Path("artifacts/additive-f-r2-stage-v2/step-152964.pt")
ACTOR_SHA = "12f4a3675dd6880203a2968431d3c7bdd44a45b586e92937a25373facc32803e"
PROTOCOL = Path("research/328-root-only-hard-f-yield-micro-pilot-v1.md")
SEED = 20271330
CONFIG = {"insertion_candidates": 4, "widening_batch": 4,
          "c_puct": 1.5, "temperature": 1.0, "simulations": 64}


def read(path):
    return json.loads(path.read_text())


def load_policy():
    net, objective = load_actor()
    saved = torch.load(ACTOR, map_location="cpu", weights_only=False)
    if saved["step"] != 152964:
        raise ValueError("frozen R2 step differs")
    net.encoder.graph_rounds = 2
    net.load_state_dict(saved["model"], strict=True)
    objective.load_state_dict(saved["objective"], strict=True)
    net.eval(); objective.eval()
    return net, objective


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
    roots = roster["stages"][0][:2] + roster["stages"][1][:2]
    if (len(roots) != 4 or len({tuple(row["query_group"]) for row in roots}) != 4):
        raise ValueError("fixed distinct-root selection differs")
    tasks = [(root_index, weight_index)
             for root_index in range(4) for weight_index in range(2)]
    pins = (ROSTER, AUDIT, ACTOR, PROTOCOL, Path(__file__),
            Path("scripts/bootstrap_cost_value_teacher_v1.py"),
            Path("scripts/resource_cost_search.py"))
    inputs = {str(path): sha(path) for path in pins}
    contract = {
        "schema": "rf-knots-root-only-hard-F-yield-micro-v1",
        "role": "TRAIN-only search outcomes, not optimizer training",
        "root_generator_indices": [root["generator_index"] for root in roots],
        "search_seed": SEED, "config": CONFIG,
        "gradient_boundary": "root state only; no intermediate targets",
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
                    raise ValueError("saved shard SHA differs")
        else:
            state = {"contract": contract,
                     "contract_sha256": digest(contract),
                     "status": "preparing", "next_task": 0, "shards": []}
            atomic(manifest_path, state)
        if state["next_task"] == len(tasks):
            state["status"] = "completed"
            atomic(manifest_path, state)
            return
        net, objective = load_policy()
        state.update(status="collecting", pid=os.getpid())
        atomic(manifest_path, state)
        end = min(len(tasks), state["next_task"] +
                  (args.stop_after or len(tasks)))
        for position in range(state["next_task"], end):
            if any(sha(Path(path)) != expected for path, expected in inputs.items()):
                raise ValueError("pinned input changed")
            root_index, weight_index = tasks[position]
            root = roots[root_index]
            instance_dict = root["instances"][weight_index]
            instance = M.Instance(**{**instance_dict,
                                     "word": tuple(instance_dict["word"])})
            config = {**CONFIG, "seed": SEED, "horizon": instance.cap_moves}
            started_wall, started_cpu = time.monotonic(), time.process_time()
            rows, counts = run_batch(
                [{"condition": root_index, "panel_index": root["generator_index"],
                  "instance": instance_dict}], net, objective, 0, config,
                prior="distance", cached=True)
            shard = args.out / f"attempt-{position:04d}.json"
            atomic(shard, {
                "position": position, "root_index": root_index,
                "generator_index": root["generator_index"],
                "weight": instance.cc_weight, "seed": SEED,
                "row": rows[0], "counts": counts,
                "cpu_seconds": time.process_time() - started_cpu,
                "wall_seconds": time.monotonic() - started_wall,
            })
            state["shards"].append({"file": shard.name, "sha256": sha(shard)})
            state["next_task"] = position + 1
            atomic(manifest_path, state)
            print(json.dumps({"next_task": state["next_task"],
                              "root": root_index,
                              "weight": instance.cc_weight}), flush=True)
        if state["next_task"] == len(tasks):
            state["status"] = "completed"
            atomic(manifest_path, state)


if __name__ == "__main__":
    main()
