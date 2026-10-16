"""Resumable new-root R2 MCTS yield pilot; search collection, not training."""

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import torch

import build_f_incremental_roster_v2 as roster_source
from bootstrap_cost_value_teacher_v1 import load_actor, owned_path
from build_mcts_training_supervision import atomic, digest, sha
from resource_cost_search import run_batch
from rf_knots import mcts_loop as M

ROSTER = Path("artifacts/f-incremental-roster-v2/roster.json")
ROSTER_SHA = "93bbaa282ec79bd017b5c7adfbd4f55041adddebfe5505650d258d1f17ce58f8"
ACTOR = Path("artifacts/joint-f-r2-r4-smoothed-v4-R2/step-114561.pt")
ACTOR_SHA = "1ffe7346e0725fcb98125ebf03ab1faaaf7048985d69eeb00300e2828cca6946"
PROTOCOL = Path("research/318-incremental-r2-mcts-f-yield-pilot-v1.md")
SEEDS = (20271317, 20271318)
CONFIG = {"insertion_candidates": 4, "widening_batch": 4, "c_puct": 1.5,
          "temperature": 1.0, "simulations": 64}


def read(path):
    return json.loads(path.read_text())


def selected_roots(roster):
    chosen = list(roster["stages"][0][:8])
    for rows in roster["stages"][1:]:
        chosen += [row for row in rows if row["reference_cc"] == 1][:4]
        chosen += [row for row in rows if row["reference_cc"] == 2][:4]
    if len(chosen) != 32 or len({tuple(r["query_group"]) for r in chosen}) != 32:
        raise ValueError("balanced root selection failed")
    return chosen


def load_policy():
    net, objective = load_actor()
    saved = torch.load(ACTOR, map_location="cpu", weights_only=False)
    if saved["step"] != 114561:
        raise ValueError("selected R2 checkpoint step changed")
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
    if sha(ROSTER) != ROSTER_SHA or sha(ACTOR) != ACTOR_SHA:
        raise ValueError("roster or selected R2 actor changed")
    roots = selected_roots(read(ROSTER))
    tasks = [(r, weight, seed) for r in range(32)
             for weight in range(2) for seed in SEEDS]
    pins = (ROSTER, ACTOR, PROTOCOL, Path(__file__),
            Path("scripts/build_f_incremental_roster_v2.py"),
            Path("scripts/bootstrap_cost_value_teacher_v1.py"),
            Path("scripts/resource_cost_search.py"))
    inputs = {str(path): sha(path) for path in pins}
    contract = {"schema": "rf-knots-incremental-R2-F-yield-pilot-v1",
                "root_generator_indices": [r["generator_index"] for r in roots],
                "search_seeds": list(SEEDS), "config": CONFIG, "inputs": inputs,
                "role": "TRAIN-only search outcomes, not optimizer training"}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out / "manifest.json"
        if args.resume:
            state = read(manifest_path)
            if (state["contract"] != contract or
                    state["contract_sha256"] != digest(contract) or
                    state["next_task"] != len(state["shards"])):
                raise ValueError("resume contract/shards differ")
            for position, entry in enumerate(state["shards"]):
                if (entry["file"] != f"attempt-{position:04d}.json" or
                        sha(args.out / entry["file"]) != entry["sha256"]):
                    raise ValueError("committed shard SHA differs")
        else:
            state = {"contract": contract, "contract_sha256": digest(contract),
                     "status": "preparing", "next_task": 0, "shards": []}
            atomic(manifest_path, state)
        if state["next_task"] == len(tasks):
            state["status"] = "completed"; atomic(manifest_path, state); return
        excluded = roster_source.old.excluded_groups()
        for panel in roster_source.PANELS:
            excluded.update(tuple(row["query_group"])
                            for row in read(panel)["rows"])
        net, objective = load_policy()
        state.update(status="collecting", pid=os.getpid())
        atomic(manifest_path, state)
        end = min(len(tasks), state["next_task"] + (args.stop_after or len(tasks)))
        for position in range(state["next_task"], end):
            if any(sha(Path(path)) != expected for path, expected in inputs.items()):
                raise ValueError("pinned input changed")
            root_index, weight_index, seed = tasks[position]
            root = roots[root_index]
            instance_dict = root["instances"][weight_index]
            instance = M.Instance(**{**instance_dict,
                                     "word": tuple(instance_dict["word"])})
            config = {**CONFIG, "seed": seed, "horizon": instance.cap_moves}
            started_wall, started_cpu = time.monotonic(), time.process_time()
            rows, counts = run_batch(
                [{"condition": root_index, "panel_index": root["generator_index"],
                  "instance": instance_dict}], net, objective, 0, config,
                prior="distance", cached=True)
            row = rows[0]
            row["episodes_train_owned"] = [owned_path(instance, ep, excluded)
                                           for ep in row["episodes"]]
            row["eligible_solved_episodes"] = sum(
                ep["continuation"]["stop_reason"] == "solved" and eligible
                for ep, eligible in zip(row["episodes"],
                                        row["episodes_train_owned"], strict=True))
            shard = args.out / f"attempt-{position:04d}.json"
            atomic(shard, {"root_index": root_index,
                           "generator_index": root["generator_index"],
                           "weight": instance.cc_weight, "seed": seed,
                           "row": row, "counts": counts,
                           "cpu_seconds": time.process_time() - started_cpu,
                           "wall_seconds": time.monotonic() - started_wall})
            state["shards"].append({"file": shard.name, "sha256": sha(shard)})
            state["next_task"] = position + 1
            atomic(manifest_path, state)
            print(json.dumps({"next_task": state["next_task"],
                              "root": root_index, "weight": instance.cc_weight,
                              "eligible_solved_episodes": row["eligible_solved_episodes"]}),
                  flush=True)
        if state["next_task"] == len(tasks):
            state["status"] = "completed"; atomic(manifest_path, state)


if __name__ == "__main__":
    main()
