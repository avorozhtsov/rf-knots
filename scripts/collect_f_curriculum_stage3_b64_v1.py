"""Resumable TRAIN-only stage-3 b64 search; does not train a network."""

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import torch

import build_f_curriculum_roster_v1 as roster_source
import collect_f_curriculum_yield_pilot_v1 as prior
from bootstrap_cost_value_teacher_v1 import ACTOR, ACTOR_SHA, load_actor, owned_path
from build_mcts_training_supervision import atomic, sha
from resource_cost_search import run_batch
from rf_knots import mcts_loop as M

ROSTER = prior.ROSTER
ROSTER_SHA = prior.ROSTER_SHA
SEEDS = prior.SEEDS
CONFIG = {**prior.CONFIG, "simulations": 64}
PILOT_MANIFEST = Path("artifacts/f-curriculum-s10-yield-pilot-v1/manifest.json")
PILOT_MANIFEST_SHA = "1f17e91bef18b0dccc719ced35a4718eff2d8f4412220fd885ec4ef03bdcf24f"


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
    if (sha(ROSTER) != ROSTER_SHA or sha(ACTOR) != ACTOR_SHA
            or sha(PILOT_MANIFEST) != PILOT_MANIFEST_SHA):
        raise ValueError("frozen data/actor/pilot identity changed")
    roots = prior.selected_roots(prior.read(ROSTER))[24:]
    if len(roots) != 8 or any(root["stage"] != 3 for root in roots):
        raise ValueError("stage-3 sample differs")
    tasks = [(root_index, weight_index, seed) for root_index in range(8)
             for weight_index in range(2) for seed in SEEDS]
    pins = (ROSTER, ACTOR, PILOT_MANIFEST, Path(__file__),
            Path("scripts/collect_f_curriculum_yield_pilot_v1.py"),
            Path("scripts/bootstrap_cost_value_teacher_v1.py"),
            Path("scripts/resource_cost_search.py"),
            Path("research/278-f-curriculum-long-b64-escalation-v1.md"))
    inputs = {str(path): sha(path) for path in pins}
    contract = {"schema": "rf-knots-f-curriculum-stage3-b64-v1",
                "root_generator_indices": [r["generator_index"] for r in roots],
                "search_seeds": SEEDS, "config": CONFIG, "inputs": inputs,
                "role": "TRAIN teacher search, not neural training"}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = args.out / "manifest.json"
        if args.resume:
            state = prior.read(manifest)
            if state["contract"] != json.loads(json.dumps(contract)) or state["next_task"] != len(state["shards"]):
                raise ValueError("resume contract/shard count differs")
            for position, shard in enumerate(state["shards"]):
                if shard["file"] != f"attempt-{position:04d}.json" or sha(args.out / shard["file"]) != shard["sha256"]:
                    raise ValueError("committed shard changed")
        else:
            state = {"contract": contract, "status": "preparing", "next_task": 0, "shards": []}
            atomic(manifest, state)
        if state["next_task"] == len(tasks):
            state["status"] = "completed"
            atomic(manifest, state)
            return
        excluded = roster_source.excluded_groups()
        net, objective = load_actor()
        state.update(status="collecting", pid=os.getpid())
        atomic(manifest, state)
        end = min(len(tasks), state["next_task"] + (args.stop_after or len(tasks)))
        for position in range(state["next_task"], end):
            if any(sha(Path(path)) != digest for path, digest in inputs.items()):
                raise ValueError("pinned input changed")
            root_index, weight_index, seed = tasks[position]
            root = roots[root_index]
            instance_dict = root["instances"][weight_index]
            instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
            config = {**CONFIG, "seed": seed, "horizon": instance.cap_moves}
            started_wall, started_cpu = time.monotonic(), time.process_time()
            rows, counts = run_batch(
                [{"condition": root_index, "panel_index": root["generator_index"],
                  "instance": instance_dict}],
                net, objective, 0, config, prior="distance", cached=True)
            row = rows[0]
            row["episodes_train_owned"] = [owned_path(instance, ep, excluded)
                                           for ep in row["episodes"]]
            row["eligible_solved_episodes"] = sum(
                ep["continuation"]["stop_reason"] == "solved" and eligible
                for ep, eligible in zip(row["episodes"], row["episodes_train_owned"], strict=True))
            shard_path = args.out / f"attempt-{position:04d}.json"
            atomic(shard_path, {"root_index": root_index, "generator_index": root["generator_index"],
                                "weight": instance.cc_weight, "seed": seed,
                                "row": row, "counts": counts,
                                "cpu_seconds": time.process_time() - started_cpu,
                                "wall_seconds": time.monotonic() - started_wall})
            state["shards"].append({"file": shard_path.name, "sha256": sha(shard_path)})
            state["next_task"] = position + 1
            atomic(manifest, state)
            print(json.dumps({"next_task": state["next_task"], "root": root_index,
                              "weight": instance.cc_weight,
                              "eligible_solved_episodes": row["eligible_solved_episodes"]}), flush=True)
        if state["next_task"] == len(tasks):
            state["status"] = "completed"
            atomic(manifest, state)


if __name__ == "__main__":
    main()
