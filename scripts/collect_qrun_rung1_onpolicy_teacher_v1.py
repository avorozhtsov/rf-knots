"""Resumable TRAIN-only on-policy teacher from curriculum1024."""

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import torch

import build_f_curriculum_roster_v1 as roster_source
from bootstrap_cost_value_teacher_v1 import load_actor, owned_path
from build_mcts_training_supervision import atomic, digest, sha
from resource_cost_search import run_batch
from rf_knots import mcts_loop as M

ROSTER = Path("artifacts/f-curriculum-roster-v3/roster.json")
ROSTER_SHA = "268f1c4d2e8731bcfd4cc743bed767c9864fb74b2a2022ed679fbe36af134506"
ACTOR = Path("artifacts/qrun-like-policy-curriculum-rung1-v1-envfix/step-1024.pt")
ACTOR_SHA = "03506e0c572864dc207c0b4a330648dc86bef18b18af2420955a69697b9fe9e8"
TRAIN_MANIFEST = Path("artifacts/qrun-like-policy-curriculum-rung1-v1-envfix/manifest.json")
TRAIN_MANIFEST_SHA = "1ed9bad134dff67304f1d601965563cdac1293716cb51c8e5ca491da1937242d"
PROTOCOL = Path("research/289-qrun-rung1-onpolicy-teacher-v1.md")
SEED = 20271300
CONFIG = {"seed": SEED, "simulations": 64, "insertion_candidates": 4,
          "widening_batch": 4, "c_puct": 1.5, "temperature": 1.0,
          "prior": "distance", "cache": True}


def read(path):
    return json.loads(path.read_text())


def roots(roster):
    ordered = [row for stage in roster["stages"] for row in stage]
    if (len(ordered) != 1000
            or len({tuple(row["query_group"]) for row in ordered}) != 1000
            or [sum(row["stage"] == stage for row in ordered)
                for stage in range(4)] != [140, 286, 287, 287]):
        raise ValueError("frozen 1000-root curriculum roster differs")
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists() != args.resume or args.stop_after < 0:
        raise ValueError("new output without --resume; existing output with --resume")
    pgx_src = args.pgx_root.resolve() / "src"
    if not (pgx_src / "pgx_mcts_bench").is_dir():
        raise ValueError("canonical pgx-mcts-bench missing")
    sys.path.insert(0, str(pgx_src))
    torch.set_num_threads(1)
    pins = (ROSTER, ACTOR, TRAIN_MANIFEST, PROTOCOL, Path(__file__),
            Path("scripts/bootstrap_cost_value_teacher_v1.py"),
            Path("scripts/resource_cost_search.py"))
    inputs = {str(path): sha(path) for path in pins}
    if (inputs[str(ROSTER)] != ROSTER_SHA or inputs[str(ACTOR)] != ACTOR_SHA
            or inputs[str(TRAIN_MANIFEST)] != TRAIN_MANIFEST_SHA):
        raise ValueError("roster, actor or training manifest changed")
    roster = read(ROSTER)
    selected = roots(roster)
    tasks = [(root_index, weight_index) for root_index in range(1000)
             for weight_index in range(2)]
    contract = {"schema": "rf-knots-qrun-rung1-onpolicy-teacher-v1",
                "root_generator_indices": [row["generator_index"] for row in selected],
                "config": CONFIG, "inputs": inputs,
                "role": "TRAIN search teacher, not optimizer training"}
    contract_sha = digest(contract)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out / "manifest.json"
        state = read(manifest_path) if args.resume else {
            "contract": contract, "contract_sha256": contract_sha,
            "status": "preparing", "next_task": 0, "shards": []}
        if (state["contract"] != json.loads(json.dumps(contract))
                or state["contract_sha256"] != contract_sha
                or state["next_task"] != len(state["shards"])):
            raise ValueError("resume contract or coverage differs")
        for position, entry in enumerate(state["shards"]):
            if (entry["file"] != f"condition-{position:04d}.json"
                    or sha(args.out / entry["file"]) != entry["sha256"]):
                raise ValueError("committed teacher shard changed")
        if state["next_task"] == len(tasks):
            state.update(status="completed", pid=None)
            atomic(manifest_path, state)
            return
        excluded = roster_source.excluded_groups()
        net, objective = load_actor()
        checkpoint = torch.load(ACTOR, map_location="cpu", weights_only=False)
        if checkpoint["step"] != 1024:
            raise ValueError("curriculum endpoint differs")
        net.load_state_dict(checkpoint["model"], strict=True)
        objective.load_state_dict(checkpoint["objective"], strict=True)
        state.update(status="collecting", pid=os.getpid())
        atomic(manifest_path, state)
        end = min(len(tasks), state["next_task"] + (args.stop_after or len(tasks)))
        for position in range(state["next_task"], end):
            if any(sha(Path(path)) != expected for path, expected in inputs.items()):
                raise ValueError("pinned teacher input changed")
            root_index, weight_index = tasks[position]
            root = selected[root_index]
            instance_dict = root["instances"][weight_index]
            instance = M.Instance(**{**instance_dict,
                                      "word": tuple(instance_dict["word"])})
            config = {**CONFIG, "horizon": instance.cap_moves}
            started_wall, started_cpu = time.monotonic(), time.process_time()
            found, counts = run_batch(
                [{"condition": root_index, "panel_index": root["generator_index"],
                  "instance": instance_dict}], net, objective, 0, config,
                prior="distance", cached=True)
            result = found[0]
            result["episodes_train_owned"] = [owned_path(instance, episode, excluded)
                                               for episode in result["episodes"]]
            result["eligible_solved_episodes"] = sum(
                episode["continuation"]["stop_reason"] == "solved" and eligible
                for episode, eligible in zip(result["episodes"],
                                              result["episodes_train_owned"], strict=True))
            payload = {"task": position, "root_index": root_index,
                       "stage": root["stage"],
                       "generator_index": root["generator_index"],
                       "weight": instance.cc_weight, "seed": SEED,
                       "result": result, "counts": dict(counts),
                       "cpu_seconds": time.process_time() - started_cpu,
                       "wall_seconds": time.monotonic() - started_wall}
            shard_path = args.out / f"condition-{position:04d}.json"
            atomic(shard_path, payload)
            state["shards"].append({"file": shard_path.name,
                                    "sha256": sha(shard_path)})
            state.update(status="collecting", next_task=position + 1,
                         current_stage=root["stage"], pid=os.getpid())
            atomic(manifest_path, state)
            print(json.dumps({"next_task": position + 1, "stage": root["stage"],
                              "weight": instance.cc_weight,
                              "solved": result["solved"],
                              "eligible_solved_episodes": result["eligible_solved_episodes"]}),
                  flush=True)
        if state["next_task"] == len(tasks):
            state.update(status="completed", current_stage=None, pid=None)
            atomic(manifest_path, state)


if __name__ == "__main__":
    main()
