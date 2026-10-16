"""Resumable TRAIN-only hard-stage budget-yield pilot (protocol295)."""

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
from rf_knots.evidence import UnknotWitness

ROSTER = Path("artifacts/f-curriculum-roster-v3/roster.json")
ROSTER_SHA = "268f1c4d2e8731bcfd4cc743bed767c9864fb74b2a2022ed679fbe36af134506"
ACTOR = Path("artifacts/qrun-like-policy-curriculum-rung1-v1-envfix/step-1024.pt")
ACTOR_SHA = "03506e0c572864dc207c0b4a330648dc86bef18b18af2420955a69697b9fe9e8"
PROTOCOL = Path("research/295-f-hard-stage-budget-yield-pilot-v1.md")
ROOT_INDICES = (4, 8, 10, 21, 0, 1, 2, 3)
SEEDS = (20271351, 20271352)
BUDGETS = (64, 128)
BASE_CONFIG = {"insertion_candidates": 4, "widening_batch": 4,
               "c_puct": 1.5, "temperature": 1.0,
               "prior": "distance", "cache": True}


def read(path):
    return json.loads(path.read_text())


def tasks():
    return [[root, objective, seed, budget]
            for root in ROOT_INDICES for objective in range(2)
            for seed in SEEDS for budget in BUDGETS]


def selected_roots(roster):
    stage = roster["stages"][3]
    if len(stage) != 287 or len({tuple(row["query_group"]) for row in stage}) != 287:
        raise ValueError("stage-3 roster differs")
    for cap, indices in ((1, ROOT_INDICES[:4]), (2, ROOT_INDICES[4:])):
        expected = tuple(i for i, row in enumerate(stage)
                         if UnknotWitness.from_dict(row["witness"]).crossing_changes == cap)[:4]
        if tuple(indices) != expected:
            raise ValueError("score-blind CC-stratified selection differs")
    return stage


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
    pins = (ROSTER, ACTOR, PROTOCOL, Path(__file__),
            Path("scripts/bootstrap_cost_value_teacher_v1.py"),
            Path("scripts/resource_cost_search.py"))
    inputs = {str(path): sha(path) for path in pins}
    if inputs[str(ROSTER)] != ROSTER_SHA or inputs[str(ACTOR)] != ACTOR_SHA:
        raise ValueError("roster or actor changed")
    roots = selected_roots(read(ROSTER))
    schedule = tasks()
    contract = {"schema": "rf-knots-f-hard-stage-budget-yield-v1",
                "root_indices": list(ROOT_INDICES), "seeds": list(SEEDS),
                "budgets": list(BUDGETS), "base_config": BASE_CONFIG,
                "tasks": schedule, "inputs": inputs,
                "role": "TRAIN search collection, not optimizer training"}
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
                raise ValueError("committed pilot shard changed")
        if state["next_task"] == len(schedule):
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
        end = min(len(schedule), state["next_task"] + (args.stop_after or len(schedule)))
        for position in range(state["next_task"], end):
            if any(sha(Path(path)) != expected for path, expected in inputs.items()):
                raise ValueError("pinned pilot input changed")
            root_index, weight_index, seed, budget = schedule[position]
            root = roots[root_index]
            instance_dict = root["instances"][weight_index]
            instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
            config = {**BASE_CONFIG, "seed": seed, "simulations": budget,
                      "horizon": instance.cap_moves}
            started_wall, started_cpu = time.monotonic(), time.process_time()
            found, counts = run_batch(
                [{"condition": root_index, "panel_index": root["generator_index"],
                  "instance": instance_dict}], net, objective, position, config,
                prior="distance", cached=True)
            result = found[0]
            result["episodes_train_owned"] = [owned_path(instance, episode, excluded)
                                               for episode in result["episodes"]]
            result["eligible_solved_episodes"] = sum(
                episode["continuation"]["stop_reason"] == "solved" and eligible
                for episode, eligible in zip(result["episodes"],
                                             result["episodes_train_owned"], strict=True))
            payload = {"task": position, "root_index": root_index,
                       "stage": 3, "generator_index": root["generator_index"],
                       "weight": instance.cc_weight, "seed": seed,
                       "budget": budget, "result": result, "counts": dict(counts),
                       "cpu_seconds": time.process_time() - started_cpu,
                       "wall_seconds": time.monotonic() - started_wall}
            shard_path = args.out / f"condition-{position:04d}.json"
            atomic(shard_path, payload)
            state["shards"].append({"file": shard_path.name,
                                    "sha256": sha(shard_path)})
            state.update(status="collecting", next_task=position + 1,
                         pid=os.getpid())
            atomic(manifest_path, state)
            print(json.dumps({"next_task": position + 1, "root_index": root_index,
                              "weight": instance.cc_weight, "seed": seed,
                              "budget": budget, "solved": result["solved"],
                              "eligible_solved_episodes": result["eligible_solved_episodes"]}),
                  flush=True)
        if state["next_task"] == len(schedule):
            state.update(status="completed", pid=None)
            atomic(manifest_path, state)


if __name__ == "__main__":
    main()
