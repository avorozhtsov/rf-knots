"""Resumable TRAIN-only teacher adequacy pilot on three distinct representations.

Uses the already pinned long-raster actor and independent CPU rollout searches.
This is data collection, not multi-output-head training or AlphaZero search.
"""

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import torch
from bootstrap_cost_value_teacher_v1 import (
    ACTOR,
    ACTOR_SHA,
    ROOTS,
    SOURCE,
    load_actor,
    owned_path,
    read,
)
from build_mcts_training_supervision import atomic, sha
from resource_cost_search import run_batch

from rf_knots import mcts_loop as M

ROOT_PAIRS = ((70, 71), (244, 245), (160, 161))
SEEDS = tuple(20261200 + i for i in range(10))
CONFIG = {
    "insertion_candidates": 4,
    "widening_batch": 4,
    "c_puct": 1.5,
    "temperature": 1.0,
    "simulations": 32,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.stop_after < 0 or args.out.exists() != args.resume:
        raise ValueError("new output requires no --resume; existing output requires --resume")
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    torch.set_num_threads(1)
    source, roots = read(SOURCE), read(ROOTS)
    if source["status"] != "completed" or sha(ROOTS) != source["roots_sha256"]:
        raise ValueError("pinned TRAIN roots differ")
    if sha(ACTOR) != ACTOR_SHA:
        raise ValueError("selected actor checkpoint differs")
    for first, second in ROOT_PAIRS:
        a, b = roots[first], roots[second]
        if (
            a["base_root_index"] != b["base_root_index"]
            or a["instance"]["word"] != b["instance"]["word"]
            or a["instance"]["strands"] != b["instance"]["strands"]
            or {a["instance"]["cc_weight"], b["instance"]["cc_weight"]} != {10, 1000}
        ):
            raise ValueError("not a paired TRAIN representation")
    if len({roots[a]["base_root_index"] for a, _ in ROOT_PAIRS}) != len(ROOT_PAIRS):
        raise ValueError("representation pairs repeat a TRAIN root")
    heldout = {tuple(row) for row in source["heldout_query_groups"]}
    pinned = (
        ROOTS,
        SOURCE,
        ACTOR,
        Path(__file__),
        Path("scripts/bootstrap_cost_value_teacher_v1.py"),
    )
    inputs = {str(p): sha(p) for p in pinned}
    contract = dict(
        schema="rf-knots-budgeted-value-teacher-diverse-v1",
        actor_sha256=ACTOR_SHA,
        root_pairs=ROOT_PAIRS,
        seeds=SEEDS,
        config=CONFIG,
        inputs=inputs,
        role="TRAIN-only teacher adequacy; no value-head update",
    )
    tasks = [(index, seed) for pair in ROOT_PAIRS for index in pair for seed in SEEDS]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.out / "manifest.json"
        if args.resume:
            state = read(path)
            if state["contract"] != json.loads(json.dumps(contract)):
                raise ValueError("resume contract differs")
            if state["next_task"] != len(state["shards"]) or not 0 <= state["next_task"] <= len(
                tasks
            ):
                raise ValueError("resume task coverage differs")
            for position, shard in enumerate(state["shards"]):
                if (
                    shard["file"] != f"attempt-{position:03d}.json"
                    or sha(args.out / shard["file"]) != shard["sha256"]
                ):
                    raise ValueError("committed attempt changed")
        else:
            state = dict(contract=contract, status="preparing", next_task=0, shards=[])
            atomic(path, state)
        if state["next_task"] == len(tasks):
            state["status"] = "completed"
            atomic(path, state)
            return
        net, objective = load_actor()
        state.update(status="collecting", pid=os.getpid())
        atomic(path, state)
        remaining = args.stop_after or len(tasks)
        end = min(len(tasks), state["next_task"] + remaining)
        for position in range(state["next_task"], end):
            if any(sha(Path(p)) != h for p, h in inputs.items()):
                raise ValueError("teacher input/source changed during collection")
            index, seed = tasks[position]
            root = roots[index]
            instance = M.Instance(**{**root["instance"], "word": tuple(root["instance"]["word"])})
            config = {**CONFIG, "seed": seed, "horizon": instance.cap_moves}
            started = time.monotonic()
            rows, counts = run_batch(
                [
                    {
                        "condition": index,
                        "panel_index": root["corpus_index"],
                        "instance": root["instance"],
                    }
                ],
                net,
                objective,
                0,
                config,
                prior="distance",
                cached=True,
            )
            row = rows[0]
            row["episodes_train_owned"] = [
                owned_path(instance, ep, heldout) for ep in row["episodes"]
            ]
            row["eligible_solved_episodes"] = sum(
                ep["continuation"]["stop_reason"] == "solved" and eligible
                for ep, eligible in zip(row["episodes"], row["episodes_train_owned"], strict=True)
            )
            shard = args.out / f"attempt-{position:03d}.json"
            atomic(shard, dict(row=row, counts=counts, seconds=time.monotonic() - started))
            state["shards"].append(dict(file=shard.name, sha256=sha(shard)))
            state["next_task"] = position + 1
            atomic(path, state)
            print(
                json.dumps(
                    dict(
                        next_task=state["next_task"],
                        representation=root["base_root_index"],
                        weight=instance.cc_weight,
                        solved=row["solved"],
                        eligible_solved_episodes=row["eligible_solved_episodes"],
                    )
                ),
                flush=True,
            )
        if state["next_task"] == len(tasks):
            state["status"] = "completed"
            atomic(path, state)


if __name__ == "__main__":
    main()
