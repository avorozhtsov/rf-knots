"""Unselected 128-representation TRAIN teacher collection with resumable shards.

This collects CPU search outcomes; it does not train a head or implement
AlphaZero value-guided MCTS. All representations are retained in the contract.
"""

import argparse
import fcntl
import json
import os
import random
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

SEED = 20261002
SEARCH_SEEDS = tuple(20270000 + i for i in range(10))
BUDGETS = (16, 32, 128, 512)
CONFIG = {
    "insertion_candidates": 4,
    "widening_batch": 4,
    "c_puct": 1.5,
    "temperature": 1.0,
}


def length_band(n):
    return 0 if n <= 30 else 1 if n <= 60 else 2 if n <= 120 else 3


def root_order(roots):
    """Fixed breadth-first length strata; no search outcome is inspected."""
    by_length = {i: [] for i in range(4)}
    for index in range(0, len(roots), 2):
        first, second = roots[index : index + 2]
        if (
            first["base_root_index"] != second["base_root_index"]
            or first["instance"]["word"] != second["instance"]["word"]
            or first["instance"]["strands"] != second["instance"]["strands"]
            or {first["instance"]["cc_weight"], second["instance"]["cc_weight"]}
            != {10, 1000}
        ):
            raise ValueError(f"invalid lambda pair at index {index}")
        by_length[length_band(len(first["instance"]["word"]))].append(index)
    rng = random.Random(SEED)
    for group in by_length.values():
        rng.shuffle(group)
    order = []
    while any(by_length.values()):
        for group in by_length.values():
            if group:
                order.append(group.pop())
    if len(order) != len(roots) // 2:
        raise ValueError("root order lost a representation")
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.stop_after < 0 or args.out.exists() != args.resume:
        raise ValueError("new output requires no --resume; existing output requires --resume")
    if not (args.pgx_root.resolve() / "src" / "pgx_mcts_bench").exists():
        raise ValueError("canonical pgx-mcts-bench src directory not found")
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    torch.set_num_threads(1)
    source, roots = read(SOURCE), read(ROOTS)
    if source["status"] != "completed" or sha(ROOTS) != source["roots_sha256"]:
        raise ValueError("pinned TRAIN roster changed")
    if len(roots) != 256 or sha(ACTOR) != ACTOR_SHA:
        raise ValueError("expected 128 pairs and selected immutable actor")
    order = root_order(roots)
    heldout = {tuple(row) for row in source["heldout_query_groups"]}
    pins = (
        ROOTS,
        SOURCE,
        ACTOR,
        Path(__file__),
        Path("scripts/bootstrap_cost_value_teacher_v1.py"),
        Path("scripts/resource_cost_search.py"),
    )
    inputs = {str(p): sha(p) for p in pins}
    contract = {
        "schema": "rf-knots-budgeted-value-teacher-unselected-v1",
        "actor_sha256": ACTOR_SHA,
        "root_count": len(order),
        "root_order": order,
        "search_seeds": SEARCH_SEEDS,
        "budgets": BUDGETS,
        "config": CONFIG,
        "inputs": inputs,
        "role": "TRAIN-only teacher collection; no head training or selection",
    }
    # First visit every representation at the cheapest budget; interleave
    # length strata and lambda before spending more on one representation.
    tasks = [
        (index, budget, seed)
        for budget in BUDGETS
        for seed in SEARCH_SEEDS
        for first in order
        for index in (first, first + 1)
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.out / "manifest.json"
        if args.resume:
            state = read(path)
            if state["contract"] != json.loads(json.dumps(contract)):
                raise ValueError("resume contract differs")
            if state["next_task"] != len(state["shards"]) or not 0 <= state[
                "next_task"
            ] <= len(tasks):
                raise ValueError("resume task coverage differs")
            for position, shard in enumerate(state["shards"]):
                if (
                    shard["file"] != f"attempt-{position:05d}.json"
                    or sha(args.out / shard["file"]) != shard["sha256"]
                ):
                    raise ValueError("committed shard changed")
        else:
            state = {
                "contract": contract,
                "status": "preparing",
                "next_task": 0,
                "shards": [],
            }
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
            if any(sha(Path(p)) != digest for p, digest in inputs.items()):
                raise ValueError("teacher input/source changed")
            index, budget, seed = tasks[position]
            root = roots[index]
            instance = M.Instance(
                **{**root["instance"], "word": tuple(root["instance"]["word"])}
            )
            config = {**CONFIG, "seed": seed, "horizon": instance.cap_moves, "simulations": budget}
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
            shard = args.out / f"attempt-{position:05d}.json"
            atomic(
                shard,
                {
                    "root_index": index,
                    "base_root_index": root["base_root_index"],
                    "budget": budget,
                    "seed": seed,
                    "row": row,
                    "counts": counts,
                    "seconds": time.monotonic() - started,
                },
            )
            state["shards"].append({"file": shard.name, "sha256": sha(shard)})
            state["next_task"] = position + 1
            atomic(path, state)
            print(
                json.dumps(
                    {
                        "next_task": state["next_task"],
                        "representation": root["base_root_index"],
                        "weight": instance.cc_weight,
                        "budget": budget,
                        "solved": row["solved"],
                        "eligible_solved_episodes": row["eligible_solved_episodes"],
                    }
                ),
                flush=True,
            )
        if state["next_task"] == len(tasks):
            state["status"] = "completed"
            atomic(path, state)


if __name__ == "__main__":
    main()
