"""Resumable frozen-teacher DEV attempts for protocol265 calibration."""

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import torch

from bootstrap_cost_value_teacher_v1 import ACTOR, ACTOR_SHA, load_actor
from build_mcts_training_supervision import atomic, sha
from raster_log_training import digest
from resource_cost_search import run_batch
from rf_knots import mcts_loop as M
from rf_knots.embedding_replay_corpus import owner
from rf_knots.evidence import UnknotWitness


SHORT = Path("artifacts/mcts-tournament-dev-panel-v1/panel.json")
LONG = Path("artifacts/mcts-tournament-long-dev-panel-v1/panel.json")
PROTOCOL = Path("research/265-root-budget16-value-dev-calibration-v1.md")
CONFIG = {
    "seeds": [20271210, 20271211, 20271212, 20271213],
    "simulations": 16,
    "insertion_candidates": 4,
    "widening_batch": 4,
    "c_puct": 1.5,
    "temperature": 1.0,
    "prior": "distance",
    "cache": True,
}


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists() != args.resume or args.stop_after < 0:
        raise ValueError("new output requires no --resume; existing output requires --resume")
    pgx_src = args.pgx_root.resolve() / "src"
    if not (pgx_src / "pgx_mcts_bench").is_dir():
        raise ValueError("canonical dependency missing")
    sys.path.insert(0, str(pgx_src))
    torch.set_num_threads(1)
    short, long = read(SHORT), read(LONG)
    if (short["schema"] != "rf-knots-tournament-dev-screen-v1"
            or long["schema"] != "rf-knots-tournament-long-dev-panel-v1"
            or len(short["rows"]) != 16 or len(long["rows"]) != 16
            or sha(ACTOR) != ACTOR_SHA):
        raise ValueError("independent DEV panels or frozen actor missing")
    for panel in (short, long):
        for row in panel["rows"]:
            if owner(tuple((row["query_group"][0], row["query_group"][1]))) != "validation":
                raise ValueError("DEV root ownership differs")
    panels = {"short": short, "long": long}
    inputs = {str(p): sha(p) for p in (
        SHORT, LONG, ACTOR, PROTOCOL, Path(__file__),
        Path("scripts/resource_cost_search.py"),
    )}
    contract = {"schema": "rf-knots-root-budget16-dev-teacher-v1",
                "config": CONFIG, "inputs": inputs, "actor_sha256": ACTOR_SHA}
    contract_sha = digest(contract)
    tasks = [(band, row_index, weight_index, seed)
             for band in ("short", "long") for row_index in range(16)
             for weight_index in range(2) for seed in CONFIG["seeds"]]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.out / "manifest.json"
        state = read(path) if args.resume else {
            "schema": contract["schema"], "contract": contract,
            "contract_sha256": contract_sha, "status": "preparing",
            "next_task": 0, "shards": [],
        }
        if state["contract"] != contract or state["next_task"] != len(state["shards"]):
            raise ValueError("DEV teacher resume contract or coverage differs")
        for position, entry in enumerate(state["shards"]):
            if entry["file"] != f"attempt-{position:03d}.json" or sha(args.out / entry["file"]) != entry["sha256"]:
                raise ValueError("DEV teacher shard changed")
        atomic(path, state)
        net, objective = load_actor()
        end = min(len(tasks), state["next_task"] + (args.stop_after or len(tasks)))
        for position in range(state["next_task"], end):
            band, row_index, weight_index, seed = tasks[position]
            row = panels[band]["rows"][row_index]
            instance = row["instances"][weight_index]
            problem = M.Instance(**{**instance, "word": tuple(instance["word"])})
            config = {**CONFIG, "seed": seed, "horizon": problem.cap_moves}
            started_wall, started_cpu = time.monotonic(), time.process_time()
            found, counts = run_batch(
                [{"condition": position, "panel_index": row_index, "instance": instance}],
                net, objective, 0, config, prior="distance", cached=True,
            )
            outcome = found[0]
            if outcome["witness"] is not None:
                witness = UnknotWitness.from_dict(outcome["witness"])
                witness.verify()
                if ((witness.start.word, witness.start.strands) != (problem.word, problem.strands)
                        or witness.crossing_changes > problem.cap_crossing_changes
                        or witness.moves > problem.cap_moves):
                    raise ValueError("DEV teacher witness start/resource differs")
            payload = {
                "task": position, "band": band, "row_index": row_index,
                "lambda": problem.cc_weight, "seed": seed,
                "result": outcome, "counts": dict(counts),
                "wall_seconds": time.monotonic() - started_wall,
                "cpu_seconds": time.process_time() - started_cpu,
            }
            shard = args.out / f"attempt-{position:03d}.json"
            atomic(shard, payload)
            state["shards"].append({"file": shard.name, "sha256": sha(shard)})
            state.update(status="collecting", next_task=position + 1, pid=os.getpid())
            atomic(path, state)
            print(json.dumps({"task": position + 1, "band": band,
                              "solved": outcome["solved"]}, sort_keys=True), flush=True)
        if state["next_task"] == len(tasks):
            state.update(status="completed", pid=None)
            atomic(path, state)


if __name__ == "__main__":
    main()
