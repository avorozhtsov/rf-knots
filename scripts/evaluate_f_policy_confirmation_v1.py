"""Resumable matched A/F search on the frozen protocol272 panel."""

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import torch

from bootstrap_cost_value_teacher_v1 import load_actor
from build_mcts_training_supervision import atomic, sha
from raster_log_training import digest
from resource_cost_search import run_batch
from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness


PILOT = Path("artifacts/mcts-tournament-nine-arm-pilot-v1")
PANEL = Path("artifacts/f-policy-confirmation-panel-v1/panel.json")
PANEL_SHA = "2156fa98b00289501a06d3759c62892ce0ab7fa4bfc688339c390bc58bc7a5f5"
PROTOCOL = Path("research/272-f-policy-independent-dev-confirmation-v1.md")
CONFIG = {
    "arms": ["A", "F"],
    "seeds": [20271119, 20271120],
    "simulations": 128,
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
        raise ValueError("canonical pgx dependency missing")
    sys.path.insert(0, str(pgx_src))
    torch.set_num_threads(1)
    pilot, panel = read(PILOT / "manifest.json"), read(PANEL)
    if (pilot["status"] != "completed" or pilot["arm_index"] != 9
            or sha(PANEL) != PANEL_SHA
            or panel["schema"] != "rf-knots-f-policy-confirmation-panel-v1"
            or len(panel["rows"]) != 32):
        raise ValueError("completed pilot or frozen confirmation panel missing")
    for arm in CONFIG["arms"]:
        entry = pilot["checkpoints"][arm]
        if entry["step"] != 144 or sha(PILOT / entry["file"]) != entry["sha256"]:
            raise ValueError("pilot checkpoint identity differs")
    inputs = {str(path): sha(path) for path in (
        PILOT / "manifest.json", PANEL, Path(__file__),
        Path("scripts/resource_cost_search.py"), PROTOCOL,
    )}
    contract = {"config": CONFIG, "inputs": inputs}
    contract_sha = digest(contract)
    tasks = [(arm, row_index, weight_index, seed)
             for arm in CONFIG["arms"] for row_index in range(32)
             for weight_index in range(2) for seed in CONFIG["seeds"]]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.out / "manifest.json"
        state = read(path) if args.resume else {
            "schema": "rf-knots-f-policy-confirmation-evaluation-v1",
            "contract": contract, "contract_sha256": contract_sha,
            "status": "preparing", "next_task": 0, "shards": [],
        }
        if state["contract"] != contract or state["next_task"] != len(state["shards"]):
            raise ValueError("resume contract or shard coverage differs")
        for position, entry in enumerate(state["shards"]):
            if entry["file"] != f"condition-{position:03d}.json" or sha(args.out / entry["file"]) != entry["sha256"]:
                raise ValueError("saved shard changed")
        atomic(path, state)
        end = min(len(tasks), state["next_task"] + (args.stop_after or len(tasks)))
        loaded_arm = None
        net = objective = None
        for position in range(state["next_task"], end):
            arm, row_index, weight_index, seed = tasks[position]
            if arm != loaded_arm:
                net, objective = load_actor()
                entry = pilot["checkpoints"][arm]
                checkpoint = torch.load(PILOT / entry["file"], map_location="cpu", weights_only=False)
                if (checkpoint["contract_sha256"] != pilot["contract_sha256"]
                        or checkpoint["arm"] != arm or checkpoint["step"] != 144):
                    raise ValueError("saved arm state differs")
                net.load_state_dict(checkpoint["model"])
                objective.load_state_dict(checkpoint["objective"])
                loaded_arm = arm
            row = panel["rows"][row_index]
            instance = row["instances"][weight_index]
            problem = M.Instance(**{**instance, "word": tuple(instance["word"])})
            config = {**CONFIG, "seed": seed, "horizon": problem.cap_moves}
            started_wall, started_cpu = time.monotonic(), time.process_time()
            found, counts = run_batch(
                [{"condition": position, "panel_index": row["generator_index"], "instance": instance}],
                net, objective, 0, config, prior="distance", cached=True,
            )
            outcome = found[0]
            if outcome["witness"] is not None:
                witness = UnknotWitness.from_dict(outcome["witness"])
                witness.verify()
                if (witness.start.word, witness.start.strands) != (problem.word, problem.strands):
                    raise ValueError("witness starts elsewhere")
            payload = {
                "task": position, "arm": arm, "row_index": row_index, "seed": seed,
                "lambda": problem.cc_weight, "result": outcome, "counts": dict(counts),
                "wall_seconds": time.monotonic() - started_wall,
                "cpu_seconds": time.process_time() - started_cpu,
            }
            shard = args.out / f"condition-{position:03d}.json"
            atomic(shard, payload)
            state["shards"].append({"file": shard.name, "sha256": sha(shard)})
            state.update(status="evaluating", next_task=position + 1, current_arm=arm, pid=os.getpid())
            atomic(path, state)
            print(json.dumps({"task": position + 1, "arm": arm, "solved": outcome["solved"],
                              "wall_seconds": payload["wall_seconds"]}, sort_keys=True), flush=True)
        if state["next_task"] == len(tasks):
            state.update(status="completed", current_arm=None, pid=None)
            atomic(path, state)


if __name__ == "__main__":
    main()
