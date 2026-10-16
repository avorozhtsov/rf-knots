"""Matched, resumable fixed-64-simulation DEV comparison of three policy states."""

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
from rf_knots.evidence import UnknotWitness

PANEL = Path("artifacts/qrun-bootstrap-dev-panel-v3/panel.json")
PARENT = Path("artifacts/mcts-tournament-nine-arm-pilot-v1/F-step-0144.pt")
SHORT = Path("artifacts/qrun-like-policy-bootstrap-rung0-v1-envfix/step-0256.pt")
MIXED = Path("artifacts/qrun-like-policy-bootstrap-rung0-v2/step-0256.pt")
CHECKPOINTS = {"F144_parent": PARENT, "short_only_256": SHORT,
               "mixed_256": MIXED}
CONFIG = {"arms": list(CHECKPOINTS), "seed": 20271204,
          "simulations": 64, "insertion_candidates": 4,
          "widening_batch": 4, "c_puct": 1.5, "temperature": 1.0,
          "prior": "distance", "cache": True}


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
        raise ValueError("new output requires no --resume; existing requires --resume")
    pgx_src = args.pgx_root.resolve() / "src"
    if not (pgx_src / "pgx_mcts_bench").is_dir():
        raise ValueError("canonical pgx-mcts-bench missing")
    sys.path.insert(0, str(pgx_src))
    torch.set_num_threads(1)
    panel = read(PANEL)
    if (panel["schema"] != "rf-knots-qrun-bootstrap-dev-panel-v3"
            or len(panel["rows"]) != 32
            or [row["band"] for row in panel["rows"]] != [0] * 16 + [1] * 16):
        raise ValueError("frozen short/long DEV panel differs")
    parents = [Path("artifacts/qrun-like-policy-bootstrap-rung0-v1-envfix/manifest.json"),
               Path("artifacts/qrun-like-policy-bootstrap-rung0-v2/manifest.json")]
    for manifest_path, checkpoint_path in zip(parents, (SHORT, MIXED), strict=True):
        train = read(manifest_path)
        if (train["status"] != "completed" or train["optimizer_steps"] != 256
                or train["checkpoints"][-1]["file"] != checkpoint_path.name
                or train["checkpoints"][-1]["sha256"] != sha(checkpoint_path)):
            raise ValueError("completed trained checkpoint differs")
    pins = (PANEL, *CHECKPOINTS.values(), *parents, Path(__file__),
            Path("research/281-qrun-bootstrap-independent-dev-panel-v1.md"),
            Path("research/283-qrun-bootstrap-dev-panel-distinct-prefix-v3.md"),
            Path("scripts/resource_cost_search.py"))
    inputs = {str(path): sha(path) for path in pins}
    contract = {"schema": "rf-knots-qrun-bootstrap-matched-dev-v1",
                "config": CONFIG, "inputs": inputs}
    contract_sha = digest(contract)
    tasks = [(arm, row_index, weight_index) for arm in CONFIG["arms"]
             for row_index in range(32) for weight_index in range(2)]
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
            raise ValueError("resume contract/coverage differs")
        for position, entry in enumerate(state["shards"]):
            if entry["file"] != f"condition-{position:03d}.json" or sha(args.out / entry["file"]) != entry["sha256"]:
                raise ValueError("committed evaluation shard changed")
        atomic(manifest_path, state)
        end = min(len(tasks), state["next_task"] + (args.stop_after or len(tasks)))
        active = None
        net = objective = None
        for position in range(state["next_task"], end):
            arm, row_index, weight_index = tasks[position]
            if arm != active:
                net, objective = load_actor()
                checkpoint = torch.load(CHECKPOINTS[arm], map_location="cpu", weights_only=False)
                if arm == "F144_parent":
                    if checkpoint["arm"] != "F" or checkpoint["step"] != 144:
                        raise ValueError("F144 identity differs")
                elif checkpoint["step"] != 256:
                    raise ValueError("trained endpoint differs")
                net.load_state_dict(checkpoint["model"], strict=True)
                objective.load_state_dict(checkpoint["objective"], strict=True)
                active = arm
            row = panel["rows"][row_index]
            instance_dict = row["instances"][weight_index]
            instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
            config = {**CONFIG, "seed": CONFIG["seed"], "horizon": instance.cap_moves}
            started_wall, started_cpu = time.monotonic(), time.process_time()
            found, counts = run_batch(
                [{"condition": position, "panel_index": row["generator_index"],
                  "instance": instance_dict}],
                net, objective, 0, config, prior="distance", cached=True)
            result = found[0]
            if result["witness"] is not None:
                witness = UnknotWitness.from_dict(result["witness"])
                witness.verify()
                if (witness.start.word, witness.start.strands) != (instance.word, instance.strands):
                    raise ValueError("solution starts at another root")
            payload = {"task": position, "arm": arm, "row_index": row_index,
                       "lambda": instance.cc_weight, "seed": CONFIG["seed"],
                       "result": result, "counts": dict(counts),
                       "cpu_seconds": time.process_time() - started_cpu,
                       "wall_seconds": time.monotonic() - started_wall}
            shard_path = args.out / f"condition-{position:03d}.json"
            atomic(shard_path, payload)
            state["shards"].append({"file": shard_path.name, "sha256": sha(shard_path)})
            state.update(status="evaluating", next_task=position + 1,
                         current_arm=arm, pid=os.getpid())
            atomic(manifest_path, state)
            print(json.dumps({"next_task": position + 1, "arm": arm,
                              "solved": result["solved"]}), flush=True)
        if state["next_task"] == len(tasks):
            state.update(status="completed", current_arm=None, pid=None)
            atomic(manifest_path, state)


if __name__ == "__main__":
    main()
