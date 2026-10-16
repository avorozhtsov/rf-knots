"""Matched R2/R4 search on the fresh bounded-replication DEV panel."""

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


PANEL = Path("artifacts/f-r2-r4-replication-panel-v1/panel.json")
PANEL_SHA = "dd7c2d517514b6efbfd77e7f578fcf5b9088be286271e7cecd0f46045beda2de"
TRAINING_SOURCE_SHA = "07895f0d7dea6815dff8bccbd4f65e00454d9260ba84be76030a860b8e035f82"
SEEDS = (20271010, 20271011)
CONFIG = {"arms": ("R2", "R4"), "seeds": SEEDS,
          "simulations": 128, "insertion_candidates": 4,
          "widening_batch": 4, "c_puct": 1.5, "temperature": 1.0,
          "prior": "distance", "cache": True}


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.step != 114561 or args.stop_after < 0 or args.out.exists() != args.resume:
        raise ValueError("invalid step or new/resume output mode")
    pgx_src = args.pgx_root.resolve() / "src"
    if not (pgx_src / "pgx_mcts_bench").is_dir():
        raise ValueError("canonical pgx-mcts-bench missing")
    sys.path.insert(0, str(pgx_src))
    torch.set_num_threads(1)
    if sha(PANEL) != PANEL_SHA:
        raise ValueError("replication DEV panel SHA differs")
    panel = read(PANEL)
    if (panel["schema"] != "rf-knots-f-r2-r4-replication-panel-v1"
            or len(panel["rows"]) != 32
            or [row["band"] for row in panel["rows"]] != [0] * 16 + [1] * 16):
        raise ValueError("fixed replication DEV panel identity differs")
    manifests = {arm: Path(f"artifacts/joint-f-r2-r4-smoothed-v4-{arm}/manifest.json")
                 for arm in CONFIG["arms"]}
    checkpoints = {arm: manifests[arm].parent / f"step-{args.step:06d}.pt"
                   for arm in CONFIG["arms"]}
    for arm in CONFIG["arms"]:
        trained = read(manifests[arm])
        if (trained["optimizer_steps"] != args.step
                or trained["active_seconds"] < 43200
                or trained["status"] != "checkpointed"
                or trained["contract"]["arm"] != arm
                or trained["contract"]["config"]["v4_source_sha256"]
                != TRAINING_SOURCE_SHA
                or trained["checkpoints"][-1]["file"] != checkpoints[arm].name
                or trained["checkpoints"][-1]["sha256"] != sha(checkpoints[arm])):
            raise ValueError(f"trained endpoint not eligible: {arm}")
    pins = (PANEL, *checkpoints.values(), *manifests.values(), Path(__file__),
            Path("research/312-f-r2-r4-bounded-dev-replication-v1.md"),
            Path("research/307-f-r2-r4-smoothed-dev-evaluation-v1.md"),
            Path("research/310-f-r2-r4-v4-dev-evaluation-adapter-v1.md"),
            Path("research/311-f-r2-r4-v4-dev-evaluator-json-normalization-fix.md"),
            Path("artifacts/joint-f-r2-r4-smoothed-v4-migration.json"),
            Path("artifacts/joint-f-r2-r4-smoothed-v3-retention.json"),
            Path("scripts/build_f_r2_r4_replication_panel_v1.py"),
            Path("scripts/audit_f_r2_r4_replication_panel_v1.py"),
            Path("scripts/resource_cost_search.py"),
            Path("scripts/train_joint_f_r2_r4_smoothed_v4.py"),
            Path("scripts/run_joint_f_r2_r4_smoothed_v4.py"))
    contract = {"schema": "rf-knots-f-r2-r4-bounded-replication-v1",
                "step": args.step, "config": CONFIG,
                "inputs": {str(path): sha(path) for path in pins}}
    contract = json.loads(json.dumps(contract))
    contract_sha = digest(contract)
    tasks = [(arm, seed, row_index, weight_index)
             for arm in CONFIG["arms"] for seed in SEEDS
             for row_index in range(32) for weight_index in range(2)]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out / "manifest.json"
        state = read(manifest_path) if args.resume else {
            "contract": contract, "contract_sha256": contract_sha,
            "status": "preparing", "next_task": 0, "shards": []}
        if (state["contract"] != contract
                or state["contract_sha256"] != contract_sha
                or state["next_task"] != len(state["shards"])):
            raise ValueError("resume contract or coverage differs")
        for position, entry in enumerate(state["shards"]):
            if (entry["file"] != f"condition-{position:03d}.json"
                    or sha(args.out / entry["file"]) != entry["sha256"]):
                raise ValueError("committed replication shard SHA differs")
        atomic(manifest_path, state)
        end = min(len(tasks), state["next_task"] + (args.stop_after or len(tasks)))
        active = None
        net = objective = None
        for position in range(state["next_task"], end):
            arm, seed, row_index, weight_index = tasks[position]
            if arm != active:
                net, objective = load_actor()
                net.encoder.graph_rounds = 2 if arm == "R2" else 4
                checkpoint = torch.load(checkpoints[arm], map_location="cpu",
                                        weights_only=False)
                if (checkpoint["step"] != args.step
                        or checkpoint["contract_sha256"]
                        != read(manifests[arm])["contract_sha256"]):
                    raise ValueError(f"checkpoint identity differs: {arm}")
                net.load_state_dict(checkpoint["model"], strict=True)
                objective.load_state_dict(checkpoint["objective"], strict=True)
                active = arm
            row = panel["rows"][row_index]
            instance_dict = row["instances"][weight_index]
            instance = M.Instance(**{**instance_dict,
                                      "word": tuple(instance_dict["word"])})
            config = {**CONFIG, "seed": seed, "horizon": instance.cap_moves}
            started_wall, started_cpu = time.monotonic(), time.process_time()
            found, counts = run_batch(
                [{"condition": position, "panel_index": row["generator_index"],
                  "instance": instance_dict}], net, objective, 0, config,
                prior="distance", cached=True)
            result = found[0]
            if result["witness"] is not None:
                witness = UnknotWitness.from_dict(result["witness"])
                witness.verify()
                if (witness.start.word, witness.start.strands) != (instance.word,
                                                                   instance.strands):
                    raise ValueError("solution starts elsewhere")
            payload = {"task": position, "arm": arm, "seed": seed,
                       "row_index": row_index, "lambda": instance.cc_weight,
                       "result": result, "counts": dict(counts),
                       "cpu_seconds": time.process_time() - started_cpu,
                       "wall_seconds": time.monotonic() - started_wall}
            shard_path = args.out / f"condition-{position:03d}.json"
            atomic(shard_path, payload)
            state["shards"].append({"file": shard_path.name,
                                    "sha256": sha(shard_path)})
            state.update(status="evaluating", next_task=position + 1,
                         current_arm=arm, pid=os.getpid())
            atomic(manifest_path, state)
            print(json.dumps({"next_task": position + 1, "arm": arm,
                              "seed": seed, "solved": result["solved"]}),
                  flush=True)
        if state["next_task"] == len(tasks):
            state.update(status="completed", current_arm=None, pid=None)
            atomic(manifest_path, state)


if __name__ == "__main__":
    main()
