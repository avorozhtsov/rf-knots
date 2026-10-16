"""Fixed matched fresh-DEV search of root-only R2 parent and endpoint."""

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

PANEL = Path("artifacts/root-only-r2-fresh-dev-panel-v1/panel.json")
PANEL_SHA = "e069ce461ee5b2993f73ebda2f8329b46e588f9c9294f00104c38d16e942d887"
PANEL_AUDIT = Path("artifacts/root-only-r2-fresh-dev-panel-audit-v1/report.json")
PANEL_AUDIT_SHA = "02478a9cd709c47d4f8733d453cba739ab1e48303751ad63df8da565a5870a35"
PROTOCOL = Path("research/348-root-only-r2-parent-endpoint-dev-v1.md")
ARMS = {
    "parent": (Path("artifacts/additive-f-r2-stage-v2/step-152964.pt"),
               "12f4a3675dd6880203a2968431d3c7bdd44a45b586e92937a25373facc32803e",
               152964),
    "endpoint": (Path("artifacts/additive-root-only-r2-v3/step-172421.pt"),
                 "b1ca2fb4ca4b0907a56b2b9372012c8a93549692b6a863aec9d1d1174c3d1e52",
                 172421),
}
SEEDS = (20271020, 20271021)
CONFIG = {"arms": tuple(ARMS), "seeds": SEEDS,
          "simulations": 128, "insertion_candidates": 4,
          "widening_batch": 4, "c_puct": 1.5,
          "temperature": 1.0, "prior": "distance", "cache": True}


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.stop_after < 0 or args.out.exists() != args.resume:
        raise ValueError("output/resume mismatch")
    pgx_src = args.pgx_root.resolve() / "src"
    if not (pgx_src / "pgx_mcts_bench").is_dir():
        raise ValueError("canonical pgx-mcts-bench missing")
    sys.path.insert(0, str(pgx_src))
    torch.set_num_threads(1)
    if sha(PANEL) != PANEL_SHA or sha(PANEL_AUDIT) != PANEL_AUDIT_SHA:
        raise ValueError("fresh panel/audit SHA differs")
    panel = read(PANEL)
    if (panel["schema"] != "rf-knots-root-only-R2-fresh-DEV-panel-v1"
            or len(panel["rows"]) != 32
            or [row["band"] for row in panel["rows"]] != [0] * 16 + [1] * 16):
        raise ValueError("panel identity differs")
    manifests = {}
    for arm, (checkpoint, expected_sha, step) in ARMS.items():
        if sha(checkpoint) != expected_sha:
            raise ValueError(f"frozen checkpoint SHA differs: {arm}")
        manifest_path = checkpoint.parent / "manifest.json"
        trained = read(manifest_path)
        if (trained["status"] != "checkpointed"
                or trained["optimizer_steps"] != step
                or trained["contract"]["arm"] != "R2"
                or trained["checkpoints"][-1]["file"] != checkpoint.name
                or trained["checkpoints"][-1]["sha256"] != expected_sha):
            raise ValueError(f"checkpoint manifest identity differs: {arm}")
        if arm == "endpoint" and trained["new_active_seconds"] < 7200:
            raise ValueError("root-only endpoint trained too briefly")
        manifests[arm] = manifest_path
    pins = (PANEL, PANEL_AUDIT, PROTOCOL, Path(__file__),
            Path("scripts/build_root_only_r2_fresh_dev_panel_v1.py"),
            Path("scripts/audit_root_only_r2_fresh_dev_panel_v1.py"),
            Path("scripts/resource_cost_search.py"),
            Path("scripts/bootstrap_cost_value_teacher_v1.py"),
            Path("scripts/train_additive_f_r2_stage_v2.py"),
            Path("scripts/train_additive_root_only_r2_v3.py"),
            *[item[0] for item in ARMS.values()], *manifests.values())
    contract = {
        "schema": "rf-knots-root-only-R2-parent-endpoint-DEV-v1",
        "config": CONFIG,
        "steps": {arm: entry[2] for arm, entry in ARMS.items()},
        "inputs": {str(path): sha(path) for path in pins},
    }
    contract = json.loads(json.dumps(contract))
    contract_sha = digest(contract)
    tasks = [(arm, seed, row_index, weight_index)
             for arm in ARMS for seed in SEEDS
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
            raise ValueError("resume contract/coverage differs")
        for position, entry in enumerate(state["shards"]):
            if (entry["file"] != f"condition-{position:03d}.json"
                    or sha(args.out / entry["file"]) != entry["sha256"]):
                raise ValueError("committed shard SHA differs")
        atomic(manifest_path, state)
        end = min(len(tasks), state["next_task"] +
                  (args.stop_after or len(tasks)))
        active = None
        net = objective = None
        for position in range(state["next_task"], end):
            arm, seed, row_index, weight_index = tasks[position]
            if any(sha(Path(path)) != expected
                   for path, expected in contract["inputs"].items()):
                raise ValueError("pinned source/input SHA changed")
            if arm != active:
                net, objective = load_actor()
                net.encoder.graph_rounds = 2
                checkpoint_path = ARMS[arm][0]
                saved = torch.load(checkpoint_path, map_location="cpu",
                                   weights_only=False)
                if (saved["step"] != ARMS[arm][2]
                        or saved["contract_sha256"]
                        != read(manifests[arm])["contract_sha256"]):
                    raise ValueError(f"checkpoint identity differs: {arm}")
                net.load_state_dict(saved["model"], strict=True)
                objective.load_state_dict(saved["objective"], strict=True)
                net.eval(); objective.eval()
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
                if (witness.start.word, witness.start.strands) != (
                        instance.word, instance.strands):
                    raise ValueError("solution starts elsewhere")
            shard = args.out / f"condition-{position:03d}.json"
            atomic(shard, {
                "task": position, "arm": arm, "seed": seed,
                "row_index": row_index, "lambda": instance.cc_weight,
                "result": result, "counts": dict(counts),
                "cpu_seconds": time.process_time() - started_cpu,
                "wall_seconds": time.monotonic() - started_wall,
            })
            state["shards"].append({"file": shard.name, "sha256": sha(shard)})
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
