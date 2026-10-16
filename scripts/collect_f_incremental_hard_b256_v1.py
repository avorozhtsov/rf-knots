"""Fixed hard-stage R2 TRAIN searches at 256 real simulations."""

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import torch

import build_f_incremental_roster_v2 as roster_source
from bootstrap_cost_value_teacher_v1 import load_actor, owned_path
from build_mcts_training_supervision import atomic, digest, sha
from resource_cost_search import run_batch
from rf_knots import mcts_loop as M

ROSTER = Path("artifacts/f-incremental-roster-v2/roster.json")
ROSTER_SHA = "93bbaa282ec79bd017b5c7adfbd4f55041adddebfe5505650d258d1f17ce58f8"
ACTOR = Path("artifacts/joint-f-r2-continuation-v1/step-133506.pt")
ACTOR_SHA = "4b11cb1078f14f64cfabd1d65c97826b8eb7fdefb16edec8d07d982c33d4cad2"
PILOT_REPORT = Path("artifacts/f-incremental-r2-yield-pilot-audit-v1/report.json")
PILOT_REPORT_SHA = "a42ae6b1b78123aa71e2f4ba9cb2e52fd8e881a83ad37176949e85af83ed60bb"
PROTOCOL = Path("research/319-new-root-b64-yield-and-hard-budget-gate.md")
SEEDS = (20271321, 20271322)
CONFIG = {"insertion_candidates": 4, "widening_batch": 4, "c_puct": 1.5,
          "temperature": 1.0, "simulations": 256}


def read(path):
    return json.loads(path.read_text())


def selected_roots(roster):
    roots = []
    for stage in (2, 3):
        rows = roster["stages"][stage]
        roots += [row for row in rows if row["reference_cc"] == 1][:4]
        roots += [row for row in rows if row["reference_cc"] == 2][:4]
    if len(roots) != 16 or len({tuple(r["query_group"]) for r in roots}) != 16:
        raise ValueError("fixed hard-root selection differs")
    return roots


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
            or sha(PILOT_REPORT) != PILOT_REPORT_SHA):
        raise ValueError("frozen roster/actor/pilot report changed")
    roots = selected_roots(read(ROSTER))
    tasks = [(r, weight, seed) for r in range(16)
             for weight in range(2) for seed in SEEDS]
    pins = (ROSTER, ACTOR, PILOT_REPORT, PROTOCOL, Path(__file__),
            Path("scripts/build_f_incremental_roster_v2.py"),
            Path("scripts/bootstrap_cost_value_teacher_v1.py"),
            Path("scripts/resource_cost_search.py"))
    inputs = {str(path): sha(path) for path in pins}
    contract = {"schema": "rf-knots-incremental-hard-b256-v1",
                "root_generator_indices": [r["generator_index"] for r in roots],
                "search_seeds": list(SEEDS), "config": CONFIG, "inputs": inputs,
                "role": "TRAIN-only hard-root search, not optimizer training"}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out / "manifest.json"
        if args.resume:
            state = read(manifest_path)
            if (state["contract"] != contract
                    or state["contract_sha256"] != digest(contract)
                    or state["next_task"] != len(state["shards"])):
                raise ValueError("resume contract/shards differ")
            for position, entry in enumerate(state["shards"]):
                if (entry["file"] != f"attempt-{position:04d}.json"
                        or sha(args.out / entry["file"]) != entry["sha256"]):
                    raise ValueError("committed shard SHA differs")
        else:
            state = {"contract": contract, "contract_sha256": digest(contract),
                     "status": "preparing", "next_task": 0, "shards": []}
            atomic(manifest_path, state)
        if state["next_task"] == len(tasks):
            state["status"] = "completed"; atomic(manifest_path, state); return
        excluded = roster_source.old.excluded_groups()
        for panel in roster_source.PANELS:
            excluded.update(tuple(row["query_group"]) for row in read(panel)["rows"])
        net, objective = load_actor()
        saved = torch.load(ACTOR, map_location="cpu", weights_only=False)
        if saved["step"] != 133506:
            raise ValueError("selected R2 checkpoint step differs")
        net.encoder.graph_rounds = 2
        net.load_state_dict(saved["model"], strict=True)
        objective.load_state_dict(saved["objective"], strict=True)
        net.eval(); objective.eval()
        state.update(status="collecting", pid=os.getpid())
        atomic(manifest_path, state)
        end = min(len(tasks), state["next_task"] + (args.stop_after or len(tasks)))
        for position in range(state["next_task"], end):
            if any(sha(Path(path)) != expected for path, expected in inputs.items()):
                raise ValueError("pinned input changed")
            root_index, weight_index, seed = tasks[position]
            root = roots[root_index]
            instance_dict = root["instances"][weight_index]
            instance = M.Instance(**{**instance_dict,
                                     "word": tuple(instance_dict["word"])})
            config = {**CONFIG, "seed": seed, "horizon": instance.cap_moves}
            started_wall, started_cpu = time.monotonic(), time.process_time()
            rows, counts = run_batch(
                [{"condition": root_index, "panel_index": root["generator_index"],
                  "instance": instance_dict}], net, objective, 0, config,
                prior="distance", cached=True)
            row = rows[0]
            row["episodes_train_owned"] = [owned_path(instance, ep, excluded)
                                           for ep in row["episodes"]]
            row["eligible_solved_episodes"] = sum(
                ep["continuation"]["stop_reason"] == "solved" and eligible
                for ep, eligible in zip(row["episodes"],
                                        row["episodes_train_owned"], strict=True))
            shard = args.out / f"attempt-{position:04d}.json"
            atomic(shard, {"root_index": root_index,
                           "generator_index": root["generator_index"],
                           "weight": instance.cc_weight, "seed": seed,
                           "row": row, "counts": counts,
                           "cpu_seconds": time.process_time() - started_cpu,
                           "wall_seconds": time.monotonic() - started_wall})
            state["shards"].append({"file": shard.name, "sha256": sha(shard)})
            state["next_task"] = position + 1
            atomic(manifest_path, state)
            print(json.dumps({"next_task": state["next_task"],
                              "root": root_index, "weight": instance.cc_weight,
                              "eligible_solved_episodes": row["eligible_solved_episodes"]}),
                  flush=True)
        if state["next_task"] == len(tasks):
            state["status"] = "completed"; atomic(manifest_path, state)


if __name__ == "__main__":
    main()
