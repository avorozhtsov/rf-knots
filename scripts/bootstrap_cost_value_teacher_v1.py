"""Small TRAIN-only CPU teacher pilot from an immutable long-raster checkpoint.

This collects independent rollout-search outcomes to bootstrap a NEW value
head. It is not AlphaZero search, value-head training, or the eight-arm sweep.
"""

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import torch
from build_mcts_training_supervision import atomic, sha
from raster_log_training import inputs_and_sources, matched_initializations
from resource_cost_search import run_batch

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.cost_mcts_rollout import advance
from rf_knots.embedding_metric import MetricEmbeddingObjective
from rf_knots.embedding_replay_corpus import invariants, owner

ACTOR = Path(
    "artifacts/raster-long-general-training-continuation-v1/long/step-032768-9b8ebee990c1986e.pt"
)
ACTOR_SHA = "9b8ebee990c1986eec3cda30a00105da9819892dd8649dcc438468a197685cba"
ROOTS = Path("artifacts/diverse-cost-mcts-training-v1/roots.json")
SOURCE = Path("artifacts/diverse-cost-mcts-training-v1/manifest.json")
ROOT_INDICES = (10, 11)  # the same TRAIN representation, L10 and L1000
SEEDS = tuple(20261100 + i for i in range(10))
SIMULATIONS = 16  # adequacy smoke test, not a claim of useful search depth
CONFIG = {
    "insertion_candidates": 4,
    "widening_batch": 4,
    "c_puct": 1.5,
    "temperature": 1.0,
    "simulations": SIMULATIONS,
}


def read(path):
    return json.loads(path.read_text())


def owned_path(instance, episode, heldout):
    state = M.SearchState(instance.word, instance.strands, 0, 0)
    groups = {invariants(state.word, state.strands)}
    for action in episode["tree_prefix"] + episode["continuation"]["actions"]:
        state = advance(instance, state, action)
        if instance.spec.decode(action)[0] == CROSSING_CHANGE:
            groups.add(invariants(state.word, state.strands))
    return all(owner(fp) == "train" and (fp[0], abs(fp[1])) not in heldout for fp in groups)


def load_actor():
    if sha(ACTOR) != ACTOR_SHA:
        raise ValueError("selected actor checkpoint hash changed")
    pins, _ = inputs_and_sources()
    net, _, architecture, _ = matched_initializations(pins)
    payload = torch.load(ACTOR, map_location="cpu", weights_only=False)
    if (
        payload["arm"] != "long"
        or payload["steps"] != 32768
        or payload["architecture"] != architecture
    ):
        raise ValueError("selected actor identity differs")
    net.load_state_dict(payload["model"], strict=True)
    objective = MetricEmbeddingObjective()
    objective.load_state_dict(payload["objective"], strict=True)
    return net, objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-after", type=int, default=0, help="operational limit, not a contract change"
    )
    args = parser.parse_args()
    if args.stop_after < 0 or args.out.exists() != args.resume:
        raise ValueError("new output requires no --resume; existing output requires --resume")
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    torch.set_num_threads(1)
    roots_all = read(ROOTS)
    source = read(SOURCE)
    if source["status"] != "completed" or sha(ROOTS) != source["roots_sha256"]:
        raise ValueError("TRAIN root source lacks completed pinned identity")
    if len(roots_all) <= max(ROOT_INDICES):
        raise ValueError("selected TRAIN roots absent")
    roots = [roots_all[i] for i in ROOT_INDICES]
    if roots[0]["base_root_index"] != roots[1]["base_root_index"] or {
        r["instance"]["cc_weight"] for r in roots
    } != {10, 1000}:
        raise ValueError("selected roots are not a paired L10/L1000 representation")
    heldout = {tuple(row) for row in source["heldout_query_groups"]}
    pins = {str(p): sha(p) for p in (ROOTS, SOURCE, Path(__file__))}
    pins[str(ACTOR)] = ACTOR_SHA
    contract = dict(
        schema="rf-knots-budgeted-value-teacher-smoke-v1",
        actor_sha256=ACTOR_SHA,
        roots=ROOT_INDICES,
        seeds=SEEDS,
        simulations=SIMULATIONS,
        config=CONFIG,
        inputs=pins,
        role="TRAIN-only teacher outcomes; not value-head training or variant selection",
    )
    tasks = [(index, seed) for index in ROOT_INDICES for seed in SEEDS]
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
            for shard in state["shards"]:
                if sha(args.out / shard["file"]) != shard["sha256"]:
                    raise ValueError("committed teacher shard changed")
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
        for position in range(state["next_task"], min(len(tasks), state["next_task"] + remaining)):
            if any(sha(Path(p)) != h for p, h in pins.items()):
                raise ValueError("teacher inputs changed")
            index, seed = tasks[position]
            root = roots_all[index]
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
