"""Resumable TRAIN-only reference-directed R2 cost-MCTS bridge pilot."""

import argparse
import fcntl
import json
import os
import signal
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from audit_cost_mcts_continuations import audit_episode
from bootstrap_cost_value_teacher_v1 import load_actor
from build_mcts_training_supervision import atomic, digest, sha
from pretrained_raster_policy import RasterPolicyEvaluator
from reference_guided_bridge_v1 import (
    ReferenceCorridorPrior, completed_bridges, reference_actions_states,
)
from rf_knots import mcts_loop as M
from rf_knots.cost_mcts_rollout import (
    BatchedPolicyContinuation, run_continuation_cost_searches,
)
from rf_knots.evidence import UnknotWitness


ROSTER = Path("artifacts/f-root-only-roster-v1/roster.json")
ROSTER_SHA = "3c4a8eb62a63d9f4e22a4314aa94434164bb5a4c57fa70282783b9c602e23a81"
AUDIT = Path("artifacts/f-root-only-roster-audit-v1/report.json")
AUDIT_SHA = "755d48594e69c76c16041fce7279a258532b82e9ebc65254d64666d47511dd66"
ACTOR = Path("artifacts/additive-f-r2-stage-v2/step-152964.pt")
ACTOR_SHA = "12f4a3675dd6880203a2968431d3c7bdd44a45b586e92937a25373facc32803e"
PROTOCOL = Path("research/336-reference-guided-root-f-mcts-pilot-v1.md")
GENERATORS = (0, 90, 35, 63)
SEEDS = (20271336, 20271337)
CONFIG = {"simulations": 32, "insertion_candidates": 4,
          "widening_batch": 4, "c_puct": 1.5,
          "corridor_temperature": 2.0, "reference_first_suppression": 6.0}
CPU_CEILING = 1800.0
PER_TASK_CPU_CEILING = 180.0


class CpuCeiling(Exception):
    pass


def expired(_signum, _frame):
    raise CpuCeiling("per-task CPU ceiling reached")


def read(path):
    return json.loads(path.read_text())


def selected_roots(roster):
    selected = []
    for stage in roster["stages"]:
        ordered = sorted(stage, key=lambda row:
                         (row["instances"][0]["cap_crossing_changes"],
                          row["generator_index"]))
        selected.extend((ordered[0], ordered[-1]))
    if (len(selected) != 4 or tuple(row["generator_index"] for row in selected)
            != GENERATORS or len({tuple(row["query_group"]) for row in selected}) != 4):
        raise ValueError("frozen root selection differs")
    return selected


def load_policy():
    net, objective = load_actor()
    saved = torch.load(ACTOR, map_location="cpu", weights_only=False)
    if saved["step"] != 152964:
        raise ValueError("R2 checkpoint step differs")
    net.encoder.graph_rounds = 2
    net.load_state_dict(saved["model"], strict=True)
    objective.load_state_dict(saved["objective"], strict=True)
    net.eval(); objective.eval()
    return net


def search_task(root, instance, net, seed):
    witness = UnknotWitness.from_dict(root["witness"])
    reference_actions, reference_states = reference_actions_states(instance, witness)
    prior = ReferenceCorridorPrior(
        instance, reference_actions, reference_states,
        temperature=CONFIG["corridor_temperature"],
        suppress_first=CONFIG["reference_first_suppression"])
    neural = RasterPolicyEvaluator(net, microbatch=2)
    continuation = BatchedPolicyContinuation(
        neural, horizon=instance.cap_moves,
        insertion_candidates=CONFIG["insertion_candidates"])
    tree = run_continuation_cost_searches(
        [instance], prior, continuation,
        simulations=CONFIG["simulations"], seed=seed,
        widening_batch=CONFIG["widening_batch"],
        c_puct=CONFIG["c_puct"])[0]
    episodes, bridge_counts = [], Counter()
    best_strict = None
    replay_config = {"horizon": instance.cap_moves,
                     "insertion_candidates": CONFIG["insertion_candidates"]}
    for simulation, prefix, roll in tree.continuations:
        episode = json.loads(json.dumps({
            "simulation": simulation, "tree_prefix": list(prefix),
            "continuation": asdict(roll)}))
        audit_episode(instance, episode, replay_config)
        counts, bridges = completed_bridges(
            instance, reference_actions, reference_states, episode)
        bridge_counts.update(counts)
        for candidate in bridges:
            if candidate["strict_cost_pair"] and (
                    best_strict is None or candidate["cost"] < best_strict["cost"]):
                best_strict = candidate
        episodes.append(episode)
    if neural.calls != continuation.state_evaluations:
        raise ValueError("neural continuation evaluation count differs")
    if tree.best_witness is not None:
        tree.best_witness.verify()
    return {
        "episodes": episodes, "simulations": tree.simulations,
        "expansions": tree.expansions,
        "natural_witness": tree.best_witness.to_dict() if tree.best_witness else None,
        "bridge_counts": dict(bridge_counts), "best_strict_bridge": best_strict,
        "policy_neural_evaluations": neural.calls,
        "policy_neural_batches": neural.forwards,
        "corridor_action_scores": prior.action_scores,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.stop_after < 0 or args.out.exists() != args.resume:
        raise ValueError("output/resume mismatch")
    pgx_source = args.pgx_root.resolve() / "src/pgx_mcts_bench/search.py"
    if not pgx_source.exists():
        raise ValueError("canonical pgx-mcts-bench source missing")
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    torch.set_num_threads(1)
    if sha(ROSTER) != ROSTER_SHA or sha(AUDIT) != AUDIT_SHA or sha(ACTOR) != ACTOR_SHA:
        raise ValueError("frozen roster/audit/R2 SHA differs")
    roster = read(ROSTER)
    roots = selected_roots(roster)
    tasks = [(root_index, weight_index, seed)
             for root_index in range(4) for weight_index in range(2)
             for seed in SEEDS]
    pins = (ROSTER, AUDIT, ACTOR, PROTOCOL, Path(__file__),
            Path("scripts/reference_guided_bridge_v1.py"),
            Path("scripts/bootstrap_cost_value_teacher_v1.py"),
            Path("scripts/pretrained_raster_policy.py"),
            Path("scripts/audit_cost_mcts_continuations.py"),
            Path("src/rf_knots/cost_mcts_rollout.py"), pgx_source)
    inputs = {str(path): sha(path) for path in pins}
    contract = {"schema": "rf-knots-reference-guided-root-F-pilot-v1",
                "role": "TRAIN-only search yield, not optimizer training",
                "generator_indices": list(GENERATORS), "seeds": list(SEEDS),
                "config": CONFIG, "cpu_ceiling": CPU_CEILING,
                "per_task_cpu_ceiling": PER_TASK_CPU_CEILING,
                "gradient_boundary": "exact root only; no intermediate labels",
                "inputs": inputs}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.out / "manifest.json"
        if args.resume:
            state = read(path)
            if (state["contract"] != contract or state["contract_sha256"] != digest(contract)
                    or state["next_task"] != len(state["shards"])):
                raise ValueError("resume contract differs")
            for index, entry in enumerate(state["shards"]):
                shard = args.out / f"attempt-{index:04d}.json"
                if entry["file"] != shard.name or sha(shard) != entry["sha256"]:
                    raise ValueError("saved shard SHA differs")
        else:
            state = {"contract": contract, "contract_sha256": digest(contract),
                     "status": "preparing", "next_task": 0, "shards": [],
                     "cpu_seconds": 0.0}
            atomic(path, state)
        if state["next_task"] == len(tasks):
            state["status"] = "completed"
            atomic(path, state)
            return
        net = load_policy()
        state.update(status="collecting", pid=os.getpid())
        atomic(path, state)
        end = min(len(tasks), state["next_task"] + (args.stop_after or len(tasks)))
        signal.signal(signal.SIGPROF, expired)
        for index in range(state["next_task"], end):
            if any(sha(Path(name)) != expected for name, expected in inputs.items()):
                raise ValueError("pinned source/input changed")
            if state["cpu_seconds"] >= CPU_CEILING:
                state["status"] = "cpu_ceiling"
                atomic(path, state)
                return
            root_index, weight_index, seed = tasks[index]
            root = roots[root_index]
            instance_dict = root["instances"][weight_index]
            instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
            started_cpu, started_wall = time.process_time(), time.monotonic()
            try:
                signal.setitimer(signal.ITIMER_PROF, PER_TASK_CPU_CEILING)
                result = search_task(root, instance, net, seed)
                task_status = "searched"
            except CpuCeiling:
                result = None
                task_status = "cpu_censored"
            finally:
                signal.setitimer(signal.ITIMER_PROF, 0)
            shard = args.out / f"attempt-{index:04d}.json"
            cpu = time.process_time() - started_cpu
            atomic(shard, {"task": index, "root_index": root_index,
                           "generator_index": root["generator_index"],
                           "stage": root["stage"], "weight": instance.cc_weight,
                           "seed": seed, "status": task_status, "result": result,
                           "cpu_seconds": cpu,
                           "wall_seconds": time.monotonic() - started_wall})
            state["shards"].append({"file": shard.name, "sha256": sha(shard)})
            state["next_task"] = index + 1
            state["cpu_seconds"] += cpu
            atomic(path, state)
            print(json.dumps({"next_task": state["next_task"],
                              "status": task_status, "cpu_seconds": state["cpu_seconds"]}),
                  flush=True)
        if state["next_task"] == len(tasks):
            state["status"] = "completed"
            atomic(path, state)


if __name__ == "__main__":
    main()
