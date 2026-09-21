"""Protocol70 full-policy cost MCTS with size-scaled per-root resource envelopes."""

import argparse
import fcntl
import json
import os
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
from audit_graph_mixture_short_u_panel import stratum
from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, sha
from evaluate_embedding_cost_search import query_instance
from evaluate_joint_cost_mcts import committed_rows, run_batch

from rf_knots.embedding_evaluation import verify_evaluation_pair

PANEL = Path("artifacts/graph-mixture-fresh-short-u-test-v1/panel.json")
RUNS = {
    "initial": Path("artifacts/mcts-policy-metric-joint-v1"),
    "graph": Path("artifacts/graph-mixture-graph-v1"),
    "control": Path("artifacts/graph-mixture-control-v1"),
}
CONFIG = {
    "seed": 20261034,
    "cc_weights": [10, 1000],
    "batch_conditions": 8,
    "simulations": 8,
    "horizon": "per-root-move-cap",
    "insertion_candidates": 4,
    "widening_batch": 4,
    "c_puct": 1.5,
    "temperature": 1.0,
}


def build_roots(panel):
    if not panel.get("complete") or panel.get("pair_kind") != "short-u":
        raise ValueError("require adequate prospective short-u panel")
    roots = []
    for index, row in enumerate(panel["validation"]):
        p = row["interval"]
        if p["right"].get("word") != [] or p["right"].get("strands") != 1:
            raise ValueError("query is not an explicit unknot task")
        witness = verify_evaluation_pair(row)
        instance, fits = query_instance(index, witness, witness_caps=True)
        if not fits:
            raise ValueError("reference does not fit")
        twice_q0 = len(instance.word) + instance.strands - 1
        if twice_q0 % 2:
            raise ValueError("odd knot cost floor")
        instance = replace(
            instance,
            cap_moves=max(128, 2 * twice_q0 + 32, witness.moves),
            growth_len=max(8, instance.growth_len),
            growth_strands=max(4, instance.growth_strands),
        )
        for weight in CONFIG["cc_weights"]:
            i = replace(instance, cc_weight=weight, uid=f"test-{index}-lambda-{weight}")
            roots.append(
                {
                    "condition": len(roots),
                    "panel_index": index,
                    "fingerprint": row["left_fingerprint"],
                    "stratum": stratum(len(i.word), i.strands),
                    "policy_stratum": stratum(
                        len(i.word), i.strands, max_length=208, max_strands=17
                    ),
                    "q0": twice_q0 // 2,
                    "horizon": i.cap_moves,
                    "instance": i.to_dict(),
                    "reference_L10": 10 * witness.crossing_changes + witness.moves,
                    "reference_L1000": 1000 * witness.crossing_changes + witness.moves,
                }
            )
    return roots


def batch_config(roots):
    # The shared continuation implementation accepts one horizon per batch.
    # Each instance stops at its own cap first, so this is exactly the declared
    # per-root-cap horizon, not extra moves for a shorter root.
    if not roots:
        raise ValueError("empty search batch")
    return {**CONFIG, "horizon": max(r["instance"]["cap_moves"] for r in roots)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=(*RUNS, "uniform"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("new output required, or explicit resume")
    run = RUNS["initial" if args.arm == "uniform" else args.arm]
    audit_path = PANEL.parent / "independence-audit.json"
    audit = json.loads(audit_path.read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    if (
        audit["status"] != "passed"
        or not audit["evaluation_ready"]
        or audit["inputs"].get(str(PANEL)) != sha(PANEL)
        or manifest["status"] != "completed"
        or manifest["checkpoint_sha256"] != sha(run / "checkpoint.pt")
    ):
        raise ValueError("require completed model and adequate audited independent panel")
    sources = [
        Path(__file__).resolve(),
        Path("scripts/evaluate_joint_cost_mcts.py"),
        Path("scripts/audit_cost_mcts_continuations.py"),
        Path("scripts/check_cost_mcts_continuations.py"),
        Path("scripts/evaluate_embedding_cost_search.py"),
        Path("scripts/audit_graph_mixture_short_u_panel.py"),
    ]
    pins = dict(audit["inputs"])
    merge_pins(
        pins,
        {
            str(p): sha(p)
            for p in [audit_path, run / "manifest.json", run / "checkpoint.pt", *sources]
        },
    )
    check_pins(pins)
    roots = build_roots(json.loads(PANEL.read_text()))
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mp, rp = args.out / "manifest.json", args.out / "roots.json"
    if args.resume:
        state = json.loads(mp.read_text())
        if (state["arm"], state["config"], state["inputs"]) != (args.arm, CONFIG, pins):
            raise ValueError("resume search contract differs")
        if state["roots_sha256"] != sha(rp) or json.loads(rp.read_text()) != roots:
            raise ValueError("search roots changed")
        committed_rows(state, args.out)
        if state["status"] == "completed":
            if state["next_condition"] != len(roots):
                raise ValueError("completed search lacks query coverage")
            print("already completed, frozen artifacts verified")
            return
    else:
        atomic(rp, roots)
        state = {
            "schema": "rf-knots-graph-mixture-cost-mcts-v1",
            "arm": args.arm,
            "config": CONFIG,
            "inputs": pins,
            "shards": [],
            "counts": {},
            "roots_sha256": sha(rp),
            "next_condition": 0,
            "seconds_committed_batches": 0.0,
            "value_head_used": False,
        }
    state.update(status="evaluating", pid=os.getpid())
    atomic(mp, state)
    torch.set_num_threads(1)
    checkpoint = (
        None
        if args.arm == "uniform"
        else torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    )
    for begin in range(state["next_condition"], len(roots), CONFIG["batch_conditions"]):
        batch = roots[begin : begin + CONFIG["batch_conditions"]]
        started = time.monotonic()
        rows, counts = run_batch(batch, checkpoint, begin, batch_config(batch))
        seconds = time.monotonic() - started
        shard = args.out / f"conditions-{begin:05d}.json"
        if shard.exists() and json.loads(shard.read_text()) != json.loads(json.dumps(rows)):
            raise ValueError("uncommitted batch differs; preserving existing evidence")
        if not shard.exists():
            atomic(shard, rows)
        check_pins(pins)
        total = Counter(state["counts"])
        total.update(counts)
        state["shards"].append({"file": shard.name, "sha256": sha(shard), "seconds": seconds})
        state.update(
            next_condition=begin + len(rows),
            counts=dict(total),
            seconds_committed_batches=state["seconds_committed_batches"] + seconds,
        )
        atomic(mp, state)
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "next_condition": state["next_condition"],
                    "counts": dict(total),
                    "seconds": seconds,
                }
            ),
            flush=True,
        )
    committed_rows(state, args.out)
    check_pins(pins)
    state["status"] = "completed"
    atomic(mp, state)


if __name__ == "__main__":
    main()
