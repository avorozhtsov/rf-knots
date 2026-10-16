"""Protocol97 full128-condition cost search; frozen70 resources, raw raster embeddings."""

import argparse
import fcntl
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, sha
from evaluate_graph_mixture_cost_mcts import CONFIG, PANEL, batch_config, build_roots
from evaluate_joint_cost_mcts import committed_rows
from raster_cost_search import run_batch
from raster_log_training import matched_initializations

from rf_knots.embedding_metric import MetricEmbeddingObjective


def validate_resume(state, arm, pins, roots, directory, model, pair_audit):
    if (
        state["schema"] != "rf-knots-raster-log-cost-mcts-v1"
        or (state["arm"], state["config"], state["inputs"]) != (arm, CONFIG, pins)
        or state["model_checkpoint"] != str(model)
        or state["checkpoint_sha256"] != pins[str(model)]
        or state["training_pair_audit"] != str(pair_audit)
        or state["value_head_used"] is not False
        or state["status"] not in ("evaluating", "completed")
        or type(state["next_condition"]) is not int
        or not 0 <= state["next_condition"] <= len(roots)
        or state["next_condition"] % CONFIG["batch_conditions"]
    ):
        raise ValueError("resume search contract differs")
    rp = directory / "roots.json"
    if state["roots_sha256"] != sha(rp) or json.loads(rp.read_text()) != roots:
        raise ValueError("search roots changed")
    committed_rows(state, directory)
    if state["status"] == "completed" and state["next_condition"] != len(roots):
        raise ValueError("completed search lacks query coverage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("q254", "random"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("new output required, or explicit resume")
    run = args.run / args.arm
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
    pair = json.loads(args.pair_audit.read_text())
    if (
        pair["status"] != "passed"
        or set(pair["arms"]) != {"q254", "random"}
        or pair["arms"][args.arm]["checkpoint_sha256"] != sha(run / "checkpoint.pt")
        or pair["inputs"].get(str(run / "checkpoint.pt")) != sha(run / "checkpoint.pt")
    ):
        raise ValueError("require exact final paired training audit")
    sources = [
        Path(__file__).resolve(),
        Path("scripts/evaluate_joint_cost_mcts.py"),
        Path("scripts/evaluate_graph_mixture_cost_mcts.py"),
        Path("scripts/raster_cost_search.py"),
        Path("tests/test_raster_cost_search.py"),
        Path("tests/test_raster_log_search.py"),
        Path("scripts/audit_cost_mcts_continuations.py"),
        Path("scripts/check_cost_mcts_continuations.py"),
        Path("scripts/evaluate_embedding_cost_search.py"),
        Path("scripts/audit_graph_mixture_short_u_panel.py"),
    ]
    pins = dict(audit["inputs"])
    merge_pins(pins, pair["inputs"])
    merge_pins(
        pins,
        {
            str(p): sha(p)
            for p in [
                args.pair_audit,
                audit_path,
                run / "manifest.json",
                run / "checkpoint.pt",
                *sources,
            ]
        },
    )
    check_pins(pins)
    roots = build_roots(json.loads(PANEL.read_text()))
    if len(roots) != 128:
        raise ValueError("require all64 roots at both cost weights")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mp, rp = args.out / "manifest.json", args.out / "roots.json"
    if args.resume:
        state = json.loads(mp.read_text())
        validate_resume(
            state, args.arm, pins, roots, args.out, run / "checkpoint.pt", args.pair_audit
        )
        if state["status"] == "completed":
            print("already completed, frozen artifacts verified")
            return
    else:
        atomic(rp, roots)
        state = {
            "schema": "rf-knots-raster-log-cost-mcts-v1",
            "training_pair_audit": str(args.pair_audit),
            "model_checkpoint": str(run / "checkpoint.pt"),
            "checkpoint_sha256": sha(run / "checkpoint.pt"),
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
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    net, _, architecture, _ = matched_initializations(pins)
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint["architecture"] != architecture:
        raise ValueError("checkpoint architecture differs")
    net.load_state_dict(checkpoint["model"], strict=True)
    objective = MetricEmbeddingObjective()
    objective.load_state_dict(checkpoint["objective"], strict=True)
    for begin in range(state["next_condition"], len(roots), CONFIG["batch_conditions"]):
        batch = roots[begin : begin + CONFIG["batch_conditions"]]
        started = time.monotonic()
        rows, counts = run_batch(batch, net, objective, begin, batch_config(batch))
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
