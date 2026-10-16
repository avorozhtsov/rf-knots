"""Resumable139 MCTS on fresh136 roots, gated by the complete135 pair and cache parity."""

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
from build_mcts_training_supervision import atomic, digest, sha
from evaluate_graph_mixture_cost_mcts import CONFIG as OLD_CONFIG
from evaluate_graph_mixture_cost_mcts import build_roots
from evaluate_joint_cost_mcts import committed_rows
from resource_cost_search import run_batch
from resource_evaluation_inputs import load_model, load_panel

CONFIG = {
    **OLD_CONFIG,
    "seed": 20261074,
    "device": "cpu",
    "threads": 1,
    "cached_local_tiles": True,
    "cache_capacity": 512,
}
SCHEMA = "rf-knots-resource-units-search-v1"


def validate_resume(state, contract, roots, directory):
    if (
        state["schema"] != SCHEMA
        or state["contract"] != contract
        or state["contract_sha256"] != digest(contract)
        or state["status"] not in ("evaluating", "completed")
        or state["value_head_used"] is not False
        or type(state["next_condition"]) is not int
        or not 0 <= state["next_condition"] <= len(roots)
        or state["next_condition"] % CONFIG["batch_conditions"]
    ):
        raise ValueError("search resume contract differs")
    path = directory / "roots.json"
    if sha(path) != state["roots_sha256"] or json.loads(path.read_text()) != roots:
        raise ValueError("search roots changed")
    committed_rows(state, directory)
    if state["status"] == "completed" and state["next_condition"] != len(roots):
        raise ValueError("completed search has missing conditions")


def require_cache_gate(path, checkpoint, arm, stage, pins):
    gate = json.loads(path.read_text())
    expected = {f"{a}-{s}" for a in ("raw", "units") for s in ("initial", "final")}
    if (
        gate["status"] != "passed"
        or set(gate["models"]) != expected
        or gate["cache_capacity"] != CONFIG["cache_capacity"]
    ):
        raise ValueError("require complete checkpoint-specific cache parity")
    record = gate["models"][f"{arm}-{stage}"]
    if (
        record["status"] != "passed"
        or record["checkpoint"] != str(checkpoint)
        or record["checkpoint_sha256"] != sha(checkpoint)
        or record["weights_unchanged"] is not True
    ):
        raise ValueError("cache gate used a different checkpoint")
    merge_pins(pins, gate["inputs"])
    merge_pins(pins, {str(path): sha(path)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--cache-gate", type=Path, required=True)
    parser.add_argument("--arm", choices=("raw", "units"), required=True)
    parser.add_argument("--stage", choices=("initial", "final"), required=True)
    parser.add_argument("--prior", choices=("distance", "uniform"), required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("existing output requires explicit resume")
    if args.stage == "initial" and args.prior != "distance":
        raise ValueError("uniform prior is a declared final-only comparison")
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    panel, pins, panel_path = load_panel("short-u")
    net, objective, model_pins, checkpoint = load_model(
        args.run, args.pair_audit, args.arm, args.stage
    )
    merge_pins(pins, model_pins)
    require_cache_gate(args.cache_gate, checkpoint, args.arm, args.stage, pins)
    for source in (
        Path(__file__).resolve(),
        Path("scripts/resource_cost_search.py"),
        Path("scripts/resource_evaluation_inputs.py"),
        Path("scripts/check_resource_search_cache.py"),
        Path("scripts/evaluate_graph_mixture_cost_mcts.py"),
        Path("scripts/evaluate_joint_cost_mcts.py"),
        Path("scripts/audit_cost_mcts_continuations.py"),
        Path("scripts/probe_raster_tile_cache.py"),
        Path("tests/test_resource_search_resume.py"),
        Path("research/139-resource-units-evaluation-protocol.md"),
    ):
        merge_pins(pins, {str(source): sha(source)})
    check_pins(pins)
    roots = build_roots(panel)
    if len(roots) != 128:
        raise ValueError("require all64 short-u roots at both costs")
    contract = dict(
        config=CONFIG,
        arm=args.arm,
        stage=args.stage,
        prior=args.prior,
        training_run=str(args.run),
        pair_audit=str(args.pair_audit),
        cache_gate=str(args.cache_gate),
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),
        panel=str(panel_path),
        inputs=pins,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path, roots_path = args.out / "manifest.json", args.out / "roots.json"
    if args.resume:
        state = json.loads(path.read_text())
        validate_resume(state, contract, roots, args.out)
        if state["status"] == "completed":
            print("completed search verified; no duplicate", flush=True)
            return
    else:
        atomic(roots_path, roots)
        state = dict(
            schema=SCHEMA,
            contract=contract,
            contract_sha256=digest(contract),
            roots_sha256=sha(roots_path),
            shards=[],
            counts={},
            next_condition=0,
            seconds_committed_batches=0.0,
            value_head_used=False,
        )
    state.update(status="evaluating", pid=os.getpid())
    atomic(path, state)
    for begin in range(state["next_condition"], len(roots), CONFIG["batch_conditions"]):
        batch = roots[begin : begin + CONFIG["batch_conditions"]]
        config = {**CONFIG, "horizon": max(r["instance"]["cap_moves"] for r in batch)}
        started = time.monotonic()
        rows, counts = run_batch(
            batch, net, objective, begin, config, prior=args.prior, cached=True
        )
        elapsed = time.monotonic() - started
        shard = args.out / f"conditions-{begin:05d}.json"
        if shard.exists() and json.loads(shard.read_text()) != json.loads(json.dumps(rows)):
            raise ValueError("uncommitted shard differs; existing evidence preserved")
        if not shard.exists():
            atomic(shard, rows)
        check_pins(pins)
        total = Counter(state["counts"])
        total.update(counts)
        state["shards"].append(dict(file=shard.name, sha256=sha(shard), seconds=elapsed))
        state.update(
            next_condition=begin + len(rows),
            counts=dict(total),
            seconds_committed_batches=state["seconds_committed_batches"] + elapsed,
        )
        atomic(path, state)
        print(
            json.dumps(
                dict(
                    arm=args.arm,
                    stage=args.stage,
                    prior=args.prior,
                    next_condition=state["next_condition"],
                    counts=dict(total),
                )
            ),
            flush=True,
        )
    committed_rows(state, args.out)
    check_pins(pins)
    state.update(status="completed")
    atomic(path, state)


if __name__ == "__main__":
    main()
