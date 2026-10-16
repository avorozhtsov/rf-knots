"""Resumable149 terminal-cost MCTS; fixed151 matrix and all150 original roots."""

import argparse
import fcntl
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, digest, sha
from check_long_general_search_cache import MODELS
from check_long_general_search_cache import SCHEMA as CACHE_SCHEMA
from evaluate_graph_mixture_cost_mcts import build_roots
from evaluate_joint_cost_mcts import committed_rows
from long_general_evaluation_inputs import load_inputs, load_model, load_panel
from probe_raster_cache_search import SEEDS
from resource_cost_search import run_batch
from train_long_general_pair import SEARCH_CONFIG as CONFIG
from train_long_general_pair import SEARCH_MATRIX

SCHEMA = "rf-knots-long-general-search-v1"


def read(path):
    return json.loads(path.read_text())


def declared(arm, stage, prior):
    if dict(arm=arm, stage=stage, prior=prior) not in SEARCH_MATRIX:
        raise ValueError("search is not one of the five frozen151 configurations")


def require_cache_gate(path, checkpoint, arm, stage, pins, training_contract):
    gate = read(path)
    if (
        gate["schema"] != CACHE_SCHEMA
        or gate["status"] != "passed"
        or set(gate["models"]) != {f"{a}-{s}" for a, s in MODELS}
        or gate["training_contract_sha256"] != training_contract
        or gate["cache_capacity"] != CONFIG["cache_capacity"]
        or gate["device"] != "cpu"
        or gate["threads"] != 1
        or gate["torch_version"] != torch.__version__
    ):
        raise ValueError("require matching complete149 cache parity")
    expected_searches = {(prior, seed) for prior in ("distance", "uniform") for seed in SEEDS}
    for name, row in gate["models"].items():
        if (
            row["status"] != "passed"
            or row["weights_unchanged"] is not True
            or row["cache_capacity"] != 512
            or row["forward_cases"] != 102
            or len(row["searches"]) != len(expected_searches)
            or {(r["prior"], r["seed"]) for r in row["searches"]} != expected_searches
            or gate["inputs"].get(row["checkpoint"]) != row["checkpoint_sha256"]
        ):
            raise ValueError("cache parity model/fixture coverage differs")
        if name == "control-initial":
            identity = row["shared_initial_identity"]
            if not identity or any(
                identity[k] is not True for k in ("exact_tensors", "encoder_equal", "policy_equal")
            ):
                raise ValueError("shared initial function identity is missing")
    record = gate["models"][f"{arm}-{stage}"]
    if record["checkpoint"] != str(checkpoint) or record["checkpoint_sha256"] != sha(checkpoint):
        raise ValueError("cache parity used a different checkpoint")
    source = Path("scripts/check_long_general_search_cache.py").resolve()
    if gate["inputs"].get(str(source)) != sha(source):
        raise ValueError("cache parity source changed")
    merge_pins(pins, gate["inputs"])
    merge_pins(pins, {str(path): sha(path)})
    check_pins(pins)


def validate_resume(state, contract, roots, directory):
    declared(contract["arm"], contract["stage"], contract["prior"])
    if (
        contract["config"] != CONFIG
        or state["schema"] != SCHEMA
        or state["contract"] != contract
        or state["contract_sha256"] != digest(contract)
        or state["status"] not in ("evaluating", "completed")
        or state["value_head_used"] is not False
        or type(state["next_condition"]) is not int
        or not 0 <= state["next_condition"] <= len(roots)
        or state["next_condition"] % CONFIG["batch_conditions"]
    ):
        raise ValueError("149 search resume contract differs")
    path = directory / "roots.json"
    if sha(path) != state["roots_sha256"] or read(path) != roots:
        raise ValueError("149 search roots changed")
    expected = [
        f"conditions-{i:05d}.json"
        for i in range(0, state["next_condition"], CONFIG["batch_conditions"])
    ]
    if [s["file"] for s in state["shards"]] != expected:
        raise ValueError("committed search shard names/coverage differ")
    committed_rows(state, directory)
    if state["counts"].get("conditions", 0) != state["next_condition"] or any(
        type(value) is not int or value < 0 for value in state["counts"].values()
    ):
        raise ValueError("committed search counters differ")
    seconds = [s["seconds"] for s in state["shards"]]
    if any(not math.isfinite(s) or s < 0 for s in seconds) or not math.isclose(
        sum(seconds), state["seconds_committed_batches"]
    ):
        raise ValueError("committed search durations differ")
    if state["status"] == "completed" and state["next_condition"] != len(roots):
        raise ValueError("completed search lacks conditions")


def run_search(directory, roots, net, objective, contract, pins, *, resume=False):
    declared(contract["arm"], contract["stage"], contract["prior"])
    if contract["config"] != CONFIG or len(roots) != 128:
        raise ValueError("require unchanged config and all128 conditions")
    if contract["inputs"] != pins:
        raise ValueError("search inputs differ from frozen contract")
    check_pins(pins)
    if directory.exists() != resume:
        raise ValueError("new search output required, or explicit resume")
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path, roots_path = directory / "manifest.json", directory / "roots.json"
    if resume:
        state = read(path)
        validate_resume(state, contract, roots, directory)
        if state["status"] == "completed":
            return state
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
            batch, net, objective, begin, config, prior=contract["prior"], cached=True
        )
        elapsed = time.monotonic() - started
        if [r["condition"] for r in rows] != [r["condition"] for r in batch]:
            raise ValueError("new search batch coverage differs")
        shard = directory / f"conditions-{begin:05d}.json"
        if shard.exists() and read(shard) != json.loads(json.dumps(rows)):
            raise ValueError("uncommitted search shard differs; preserving evidence")
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
                    arm=contract["arm"],
                    stage=contract["stage"],
                    prior=contract["prior"],
                    next_condition=state["next_condition"],
                    counts=dict(total),
                )
            ),
            flush=True,
        )
    validate_resume(state, contract, roots, directory)
    check_pins(pins)
    state.update(status="completed")
    atomic(path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "pair-audit", "cache-gate", "pgx-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "long"), required=True)
    parser.add_argument("--stage", choices=("initial", "final"), required=True)
    parser.add_argument("--prior", choices=("distance", "uniform"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    declared(args.arm, args.stage, args.prior)
    if args.out.exists() != args.resume:
        raise ValueError("new search output required, or explicit resume")
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    parent, audit, pins = load_inputs(args.run, args.pair_audit)
    _, panel_path = load_panel("short-u", parent, pins)
    net, objective, checkpoint, _ = load_model(args.run, parent, audit, pins, args.arm, args.stage)
    require_cache_gate(
        args.cache_gate, checkpoint, args.arm, args.stage, pins, parent["contract_sha256"]
    )
    for name in (
        "scripts/evaluate_long_general_search.py",
        "scripts/resource_cost_search.py",
        "scripts/evaluate_graph_mixture_cost_mcts.py",
        "scripts/evaluate_joint_cost_mcts.py",
        "scripts/audit_cost_mcts_continuations.py",
        "scripts/probe_raster_tile_cache.py",
        "tests/test_long_general_search.py",
        "research/151-long-mixture-evaluation-contract.md",
    ):
        path = Path(name).resolve()
        merge_pins(pins, {str(path): sha(path)})
    check_pins(pins)
    roots = build_roots(read(panel_path))
    contract = dict(
        config=CONFIG,
        arm=args.arm,
        stage=args.stage,
        prior=args.prior,
        training_run=str(args.run),
        training_contract_sha256=parent["contract_sha256"],
        pair_audit=str(args.pair_audit),
        cache_gate=str(args.cache_gate),
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),
        panel=str(panel_path),
        inputs=pins,
        torch_version=torch.__version__,
    )
    run_search(args.out, roots, net, objective, contract, pins, resume=args.resume)


if __name__ == "__main__":
    main()
