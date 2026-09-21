"""Protocol69 paired full-encoder/policy training with ordinary graph metric pairs."""

import argparse
import fcntl
import json
import os
import random
import shutil
import time
from pathlib import Path

import torch
from audit_joint_policy_metric_panel import check_pins, merge_pins
from benchmark_crossing_preprocessing import bulk_crossing_adjacency, use_adjacency
from build_mcts_training_supervision import atomic, digest, sha
from graph_metric_sampling import load_graph_sampler, paired_draws
from mcts_joint_training import (
    load_metric_samplers,
    load_policy_targets,
    policy_backward,
    reference_preferences,
)
from run_strand_graph_pilot import metric_loss, pair_row
from train_mcts_policy_metric import finite_optimizer, finite_state, validate_progress

from rf_knots import mcts_loop as M
from rf_knots.embedding_metric import MetricEmbeddingObjective

INITIAL = Path("artifacts/mcts-policy-metric-joint-v1/checkpoint.pt")
INITIAL_SHA = "e58270a34569bfad11127ef4f69a8e67bcdcf4825213f88bce03a73a2233ef84"
GRAPH = Path("artifacts/ordinary-graph-metric-corpus-v1")
SUPERVISION = Path("artifacts/diverse-mcts-supervision-diamonds-v1")
CORPUS = Path("artifacts/embedding-diverse-short-u-corpus-v1/corpus.json")
RUNS = {mode: Path(f"artifacts/graph-mixture-{mode}-v1") for mode in ("graph", "control")}
CONFIG = {
    "steps": 4096,
    "seed": 20261031,
    "device": "cpu",
    "threads": 1,
    "lr": 2.5e-5,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "clip": 0.2,
    "positive_weight": 0.1,
    "relative_weight": 0.01,
    "policy_micro_batch": 64,
    "metric_batch": 128,
    "policy_batch": 32,
    "checkpoint_interval": 16,
    "adjacency": "exact-bulk-numpy",
    "graph_classes": ["exact-1", "exact-2", "interval"],
}


def inputs_and_sources():
    graph_audit_path = GRAPH / "partition-replay-audit.json"
    policy_audit_path = SUPERVISION / "certificate-audit.json"
    graph = json.loads(graph_audit_path.read_text())
    policy = json.loads(policy_audit_path.read_text())
    initial_manifest = json.loads(INITIAL.with_name("manifest.json").read_text())
    if (
        graph["status"] != "passed"
        or not graph["train_ready"]
        or policy["status"] != "passed"
        or initial_manifest["status"] != "completed"
        or initial_manifest["checkpoint_sha256"] != INITIAL_SHA
    ):
        raise ValueError("require completed/audited sources and exact initial checkpoint")
    pins = dict(graph["inputs"])
    merge_pins(pins, policy["inputs"])
    merge_pins(
        pins,
        {
            str(p): sha(p)
            for p in (
                graph_audit_path,
                policy_audit_path,
                INITIAL.with_name("manifest.json"),
                CORPUS,
            )
        },
    )
    merge_pins(pins, {str(INITIAL): INITIAL_SHA})
    if pins.get(str(SUPERVISION / "manifest.json")) != sha(SUPERVISION / "manifest.json"):
        raise ValueError("policy audit belongs to different supervision")
    sources = [
        Path(__file__).resolve(),
        Path("scripts/graph_metric_sampling.py"),
        Path("scripts/train_mcts_policy_metric.py"),
        Path("scripts/mcts_joint_training.py"),
        Path("scripts/mcts_training_supervision.py"),
        Path("scripts/balanced_metric_sampling.py"),
        Path("scripts/benchmark_crossing_preprocessing.py"),
        Path("scripts/run_strand_graph_pilot.py"),
        Path("research/69-graph-mixture-training-protocol.md"),
        Path("research/70-graph-mixture-prospective-evaluation.md"),
        *Path("src/rf_knots").glob("*.py"),
    ]
    code = {str(p): sha(p) for p in sources}
    check_pins(pins)
    check_pins(code)
    return pins, code


def optimize_step(net, objective, optimizer, metrics, chosen, relative, config):
    if len(metrics) != config["metric_batch"] or len(chosen) != config["policy_batch"]:
        raise ValueError("training draw coverage differs")
    net.train()
    optimizer.zero_grad(set_to_none=True)
    loss, _ = metric_loss(net, objective, metrics, "cpu")
    if not torch.isfinite(loss):
        raise ValueError("nonfinite metric loss")
    loss.backward()
    policy = policy_backward(
        net,
        chosen,
        relative,
        positive_weight=config["positive_weight"],
        relative_weight=config["relative_weight"],
        micro_batch=config["policy_micro_batch"],
        clip=config["clip"],
    )
    parameters = [p for p in [*net.parameters(), *objective.parameters()] if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    return {"metric_loss": float(loss.detach()), **policy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(RUNS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("new output required, or explicit resume")
    pins, sources = inputs_and_sources()
    config = {**CONFIG, "mode": args.mode}
    contract = digest({"config": config, "inputs": pins, "sources": sources})
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.out / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest["contract_sha256"] != contract or sha(args.out / "initial.pt") != INITIAL_SHA:
            raise ValueError("resume contract/initial state differs")
        if manifest["status"] == "completed":
            if sha(args.out / "checkpoint.pt") != manifest["checkpoint_sha256"]:
                raise ValueError("completed checkpoint changed")
            print("already completed, checkpoint verified")
            return
    else:
        shutil.copy2(INITIAL, args.out / "initial.pt")
        for i, name in enumerate(sources):
            path = Path(name)
            target = args.out / "source_snapshot" / f"{i:03d}-{path.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        manifest = {
            "status": "preparing",
            "completed_steps": 0,
            "config": config,
            "inputs": pins,
            "sources": sources,
            "contract_sha256": contract,
            "initial_checkpoint_sha256": INITIAL_SHA,
            "value_head_trained": False,
        }
    manifest.update(status="preparing", pid=os.getpid(), phase="policy_targets")
    atomic(manifest_path, manifest)
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    initial = torch.load(args.out / "initial.pt", map_location="cpu", weights_only=False)
    architecture = initial["architecture"]
    net = M.SitePolicyValueNet(**architecture)
    net.load_state_dict(initial["model"], strict=True)
    net.value.requires_grad_(False)
    objective = MetricEmbeddingObjective(cc_rank_weight=0.1, witness_priority=False)
    objective.load_state_dict(initial["objective"], strict=True)
    positives, raw_preferences = load_policy_targets(SUPERVISION)
    net.eval()
    with use_adjacency(bulk_crossing_adjacency):
        preferences = reference_preferences(net, raw_preferences, config["policy_micro_batch"])
    manifest.update(
        phase="metric_inputs",
        architecture=architecture,
        positive_conditions=len(positives),
        preference_conditions=len(preferences),
    )
    atomic(manifest_path, manifest)
    original, suffix = load_metric_samplers(CORPUS, SUPERVISION)
    graph = load_graph_sampler(GRAPH)
    manifest["metric_support"] = {
        "original": original.support,
        "suffix": suffix.support,
        "graph": graph.support,
    }
    parameters = [p for p in [*net.parameters(), *objective.parameters()] if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    rng = random.Random(config["seed"])
    completed = 0
    if args.resume and (args.out / "progress.pt").exists():
        progress = torch.load(args.out / "progress.pt", map_location="cpu", weights_only=False)
        completed = validate_progress(
            progress,
            manifest,
            contract,
            architecture,
            sha(args.out / "progress.pt"),
            config["steps"],
        )
        net.load_state_dict(progress["model"], strict=True)
        objective.load_state_dict(progress["objective"], strict=True)
        optimizer.load_state_dict(progress["optimizer"])
        rng.setstate(progress["rng"])
        torch.set_rng_state(progress["torch_rng"])
    finite_state(net, objective)
    finite_optimizer(optimizer)
    sentinel = pair_row(-1, "mirror", ((1, 1, 1), 2), ((-1, -1, -1), 2), 2, 2, 6)
    started = time.monotonic()
    manifest.update(status="training", phase="optimizer", completed_steps=completed)
    atomic(manifest_path, manifest)
    with use_adjacency(bulk_crossing_adjacency):
        for step in range(completed, config["steps"]):
            metrics, chosen, relative = paired_draws(
                rng, original, suffix, graph, positives, preferences, args.mode
            )
            losses = optimize_step(
                net, objective, optimizer, metrics + [sentinel], chosen, relative, config
            )
            if (step + 1) % config["checkpoint_interval"] == 0:
                finite_state(net, objective)
                finite_optimizer(optimizer)
                temporary = args.out / "progress.partial"
                torch.save(
                    {
                        "model": net.state_dict(),
                        "objective": objective.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "architecture": architecture,
                        "steps": step + 1,
                        "rng": rng.getstate(),
                        "torch_rng": torch.get_rng_state(),
                        "contract_sha256": contract,
                    },
                    temporary,
                )
                temporary.replace(args.out / "progress.pt")
                check_pins(pins)
                check_pins(sources)
                manifest.update(
                    completed_steps=step + 1,
                    **losses,
                    progress_sha256=sha(args.out / "progress.pt"),
                    metric_sampled_rows=128 * (step + 1),
                    policy_sampled_states=32 * (step + 1),
                    graph_sampled_rows=64 * (step + 1) if args.mode == "graph" else 0,
                    seconds_this_run=time.monotonic() - started,
                )
                atomic(manifest_path, manifest)
                print(
                    json.dumps(
                        {
                            k: manifest[k]
                            for k in (
                                "completed_steps",
                                "metric_loss",
                                "positive_loss",
                                "relative_loss",
                                "seconds_this_run",
                            )
                        }
                    ),
                    flush=True,
                )
    final = args.out / "checkpoint.pt"
    if final.exists() and sha(final) != sha(args.out / "progress.pt"):
        raise ValueError("refusing to overwrite different final checkpoint")
    if not final.exists():
        shutil.copy2(args.out / "progress.pt", final)
    manifest.update(status="completed", phase="completed", checkpoint_sha256=sha(final))
    atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
