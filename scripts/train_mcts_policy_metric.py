"""Protocol-57 fixed-data encoder/policy/metric experiment, no value relabeling."""

import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import torch
from mcts_joint_training import (
    load_metric_samplers,
    load_policy_targets,
    policy_backward,
    reference_preferences,
    sample_step,
    sha,
)
from run_strand_graph_pilot import atomic_json, metric_loss, pair_row

from rf_knots import mcts_loop as M
from rf_knots.embedding_metric import MetricEmbeddingObjective

SUPERVISION = Path("artifacts/diverse-mcts-supervision-diamonds-v1")
CORPUS = Path("artifacts/embedding-diverse-short-u-corpus-v1/corpus.json")
ACTOR = Path("artifacts/diverse-cost-mcts-training-v1/actor.pt")
ACTOR_SHA = "964199130db5645e41195905f8629636207fcf40e4bcf2a5683b9319fec7044e"
CORPUS_SHA = "f3655e7dbfcc5b6ec7448b1f747dc2a4e1235fe036ab07fec61cc18df05c8f14"


def finite_state(net, objective):
    if not all(
        torch.isfinite(t).all()
        for t in [*net.state_dict().values(), *objective.state_dict().values()]
    ):
        raise ValueError("nonfinite model/objective tensors")


def validate_progress(progress, manifest, contract, architecture, progress_sha, steps=1024):
    completed = progress["steps"]
    if (
        progress["contract_sha256"] != contract
        or progress["architecture"] != architecture
        or not manifest["completed_steps"]
        <= completed
        <= min(steps, manifest["completed_steps"] + 16)
    ):
        raise ValueError("progress checkpoint contract/step differs")
    if completed == manifest["completed_steps"] and progress_sha != manifest["progress_sha256"]:
        raise ValueError("committed progress hash changed")
    if completed > manifest["completed_steps"]:
        manifest["recovered_checkpoint_ahead_of_manifest"] = completed
    # Publish step and hash together, including recovery of an atomic checkpoint
    # written just before a crash. A second interruption must remain resumable.
    manifest.update(completed_steps=completed, progress_sha256=progress_sha)
    return completed


def finite_optimizer(optimizer):
    if any(
        not torch.isfinite(value).all()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ):
        raise ValueError("nonfinite optimizer tensors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("joint", "metric-control"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("existing output requires explicit resume; new output cannot resume")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {
        "mode": args.mode,
        "steps": 1024,
        "seed": 20261023,
        "device": "cpu",
        "threads": 1,
        "lr": 2.5e-5,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "clip": 0.2,
        "positive_weight": 0.1,
        "relative_weight": 0.01,
        "policy_micro_batch": 64,
        "metric_original": 31,
        "metric_suffix": 32,
        "policy_batch": 32,
    }
    supervision = json.loads((SUPERVISION / "manifest.json").read_text())
    audit_path = SUPERVISION / "certificate-audit.json"
    audit = json.loads(audit_path.read_text())
    if supervision["status"] != "completed" or audit["status"] != "passed":
        raise ValueError("require completed proof-augmented data and passed audit")
    inputs = {
        **audit["inputs"],
        str(audit_path): sha(audit_path),
        str(CORPUS): CORPUS_SHA,
        str(ACTOR): ACTOR_SHA,
    }
    if inputs.get(str(SUPERVISION / "manifest.json")) != sha(SUPERVISION / "manifest.json"):
        raise ValueError("certificate audit is not linked to supervision")
    if any(sha(Path(p)) != h for p, h in inputs.items()):
        raise ValueError("pinned supervision/actor/corpus input changed")
    source_paths = sorted(Path("src/rf_knots").glob("*.py")) + [
        Path(__file__).resolve(),
        Path("scripts/mcts_joint_training.py"),
        Path("scripts/mcts_training_supervision.py"),
        Path("scripts/build_mcts_training_supervision.py"),
        Path("scripts/balanced_metric_sampling.py"),
        Path("scripts/run_strand_graph_pilot.py"),
        Path("research/57-joint-policy-metric-training-protocol.md"),
    ]
    sources = {str(p): sha(p) for p in source_paths}
    contract = hashlib.sha256(
        json.dumps([config, inputs, sources], sort_keys=True).encode()
    ).hexdigest()
    manifest_path = args.out / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest["contract_sha256"] != contract:
            raise ValueError("resume contract changed")
        if sha(args.out / "initial.pt") != ACTOR_SHA:
            raise ValueError("initial tensors changed")
        if manifest["status"] == "completed":
            if sha(args.out / "checkpoint.pt") != manifest["checkpoint_sha256"]:
                raise ValueError("completed checkpoint changed")
            print("already completed")
            return
    else:
        shutil.copy2(ACTOR, args.out / "initial.pt")
        for p in source_paths:
            relative = p.relative_to(Path.cwd()) if p.is_absolute() else p
            destination = args.out / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, destination)
        manifest = {
            "status": "preparing",
            "config": config,
            "inputs": inputs,
            "sources": sources,
            "contract_sha256": contract,
            "initial_checkpoint_sha256": ACTOR_SHA,
            "completed_steps": 0,
            "value_head_trained": False,
        }
    manifest.update(status="preparing", pid=os.getpid(), phase="policy_targets")
    atomic_json(manifest_path, manifest)
    torch.set_num_threads(1)
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
    preferences = reference_preferences(net, raw_preferences, config["policy_micro_batch"])
    manifest.update(
        phase="metric_source_replay",
        positive_conditions=len(positives),
        preference_conditions=len(preferences),
        architecture=architecture,
    )
    atomic_json(manifest_path, manifest)
    original_sampler, suffix_sampler = load_metric_samplers(CORPUS, SUPERVISION)
    manifest["metric_support"] = {
        "original": original_sampler.support,
        "suffix": suffix_sampler.support,
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
    mirror = pair_row(-1, "mirror", ((1, 1, 1), 2), ((-1, -1, -1), 2), 2, 2, 6)
    started = time.monotonic()
    manifest.update(status="training", phase="optimizer", completed_steps=completed)
    atomic_json(manifest_path, manifest)
    for step in range(completed, config["steps"]):
        net.train()
        metrics, chosen, relative = sample_step(
            rng, original_sampler, suffix_sampler, positives, preferences
        )
        optimizer.zero_grad(set_to_none=True)
        loss, _ = metric_loss(net, objective, metrics + [mirror], "cpu")
        if not torch.isfinite(loss):
            raise ValueError("nonfinite metric loss")
        loss.backward()
        policy = {"positive_loss": None, "relative_loss": None}
        if args.mode == "joint":
            policy = policy_backward(
                net,
                chosen,
                relative,
                positive_weight=config["positive_weight"],
                relative_weight=config["relative_weight"],
                micro_batch=config["policy_micro_batch"],
                clip=config["clip"],
            )
        torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 16 == 0:
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
            if any(sha(Path(p)) != h for p, h in {**inputs, **sources}.items()):
                raise ValueError("frozen input/source changed during training")
            manifest.update(
                completed_steps=step + 1,
                metric_loss=float(loss.detach()),
                **policy,
                progress_sha256=sha(args.out / "progress.pt"),
                metric_sampled_rows=64 * (step + 1),
                policy_sampled_states=32 * (step + 1) if args.mode == "joint" else 0,
                seconds_this_run=time.monotonic() - started,
            )
            atomic_json(manifest_path, manifest)
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
        raise ValueError("refusing to overwrite a different final checkpoint")
    if not final.exists():
        shutil.copy2(args.out / "progress.pt", final)
    manifest.update(status="completed", phase="completed", checkpoint_sha256=sha(final))
    atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
