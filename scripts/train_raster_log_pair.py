"""Protocol97 resumable log-preference arm with frozen sources and exclusive writer."""

import argparse
import fcntl
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from build_mcts_training_supervision import atomic
from check_qrun_common_witnesses import check_pins, sha
from graph_metric_sampling import load_graph_sampler
from mcts_joint_training import load_metric_samplers, load_policy_targets
from pretrained_raster_policy import policy_backward, reference_preferences
from profile_pretrained_raster_training import backward_metric
from raster_log_training import (
    CONFIG,
    CORPUS,
    GRAPH,
    SUPERVISION,
    digest,
    draw_batch,
    inputs_and_sources,
    matched_initializations,
    relative_due,
    validate_relative_counts,
)
from train_mcts_policy_metric import finite_optimizer, finite_state, validate_progress
from trajectory_log_preference import relative_backward

from rf_knots.embedding_metric import MetricEmbeddingObjective


def optimize(net, objective, optimizer, rows, positives, relative):
    net.train()
    optimizer.zero_grad(set_to_none=True)
    metric = backward_metric(net, objective, rows, CONFIG["metric_microbatch"])
    losses = policy_backward(
        net,
        positives,
        None,
        microbatch=CONFIG["policy_microbatch"],
        positive_weight=CONFIG["positive_weight"],
    )
    if relative is not None:
        losses.update(
            relative_backward(
                net,
                relative,
                microbatch=CONFIG["policy_microbatch"],
                weight=CONFIG["relative_weight"] * CONFIG["relative_every"],
            )
        )
    norm = torch.nn.utils.clip_grad_norm_(
        [*net.parameters(), *objective.parameters()],
        CONFIG["grad_clip"],
        error_if_nonfinite=True,
    )
    optimizer.step()
    return {"metric_loss": metric, **losses, "gradient_norm_before_clip": float(norm)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("q254", "random"), required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("existing output requires explicit resume")
    pins, sources = inputs_and_sources()
    config = {**CONFIG, "arm": args.arm}
    contract = digest({"config": config, "inputs": pins, "sources": sources})
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.out / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest["contract_sha256"] != contract:
            raise ValueError("resume contract changed")
        if manifest["status"] == "completed":
            if sha(args.out / "checkpoint.pt") != manifest["checkpoint_sha256"]:
                raise ValueError("completed checkpoint hash differs")
            print("already completed; final checkpoint verified", flush=True)
            return
    else:
        manifest = {
            "status": "preparing",
            "completed_steps": 0,
            "config": config,
            "inputs": pins,
            "sources": sources,
            "contract_sha256": contract,
        }
        for i, name in enumerate(sources):
            path = Path(name)
            target = args.out / "source_snapshot" / f"{i:03d}-{path.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    manifest.update(pid=os.getpid(), phase="initialization", status="preparing")
    atomic(manifest_path, manifest)
    torch.set_num_threads(CONFIG["threads"])
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    net, states, architecture, differences = matched_initializations(pins)
    net.load_state_dict(states[args.arm], strict=True)
    objective = MetricEmbeddingObjective()
    initial_path = args.out / "initial.pt"
    if initial_path.exists():
        initial_sha = sha(initial_path)
        expected_sha = manifest.get("initial_checkpoint_sha256")
        if expected_sha is not None and initial_sha != expected_sha:
            raise ValueError("initial checkpoint changed")
        initial = torch.load(initial_path, map_location="cpu", weights_only=False)
        if (
            set(initial["model"]) != set(states[args.arm])
            or any(not torch.equal(v, initial["model"][k]) for k, v in states[args.arm].items())
            or initial["architecture"] != architecture
            or initial["arm"] != args.arm
            or set(initial["objective"]) != set(objective.state_dict())
            or any(
                not torch.equal(v, initial["objective"][k])
                for k, v in objective.state_dict().items()
            )
        ):
            raise ValueError("saved initialization differs from fixed constructor")
        if expected_sha is None:
            if manifest["completed_steps"]:
                raise ValueError("trained run has no committed initial identity")
            manifest.update(
                initial_checkpoint_sha256=initial_sha, recovered_initial_before_manifest=True
            )
    else:
        if manifest.get("initial_checkpoint_sha256") or manifest["completed_steps"]:
            raise ValueError("committed initial checkpoint is missing")
        temporary_initial = args.out / "initial.partial"
        torch.save(
            {
                "model": states[args.arm],
                "objective": objective.state_dict(),
                "architecture": architecture,
                "arm": args.arm,
            },
            temporary_initial,
        )
        temporary_initial.replace(initial_path)
        manifest["initial_checkpoint_sha256"] = sha(initial_path)
    manifest.update(
        architecture=architecture,
        initialization_changed_keys=differences,
        phase="policy_reference",
        value_head_trained=False,
    )
    atomic(manifest_path, manifest)
    positives, raw_preferences = load_policy_targets(SUPERVISION)
    preferences = reference_preferences(
        net, raw_preferences, microbatch=CONFIG["policy_microbatch"]
    )
    group_sizes = [len(group) for group, _ in preferences]
    reference_hash = digest([old.tolist() for _, old in preferences])
    if manifest.get("reference_log_probabilities_sha256", reference_hash) != reference_hash:
        raise ValueError("initial fixed reference probabilities changed")
    manifest.update(
        reference_log_probabilities_sha256=reference_hash,
        phase="metric_inputs",
        positive_conditions=len(positives),
        relative_groups=len(preferences),
    )
    atomic(manifest_path, manifest)
    original, suffix = load_metric_samplers(CORPUS, SUPERVISION)
    graph = load_graph_sampler(GRAPH)
    optimizer = torch.optim.AdamW(
        [*net.parameters(), *objective.parameters()],
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    rng = random.Random(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    completed, draw_digest = 0, digest([])
    relative_counts = [0] * len(preferences)
    progress_path = args.out / "progress.pt"
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        completed = validate_progress(
            progress, manifest, contract, architecture, sha(progress_path), CONFIG["steps"]
        )
        net.load_state_dict(progress["model"], strict=True)
        objective.load_state_dict(progress["objective"], strict=True)
        optimizer.load_state_dict(progress["optimizer"])
        rng.setstate(progress["rng"])
        torch.set_rng_state(progress["torch_rng"])
        draw_digest = progress["draw_digest"]
        relative_counts = progress["relative_group_update_counts"]
        validate_relative_counts(relative_counts, len(preferences), completed)
    elif manifest["completed_steps"]:
        raise ValueError("committed progress is missing")
    finite_state(net, objective)
    finite_optimizer(optimizer)
    started = time.monotonic()
    manifest.update(status="training", phase="optimizer", completed_steps=completed)
    atomic(manifest_path, manifest)
    for step in range(completed, CONFIG["steps"]):
        rows, chosen, relative, step_digest, group_index = draw_batch(
            rng, original, suffix, graph, positives, preferences
        )
        if relative_due(step):
            relative_counts[group_index] += 1
        else:
            relative = None
        losses = optimize(net, objective, optimizer, rows, chosen, relative)
        draw_digest = digest([draw_digest, step_digest])
        if (step + 1) % CONFIG["checkpoint_interval"] == 0:
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
                    "draw_digest": draw_digest,
                    "relative_group_update_counts": relative_counts.copy(),
                    "contract_sha256": contract,
                },
                temporary,
            )
            temporary.replace(progress_path)
            check_pins(pins)
            check_pins(sources)
            manifest.update(
                completed_steps=step + 1,
                progress_sha256=sha(progress_path),
                draw_digest=draw_digest,
                metric_sampled_rows=32 * (step + 1),
                positive_sampled_states=8 * (step + 1),
                relative_updates=sum(relative_counts),
                relative_group_update_counts=relative_counts.copy(),
                relative_group_sizes=group_sizes,
                relative_full_states=sum(
                    n * size for n, size in zip(relative_counts, group_sizes, strict=True)
                ),
                **losses,
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
    if final.exists() and sha(final) != sha(progress_path):
        raise ValueError("refusing to overwrite unrelated final checkpoint")
    if not final.exists():
        shutil.copy2(progress_path, final)
    manifest.update(status="completed", phase="completed", checkpoint_sha256=sha(final))
    atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
