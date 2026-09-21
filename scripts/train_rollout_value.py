"""One fixed-data joint policy/value/metric improvement iteration.

Value evaluates frozen rollout actor pi0, not pi1 or a 512-simulation MCTS.
Relative policy terms use complete same-root/resource/lambda groups, exact
prefix cancellation, and a PPO-style clipped ratio (not vanilla PPO/GRPO).
All shared encoder/head parameters train. No residual output adapter is added.
"""

import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from balanced_metric_sampling import GroupBalancedMetricSampler, prepare_rows
from run_strand_graph_pilot import (
    atomic_json,
    load_frozen_corpus,
    metric_loss,
    metric_report,
    pair_row,
)

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.embedding_metric import MetricEmbeddingObjective
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.rollout_value import (
    clipped_relative_loss,
    episode_states,
    relative_terms,
    rollout_candidates,
    verify_episode,
)
from rf_knots.site_head import SiteHead


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forward(net, requests):
    batch = M.build_batch(requests, "cpu", budget_dim=net.budget_dim)
    logits, values, mask = net(**batch)
    return SiteHead.masked_log_policy(logits, mask), values


def validate_episode_coverage(episodes, root_count, rollouts):
    if root_count < 1 or rollouts < 8:
        raise ValueError("require roots and at least eight complete rollout seeds")
    groups = defaultdict(list)
    for episode in episodes:
        if not 0 <= episode["group"] < root_count:
            raise ValueError("unexpected rollout group")
        groups[episode["group"]].append(episode["seed"])
    if set(groups) != set(range(root_count)) or any(
        sorted(seeds) != list(range(rollouts)) for seeds in groups.values()
    ):
        raise ValueError("missing or duplicate rollout seeds/groups")


def selected_policy_chunks(net, requests, indices, micro_batch):
    if micro_batch < 1 or not requests or len(requests) != len(indices):
        raise ValueError("invalid policy micro-batch or empty/mismatched group")
    for start in range(0, len(requests), micro_batch):
        stop = min(len(requests), start + micro_batch)
        log_prob, _ = forward(net, requests[start:stop])
        yield start, stop, log_prob[torch.arange(stop-start), indices[start:stop]]


def relative_policy_backward(net, group, *, weight=0.1, micro_batch=64, clip=0.2):
    """Exact sum across a previously prefix-cancelled group; never per-route means."""
    requests, indices, old_prob, coefficients = group
    total, ratio_sum, clipped = 0.0, 0.0, 0
    for start, stop, selected in selected_policy_chunks(net, requests, indices, micro_batch):
        loss = clipped_relative_loss(selected, old_prob[start:stop], coefficients[start:stop], clip)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite relative policy loss")
        (weight * loss).backward()
        total += float(loss.detach())
        delta = selected.detach() - old_prob[start:stop]
        ratio_sum += float(delta.abs().sum())
        clipped += int((delta.exp().sub(1).abs() > clip).sum())
    return {"loss": total, "mean_abs_log_ratio": ratio_sum / len(indices),
            "clip_fraction": clipped / len(indices)}


@torch.no_grad()
def value_report(net, instances, episodes, constant):
    grouped = defaultdict(list)
    for e in episodes:
        if e["seed"] >= 6:
            grouped[e["group"]].append(e)
    net.eval()
    rows = []
    for group, trials in sorted(grouped.items()):
        problem = instances[group]
        state = M.SearchState(problem.word, problem.strands, 0, 0)
        _, logits = forward(net, [(problem, state, ())])
        probability = float(logits.sigmoid()[0])
        rows.append(
            {
                "group": group,
                "probability": probability,
                "trials": len(trials),
                "successes": sum(e["success"] for e in trials),
            }
        )
    count = sum(r["trials"] for r in rows)

    def brier(p, r):
        return r["successes"] * (1 - p) ** 2 + (r["trials"] - r["successes"]) * p**2

    return {
        "split": "fresh rollout seeds on SAME training roots, NOT knot-disjoint test",
        "trials": count,
        "successes": sum(r["successes"] for r in rows),
        "brier": sum(brier(r["probability"], r) for r in rows) / count,
        "constant_training_root_rate": constant,
        "constant_brier": sum(brier(constant, r) for r in rows) / count,
        "rows": rows,
        "caveat": "All heldout-seed outcomes included, including paths censored from TRAIN. "
        "Training-path exclusions may induce selection bias. Target behaviour remains pi0.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--metric-corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--window-size", type=int, default=0)
    p.add_argument("--hierarchy-levels", type=int, default=0)
    p.add_argument("--policy-micro-batch", type=int, default=64)
    p.add_argument("--min-relative-groups", type=int, default=1)
    p.add_argument("--min-long-relative-groups", type=int, default=0)
    p.add_argument("--long-root-length", type=int, default=96)
    p.add_argument("--metric-bounds", choices=("signature", "determinant"), default="signature")
    p.add_argument("--metric-sampling", choices=("uniform", "group-balanced"), default="uniform")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.policy_micro_batch < 1 or args.min_relative_groups < 1 or args.min_long_relative_groups < 0 or args.long_root_length < 1:
        raise ValueError("invalid training micro-batch or adequacy gate")
    if bool(args.window_size) != bool(args.hierarchy_levels):
        raise ValueError("--window-size and --hierarchy-levels must be enabled together")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"
    }
    config.update(batch_size=64, lr=2.5e-5, policy_weight=0.1, clip=0.2, train_seeds=list(range(6)))
    data_manifest = json.loads((args.data / "manifest.json").read_text())
    if data_manifest["status"] != "completed" or data_manifest.get("purpose") != "training":
        raise ValueError("require completed rollout corpus")
    inputs = {
        str(path): sha(path)
        for path in (
            args.data / "manifest.json",
            args.data / "actor.pt",
            args.data / "roots.json",
            args.metric_corpus,
        )
    }
    sources = {
        str(path.resolve().relative_to(Path.cwd())): sha(path)
        for path in (
            [Path(__file__), Path("scripts/run_strand_graph_pilot.py"), Path("scripts/balanced_metric_sampling.py")]
            + sorted(Path("src/rf_knots").rglob("*.py"))
        )
    }
    manifest_path = args.out / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous:
        if not args.resume or (previous["config"], previous["inputs"], previous["sources"]) != (
            config,
            inputs,
            sources,
        ):
            raise ValueError("existing run or changed resume contract")
        if previous["status"] == "completed":
            print("already completed")
            return
    elif args.resume:
        raise ValueError("nothing to resume")
    if (
        inputs[str(args.data / "actor.pt")] != data_manifest["actor_sha256"]
        or inputs[str(args.data / "roots.json")] != data_manifest["roots_sha256"]
    ):
        raise ValueError("rollout input hashes differ")
    episodes = []
    for shard in data_manifest["shards"]:
        path = args.data / shard["file"]
        if sha(path) != shard["sha256"]:
            raise ValueError("episode shard changed")
        episodes.extend(json.loads(path.read_text()))
    roots = json.loads((args.data / "roots.json").read_text())
    validate_episode_coverage(episodes, len(roots), data_manifest["config"]["rollouts"])
    instances = [
        M.Instance(**{**r["instance"], "word": tuple(r["instance"]["word"])}) for r in roots
    ]
    preparation = {"status": "preparing", "pid": os.getpid(), "inputs": inputs,
                   "phase": "replaying_episodes", "episode_count": len(episodes), "verified": 0}
    atomic_json(args.out / "preparation.json", preparation)
    values, grouped = [], defaultdict(list)
    for episode_index, e in enumerate(episodes):
        problem = instances[e["group"]]
        verify_episode(e, problem)
        if (episode_index + 1) % 256 == 0:
            preparation["verified"] = episode_index + 1
            atomic_json(args.out / "preparation.json", preparation)
            print(json.dumps({"phase": preparation["phase"], "verified": episode_index+1}), flush=True)
        if e["seed"] >= 6:
            continue
        grouped[e["group"]].append(e)
        if not e["eligible"]:
            continue
        fp = invariants(problem.word, problem.strands)
        if owner(fp) != "train":
            raise ValueError("non-train root")
        for state, action in episode_states(e, problem):
            if owner(fp) != "train":
                raise ValueError("non-train intermediate state")
            values.append((problem, state, (), float(e["success"])))
            if problem.spec.decode(action)[0] == CROSSING_CHANGE:
                from rf_knots import reference

                word, n = reference.apply(problem.spec, state.word, state.strands, action)
                fp = invariants(word, n)
                if owner(fp) != "train":
                    raise ValueError("non-train crossing endpoint")
    terms = []
    preparation.update(phase="building_relative_groups", verified=len(episodes))
    atomic_json(args.out / "preparation.json", preparation)
    for group, trials in sorted(grouped.items()):
        if len(trials) != 6:
            raise ValueError("missing episode in fixed training seed group")
        entries = relative_terms(trials, instances[group])
        if entries:
            terms.append((instances[group], entries))
    if not values or not terms:
        preparation.update(status="insufficient_supervision", phase="supervision_gate",
                           value_rows=len(values), relative_groups=len(terms))
        atomic_json(args.out / "preparation.json", preparation)
        raise ValueError("no value or relative-policy supervision")
    long_relative_groups = sum(len(problem.word) >= args.long_root_length for problem, _ in terms)
    if len(terms) < args.min_relative_groups or long_relative_groups < args.min_long_relative_groups:
        preparation.update(status="insufficient_supervision", phase="supervision_gate",
                           value_rows=len(values), relative_groups=len(terms),
                           long_relative_groups=long_relative_groups)
        atomic_json(args.out / "preparation.json", preparation)
        raise ValueError(f"insufficient relative supervision: {len(terms)} groups, {long_relative_groups} long groups")
    actor = torch.load(args.data / "actor.pt", map_location="cpu", weights_only=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    architecture = dict(actor["architecture"])
    if args.window_size:
        architecture.update(window_size=args.window_size, hierarchy_levels=args.hierarchy_levels)
    net = M.SitePolicyValueNet(**architecture)
    initial_load = net.load_state_dict(actor["model"], strict=not bool(args.window_size))
    objective = MetricEmbeddingObjective(cc_rank_weight=0.1, witness_priority=False)
    objective.load_state_dict(actor["objective"], strict=True)
    # Old per-action probabilities are pinned to pi0, recomputed before updates.
    policy_groups = []
    preparation.update(phase="behaviour_log_probabilities", relative_groups=len(terms), prepared_groups=0)
    atomic_json(args.out / "preparation.json", preparation)
    net.eval()
    with torch.no_grad():
        for problem, entries in terms:
            requests = [(problem, s, rollout_candidates(problem, s)) for s, _, _ in entries]
            indices = torch.tensor(
                [r[2].index(a) for r, (_, a, _) in zip(requests, entries, strict=True)]
            )
            selected = torch.cat([chunk for _, _, chunk in
                selected_policy_chunks(net, requests, indices, args.policy_micro_batch)])
            policy_groups.append(
                (requests, indices, selected.clone(), torch.tensor([w for _, _, w in entries]))
            )
            if len(policy_groups) % 16 == 0:
                preparation["prepared_groups"] = len(policy_groups)
                atomic_json(args.out / "preparation.json", preparation)
    preparation.update(phase="metric_corpus", prepared_groups=len(policy_groups))
    atomic_json(args.out / "preparation.json", preparation)
    raw_corpus, corpus = load_frozen_corpus(args.metric_corpus)
    metric_pool = prepare_rows(corpus["train"][0], raw_corpus["train"], bounds=args.metric_bounds)
    metric_sampler = (GroupBalancedMetricSampler(metric_pool, raw_corpus["train"])
                      if args.metric_sampling == "group-balanced" else None)
    atomic_json(args.out / "metric_sampling.json", {
        "bounds": args.metric_bounds, "sampling": args.metric_sampling,
        "support": metric_sampler.support if metric_sampler else None,
        "random_draws_per_sample": 3 if metric_sampler else 1,
        "contract": "63 metric samples plus the fixed mirror-distance-two anchor; TRAIN only"})
    validation, large = corpus["validation"][0], corpus["size_shift"][0]
    mirror = pair_row(-1, "mirror", ((1, 1, 1), 2), ((-1, -1, -1), 2), 2, 2, 6)
    train_episodes = [e for e in episodes if e["seed"] < 6 and e["eligible"]]
    constant = sum(e["success"] for e in train_episodes) / len(train_episodes)
    optimizer = torch.optim.AdamW(
        list(net.parameters()) + list(objective.parameters()), lr=config["lr"]
    )
    rng = random.Random(args.seed)
    resume = (
        torch.load(args.out / "checkpoint.pt", map_location="cpu", weights_only=False)
        if previous
        else None
    )
    manifest = previous or {
        "schema": "rf-knots-rollout-joint-v1",
        "config": config,
        "inputs": inputs,
        "sources": sources,
        "value_rows": len(values),
        "relative_groups": len(policy_groups),
        "long_relative_groups": long_relative_groups,
        "architecture": architecture,
        "initial_random_initialized_keys": list(initial_load.missing_keys),
        "contract": __doc__,
        "completed_steps": 0,
    }
    if resume:
        if sha(args.out / "initial.pt") != manifest["initial_checkpoint_sha256"]:
            raise ValueError("saved initial tensors changed")
        net.load_state_dict(resume["model"])
        objective.load_state_dict(resume["objective"])
        optimizer.load_state_dict(resume["optimizer"])
        rng.setstate(resume["rng"])
        torch.set_rng_state(resume["torch_rng"])
    else:
        torch.save({"architecture": architecture, "model": net.state_dict(),
                    "objective": objective.state_dict()}, args.out / "initial.pt")
        manifest["initial_checkpoint_sha256"] = sha(args.out / "initial.pt")
        for name in sources:
            target = args.out / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, target)
        atomic_json(
            args.out / "before.json",
            {
                "validation": metric_report(net, objective, validation, "cpu"),
                "rollout_value": value_report(net, instances, episodes, constant),
            },
        )
    manifest.update(status="training", pid=os.getpid())
    atomic_json(manifest_path, manifest)
    preparation.update(status="completed", phase="training_started")
    atomic_json(args.out / "preparation.json", preparation)
    started = time.time()
    for step in range(resume["steps"] if resume else 0, args.steps):
        net.train()
        chosen = rng.choices(values, k=64)
        group = rng.choice(policy_groups)
        metric_batch = (metric_sampler.sample(rng, 63) if metric_sampler
                        else rng.choices(metric_pool, k=63)) + [mirror]
        optimizer.zero_grad(set_to_none=True)
        _, logits = forward(net, [r[:3] for r in chosen])
        value_loss = F.binary_cross_entropy_with_logits(
            logits, torch.tensor([r[3] for r in chosen])
        )
        distance_loss, _ = metric_loss(net, objective, metric_batch, "cpu")
        loss = args.value_weight * value_loss + distance_loss
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        policy = relative_policy_backward(net, group, weight=config["policy_weight"],
                                          micro_batch=args.policy_micro_batch, clip=config["clip"])
        torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(objective.parameters()), 1.0,
                                       error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 16 == 0 or step + 1 == args.steps:
            temp = args.out / "checkpoint.partial"
            torch.save(
                {
                    "model": net.state_dict(),
                    "objective": objective.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "architecture": architecture,
                    "steps": step + 1,
                    "rng": rng.getstate(),
                    "torch_rng": torch.get_rng_state(),
                },
                temp,
            )
            temp.replace(args.out / "checkpoint.pt")
            manifest.update(
                completed_steps=step + 1,
                value_loss=float(value_loss.detach()),
                policy_loss=policy["loss"],
                metric_loss=float(distance_loss.detach()),
                policy_mean_abs_log_ratio=policy["mean_abs_log_ratio"],
                policy_clip_fraction=policy["clip_fraction"],
                seconds_this_run=time.time() - started,
            )
            atomic_json(manifest_path, manifest)
            print(
                json.dumps(
                    {
                        k: manifest[k]
                        for k in (
                            "completed_steps",
                            "value_loss",
                            "policy_loss",
                            "metric_loss",
                            "seconds_this_run",
                        )
                    }
                ),
                flush=True,
            )
    atomic_json(
        args.out / "after.json",
        {
            "validation": metric_report(net, objective, validation, "cpu"),
            "size_shift": metric_report(net, objective, large, "cpu"),
            "rollout_value": value_report(net, instances, episodes, constant),
        },
    )
    if any(sha(Path(name)) != digest for name, digest in sources.items()):
        raise ValueError("training source changed during execution")
    manifest.update(status="completed", checkpoint_sha256=sha(args.out / "checkpoint.pt"))
    atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
