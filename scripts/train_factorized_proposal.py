"""Train the opt-in factorized proposal head on replayed relative supervision.

The encoder and historical SiteHead/value are frozen.  Only factorized
kind/position/generator/sign scores are updated.  Unopened actions never enter
the loss: each row's denominator is exactly the frozen behavioural proposal set
and labels come only from replayed relative winner/loser terms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from run_strand_graph_pilot import atomic_json
from rf_knots import mcts_loop as M
from rf_knots.lazy_actions import factorized_action_candidates
from rf_knots.rollout_value import (
    episode_states,
    relative_terms,
    rollout_candidates,
    verify_episode,
)
from rf_knots.site_head import SiteHead, masked_log_policy


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_relative_loss(log_prob, old_log_prob, coefficients, clip: float = 0.2):
    """Sign-correct clipped relative objective for positive and loser weights."""
    ratio = torch.exp((log_prob - old_log_prob).clamp(-20, 20))
    clipped = ratio.clamp(1 - clip, 1 + clip)
    unclipped = ratio * coefficients
    clipped_term = clipped * coefficients
    surrogate = torch.where(
        coefficients >= 0,
        torch.minimum(unclipped, clipped_term),
        torch.maximum(unclipped, clipped_term),
    )
    return -surrogate.sum()


def factor_log_probs(net, requests):
    batch = M.build_batch(requests, "cpu", budget_dim=net.budget_dim)
    with torch.no_grad():
        encoded = net.encoder.encode_sites(
            batch["word"], batch["lengths"], batch["strands"]
        )
    globals_ = {name: encoded[name] for name in ("knot", "isotopy", "layout")}
    factors = net.factorized_scores(encoded, globals_, batch["budgets"], batch["strands"])
    scores = net.factorized_head.action_scores(factors, batch["sites"])
    return masked_log_policy(scores, batch["site_mask"]), batch


@torch.no_grad()
def proposal_recall(net, samples, top_positions: int, top_generators: int) -> dict:
    hits = 0
    total = 0
    proposed = 0
    net.eval()
    for problem, state, action in samples:
        log_probs, batch = factor_log_probs(
            net, [(problem, state, rollout_candidates(problem, state))]
        )
        encoded = net.encoder.encode_sites(
            batch["word"], batch["lengths"], batch["strands"]
        )
        globals_ = {name: encoded[name] for name in ("knot", "isotopy", "layout")}
        factors = net.factorized_scores(encoded, globals_, batch["budgets"], batch["strands"])
        row = {
            key: value[0].detach().cpu().numpy()
            for key, value in factors.items()
            if key.endswith("_scores")
        }
        proposals = factorized_action_candidates(
            problem.spec,
            state.word,
            state.strands,
            kind_scores=row["kind_scores"],
            position_scores=row["position_scores"],
            generator_scores=row["generator_scores"],
            sign_scores=row["sign_scores"],
            top_positions=top_positions,
            top_generators=top_generators,
        )
        proposed_actions = {candidate for candidate, _ in proposals}
        hits += int(action in proposed_actions)
        proposed += len(proposals)
        total += 1
    return {
        "samples": total,
        "behavior_action_recall": hits / max(total, 1),
        "mean_proposed": proposed / max(total, 1),
        "top_positions": top_positions,
        "top_generators": top_generators,
        "caveat": "heldout-seed behavior actions are not optimal labels and remain same-root diagnostics",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("refusing to overwrite an existing proposal run")
    args.out.mkdir(parents=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.set_num_threads(1)

    data_manifest = json.loads((args.data / "manifest.json").read_text())
    base_manifest = json.loads((args.base_run / "manifest.json").read_text())
    if data_manifest.get("status") != "completed" or base_manifest.get("status") != "completed":
        raise ValueError("both data and base run must be completed")
    base_sha = sha(args.base_run / "checkpoint.pt")
    if base_sha != base_manifest["checkpoint_sha256"]:
        raise ValueError("base checkpoint hash does not match its manifest")

    inputs = {
        str(path): sha(path)
        for path in (
            args.data / "manifest.json",
            args.data / "actor.pt",
            args.data / "roots.json",
            args.base_run / "manifest.json",
            args.base_run / "checkpoint.pt",
        )
    }
    episodes = []
    for shard in data_manifest["shards"]:
        path = args.data / shard["file"]
        if sha(path) != shard["sha256"]:
            raise ValueError("episode shard changed")
        episodes.extend(json.loads(path.read_text()))
    roots = json.loads((args.data / "roots.json").read_text())
    instances = [
        M.Instance(**{**row["instance"], "word": tuple(row["instance"]["word"])})
        for row in roots
    ]

    grouped = defaultdict(list)
    heldout_samples = []
    for episode in episodes:
        problem = instances[episode["group"]]
        verify_episode(episode, problem)
        if episode["seed"] < 6:
            grouped[episode["group"]].append(episode)
        elif episode["success"]:
            heldout_samples.extend(
                (problem, state, action)
                for state, action in episode_states(episode, problem)
            )
    terms = []
    for group, trials in sorted(grouped.items()):
        if len(trials) != 6:
            raise ValueError("missing fixed training seed in a group")
        entries = relative_terms(trials, instances[group])
        if entries:
            terms.append((instances[group], entries))
    if not terms:
        raise ValueError("no relative proposal terms")

    base = torch.load(args.base_run / "checkpoint.pt", map_location="cpu", weights_only=False)
    architecture = {**base["architecture"], "factorized_actions": True}
    net = M.SitePolicyValueNet(**architecture)
    loaded = net.load_state_dict(base["model"], strict=False)
    if any(not key.startswith("factorized_head.") for key in loaded.missing_keys):
        raise ValueError(f"unexpected missing base parameters: {loaded.missing_keys}")
    if loaded.unexpected_keys:
        raise ValueError(f"unexpected base parameters: {loaded.unexpected_keys}")
    for name, parameter in net.named_parameters():
        parameter.requires_grad = name.startswith("factorized_head.")
    optimizer = torch.optim.AdamW(net.factorized_head.parameters(), lr=1e-4)

    # Pin the old factorized probabilities before any update for PPO-style
    # relative comparisons.  Every denominator is the replayed proposal set.
    pinned = []
    for problem, entries in terms:
        requests = [(problem, state, rollout_candidates(problem, state)) for state, _, _ in entries]
        actions = [action for _, action, _ in entries]
        indices = torch.tensor(
            [request[2].index(action) for request, action in zip(requests, actions, strict=True)]
        )
        log_prob, _ = factor_log_probs(net, requests)
        pinned.append(
            (
                requests,
                indices,
                log_prob[torch.arange(len(indices)), indices].detach(),
                torch.tensor([weight for _, _, weight in entries], dtype=torch.float32),
            )
        )

    report_before = {
        "train_groups": len(pinned),
        "train_terms": sum(len(group[1]) for group in terms),
        "heldout_behavior_recall": proposal_recall(net, heldout_samples[:512], 8, 8),
        "base_checkpoint_sha256": base_sha,
        "random_initialized_keys": list(loaded.missing_keys),
    }
    atomic_json(args.out / "before.json", report_before)
    sources = (
        Path(__file__).resolve(),
        Path("src/rf_knots/site_head.py").resolve(),
        Path("src/rf_knots/mcts_loop.py").resolve(),
    )
    for source in sources:
        relative = source.relative_to(Path.cwd())
        target = args.out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    manifest = {
        "schema": "rf-knots-factorized-proposal-v1",
        "status": "training",
        "pid": os.getpid(),
        "config": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "lr": 1e-4,
            "top_positions": 8,
            "top_generators": 8,
            "loss_contract": "sign-correct clipped relative objective",
        },
        "inputs": inputs,
        "architecture": architecture,
        "base_checkpoint_sha256": base_sha,
        "random_initialized_keys": list(loaded.missing_keys),
        "train_groups": len(pinned),
        "train_terms": sum(len(group[1]) for group in terms),
        "completed_steps": 0,
    }
    manifest_path = args.out / "manifest.json"
    atomic_json(manifest_path, manifest)
    started = time.time()
    for step in range(args.steps):
        requests, indices, old_log_prob, coefficients = rng.choice(pinned)
        net.train()
        log_prob, _ = factor_log_probs(net, requests)
        selected = log_prob[torch.arange(len(indices)), indices]
        loss = stable_relative_loss(selected, old_log_prob, coefficients) / max(len(indices), 1)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite factorized proposal loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.factorized_head.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 16 == 0 or step + 1 == args.steps:
            checkpoint = {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "architecture": architecture,
                "steps": step + 1,
                "seed": args.seed,
            }
            torch.save(checkpoint, args.out / "checkpoint.partial")
            (args.out / "checkpoint.partial").replace(args.out / "checkpoint.pt")
            manifest.update(
                completed_steps=step + 1,
                loss=float(loss.detach()),
                seconds_this_run=time.time() - started,
            )
            atomic_json(manifest_path, manifest)
            print(json.dumps({k: manifest[k] for k in ("completed_steps", "loss", "seconds_this_run")}), flush=True)

    checkpoint_sha = sha(args.out / "checkpoint.pt")
    net.eval()
    after = {
        "heldout_behavior_recall": proposal_recall(net, heldout_samples[:512], 8, 8),
        "train_groups": len(pinned),
        "train_terms": sum(len(group[1]) for group in terms),
    }
    atomic_json(args.out / "after.json", after)
    manifest.update(status="completed", checkpoint_sha256=checkpoint_sha)
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "completed", "checkpoint_sha256": checkpoint_sha, "after": after}, indent=2))


if __name__ == "__main__":
    main()
