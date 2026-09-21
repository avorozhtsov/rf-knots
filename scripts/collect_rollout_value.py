"""Frozen-policy complete episodes for resource-conditioned Monte Carlo value.

Records stochastic rollout failures, NOT unsolvability or failed-MCTS labels.
Any episode visiting a non-train knot group is retained but excluded from train.
This censoring can bias the accepted distribution and is explicitly reported.
"""

import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from rf_knots import mcts_loop as M
from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.embedding_replay_corpus import invariants, lift, owner, verify_record
from rf_knots.embedding_suffix_supervision import verify_u_view
from rf_knots.evidence import UnknotWitness
from rf_knots.rollout_value import expanded_resource_model, rollout_candidates, verify_episode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, payload):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


def choose_training_views(index, count, seed, *, maximum_moves=64, stratify_size=False):
    """Model-blind round robin across invariant/depth (and optional size) bins."""
    if count < 1 or maximum_moves < 1:
        raise ValueError("positive root count and move limit required")
    buckets = defaultdict(list)
    for i, view in enumerate(index["views"]):
        if not 1 <= view["remaining_moves"] <= maximum_moves:
            continue
        det, sig = view["fingerprint"]
        band = min(3, (view["remaining_moves"] - 1) // 8)
        key = (det, abs(sig), band)
        if stratify_size:
            key += ((len(view["word"])-1)//32, (view["strands"]-1)//8)
        buckets[key].append(i)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    choices = []
    while len(choices) < count:
        before = len(choices)
        for key in sorted(buckets):
            if buckets[key] and len(choices) < count:
                choices.append(buckets[key].pop())
        if len(choices) == before:
            raise ValueError("insufficient distinct train views")
    return choices


def condition_training_root(problem, root_index, *, paired=False):
    """No cross-cost winner/loser group: each condition is its own experiment."""
    if not paired:
        return [problem]
    return [replace(problem, cc_weight=weight, uid=f"rollout-{root_index}-{weight}")
            for weight in (10, 1000)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--index", type=Path)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--evaluation-panel", type=Path)
    p.add_argument("--evaluation-view", choices=("validation", "size_shift"), default="validation")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--roots", type=int, default=128)
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--paired-cc-weights", action="store_true")
    p.add_argument("--max-root-moves", type=int, default=64)
    p.add_argument("--stratify-size", action="store_true")
    p.add_argument("--no-extra-lifts", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.roots < 1 or args.rollouts < 1 or args.max_root_moves < 1:
        raise ValueError("positive roots, rollouts and maximum root moves required")
    if args.evaluation_panel and (args.paired_cc_weights or args.stratify_size or args.no_extra_lifts
                                  or args.max_root_moves != 64):
        raise ValueError("training root-selection options do not apply to evaluation panels")
    if not args.evaluation_panel and (not args.index or not args.corpus):
        raise ValueError("provide train index/corpus or an evaluation-only panel")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"
    }
    inputs = {
        str(path): sha(path)
        for path in (args.run / "checkpoint.pt", args.index, args.corpus, args.evaluation_panel)
        if path is not None
    }
    sources = {
        str(path.resolve().relative_to(Path.cwd())): sha(path)
        for path in [Path(__file__), *sorted(Path("src/rf_knots").rglob("*.py"))]
    }
    manifest_path = args.out / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous:
        if not args.resume:
            raise ValueError("output exists; explicit resume required")
        if (previous["config"], previous["inputs"], previous["sources"]) != (
            config,
            inputs,
            sources,
        ):
            raise ValueError("resume contract changed")
        if previous["status"] == "completed":
            print("already completed")
            return
    elif args.resume:
        raise ValueError("nothing to resume")
    parent = json.loads((args.run / "manifest.json").read_text())
    if (
        parent["status"] != "completed"
        or parent["checkpoint_sha256"] != inputs[str(args.run / "checkpoint.pt")]
    ):
        raise ValueError("actor is not a completed pinned checkpoint")
    checkpoint = torch.load(args.run / "checkpoint.pt", map_location="cpu", weights_only=False)
    torch.set_num_threads(1)
    if checkpoint["architecture"].get("budget_dim", 2) == 5:
        architecture = checkpoint["architecture"]
        net = M.SitePolicyValueNet(**architecture)
        net.load_state_dict(checkpoint["model"], strict=True)
    else:
        net, architecture = expanded_resource_model(checkpoint)
    evaluator = M.NetworkEvaluator(net, "cpu")
    index = json.loads(args.index.read_text()) if args.index else None
    records = json.loads(args.corpus.read_text())["train"] if args.corpus else None
    if previous:
        roots = json.loads((args.out / "roots.json").read_text())
        if sha(args.out / "roots.json") != previous["roots_sha256"]:
            raise ValueError("root selection changed on resume")
    elif args.evaluation_panel:
        from evaluate_embedding_cost_search import query_instance

        panel = json.loads(args.evaluation_panel.read_text())
        if panel.get("complete") is False or panel.get("partition", "validation") not in (
            "validation",
            "test",
        ):
            raise ValueError("require a completed heldout panel")
        roots = []
        for i, record in enumerate(panel[args.evaluation_view][: args.roots]):
            w = verify_record(record)
            if owner(record["left_fingerprint"]) not in ("validation", "test"):
                raise ValueError("evaluation root belongs to train/quarantine")
            problem, fits = query_instance(i, w, witness_caps=True)
            assert fits
            for weight in (10, 1000):
                conditioned = replace(
                    problem, cc_weight=weight, uid=f"eval-{args.evaluation_view}-{i}-{weight}"
                )
                roots.append(
                    {
                        "instance": conditioned.to_dict(),
                        "panel_index": i,
                        "fingerprint": record["left_fingerprint"],
                        "witness": w.to_dict(),
                    }
                )
        save(args.out / "roots.json", roots)
        shutil.copyfile(args.evaluation_panel, args.out / "source-panel.json")
    else:
        choices = choose_training_views(index, args.roots, args.seed,
                                        maximum_moves=args.max_root_moves,
                                        stratify_size=args.stratify_size)
        roots = []
        for group, i in enumerate(choices):
            view = index["views"][i]
            verify_u_view(view, records, index["parent_witness_sha256"])
            w = verify_record(records[view["parent_index"]])
            depth = view["suffix_index"]
            start = w.start if depth == 0 else w.steps[depth - 1].after
            suffix = UnknotWitness(start, w.max_len, w.max_strands, w.steps[depth:])
            if group % 4 == 0 and not args.no_extra_lifts:
                suffix, _ = lift(suffix, max(9, start.strands + 2), max(60, len(start.word) + 16))
            suffix.verify()
            states = [suffix.start] + [s.after for s in suffix.steps]
            problem = M.Instance(
                f"rollout-{group}",
                suffix.start.word,
                suffix.start.strands,
                suffix.crossing_changes,
                cap_moves=suffix.moves + 8,
                growth_len=max(2, max(len(s.word) for s in states) - len(suffix.start.word)),
                growth_strands=max(1, max(s.strands for s in states) - suffix.start.strands),
                uid=f"rollout-{group}",
                cc_weight=10 if group % 2 == 0 else 1000,
            )
            for conditioned in condition_training_root(problem, group, paired=args.paired_cc_weights):
                roots.append(
                    {
                        "instance": conditioned.to_dict(),
                        "base_root_index": group,
                        "view_index": i,
                        "fingerprint": view["fingerprint"],
                        "witness": suffix.to_dict(),
                    }
                )
        save(args.out / "roots.json", roots)
    if not previous:
        torch.save(
            {
                "architecture": architecture,
                "model": net.state_dict(),
                "objective": checkpoint["objective"],
            },
            args.out / "actor.pt",
        )
        for name in sources:
            target = args.out / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, target)
    manifest = previous or {
        "schema": "rf-knots-rollout-value-v1",
        "purpose": "evaluation-only" if args.evaluation_panel else "training",
        "config": config,
        "inputs": inputs,
        "sources": sources,
        "roots_sha256": sha(args.out / "roots.json"),
        "actor_sha256": sha(args.out / "actor.pt"),
        "next_group": 0,
        "shards": [],
        "counts": {},
        "contract": "One categorical rollout of frozen actor with deterministic "
        "O(L)+16 proposals, within moves/CC/length/strands caps. Terminal episode return only. "
        "Not probability of MCTS success and not unsolvability. Any non-train intermediate "
        "fingerprint excludes the episode from training. Accepted data can be selection-biased.",
    }
    started = time.time()
    manifest.update(status="collecting", pid=os.getpid())
    save(manifest_path, manifest)
    for begin in range(manifest["next_group"], len(roots), 8):
        jobs = []
        for group in range(begin, min(begin + 8, len(roots))):
            row = roots[group]["instance"]
            problem = M.Instance(**{**row, "word": tuple(row["word"])})
            for seed in range(args.rollouts):
                jobs.append(
                    {
                        "group": group,
                        "seed": seed,
                        "instance": problem,
                        "state": M.SearchState(problem.word, problem.strands, 0, 0),
                        "actions": [],
                        "rng": random.Random(args.seed + 100003 * group + seed),
                        "eligible": not bool(args.evaluation_panel),
                        "excluded_fingerprints": [],
                        "done": False,
                    }
                )
        while any(not j["done"] for j in jobs):
            active, requests = [], []
            for job in jobs:
                if job["done"]:
                    continue
                actions = rollout_candidates(job["instance"], job["state"])
                if not actions:
                    job["done"] = True
                    continue
                active.append(job)
                requests.append((job["instance"], job["state"], actions))
            for job, request, prediction in zip(
                active, requests, evaluator.evaluate(requests), strict=True
            ):
                draw, acc = job["rng"].random(), 0.0
                action = request[2][-1]
                for candidate, probability in zip(request[2], prediction.priors, strict=True):
                    acc += probability
                    if draw < acc:
                        action = candidate
                        break
                problem, state = job["instance"], job["state"]
                word, strands = reference.apply(problem.spec, state.word, state.strands, action)
                is_cc = problem.spec.decode(action)[0] == CROSSING_CHANGE
                if is_cc and not args.evaluation_panel:
                    fp = invariants(word, strands)
                    if owner(fp) != "train":
                        job["eligible"] = False
                        job["excluded_fingerprints"].append(list(fp))
                job["state"] = M.SearchState(
                    word, strands, state.used_cc + int(is_cc), state.used_moves + 1
                )
                job["actions"].append(action)
        episodes = []
        for job in jobs:
            state = job["state"]
            episode = {
                k: job[k] for k in ("group", "seed", "actions", "eligible", "excluded_fingerprints")
            }
            episode.update(success=state.key == M.SOLVED, cc=state.used_cc, moves=state.used_moves)
            verify_episode(episode, job["instance"])
            episodes.append(episode)
        shard = args.out / f"episodes-{begin:05d}.json"
        if shard.exists() and json.loads(shard.read_text()) != episodes:
            raise ValueError("recomputed uncommitted shard differs")
        save(shard, episodes)
        counts = Counter(manifest["counts"])
        counts["episodes"] += len(episodes)
        counts["transitions"] += sum(e["moves"] for e in episodes)
        counts["successes"] += sum(e["success"] for e in episodes)
        counts["eligible"] += sum(e["eligible"] for e in episodes)
        counts["eligible_successes"] += sum(e["eligible"] and e["success"] for e in episodes)
        manifest["shards"].append({"file": shard.name, "sha256": sha(shard)})
        manifest.update(
            next_group=min(begin + 8, len(roots)),
            counts=dict(counts),
            seconds_this_run=time.time() - started,
            evaluator_requests=evaluator.calls,
        )
        save(manifest_path, manifest)
        print(
            json.dumps({k: manifest[k] for k in ("next_group", "counts", "seconds_this_run")}),
            flush=True,
        )
    if any(sha(Path(name)) != digest for name, digest in sources.items()):
        raise ValueError("source changed while collecting")
    manifest["status"] = "completed"
    save(manifest_path, manifest)


if __name__ == "__main__":
    main()
