"""Resumable protocol-53 collection; distinct schema, no implicit RL labels."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import torch
from audit_cost_mcts_continuations import audit_episode
from check_cost_mcts_continuations import check_logits
from evaluate_embedding_cost_search import KnotDistanceHeuristic, query_instance

from rf_knots import mcts_loop as M
from rf_knots.cost_mcts_collection import annotate_path_ownership, choose_distinct_training_roots
from rf_knots.cost_mcts_rollout import BatchedPolicyContinuation, run_continuation_cost_searches
from rf_knots.embedding_mcts_prior import EmbeddingDistancePrior
from rf_knots.embedding_metric import MetricEmbeddingObjective
from rf_knots.embedding_replay_corpus import owner, verify_record

CORPUS = Path("artifacts/embedding-diverse-short-u-corpus-v1/corpus.json")
ACTOR = Path("artifacts/embedding-diverse-metric-joint-v1/checkpoint.pt")
EXPECTED = {
    str(CORPUS): "f3655e7dbfcc5b6ec7448b1f747dc2a4e1235fe036ab07fec61cc18df05c8f14",
    str(ACTOR): "964199130db5645e41195905f8629636207fcf40e4bcf2a5683b9319fec7044e",
}
PANELS = [
    Path("artifacts/embedding-diverse-metric-short-u-test-v1/panel.json"),
    Path("artifacts/embedding-common-block-gordian-supplement-v1/panel.json"),
]
CONFIG = {
    "root_groups": 128,
    "seed": 20261022,
    "cc_weights": [10, 1000],
    "batch_conditions": 8,
    "simulations": 8,
    "horizon": 128,
    "insertion_candidates": 4,
    "widening_batch": 4,
    "c_puct": 1.5,
    "temperature": 1.0,
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    temp = p.with_suffix(".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    existed = args.out.exists()
    if existed and not args.resume:
        raise ValueError("output exists; explicit verified resume required")
    if not existed and args.resume:
        raise ValueError("nothing to resume")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inputs = {str(p): sha(p) for p in [CORPUS, ACTOR, ACTOR.parent / "manifest.json", *PANELS]}
    if any(inputs[p] != h for p, h in EXPECTED.items()):
        raise ValueError("frozen actor or corpus changed")
    actor_manifest = json.loads((ACTOR.parent / "manifest.json").read_text())
    if (
        actor_manifest["status"] != "completed"
        or actor_manifest["checkpoint_sha256"] != inputs[str(ACTOR)]
    ):
        raise ValueError("actor is not a verified completed checkpoint")
    sources = sorted(Path("src/rf_knots").glob("*.py")) + [
        Path(__file__).resolve(),
        Path("scripts/audit_cost_mcts_continuations.py"),
        Path("scripts/check_cost_mcts_continuations.py"),
        Path("scripts/evaluate_embedding_cost_search.py"),
        Path("research/53-diverse-cost-mcts-training-collection.md"),
    ]
    source_pins = {str(p): sha(p) for p in sources}
    manifest_path = args.out / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if (manifest["config"], manifest["inputs"], manifest["sources"]) != (
            CONFIG,
            inputs,
            source_pins,
        ):
            raise ValueError("resume configuration/source/input contract changed")
        committed = []
        for shard in manifest["shards"]:
            if sha(args.out / shard["file"]) != shard["sha256"]:
                raise ValueError("committed shard changed")
            committed.extend(
                r["condition"] for r in json.loads((args.out / shard["file"]).read_text())
            )
        if committed != list(range(manifest["next_condition"])):
            raise ValueError("resume shard coverage differs from progress")
        if sha(args.out / "actor.pt") != inputs[str(ACTOR)]:
            raise ValueError("saved actor changed")
        if (
            manifest.get("roots_sha256")
            and sha(args.out / "roots.json") != manifest["roots_sha256"]
        ):
            raise ValueError("selected roots changed")
        if manifest["status"] == "completed":
            if manifest["next_condition"] != CONFIG["root_groups"] * len(CONFIG["cc_weights"]):
                raise ValueError("completed manifest lacks full coverage")
            print("already completed")
            return
    else:
        for p in sources:
            relative = p.relative_to(Path.cwd()) if p.is_absolute() else p
            target = args.out / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        shutil.copy2(ACTOR, args.out / "actor.pt")
        manifest = {
            "schema": "rf-knots-cost-mcts-trajectories-v1",
            "status": "preparing",
            "config": CONFIG,
            "inputs": inputs,
            "sources": source_pins,
            "shards": [],
            "next_condition": 0,
            "counts": {},
            "actor_sha256": sha(args.out / "actor.pt"),
        }
    manifest.update(pid=os.getpid(), status="preparing")
    save(manifest_path, manifest)
    if sha(args.out / "actor.pt") != inputs[str(ACTOR)]:
        raise ValueError("saved actor changed")
    corpus = json.loads(CORPUS.read_text())
    heldout = set()
    for panel in [corpus, *(json.loads(p.read_text()) for p in PANELS)]:
        if panel.get("complete") is False:
            raise ValueError("partial heldout panel")
        for view in ("validation", "size_shift"):
            for row in panel[view]:
                det, sig = row["left_fingerprint"]
                if owner((det, sig)) == "train":
                    raise ValueError("heldout query is TRAIN-owned")
                heldout.add((det, abs(sig)))
    roots_path = args.out / "roots.json"
    if manifest.get("roots_sha256"):
        if sha(roots_path) != manifest["roots_sha256"]:
            raise ValueError("selected roots changed")
        roots = json.loads(roots_path.read_text())
    else:
        roots = []
        indices = choose_distinct_training_roots(corpus["train"], 128, CONFIG["seed"])
        for group, index in enumerate(indices):
            record = corpus["train"][index]
            w = verify_record(record)
            base = query_instance(index, w, witness_caps=True)[0]
            for weight in CONFIG["cc_weights"]:
                problem = replace(base, cc_weight=weight, uid=f"mcts-train-{group}-{weight}")
                roots.append(
                    {
                        "base_root_index": group,
                        "corpus_index": index,
                        "fingerprint": record["left_fingerprint"],
                        "instance": problem.to_dict(),
                        "reference_witness": w.to_dict(),
                    }
                )
            if (group + 1) % 16 == 0:
                manifest["prepared_root_groups"] = group + 1
                save(manifest_path, manifest)
        if roots_path.exists() and json.loads(roots_path.read_text()) != roots:
            raise ValueError("uncommitted root selection differs")
        if not roots_path.exists():
            save(roots_path, roots)
        manifest.update(
            roots_sha256=sha(roots_path),
            selected_root_groups=128,
            heldout_query_groups=[list(fp) for fp in sorted(heldout)],
            root_size_range={
                "min_length": min(len(r["instance"]["word"]) for r in roots),
                "max_length": max(len(r["instance"]["word"]) for r in roots),
                "min_strands": min(r["instance"]["strands"] for r in roots),
                "max_strands": max(r["instance"]["strands"] for r in roots),
            },
        )
    manifest["status"] = "collecting"
    save(manifest_path, manifest)
    checkpoint = torch.load(args.out / "actor.pt", map_location="cpu", weights_only=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    for begin in range(manifest["next_condition"], len(roots), CONFIG["batch_conditions"]):
        selected = roots[begin : begin + CONFIG["batch_conditions"]]
        instances = [
            M.Instance(**{**r["instance"], "word": tuple(r["instance"]["word"])}) for r in selected
        ]
        # Batch-local model/cache makes a resumed batch independent of earlier cache history.
        net = M.SitePolicyValueNet(**checkpoint["architecture"])
        net.load_state_dict(checkpoint["model"], strict=True)
        objective = MetricEmbeddingObjective(cc_rank_weight=0.1, witness_priority=False)
        objective.load_state_dict(checkpoint["objective"], strict=True)
        heuristic = KnotDistanceHeuristic(net, objective)
        tree_evaluator, policy_evaluator = (
            EmbeddingDistancePrior(heuristic),
            M.NetworkEvaluator(net),
        )
        hook = net.register_forward_hook(check_logits)
        policy = BatchedPolicyContinuation(
            policy_evaluator,
            horizon=CONFIG["horizon"],
            insertion_candidates=CONFIG["insertion_candidates"],
        )
        try:
            trees = run_continuation_cost_searches(
                instances,
                tree_evaluator,
                policy,
                simulations=CONFIG["simulations"],
                seed=CONFIG["seed"] + begin,
                widening_batch=CONFIG["widening_batch"],
                c_puct=CONFIG["c_puct"],
            )
        finally:
            hook.remove()
        rows, counts = [], Counter(manifest["counts"])
        for local_index, tree in enumerate(trees):
            episodes = []
            for simulation, prefix, result in tree.continuations:
                episode = {
                    "simulation": simulation,
                    "tree_prefix": list(prefix),
                    "continuation": asdict(result),
                }
                # Auditor consumes the JSON representation, not Python tuple identity.
                audit_episode(tree.instance, json.loads(json.dumps(episode)), CONFIG)
                episode.update(annotate_path_ownership(tree.instance, prefix, result, heldout))
                episodes.append(episode)
                counts["episodes"] += 1
                counts["full_path_moves"] += len(prefix) + len(result.actions)
                counts["continuation_moves"] += len(result.actions)
                counts["eligible_episodes"] += int(episode["eligible"])
                counts["successful_episodes"] += int(result.stop_reason == "solved")
                counts["eligible_successful_episodes"] += int(
                    episode["eligible"] and result.stop_reason == "solved"
                )
            witness = tree.best_witness
            if witness is not None:
                witness.verify()
            eligible = bool(episodes) and all(e["eligible"] for e in episodes)
            row = {
                "condition": begin + local_index,
                "search_seed": CONFIG["seed"] + begin,
                "tree_seed": tree.seed,
                "episodes": episodes,
                "all_episodes_train_owned": eligible,
                "simulations": tree.simulations,
                "expansions": tree.expansions,
                "maximum_tree_depth": tree.maximum_tree_depth,
                "maximum_observed_depth": tree.maximum_observed_depth,
                "incumbents": tree.incumbents,
                "solved": witness is not None,
                "witness": witness.to_dict() if witness else None,
                "L10": 10 * witness.crossing_changes + witness.moves if witness else None,
                "L1000": 1000 * witness.crossing_changes + witness.moves if witness else None,
            }
            rows.append(row)
            counts["conditions"] += 1
            counts["solved_conditions"] += int(witness is not None)
            counts["all_episodes_train_conditions"] += int(eligible)
        counts["policy_neural_evaluations"] += policy_evaluator.calls
        counts["policy_neural_batches"] += policy_evaluator.forwards
        counts["tree_embedding_evaluations"] += heuristic.evaluations
        counts["policy_action_scores"] += policy.action_scores
        counts["tree_action_scores"] += tree_evaluator.action_scores
        shard = args.out / f"conditions-{begin:05d}.json"
        if shard.exists() and json.loads(shard.read_text()) != json.loads(json.dumps(rows)):
            raise ValueError("uncommitted recomputed shard differs; preserving both states")
        if not shard.exists():
            save(shard, rows)
        if not all(sha(Path(p)) == h for p, h in {**inputs, **source_pins}.items()):
            raise ValueError("frozen source/input changed during collection")
        manifest["shards"].append({"file": shard.name, "sha256": sha(shard)})
        manifest.update(
            next_condition=begin + len(rows),
            counts=dict(counts),
            seconds_this_run=time.monotonic() - started,
        )
        save(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "next_condition": manifest["next_condition"],
                    "counts": dict(counts),
                    "seconds_this_run": manifest["seconds_this_run"],
                }
            ),
            flush=True,
        )
    manifest["status"] = "completed"
    save(manifest_path, manifest)


if __name__ == "__main__":
    main()
