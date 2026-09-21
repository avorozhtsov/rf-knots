"""Fixed prospective full-policy cost-MCTS evaluation (protocol 58), resumable."""

import argparse
import fcntl
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import torch
from audit_cost_mcts_continuations import audit_episode
from audit_joint_policy_metric_panel import check_pins, merge_pins, size_stratum
from build_mcts_training_supervision import atomic, sha
from check_cost_mcts_continuations import check_logits
from evaluate_embedding_cost_search import KnotDistanceHeuristic, query_instance

from rf_knots import mcts_loop as M
from rf_knots.cost_mcts_rollout import BatchedPolicyContinuation, run_continuation_cost_searches
from rf_knots.embedding_evaluation import verify_evaluation_pair
from rf_knots.embedding_mcts_prior import EmbeddingDistancePrior
from rf_knots.embedding_metric import MetricEmbeddingObjective

PANEL = Path("artifacts/mcts-policy-metric-fresh-short-u-test-v1/panel.json")
RUNS = {
    "parent": Path("artifacts/embedding-diverse-metric-joint-v1"),
    "joint": Path("artifacts/mcts-policy-metric-joint-v1"),
    "metric-control": Path("artifacts/mcts-policy-metric-control-v1"),
}
CONFIG = {
    "seed": 20261026,
    "cc_weights": [10, 1000],
    "batch_conditions": 8,
    "simulations": 8,
    "horizon": 192,
    "insertion_candidates": 4,
    "widening_batch": 4,
    "c_puct": 1.5,
    "temperature": 1.0,
}


class CountedUniform(M.UniformEvaluator):
    def __init__(self):
        super().__init__()
        self.requests = 0
        self.action_scores = 0

    def evaluate(self, requests):
        self.requests += len(requests)
        self.action_scores += sum(len(actions) for _, _, actions in requests)
        return super().evaluate(requests)


def build_roots(panel):
    if not panel.get("complete") or panel.get("pair_kind") != "short-u":
        raise ValueError("require complete short-u evaluation panel")
    roots = []
    for index, row in enumerate(panel["validation"]):
        if row["interval"]["right"] not in (
            {"word": [], "strands": 1},
            {"word": [], "strands": 1, "cyclic_band_generators": False},
        ):
            raise ValueError("search query is not an explicit-unknot task")
        witness = verify_evaluation_pair(row)
        instance, fits = query_instance(index, witness, witness_caps=True)
        if not fits:
            raise ValueError("reference solution does not fit common caps")
        for weight in CONFIG["cc_weights"]:
            i = replace(instance, cc_weight=weight, uid=f"test-{index}-lambda-{weight}")
            roots.append(
                {
                    "condition": len(roots),
                    "panel_index": index,
                    "fingerprint": row["left_fingerprint"],
                    "stratum": size_stratum(i.word, i.strands),
                    "instance": i.to_dict(),
                    "reference_L10": 10 * witness.crossing_changes + witness.moves,
                    "reference_L1000": 1000 * witness.crossing_changes + witness.moves,
                }
            )
    return roots


def run_batch(roots, checkpoint, begin, config=CONFIG):
    instances = [
        M.Instance(**{**r["instance"], "word": tuple(r["instance"]["word"])}) for r in roots
    ]
    hook, heuristic = None, None
    if checkpoint is None:
        tree_evaluator, policy_evaluator = CountedUniform(), CountedUniform()
    else:
        torch.manual_seed(config["seed"] + begin)
        net = M.SitePolicyValueNet(**checkpoint["architecture"])
        net.load_state_dict(checkpoint["model"], strict=True)
        objective = MetricEmbeddingObjective(cc_rank_weight=0.1, witness_priority=False)
        objective.load_state_dict(checkpoint["objective"], strict=True)
        if any(
            not torch.isfinite(t).all()
            for t in [*net.state_dict().values(), *objective.state_dict().values()]
        ):
            raise ValueError("nonfinite checkpoint tensors")
        heuristic = KnotDistanceHeuristic(net, objective)
        tree_evaluator = EmbeddingDistancePrior(heuristic, temperature=config["temperature"])
        policy_evaluator = M.NetworkEvaluator(net)
        hook = net.register_forward_hook(check_logits)
    policy = BatchedPolicyContinuation(
        policy_evaluator,
        horizon=config["horizon"],
        insertion_candidates=config["insertion_candidates"],
    )
    try:
        trees = run_continuation_cost_searches(
            instances,
            tree_evaluator,
            policy,
            simulations=config["simulations"],
            seed=config["seed"] + begin,
            widening_batch=config["widening_batch"],
            c_puct=config["c_puct"],
        )
    finally:
        if hook is not None:
            hook.remove()
    rows, counts = [], Counter()
    for root, tree in zip(roots, trees, strict=True):
        episodes = []
        for simulation, prefix, continuation in tree.continuations:
            episode = json.loads(
                json.dumps(
                    {
                        "simulation": simulation,
                        "tree_prefix": list(prefix),
                        "continuation": asdict(continuation),
                    }
                )
            )
            steps, scores, _ = audit_episode(tree.instance, episode, config)
            counts["episodes"] += 1
            counts["continuation_moves"] += steps
            counts["full_path_moves"] += steps + len(prefix)
            counts["policy_action_scores"] += scores
            episodes.append(episode)
        witness = tree.best_witness
        if witness is not None:
            witness.verify()
            if (witness.start.word, witness.start.strands) != (
                tree.instance.word,
                tree.instance.strands,
            ):
                raise ValueError("solution starts at a different query")
            if (
                witness.crossing_changes > tree.instance.cap_crossing_changes
                or witness.moves > tree.instance.cap_moves
            ):
                raise ValueError("solution exceeds common resources")
        rows.append(
            {
                "condition": root["condition"],
                "panel_index": root["panel_index"],
                "cc_weight": tree.instance.cc_weight,
                "search_seed": config["seed"] + begin,
                "tree_seed": tree.seed,
                "episodes": episodes,
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
        )
        counts["conditions"] += 1
        counts["solved_conditions"] += int(witness is not None)
    if counts["continuation_moves"] != policy.state_evaluations:
        raise ValueError("policy-state accounting differs from replay")
    if counts["policy_action_scores"] != policy.action_scores:
        raise ValueError("policy-action accounting differs from replay")
    counts.update(
        {
            "policy_neural_evaluations": policy_evaluator.calls if checkpoint is not None else 0,
            "policy_neural_batches": policy_evaluator.forwards if checkpoint is not None else 0,
            "tree_embedding_evaluations": heuristic.evaluations if heuristic else 0,
            "tree_action_scores": tree_evaluator.action_scores,
        }
    )
    return rows, counts


def committed_rows(manifest, directory):
    rows = []
    for shard in manifest["shards"]:
        path = directory / shard["file"]
        if sha(path) != shard["sha256"]:
            raise ValueError("committed evaluation shard changed")
        rows.extend(json.loads(path.read_text()))
    if [r["condition"] for r in rows] != list(range(manifest["next_condition"])):
        raise ValueError("committed evaluation coverage differs")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=(*RUNS, "uniform"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = RUNS["parent" if args.arm == "uniform" else args.arm]
    audit_path = PANEL.parent / "independence-audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "passed" or audit["inputs"].get(str(PANEL)) != sha(PANEL):
        raise ValueError("require a passed independence audit for this exact panel")
    model_manifest = json.loads((run / "manifest.json").read_text())
    if (
        model_manifest["status"] != "completed"
        or sha(run / "checkpoint.pt") != model_manifest["checkpoint_sha256"]
    ):
        raise ValueError("require completed hash-verified model")
    sources = [
        Path(__file__).resolve(),
        Path("scripts/audit_joint_policy_metric_panel.py"),
        Path("scripts/build_mcts_training_supervision.py"),
        Path("scripts/audit_cost_mcts_continuations.py"),
        Path("scripts/check_cost_mcts_continuations.py"),
        Path("scripts/evaluate_embedding_cost_search.py"),
        Path("research/58-prospective-joint-policy-metric-test.md"),
        *sorted(Path("src/rf_knots").glob("*.py")),
    ]
    pins = dict(audit["inputs"])
    merge_pins(
        pins,
        {
            str(p): sha(p)
            for p in [audit_path, run / "manifest.json", run / "checkpoint.pt", *sources]
        },
    )
    check_pins(pins)
    roots = build_roots(json.loads(PANEL.read_text()))
    if args.out.exists() != args.resume:
        raise ValueError("existing output requires explicit resume; new output cannot resume")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path, roots_path = args.out / "manifest.json", args.out / "roots.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if (manifest["arm"], manifest["config"], manifest["inputs"]) != (args.arm, CONFIG, pins):
            raise ValueError("resume evaluation contract changed")
        if (
            sha(roots_path) != manifest["roots_sha256"]
            or json.loads(roots_path.read_text()) != roots
        ):
            raise ValueError("evaluation roots changed")
        committed_rows(manifest, args.out)
        if manifest["status"] == "completed":
            if manifest["next_condition"] != len(roots):
                raise ValueError("completed evaluation lacks full coverage")
            print("already completed")
            return
    else:
        atomic(roots_path, roots)
        for p in sources:
            relative = p.relative_to(Path.cwd()) if p.is_absolute() else p
            destination = args.out / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, destination)
        manifest = {
            "schema": "rf-knots-prospective-cost-mcts-v1",
            "arm": args.arm,
            "config": CONFIG,
            "inputs": pins,
            "shards": [],
            "counts": {},
            "roots_sha256": sha(roots_path),
            "next_condition": 0,
            "seconds_committed_batches": 0.0,
            "value_head_used": False,
        }
    manifest.update(status="evaluating", pid=os.getpid())
    atomic(manifest_path, manifest)
    torch.set_num_threads(1)
    checkpoint = (
        None
        if args.arm == "uniform"
        else torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    )
    for begin in range(manifest["next_condition"], len(roots), CONFIG["batch_conditions"]):
        started = time.monotonic()
        rows, counts = run_batch(
            roots[begin : begin + CONFIG["batch_conditions"]], checkpoint, begin
        )
        seconds = time.monotonic() - started
        shard = args.out / f"conditions-{begin:05d}.json"
        if shard.exists() and json.loads(shard.read_text()) != json.loads(json.dumps(rows)):
            raise ValueError("recomputed uncommitted evaluation differs; preserving it")
        if not shard.exists():
            atomic(shard, rows)
        check_pins(pins)
        total = Counter(manifest["counts"])
        total.update(counts)
        manifest["shards"].append({"file": shard.name, "sha256": sha(shard), "seconds": seconds})
        manifest.update(
            next_condition=begin + len(rows),
            counts=dict(total),
            seconds_committed_batches=manifest["seconds_committed_batches"] + seconds,
        )
        atomic(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "next_condition": manifest["next_condition"],
                    "counts": dict(total),
                    "seconds": seconds,
                }
            ),
            flush=True,
        )
    committed_rows(manifest, args.out)
    manifest["status"] = "completed"
    atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
