"""Frozen TRAIN-only full-policy continuations, contract 51; no model training."""

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from evaluate_embedding_cost_search import KnotDistanceHeuristic, query_instance

from rf_knots import mcts_loop as M
from rf_knots.cost_mcts_rollout import BatchedPolicyContinuation, run_continuation_cost_searches
from rf_knots.embedding_mcts_prior import EmbeddingDistancePrior
from rf_knots.embedding_metric import MetricEmbeddingObjective
from rf_knots.embedding_replay_corpus import verify_record

SELECTION = Path("artifacts/embedding-cost-mcts-integration-v1/report.json")
SELECTION_SHA = "df3e0e2d3c05f00d03d6ab15992c7375d4b8c7446d973833ba6a348191bb6b09"
CORPUS = Path("artifacts/embedding-diverse-short-u-corpus-v1/corpus.json")
MODELS = {
    "parent": Path("artifacts/embedding-multisize-clean-joint-v1"),
    "candidate": Path("artifacts/embedding-diverse-metric-joint-v1"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_logits(module, inputs, output):
    # NetworkEvaluator's historical fallback must not conceal NaN logits here.
    logits, _, effective = output
    if not effective.any(dim=1).all() or not torch.isfinite(logits[effective]).all():
        raise ValueError("nonfinite or empty effective full-policy logits")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sha(SELECTION) != SELECTION_SHA:
        raise ValueError("selection report changed")
    previous = json.loads(SELECTION.read_text())
    pins = dict(previous["inputs"])
    pins[str(SELECTION)] = SELECTION_SHA
    if not all(sha(Path(p)) == h for p, h in pins.items()):
        raise ValueError("pinned corpus/checkpoint inputs changed")
    records = json.loads(CORPUS.read_text())["train"]
    selected = previous["selected"]
    if [row["corpus_index"] for row in selected] != [0, 2, 5, 8]:
        raise ValueError("unexpected TRAIN roots")
    witnesses = [verify_record(records[row["corpus_index"]]) for row in selected]
    instances = []
    for row, witness in zip(selected, witnesses, strict=True):
        if witness.start.to_dict() != row["start"]:
            raise ValueError("selection root differs from replay")
        instances.append(query_instance(row["corpus_index"], witness, witness_caps=True)[0])
    sources = sorted(Path("src/rf_knots").glob("*.py")) + [
        Path(__file__).resolve(),
        Path("scripts/evaluate_embedding_cost_search.py"),
        Path("research/48-anytime-cost-mcts-contract.md"),
        Path("research/51-budgeted-policy-continuation-contract.md"),
    ]
    source_pins = {str(p): sha(p) for p in sources}
    args.out.mkdir(parents=True, exist_ok=False)
    for p in sources:
        relative = p.relative_to(Path.cwd()) if p.is_absolute() else p
        target = args.out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    report = {
        "status": "running",
        "pid": os.getpid(),
        "scope": "TRAIN-only engineering",
        "inputs": pins,
        "sources": source_pins,
        "selected": selected,
        "instances": [i.to_dict() for i in instances],
        "config": {
            "simulations": 8,
            "horizon": 128,
            "insertion_candidates": 4,
            "widening_batch": 4,
            "c_puct": 1.5,
            "seed": 20260922,
            "cc_weight": 1000,
            "temperature": 1,
        },
        "arms": [],
    }

    def save():
        temp = args.out / "report.tmp"
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(args.out / "report.json")

    torch.set_num_threads(1)
    save()
    for mode in ("uniform", "parent", "candidate"):
        report["current"] = mode
        save()
        heuristic, hook = None, None
        if mode == "uniform":
            tree_evaluator, policy_evaluator = M.UniformEvaluator(), M.UniformEvaluator()
        else:
            c = torch.load(MODELS[mode] / "checkpoint.pt", map_location="cpu", weights_only=False)
            net = M.SitePolicyValueNet(**c["architecture"])
            net.load_state_dict(c["model"], strict=True)
            objective = MetricEmbeddingObjective(cc_rank_weight=0.1, witness_priority=False)
            objective.load_state_dict(c["objective"], strict=True)
            heuristic = KnotDistanceHeuristic(net, objective)
            tree_evaluator = EmbeddingDistancePrior(heuristic)
            policy_evaluator = M.NetworkEvaluator(net)
            hook = net.register_forward_hook(check_logits)
        policy = BatchedPolicyContinuation(policy_evaluator, horizon=128, insertion_candidates=4)
        started = time.monotonic()
        try:
            trees = run_continuation_cost_searches(
                instances,
                tree_evaluator,
                policy,
                simulations=8,
                seed=20260922,
                widening_batch=4,
                c_puct=1.5,
            )
        finally:
            if hook is not None:
                hook.remove()
        rows = []
        for selected_row, tree in zip(selected, trees, strict=True):
            witness = tree.best_witness
            if witness is not None:
                witness.verify()
            episodes = [
                {
                    "simulation": simulation,
                    "tree_prefix": list(prefix),
                    "continuation": asdict(result),
                }
                for simulation, prefix, result in tree.continuations
            ]
            rows.append(
                {
                    "corpus_index": selected_row["corpus_index"],
                    "simulations": tree.simulations,
                    "expansions": tree.expansions,
                    "nodes": tree.nodes,
                    "maximum_tree_depth": tree.maximum_tree_depth,
                    "maximum_observed_depth": tree.maximum_observed_depth,
                    "continuation_stops": dict(
                        Counter(e["continuation"]["stop_reason"] for e in episodes)
                    ),
                    "episodes": episodes,
                    "solved": witness is not None,
                    "incumbents": tree.incumbents,
                    "witness": witness.to_dict() if witness else None,
                    "L10": 10 * witness.crossing_changes + witness.moves if witness else None,
                    "L1000": 1000 * witness.crossing_changes + witness.moves if witness else None,
                }
            )
        report["arms"].append(
            {
                "mode": mode,
                "seconds": time.monotonic() - started,
                "tree_embedding_evaluations": heuristic.evaluations if heuristic else 0,
                "tree_action_scores": getattr(tree_evaluator, "action_scores", 0),
                "continuation_state_evaluations": policy.state_evaluations,
                "continuation_action_scores": policy.action_scores,
                "continuation_neural_evaluations": policy_evaluator.calls
                if mode != "uniform"
                else 0,
                "continuation_neural_batches": getattr(policy_evaluator, "forwards", 0),
                "rows": rows,
            }
        )
        if not all(sha(Path(p)) == h for p, h in {**pins, **source_pins}.items()):
            raise ValueError("pinned sources/inputs changed during run")
        print(
            json.dumps(
                {
                    "mode": mode,
                    "solved": sum(r["solved"] for r in rows),
                    "max_depth": [r["maximum_observed_depth"] for r in rows],
                    "seconds": report["arms"][-1]["seconds"],
                }
            ),
            flush=True,
        )
        save()
    report.update(status="completed", current=None)
    save()


if __name__ == "__main__":
    main()
