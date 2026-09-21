"""Fixed TRAIN-only integration diagnostic, not a promotion benchmark."""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch
from evaluate_embedding_cost_search import KnotDistanceHeuristic, query_instance

from rf_knots import mcts_loop as M
from rf_knots.anytime_cost_mcts import run_anytime_cost_searches
from rf_knots.embedding_mcts_prior import EmbeddingDistancePrior
from rf_knots.embedding_metric import MetricEmbeddingObjective
from rf_knots.embedding_replay_corpus import verify_record

CORPUS = Path("artifacts/embedding-diverse-short-u-corpus-v1/corpus.json")
CORPUS_SHA = "f3655e7dbfcc5b6ec7448b1f747dc2a4e1235fe036ab07fec61cc18df05c8f14"
MODELS = {
    "parent": (
        "embedding-multisize-clean-joint-v1",
        "076a12a86ca83c03d736edcaea09b31234cae329c9d8d5c743e8c8bef2fdc723",
    ),
    "candidate": (
        "embedding-diverse-metric-joint-v1",
        "964199130db5645e41195905f8629636207fcf40e4bcf2a5683b9319fec7044e",
    ),
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def select(records):
    indices = []
    for min_l, min_n in ((0, 0), (96, 0), (0, 12), (96, 12)):
        found = next(
            (
                i
                for i, r in enumerate(records)
                if i not in indices
                and r["split"] == "train"
                and len(r["witness"]["start"]["word"]) >= min_l
                and r["witness"]["start"]["strands"] >= min_n
            ),
            None,
        )
        if found is None:
            raise ValueError("TRAIN size condition unavailable")
        indices.append(found)
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sha(CORPUS) != CORPUS_SHA:
        raise ValueError("TRAIN corpus changed")
    records = json.loads(CORPUS.read_text())["train"]
    indices = select(records)
    witnesses = [verify_record(records[i]) for i in indices]
    instances = [
        query_instance(i, w, witness_caps=True)[0] for i, w in zip(indices, witnesses, strict=True)
    ]
    assert all(i.cc_weight == 1000 for i in instances)
    pins = {str(CORPUS): CORPUS_SHA}
    for name, (directory, expected) in MODELS.items():
        p = Path("artifacts") / directory
        manifest = json.loads((p / "manifest.json").read_text())
        if (
            manifest["status"] != "completed"
            or manifest["checkpoint_sha256"] != expected
            or sha(p / "checkpoint.pt") != expected
        ):
            raise ValueError(f"invalid final model {name}")
        pins[str(p / "checkpoint.pt")] = expected
        pins[str(p / "manifest.json")] = sha(p / "manifest.json")
    sources = [
        Path(__file__),
        Path("scripts/evaluate_embedding_cost_search.py"),
        Path("research/48-anytime-cost-mcts-contract.md"),
        Path("research/49-embedding-cost-mcts-integration.md"),
    ]
    sources += sorted(Path("src/rf_knots").glob("*.py"))
    source_pins = {str(p): sha(p) for p in sources}
    args.out.mkdir(parents=True, exist_ok=False)
    for p in sources:
        relative = p if not p.is_absolute() else p.relative_to(Path.cwd())
        target = args.out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    report = {
        "status": "running",
        "pid": os.getpid(),
        "inputs": pins,
        "sources": source_pins,
        "scope": "TRAIN-only engineering diagnostic",
        "config": {
            "simulations": 512,
            "seed": 20260921,
            "cc_weight": 1000,
            "widening_batch": 4,
            "c_puct": 1.5,
            "temperature": 1.0,
        },
        "selected": [
            {
                "corpus_index": i,
                "start": w.start.to_dict(),
                "fingerprint": records[i]["left_fingerprint"],
                "reference_cc": w.crossing_changes,
                "reference_moves": w.moves,
            }
            for i, w in zip(indices, witnesses, strict=True)
        ],
        "arms": [],
    }

    def save():
        tmp = args.out / "report.tmp"
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(args.out / "report.json")

    torch.set_num_threads(1)
    save()
    for mode in ("uniform", "parent", "candidate"):
        report["current"] = mode
        save()
        heuristic = None
        if mode == "uniform":
            evaluator = M.UniformEvaluator()
        else:
            path = Path("artifacts") / MODELS[mode][0] / "checkpoint.pt"
            c = torch.load(path, map_location="cpu", weights_only=False)
            net = M.SitePolicyValueNet(**c["architecture"])
            objective = MetricEmbeddingObjective(cc_rank_weight=0.1, witness_priority=False)
            net.load_state_dict(c["model"], strict=True)
            objective.load_state_dict(c["objective"], strict=True)
            heuristic = KnotDistanceHeuristic(net, objective)
            evaluator = EmbeddingDistancePrior(heuristic, temperature=1.0)
        started = time.monotonic()
        trees = run_anytime_cost_searches(
            instances, evaluator, simulations=512, seed=20260921, widening_batch=4, c_puct=1.5
        )
        rows = []
        for i, t in zip(indices, trees, strict=True):
            w = t.best_witness
            if w is not None:
                w.verify()
            rows.append(
                {
                    "corpus_index": i,
                    "simulations": t.simulations,
                    "expansions": t.expansions,
                    "nodes": t.nodes,
                    "solved": w is not None,
                    "incumbents": t.incumbents,
                    "witness": w.to_dict() if w else None,
                    "L10": 10 * w.crossing_changes + w.moves if w else None,
                    "L1000": 1000 * w.crossing_changes + w.moves if w else None,
                    "exhausted": t.exhausted,
                }
            )
        report["arms"].append(
            {
                "mode": mode,
                "seconds": time.monotonic() - started,
                "neural_state_evaluations": heuristic.evaluations if heuristic else 0,
                "action_scores": getattr(evaluator, "action_scores", 0),
                "rows": rows,
            }
        )
        if not all(sha(Path(p)) == h for p, h in {**pins, **source_pins}.items()):
            raise ValueError("pinned inputs/sources changed during diagnostic")
        print(
            json.dumps(
                {
                    "mode": mode,
                    "solved": sum(r["solved"] for r in rows),
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
