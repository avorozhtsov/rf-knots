#!/usr/bin/env python3
"""Fit frozen linear value probes on identity-disjoint proof-graph U labels."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import torch

from rf_knots.embedding_route_retrieval import (
    fit_ridge,
    predict_ridge,
    regression_metrics,
    sha256,
)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_examples(
    *, identification_path: Path, pairs_path: Path, node_ids: list[int]
) -> list[dict]:
    identification = sqlite3.connect(f"file:{identification_path}?mode=ro", uri=True)
    pairs = sqlite3.connect(f"file:{pairs_path}?mode=ro", uri=True)
    try:
        representation_by_mirror = {}
        for mirror_key, split, word_length, strands in pairs.execute(
            "SELECT mirror_key,split,word_length,strands FROM representations "
            "ORDER BY representation_id"
        ):
            representation_by_mirror.setdefault(
                bytes(mirror_key), (str(split), int(word_length), int(strands))
            )
        allowed_nodes = set(node_ids)
        rows = []
        for node_id, knot_id, u_upper, rep_key in identification.execute(
            "SELECT v.node_id,m.knot_id,v.u_upper,v.rep_key "
            "FROM graph_vertex_knot_map m JOIN graph_vertices v USING(rep_key)"
        ):
            representation = representation_by_mirror.get(bytes(rep_key))
            if int(node_id) not in allowed_nodes or representation is None:
                continue
            split, word_length, strands = representation
            rows.append(
                {
                    "node_id": int(node_id),
                    "knot_id": str(knot_id),
                    "u_upper": int(u_upper),
                    "split": split,
                    "word_length": word_length,
                    "strands": strands,
                }
            )
    finally:
        identification.close()
        pairs.close()
    best_by_knot = {}
    for row in rows:
        previous = best_by_knot.get(row["knot_id"])
        if previous is None or (row["word_length"], row["node_id"]) < (
            previous["word_length"],
            previous["node_id"],
        ):
            best_by_knot[row["knot_id"]] = row
    return sorted(best_by_knot.values(), key=lambda row: row["knot_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--identification", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-u", type=int, default=10)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()
    if args.clip_u < 1 or args.ridge_alpha < 0:
        raise ValueError("invalid value-probe configuration")

    cache = torch.load(args.embedding_cache, map_location="cpu", weights_only=False)
    node_ids = [int(node_id) for node_id in cache["node_ids"]]
    embeddings = cache["knot_embeddings"].float()
    index_by_node = {node_id: index for index, node_id in enumerate(node_ids)}
    examples = load_examples(
        identification_path=args.identification,
        pairs_path=args.pairs,
        node_ids=node_ids,
    )
    unknot = next(row for row in examples if row["knot_id"] == "knot:0_1")
    unknot_embedding = embeddings[index_by_node[unknot["node_id"]]]

    datasets = {}
    for split in ("train", "validation"):
        rows = [row for row in examples if row["split"] == split]
        indices = torch.tensor([index_by_node[row["node_id"]] for row in rows])
        knot = embeddings[indices]
        structural = torch.tensor(
            [[torch.log1p(torch.tensor(row["word_length"])).item(), row["strands"]] for row in rows]
        )
        distance_to_unknot = (1.0 - knot @ unknot_embedding).unsqueeze(1)
        targets = torch.tensor(
            [min(row["u_upper"], args.clip_u) for row in rows], dtype=torch.float32
        )
        datasets[split] = {
            "rows": rows,
            "targets": targets,
            "constant": torch.ones((len(rows), 1)),
            "structural": structural,
            "distance_to_unknot": distance_to_unknot,
            "embedding": knot,
            "combined": torch.cat((knot, structural), dim=1),
        }

    metrics = {}
    validation_targets = datasets["validation"]["targets"]
    for name in ("constant", "structural", "distance_to_unknot", "embedding", "combined"):
        model = fit_ridge(
            datasets["train"][name],
            datasets["train"]["targets"],
            alpha=args.ridge_alpha,
        )
        metrics[name] = regression_metrics(
            validation_targets,
            predict_ridge(model, datasets["validation"][name]),
        )
    report = {
        "schema": "rf-knots-frozen-embedding-value-probe-v0",
        "status": "offline_probe_only",
        "inputs": {
            "embedding_cache_sha256": sha256(args.embedding_cache),
            "checkpoint_sha256": cache["checkpoint_sha256"],
            "proof_sha256": cache["proof_sha256"],
            "identification_sha256": sha256(args.identification),
            "pairs_sha256": sha256(args.pairs),
        },
        "protocol": {
            "one_shortest_representation_per_identified_knot": True,
            "identity_disjoint_sidecar_split": True,
            "target": f"min(replayed_u_upper,{args.clip_u})",
            "ridge_alpha": args.ridge_alpha,
            "test_split_used": False,
            "test_rows_reserved": sum(row["split"] == "test" for row in examples),
        },
        "rows": {
            "all": len(examples),
            "train": len(datasets["train"]["rows"]),
            "validation": len(datasets["validation"]["rows"]),
        },
        "validation_metrics": metrics,
        "interpretation_contract": (
            "This tests frozen linear value signal only. It does not establish a policy or "
            "MCTS improvement, and graph U labels are replayed upper bounds rather than exact U."
        ),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
