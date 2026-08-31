#!/usr/bin/env python3
"""Evaluate and apply a frozen knot embedding as a proof-graph candidate generator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from rf_knots.embedding_route_retrieval import (
    deterministic_rank,
    embed_graph_representations,
    load_checkpoint_model,
    load_gap_sources,
    load_graph_representations,
    load_hidden_exact_cc1_edges,
    sha256,
    summarize_ranks,
    validate_snapshot_contract,
)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def evaluate_hidden_edges(representations, embeddings, cases, *, source_split: str) -> dict:
    index_by_node = {
        representation.node_id: index
        for index, representation in enumerate(representations)
    }
    node_ids = torch.tensor([representation.node_id for representation in representations])
    u_upper = torch.tensor([representation.u_upper for representation in representations])
    ranks = []
    structural_ranks = []
    pool_sizes = []
    details = []
    for case in cases:
        if source_split != "all" and case.source_split != source_split:
            continue
        source_index = index_by_node.get(case.source_node)
        target_index = index_by_node.get(case.target_node)
        if source_index is None or target_index is None:
            continue
        candidates = torch.nonzero(u_upper == case.target_u, as_tuple=False).flatten()
        target_position = torch.nonzero(candidates == target_index, as_tuple=False).flatten()
        if target_position.numel() != 1:
            continue
        similarities = embeddings[candidates] @ embeddings[source_index]
        candidate_nodes = node_ids[candidates]
        rank = deterministic_rank(similarities, candidate_nodes, int(target_position[0]))
        source = representations[source_index]
        structural_scores = -(
            torch.tensor(
                [
                    abs(representations[int(index)].word_length - source.word_length)
                    + 4 * abs(representations[int(index)].strands - source.strands)
                    for index in candidates
                ],
                dtype=torch.float32,
            )
        )
        structural_rank = deterministic_rank(
            structural_scores, candidate_nodes, int(target_position[0])
        )
        ranks.append(rank)
        structural_ranks.append(structural_rank)
        pool_sizes.append(len(candidates))
        details.append(
            {
                "edge_id": case.edge_id,
                "source_node": case.source_node,
                "target_node": case.target_node,
                "source_split": case.source_split,
                "target_split": case.target_split,
                "source_u": case.source_u,
                "target_u": case.target_u,
                "cosine": float(similarities[int(target_position[0])]),
                "embedding_rank": rank,
                "structural_rank": structural_rank,
                "candidate_pool": len(candidates),
                "provenance": case.provenance,
            }
        )
    return {
        "embedding": summarize_ranks(ranks, pool_sizes),
        "structural_baseline": summarize_ranks(structural_ranks, pool_sizes),
        "cases": details,
    }


def existing_neighbors(proof_path: Path, source_nodes: set[int]) -> dict[int, set[int]]:
    import sqlite3

    connection = sqlite3.connect(f"file:{proof_path}?mode=ro", uri=True)
    result = {node: set() for node in source_nodes}
    try:
        for source, target in connection.execute("SELECT source_node,target_node FROM edges"):
            source = int(source)
            target = int(target)
            if source in result:
                result[source].add(target)
            if target in result:
                result[target].add(source)
    finally:
        connection.close()
    return result


def generate_shortcut_candidates(
    representations,
    embeddings,
    gap_sources,
    *,
    proof_path: Path,
    cc_scale: float,
    per_source: int,
) -> tuple[list[dict], dict]:
    index_by_node = {
        representation.node_id: index
        for index, representation in enumerate(representations)
    }
    source_nodes = {row[0] for row in gap_sources if row[0] in index_by_node}
    neighbors = existing_neighbors(proof_path, source_nodes)
    u_upper = torch.tensor([representation.u_upper for representation in representations])
    rows = []
    skipped_missing_source = 0
    for source_node, source_knot, source_u, source_lower in gap_sources:
        source_index = index_by_node.get(source_node)
        if source_index is None:
            skipped_missing_source += 1
            continue
        mask = u_upper + 1 < source_u
        candidate_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            continue
        similarities = embeddings[candidate_indices] @ embeddings[source_index]
        predicted_cc = cc_scale * (1.0 - similarities)
        predicted_total = predicted_cc + u_upper[candidate_indices]
        order = torch.argsort(predicted_total)
        emitted_keys = set()
        emitted = 0
        for order_index in order.tolist():
            candidate_index = int(candidate_indices[order_index])
            target = representations[candidate_index]
            if target.node_id in neighbors.get(source_node, set()):
                continue
            if target.knot_id == source_knot:
                continue
            dedupe_key = target.knot_id or f"node:{target.node_id}"
            if dedupe_key in emitted_keys:
                continue
            emitted_keys.add(dedupe_key)
            rows.append(
                {
                    "source_node": source_node,
                    "source_knot": source_knot,
                    "source_u_upper": source_u,
                    "source_u_lower": source_lower,
                    "target_node": target.node_id,
                    "target_knot": target.knot_id,
                    "target_u_upper": target.u_upper,
                    "predicted_cc": float(predicted_cc[order_index]),
                    "cosine": float(similarities[order_index]),
                    "predicted_route_total": float(predicted_total[order_index]),
                    "verified_gain_if_cc1": source_u - (1 + target.u_upper),
                    "source_representation": representations[source_index].representation_id,
                    "target_representation": target.representation_id,
                    "target_split": target.split,
                }
            )
            emitted += 1
            if emitted >= per_source:
                break
    rows.sort(
        key=lambda row: (
            row["predicted_cc"],
            -row["verified_gain_if_cc1"],
            row["source_node"],
            row["target_node"],
        )
    )
    return rows, {
        "gap_sources": len(gap_sources),
        "mapped_gap_sources": len(source_nodes),
        "skipped_missing_source": skipped_missing_source,
        "candidates": len(rows),
        "per_source": per_source,
    }


def write_candidates(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    temporary = path.with_name(f".{path.name}.part")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--identification", type=Path, required=True)
    parser.add_argument("--lower-bounds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--per-source", type=int, default=20)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    hashes = validate_snapshot_contract(
        proof_path=args.proof,
        pairs_path=args.pairs,
        identification_path=args.identification,
        lower_bounds_path=args.lower_bounds,
    )
    device = choose_device(args.device)
    representations = load_graph_representations(args.proof, args.pairs)
    model, cc_scale, checkpoint = load_checkpoint_model(args.checkpoint, device)
    embeddings = embed_graph_representations(
        model, representations, device=device, batch_size=args.batch_size
    )
    cache_path = args.output / "graph-embeddings.pt"
    torch.save(
        {
            "checkpoint_sha256": sha256(args.checkpoint),
            "proof_sha256": hashes["proof"],
            "node_ids": [representation.node_id for representation in representations],
            "representation_ids": [
                representation.representation_id for representation in representations
            ],
            "knot_embeddings": embeddings,
        },
        cache_path,
    )

    hidden = load_hidden_exact_cc1_edges(args.proof, args.pairs)
    validation = evaluate_hidden_edges(
        representations, embeddings, hidden, source_split="validation"
    )
    all_non_test = evaluate_hidden_edges(
        representations,
        embeddings,
        [case for case in hidden if case.source_split != "test"],
        source_split="all",
    )
    gap_sources = load_gap_sources(args.identification, args.lower_bounds)
    candidates, candidate_summary = generate_shortcut_candidates(
        representations,
        embeddings,
        gap_sources,
        proof_path=args.proof,
        cc_scale=cc_scale,
        per_source=args.per_source,
    )
    candidates_path = args.output / "shortcut-candidates.tsv"
    write_candidates(candidates_path, candidates)
    report = {
        "schema": "rf-knots-embedding-route-retrieval-v0",
        "inputs": {
            **{f"{name}_sha256": value for name, value in hashes.items()},
            "checkpoint_sha256": sha256(args.checkpoint),
        },
        "device": str(device),
        "cc_scale": cc_scale,
        "checkpoint_selection_score": checkpoint.get("best_selection_score"),
        "graph_representations": len(representations),
        "hidden_exact_cc1_edges": len(hidden),
        "test_edges_reserved": sum(case.source_split == "test" for case in hidden),
        "retrospective_validation": validation,
        "retrospective_non_test": all_non_test,
        "prospective": {
            **candidate_summary,
            "top_candidates": candidates[:100],
            "full_candidates_tsv": str(candidates_path.resolve()),
        },
        "embedding_cache": {
            "path": str(cache_path.resolve()),
            "sha256": sha256(cache_path),
        },
        "proof_contract": (
            "Candidates are heuristic only. No U upper bound changes without an exact "
            "constructed bridge and replay verification."
        ),
    }
    report_path = args.output / "report.json"
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
