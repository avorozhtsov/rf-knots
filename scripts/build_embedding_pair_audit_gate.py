#!/usr/bin/env python3
"""Calibrate a frozen pair-level leakage gate without using test outcomes."""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
from pathlib import Path

import torch

from rf_knots.embedding_audit_gate import (
    SCHEMA,
    FrozenPairAuditGate,
    load_audit_encoder,
    sha256,
)
from rf_knots.embedding_augmentation import scramble_zero_cc
from rf_knots.embedding_supervision import UnknotDBEmbeddingSidecar


def padded(rows: list[tuple[tuple[int, ...], int]], device: torch.device):
    lengths = torch.tensor([len(word) for word, _ in rows], dtype=torch.long)
    width = max(int(lengths.max()), 1)
    words = torch.zeros((len(rows), width), dtype=torch.long)
    for index, (word, _strands) in enumerate(rows):
        if word:
            words[index, : len(word)] = torch.tensor(word, dtype=torch.long)
    strands = torch.tensor([strands for _word, strands in rows], dtype=torch.long)
    return words.to(device), lengths.to(device), strands.to(device)


@torch.no_grad()
def encode(model, rows, device, batch_size):
    result = []
    for start in range(0, len(rows), batch_size):
        batch = padded(rows[start : start + batch_size], device)
        result.append(model(*batch)["knot"].cpu())
    return torch.cat(result)


def quantile(values: torch.Tensor, probability: float) -> float:
    return float(torch.quantile(values, probability))


def atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--negative-pairs", type=int, default=4096)
    parser.add_argument("--negative-quantile", type=float, default=0.999)
    parser.add_argument("--minimum-positive-recall", type=float, default=0.90)
    parser.add_argument("--maximum-negative-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if not 0.5 < args.negative_quantile < 1.0:
        raise ValueError("negative quantile must lie strictly between 0.5 and 1")
    device = torch.device(args.device)
    model, checkpoint = load_audit_encoder(args.checkpoint, device)

    protocol = sqlite3.connect(f"file:{args.protocol}?mode=ro", uri=True)
    anchor_ids = {
        str(row[0]) for row in protocol.execute("SELECT knot_id FROM anchors")
    }
    quarantined = {
        int(row[0])
        for row in protocol.execute(
            "SELECT DISTINCT representation_id FROM quarantine_representations"
        )
    }
    protocol.close()
    raw = sqlite3.connect(f"file:{args.pairs}?mode=ro", uri=True)
    knot_by_representation = {
        int(row[0]): row[1]
        for row in raw.execute("SELECT representation_id,knot_id FROM representations")
    }
    train_ids = [
        int(row[0])
        for row in raw.execute(
            "SELECT representation_id FROM representations WHERE split='train' "
            "ORDER BY representation_id"
        )
        if int(row[0]) not in quarantined
    ]
    anchor_representations = sorted(
        representation_id
        for representation_id, knot_id in knot_by_representation.items()
        if knot_id in anchor_ids
    )
    raw.close()
    if not anchor_representations or len(train_ids) < 2:
        raise RuntimeError("insufficient audit anchor or negative representations")

    references_left = []
    references_right = []
    positives_left = []
    positives_right = []
    with UnknotDBEmbeddingSidecar(args.pairs) as sidecar:
        for index, representation_id in enumerate(anchor_representations):
            root = sidecar.representation(representation_id)
            variants = [
                scramble_zero_cc(root, args.seed + 101 * index + depth, depth)
                for depth in (1, 2, 5, 6)
            ]
            references_left.append((variants[0].word, variants[0].strands))
            references_right.append((variants[3].word, variants[3].strands))
            positives_left.append((variants[1].word, variants[1].strands))
            positives_right.append((variants[2].word, variants[2].strands))
        rng = random.Random(args.seed)
        negatives_left = []
        negatives_right = []
        for _ in range(args.negative_pairs):
            left_id, right_id = rng.sample(train_ids, 2)
            left = sidecar.representation(left_id)
            right = sidecar.representation(right_id)
            negatives_left.append((left.word, left.strands))
            negatives_right.append((right.word, right.strands))

    reference_left = encode(model, references_left, device, args.batch_size)
    reference_right = encode(model, references_right, device, args.batch_size)
    provisional = FrozenPairAuditGate(
        reference_left,
        reference_right,
        1.0,
        True,
        sha256(args.checkpoint),
        sha256(args.protocol),
    )
    positive_scores = provisional.score(
        encode(model, positives_left, device, args.batch_size),
        encode(model, positives_right, device, args.batch_size),
    )
    negative_scores = provisional.score(
        encode(model, negatives_left, device, args.batch_size),
        encode(model, negatives_right, device, args.batch_size),
    )
    threshold = min(1.0, quantile(negative_scores, args.negative_quantile) + 1e-6)
    positive_recall = float((positive_scores >= threshold).float().mean())
    negative_rate = float((negative_scores >= threshold).float().mean())
    enabled = (
        positive_recall >= args.minimum_positive_recall
        and negative_rate <= args.maximum_negative_rate
    )
    payload = {
        "schema": SCHEMA,
        "enabled": enabled,
        "threshold": threshold,
        "reference_left": reference_left,
        "reference_right": reference_right,
        "checkpoint_sha256": sha256(args.checkpoint),
        "protocol_sha256": sha256(args.protocol),
        "pairs_sha256": sha256(args.pairs),
        "checkpoint_examples_seen": int(checkpoint["examples_seen"]),
        "anchor_knot_ids": len(anchor_ids),
        "reference_pairs": len(reference_left),
        "negative_pairs": len(negative_scores),
        "positive_recall": positive_recall,
        "negative_rate": negative_rate,
        "positive_score_quantiles": {
            name: quantile(positive_scores, value)
            for name, value in (("p05", 0.05), ("p50", 0.5), ("p95", 0.95))
        },
        "negative_score_quantiles": {
            name: quantile(negative_scores, value)
            for name, value in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("p999", 0.999))
        },
        "selection_rule": (
            "enabled iff positive recall and unrelated-pair false-positive gates pass"
        ),
    }
    atomic_torch(args.output, payload)
    report = {key: value for key, value in payload.items() if not isinstance(value, torch.Tensor)}
    report["output"] = str(args.output.resolve())
    report["output_sha256"] = sha256(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
