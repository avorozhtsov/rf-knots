#!/usr/bin/env python3
"""Materialize deterministic embedding examples into resumable binary shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from rf_knots.embedding_augmentation import SyntheticEmbeddingIterableDataset
from rf_knots.embedding_curriculum_io import load_million_pair_curriculum
from rf_knots.embedding_shards import SCHEMA, write_materialized_shard

MANIFEST_SCHEMA = "rf-knots-embedding-materialized-corpus-v0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--geodesic", type=Path, required=True)
    parser.add_argument("--hard-cases", type=Path, required=True)
    parser.add_argument("--gordian", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--virtual-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.count <= 0 or args.shard_size <= 0:
        raise ValueError("count and shard size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = None
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "complete":
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return
    completed = int(manifest.get("examples", 0)) if manifest else 0
    shards = list(manifest.get("shards", [])) if manifest else []
    curriculum, pools = load_million_pair_curriculum(
        args.pairs,
        args.geodesic,
        args.hard_cases,
        args.gordian,
        size=args.virtual_size,
        seed=args.seed,
    )
    dataset = SyntheticEmbeddingIterableDataset(
        args.pairs,
        curriculum,
        infinite=True,
        replay_check=True,
        start_position=completed,
    )
    iterator = iter(dataset)
    while completed < args.count:
        size = min(args.shard_size, args.count - completed)
        rows = [next(iterator) for _ in range(size)]
        index = len(shards)
        path = args.output_dir / f"shard-{index:06d}.pt"
        write_materialized_shard(path, rows)
        record = {
            "index": index,
            "start": completed,
            "count": size,
            "bytes": path.stat().st_size,
            "path": path.name,
            "sha256": sha256(path),
        }
        shards.append(record)
        completed += size
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "shard_schema": SCHEMA,
            "status": "complete" if completed == args.count else "in_progress",
            "examples": completed,
            "target_examples": args.count,
            "shard_size": args.shard_size,
            "seed": args.seed,
            "virtual_size": args.virtual_size,
            "source_sha256": {
                "pairs": sha256(args.pairs),
                "geodesic": sha256(args.geodesic),
                "hard_cases": sha256(args.hard_cases),
                "gordian": sha256(args.gordian),
            },
            "source_pools": pools,
            "shards": shards,
            "total_bytes": sum(row["bytes"] for row in shards),
        }
        atomic_json(manifest_path, manifest)
        print(json.dumps(record, sort_keys=True), flush=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
