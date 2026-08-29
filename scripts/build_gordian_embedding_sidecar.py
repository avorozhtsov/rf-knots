#!/usr/bin/env python3
"""Build a provenance-preserving Gordian lower-bound supervision sidecar."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
from pathlib import Path

from rf_knots.embedding_gordian import SUPPORTED_SCHEMA, canonical_knot_id

SOURCE_PAGE = "https://www.isc.meiji.ac.jp/~mackysuzuki/lowerboundofgordiandistance.html"
DATA_PAGE = "https://www.isc.meiji.ac.jp/~mackysuzuki/lowerboundbyn12table.html"
PAPER = "Friedl-Kitayama-Suzuki: Blanchfield pairings and Gordian distance"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_split(knot_id: str) -> str:
    if knot_id == "knot:0_1":
        return "train"
    bucket = hashlib.sha256(knot_id.encode()).digest()[0] % 10
    return "validation" if bucket == 8 else "test" if bucket == 9 else "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-postings", type=int, default=16)
    args = parser.parse_args()
    shards = sorted(args.source_dir.glob("source-pairs-*.tsv"))
    if not shards:
        raise ValueError("no source pair shards")
    source_pairs = []
    source_ids = set()
    for shard in shards:
        with shard.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                left, right = row["left_source_id"], row["right_source_id"]
                lower = int(row["lower_bound"])
                if lower not in {2, 3}:
                    raise ValueError(f"unexpected bound {lower}")
                source_ids.update((left, right))
                source_pairs.append((left, right, lower, row["source_semantics"]))

    source = sqlite3.connect(f"file:{args.representations}?mode=ro", uri=True)
    available: dict[str, tuple[int, ...]] = {}
    for source_id in sorted(source_ids):
        mapped = canonical_knot_id(source_id)
        if mapped is None:
            continue
        knot_id, _mirror = mapped
        available[source_id] = tuple(
            int(row[0])
            for row in source.execute(
                "SELECT representation_id FROM representations WHERE knot_id=? "
                "ORDER BY word_length,strands,representation_id LIMIT ?",
                (knot_id, args.max_postings),
            )
        )
    source.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.part")
    if temporary.exists():
        temporary.unlink()
    output = sqlite3.connect(temporary)
    output.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=FULL;
        CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE source_knots(
          source_id TEXT PRIMARY KEY,
          canonical_knot_id TEXT,
          mirror_transform INTEGER NOT NULL CHECK(mirror_transform IN (0,1)),
          split TEXT NOT NULL CHECK(split IN ('train','validation','test','unmapped')),
          mapped INTEGER NOT NULL CHECK(mapped IN (0,1)),
          posting_count INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE representation_postings(
          source_id TEXT NOT NULL,
          rank INTEGER NOT NULL,
          representation_id INTEGER NOT NULL,
          split TEXT NOT NULL CHECK(split IN ('train','validation','test')),
          PRIMARY KEY(source_id,rank)
        ) WITHOUT ROWID;
        CREATE TABLE distance_pairs(
          pair_id INTEGER PRIMARY KEY,
          left_source_id TEXT NOT NULL,
          right_source_id TEXT NOT NULL,
          distance_lower REAL NOT NULL,
          distance_upper REAL NOT NULL,
          is_exact INTEGER NOT NULL CHECK(is_exact IN (0,1)),
          split TEXT NOT NULL CHECK(
            split IN (
              'train','validation','test','excluded_cross_split','unmapped'
            )
          ),
          provenance TEXT NOT NULL,
          UNIQUE(left_source_id,right_source_id)
        );
        CREATE INDEX distance_pairs_by_split ON distance_pairs(split,pair_id);
        """
    )
    source_digest = hashlib.sha256()
    for shard in shards:
        source_digest.update(bytes.fromhex(sha256(shard)))
    metadata = {
        "schema": SUPPORTED_SCHEMA,
        "source_page": SOURCE_PAGE,
        "data_page": DATA_PAGE,
        "paper": PAPER,
        "source_shards_sha256": source_digest.hexdigest(),
        "representations_path": str(args.representations.resolve()),
        "representations_sha256": sha256(args.representations),
        "split_contract": "sha256(canonical_knot_id)[0] mod 10; 8=validation,9=test; unknot=train",
        "label_contract": "2=[2,infinity); 3=[3,3] per source upper-sum statement; 0 omitted",
        "augmentation_contract": "independent replayed zero-CC scrambles depth 0..6 at load time",
    }
    output.executemany("INSERT INTO meta VALUES(?,?)", sorted(metadata.items()))
    endpoint_splits = {}
    for source_id in sorted(source_ids):
        mapped = canonical_knot_id(source_id)
        knot_id = mapped[0] if mapped else None
        mirror = int(mapped[1]) if mapped else 0
        postings = available.get(source_id, ())
        split = identity_split(knot_id) if knot_id and postings else "unmapped"
        endpoint_splits[source_id] = split
        output.execute(
            "INSERT INTO source_knots VALUES(?,?,?,?,?,?)",
            (source_id, knot_id, mirror, split, int(bool(postings)), len(postings)),
        )
        output.executemany(
            "INSERT INTO representation_postings VALUES(?,?,?,?)",
            ((source_id, rank, rep, split) for rank, rep in enumerate(postings)),
        )
    pair_counts = {}
    provenance = f"{PAPER}; {SOURCE_PAGE}"
    for pair_id, (left, right, lower, _semantics) in enumerate(source_pairs):
        left_split, right_split = endpoint_splits[left], endpoint_splits[right]
        if "unmapped" in {left_split, right_split}:
            split = "unmapped"
        elif left_split != right_split:
            split = "excluded_cross_split"
        else:
            split = left_split
        upper = 3.0 if lower == 3 else float("inf")
        output.execute(
            "INSERT INTO distance_pairs VALUES(?,?,?,?,?,?,?,?)",
            (pair_id, left, right, float(lower), upper, int(lower == 3), split, provenance),
        )
        pair_counts[split] = pair_counts.get(split, 0) + 1
    output.execute("PRAGMA optimize")
    integrity = output.execute("PRAGMA integrity_check").fetchone()[0]
    output.commit()
    output.close()
    if integrity != "ok":
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"SQLite integrity check failed: {integrity}")
    os.replace(temporary, args.output)
    report = {
        "schema": SUPPORTED_SCHEMA,
        "source_ids": len(source_ids),
        "source_pairs": len(source_pairs),
        "mapped_source_ids": sum(bool(available.get(value)) for value in source_ids),
        "unmapped_source_ids": sum(not available.get(value) for value in source_ids),
        "pair_splits": pair_counts,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "sqlite_integrity": integrity,
        "metadata": metadata,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
