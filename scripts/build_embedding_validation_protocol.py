#!/usr/bin/env python3
"""Build a knot-identity validation/test split with a proof-graph quarantine."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "rf-knots-embedding-validation-protocol-v0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: int, value: str, salt: str) -> bytes:
    return hashlib.sha256(f"{seed}:{salt}:{value}".encode()).digest()


@dataclass(frozen=True)
class KnotStats:
    knot_id: str
    representations: int
    max_word_length: int
    mean_word_length: float
    max_u: int
    graph_nodes: tuple[int, ...]
    protected_source: bool

    @property
    def hard_score(self) -> tuple[int, int, int, int]:
        return (
            int(self.protected_source),
            self.max_word_length,
            self.max_u,
            int(math.log2(self.representations + 1) * 100),
        )


def cc0_closure(seeds: set[int], adjacency: dict[int, set[int]]) -> set[int]:
    seen = set(seeds)
    queue = deque(seeds)
    while queue:
        node = queue.popleft()
        for target in adjacency.get(node, ()):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return seen


def graph_buffer(
    anchor_nodes: set[int],
    zero: dict[int, set[int]],
    one: dict[int, set[int]],
    node_knot_ids: dict[int, set[str]],
    knot_nodes: dict[str, set[int]],
) -> tuple[set[int], set[int], set[str]]:
    exact = cc0_closure(anchor_nodes, zero)
    one_seeds = {
        target for node in exact for target in one.get(node, ()) if target not in exact
    }
    neighbour_knots = {
        knot_id for node in one_seeds for knot_id in node_knot_ids.get(node, ())
    }
    one_seeds.update(
        node for knot_id in neighbour_knots for node in knot_nodes.get(knot_id, ())
    )
    buffer = cc0_closure(one_seeds, zero) - exact
    neighbour_knots.update(
        knot_id for node in buffer for knot_id in node_knot_ids.get(node, ())
    )
    return exact, buffer, neighbour_knots


def choose_anchors(
    candidates: list[KnotStats],
    *,
    hard_count: int,
    random_count: int,
    seed: int,
    salt: str,
    forbidden: set[str] | None = None,
    forbidden_nodes: set[int] | None = None,
) -> list[tuple[KnotStats, str]]:
    forbidden = forbidden or set()
    forbidden_nodes = forbidden_nodes or set()
    eligible = [
        row
        for row in candidates
        if row.knot_id not in forbidden
        and not forbidden_nodes.intersection(row.graph_nodes)
    ]
    hard = sorted(
        eligible,
        key=lambda row: (
            tuple(-value for value in row.hard_score),
            stable_rank(seed, row.knot_id, salt + ":hard"),
        ),
    )[:hard_count]
    selected = {row.knot_id for row in hard}
    random_rows = sorted(
        (row for row in eligible if row.knot_id not in selected),
        key=lambda row: stable_rank(seed, row.knot_id, salt + ":random"),
    )[:random_count]
    if len(hard) != hard_count or len(random_rows) != random_count:
        raise RuntimeError("not enough eligible knot identities for split")
    return [(row, "hard") for row in hard] + [
        (row, "distribution_matched") for row in random_rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--hard-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--anchors", type=int, default=20)
    parser.add_argument("--random-fraction", type=float, default=0.30)
    parser.add_argument("--max-anchor-representations", type=int, default=32)
    args = parser.parse_args()
    random_count = round(args.anchors * args.random_fraction)
    hard_count = args.anchors - random_count
    if args.anchors < 2 or not 0 <= args.random_fraction < 1:
        raise ValueError("invalid anchor selection contract")

    proof = sqlite3.connect(f"file:{args.proof}?mode=ro", uri=True)
    key_to_node = {
        bytes(key): int(node)
        for key, node in proof.execute(
            "SELECT rep_key,node_id FROM node_keys WHERE key_kind=0"
        )
    }
    node_u = {
        int(node): int(value or 0)
        for node, value in proof.execute("SELECT node_id,u_upper_bound FROM nodes")
    }
    zero: dict[int, set[int]] = defaultdict(set)
    one: dict[int, set[int]] = defaultdict(set)
    for source, target, cost in proof.execute(
        "SELECT source_node,target_node,cc_cost FROM edges"
    ):
        adjacency = zero if int(cost) == 0 else one
        adjacency[int(source)].add(int(target))
        adjacency[int(target)].add(int(source))
    proof.close()

    hard = sqlite3.connect(f"file:{args.hard_cases}?mode=ro", uri=True)
    protected_ids = {
        int(row[0])
        for row in hard.execute(
            "SELECT DISTINCT representation_id FROM hard_case_representations"
        )
    }
    hard.close()
    pairs = sqlite3.connect(f"file:{args.pairs}?mode=ro", uri=True)
    representation_rows = list(
        pairs.execute(
            "SELECT representation_id,knot_id,origin_key,mirror_key,word_length "
            "FROM representations"
        )
    )
    pairs.close()
    representation_nodes: dict[int, set[int]] = {}
    node_representations: dict[int, set[int]] = defaultdict(set)
    knot_representations: dict[str, set[int]] = defaultdict(set)
    knot_nodes: dict[str, set[int]] = defaultdict(set)
    node_knot_ids: dict[int, set[str]] = defaultdict(set)
    lengths: dict[str, list[int]] = defaultdict(list)
    for representation_id, knot_id, origin, mirror, word_length in representation_rows:
        representation_id = int(representation_id)
        nodes = {
            key_to_node[key]
            for key in (bytes(origin), bytes(mirror))
            if key in key_to_node
        }
        representation_nodes[representation_id] = nodes
        for node in nodes:
            node_representations[node].add(representation_id)
        if knot_id is not None:
            knot_id = str(knot_id)
            knot_representations[knot_id].add(representation_id)
            knot_nodes[knot_id].update(nodes)
            lengths[knot_id].append(int(word_length))
            for node in nodes:
                node_knot_ids[node].add(knot_id)

    candidates = []
    for knot_id, representation_ids in knot_representations.items():
        if (
            knot_id == "knot:0_1"
            or not knot_nodes[knot_id]
            or len(representation_ids) > args.max_anchor_representations
            or bool(representation_ids.intersection(protected_ids))
        ):
            continue
        word_lengths = lengths[knot_id]
        candidates.append(
            KnotStats(
                knot_id,
                len(representation_ids),
                max(word_lengths),
                sum(word_lengths) / len(word_lengths),
                max((node_u.get(node, 0) for node in knot_nodes[knot_id]), default=0),
                tuple(sorted(knot_nodes[knot_id])),
                bool(representation_ids.intersection(protected_ids)),
            )
        )

    validation = choose_anchors(
        candidates,
        hard_count=hard_count,
        random_count=random_count,
        seed=args.seed,
        salt="validation",
    )
    validation_ids = {row.knot_id for row, _ in validation}
    validation_nodes = {
        node for row, _ in validation for node in row.graph_nodes
    }
    val_exact, val_buffer, val_neighbour_knots = graph_buffer(
        validation_nodes, zero, one, node_knot_ids, knot_nodes
    )
    test = choose_anchors(
        candidates,
        hard_count=hard_count,
        random_count=random_count,
        seed=args.seed,
        salt="test",
        forbidden=validation_ids | val_neighbour_knots,
        forbidden_nodes=val_exact | val_buffer,
    )
    test_nodes = {node for row, _ in test for node in row.graph_nodes}
    test_exact, test_buffer, test_neighbour_knots = graph_buffer(
        test_nodes, zero, one, node_knot_ids, knot_nodes
    )
    overlap = (val_exact | val_buffer).intersection(test_exact | test_buffer)

    split_data = {
        "validation": (
            validation,
            val_exact,
            val_buffer,
            val_neighbour_knots,
        ),
        "test": (test, test_exact, test_buffer, test_neighbour_knots),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.part")
    temporary.unlink(missing_ok=True)
    output = sqlite3.connect(temporary)
    output.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=FULL;
        CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE anchors(
          split TEXT NOT NULL,
          knot_id TEXT NOT NULL,
          selection_role TEXT NOT NULL,
          representation_count INTEGER NOT NULL,
          graph_node_count INTEGER NOT NULL,
          max_word_length INTEGER NOT NULL,
          mean_word_length REAL NOT NULL,
          max_u INTEGER NOT NULL,
          protected_source INTEGER NOT NULL,
          PRIMARY KEY(split,knot_id)
        ) WITHOUT ROWID;
        CREATE TABLE quarantine_nodes(
          split TEXT NOT NULL,
          node_id INTEGER NOT NULL,
          reason TEXT NOT NULL,
          cc_radius INTEGER NOT NULL CHECK(cc_radius IN (0,1)),
          PRIMARY KEY(split,node_id)
        ) WITHOUT ROWID;
        CREATE TABLE quarantine_knot_ids(
          split TEXT NOT NULL,
          knot_id TEXT NOT NULL,
          reason TEXT NOT NULL,
          cc_radius INTEGER NOT NULL CHECK(cc_radius IN (0,1)),
          PRIMARY KEY(split,knot_id)
        ) WITHOUT ROWID;
        CREATE TABLE quarantine_representations(
          split TEXT NOT NULL,
          representation_id INTEGER NOT NULL,
          reason TEXT NOT NULL,
          cc_radius INTEGER NOT NULL CHECK(cc_radius IN (0,1)),
          PRIMARY KEY(split,representation_id)
        ) WITHOUT ROWID;
        """
    )
    metadata = {
        "schema": SCHEMA,
        "seed": str(args.seed),
        "anchors_per_split": str(args.anchors),
        "hard_anchors_per_split": str(hard_count),
        "distribution_anchors_per_split": str(random_count),
        "max_anchor_representations": str(args.max_anchor_representations),
        "protected_hard_cases": "reserved challenge; excluded from identity anchors",
        "neighbourhood": "undirected CC0* then <=1 CC then CC0*, with identified-knot propagation",
        "pairs_sha256": sha256(args.pairs),
        "proof_sha256": sha256(args.proof),
        "hard_cases_sha256": sha256(args.hard_cases),
    }
    output.executemany("INSERT INTO meta VALUES(?,?)", sorted(metadata.items()))
    summary = {}
    for split, (anchors, exact, buffer, neighbour_knots) in split_data.items():
        anchor_ids = {row.knot_id for row, _ in anchors}
        output.executemany(
            "INSERT INTO anchors VALUES(?,?,?,?,?,?,?,?,?)",
            (
                (
                    split,
                    row.knot_id,
                    role,
                    row.representations,
                    len(row.graph_nodes),
                    row.max_word_length,
                    row.mean_word_length,
                    row.max_u,
                    int(row.protected_source),
                )
                for row, role in anchors
            ),
        )
        output.executemany(
            "INSERT INTO quarantine_nodes VALUES(?,?,?,?)",
            (
                (split, node, "anchor_cc0_closure", 0)
                for node in sorted(exact)
            ),
        )
        output.executemany(
            "INSERT INTO quarantine_nodes VALUES(?,?,?,?)",
            (
                (split, node, "verified_one_cc_buffer", 1)
                for node in sorted(buffer)
            ),
        )
        knot_rows = {
            knot_id: ("anchor_identity", 0)
            for knot_id in anchor_ids
        }
        knot_rows.update(
            {
                knot_id: ("identified_one_cc_neighbour", 1)
                for knot_id in neighbour_knots - anchor_ids
            }
        )
        output.executemany(
            "INSERT INTO quarantine_knot_ids VALUES(?,?,?,?)",
            (
                (split, knot_id, reason, radius)
                for knot_id, (reason, radius) in sorted(knot_rows.items())
            ),
        )
        representation_rows_out = {}
        for knot_id, (reason, radius) in knot_rows.items():
            for representation_id in knot_representations.get(knot_id, ()):
                representation_rows_out[representation_id] = (reason, radius)
        for node in exact:
            for representation_id in node_representations.get(node, ()):
                representation_rows_out[representation_id] = ("cc0_closure", 0)
        for node in buffer:
            for representation_id in node_representations.get(node, ()):
                representation_rows_out.setdefault(
                    representation_id, ("verified_one_cc_buffer", 1)
                )
        output.executemany(
            "INSERT INTO quarantine_representations VALUES(?,?,?,?)",
            (
                (split, representation_id, reason, radius)
                for representation_id, (reason, radius) in sorted(
                    representation_rows_out.items()
                )
            ),
        )
        summary[split] = {
            "anchors": len(anchors),
            "anchor_knot_ids": sorted(anchor_ids),
            "cc0_nodes": len(exact),
            "one_cc_buffer_nodes": len(buffer),
            "quarantine_knot_ids": len(knot_rows),
            "quarantine_representations": len(representation_rows_out),
        }
    output.commit()
    integrity = output.execute("PRAGMA integrity_check").fetchone()[0]
    output.close()
    if integrity != "ok":
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"SQLite integrity failed: {integrity}")
    os.replace(temporary, args.output)
    report = {
        "schema": SCHEMA,
        "status": "validated" if not overlap else "invalid_split_overlap",
        "sqlite_integrity": integrity,
        "validation_test_quarantine_node_overlap": len(overlap),
        "splits": summary,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "metadata": metadata,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if overlap:
        raise RuntimeError("validation and test quarantine buffers overlap")


if __name__ == "__main__":
    main()
