"""Independently check all source paths, TEST selection, and ordinary CC bounds."""

import argparse
import json
import random
import sqlite3
from collections import Counter, defaultdict
from functools import cache
from itertools import combinations
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, sha
from build_ordinary_graph_test_panel import CONFIG, select
from burau_integer_check import determinant
from graph_evaluation_partition import partition_inputs
from probe_ordinary_graph_pairs import SOURCE, ordinary

from rf_knots.embedding_replay_corpus import invariants, owner


def independent_paths(db):
    rows = db.execute("""SELECT p.pair_id,p.left_representation,p.right_representation,
      p.metric,p.split,p.witness_semantic_length,p.provenance,l.split,r.split
      FROM pairs p JOIN representations l ON l.representation_id=p.left_representation
      JOIN representations r ON r.representation_id=p.right_representation
      WHERE p.split != 'train' ORDER BY p.pair_id""")
    paths, neighbors = [], defaultdict(list)
    for pid, left, right, metric, split, length, provenance, ls, rs in rows:
        if not (
            metric == "cc"
            and split in {"validation", "test"}
            and ls == rs == split
            and length == 1
            and provenance in {"replayed_cc_upper_bound", "replayed_cc_distinct_attested_knots"}
        ):
            continue
        paths.append({"representations": [left, right], "pairs": [pid]})
        neighbors[left].append((right, pid))
        neighbors[right].append((left, pid))
    edge_count = len(paths)
    for middle, entries in sorted(neighbors.items()):
        if len(entries) != len({p for p, _ in entries}):
            raise ValueError("duplicate physical CC edge")
        for (a, first), (b, second) in combinations(sorted(entries), 2):
            paths.append({"representations": [a, middle, b], "pairs": [first, second]})
    random.Random(CONFIG["seed"]).shuffle(paths)
    return edge_count, paths


def trace_bounds(row, fingerprint=invariants, determinant_check=determinant):
    states, positions = row["states"], row["cc_positions"]
    if len(states) != len(positions) + 1 or len(positions) not in (1, 2):
        raise ValueError("invalid ordinary path coverage")
    computed = []
    for i, state in enumerate(states):
        w, n = tuple(state["word"]), state["strands"]
        if (
            type(n) is not int
            or n < 1
            or any(type(g) is not int or g == 0 or abs(g) >= n for g in w)
        ):
            raise ValueError("invalid physical braid state")
        fp = fingerprint(w, n)
        if fp[0] != determinant_check(w, n):
            raise ValueError("independent determinant mismatch")
        computed.append(list(fp))
        if i < len(positions):
            p = positions[i]
            if type(p) is not int or not 0 <= p < len(w):
                raise ValueError("invalid CC position")
            after = states[i + 1]
            if after["strands"] != n or tuple(after["word"]) != w[:p] + (-w[p],) + w[p + 1 :]:
                raise ValueError("not a literal signed CC path")
    if computed != row["fingerprints"]:
        raise ValueError("stored signed fingerprint differs")
    left, right = states[0], states[-1]
    a, b = tuple(left["word"]), tuple(right["word"])
    rotations = [i for i in range(max(1, len(a))) if a[i:] + a[:i] == b]
    zero = left["strands"] == right["strands"] and bool(rotations)
    lo = max(abs(computed[0][1] - computed[-1][1]) // 2, int(computed[0][0] != computed[-1][0]))
    hi = 0 if zero else len(positions)
    if (row["lower"], row["upper"]) != (lo, hi) or lo > hi:
        raise ValueError("ordinary bound/cyclic certificate differs")
    return lo, hi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite audit")
    panel = json.loads(args.panel.read_text())
    progress_path = args.panel.with_name("progress.json")
    progress = json.loads(progress_path.read_text())
    pins, training_by_source, training, excluded = partition_inputs()
    merge_pins(pins, panel["contract"]["inputs"])
    merge_pins(
        pins, {str(p): sha(p) for p in (args.panel, progress_path, Path(__file__).resolve())}
    )
    check_pins(pins)
    if (
        panel["schema"] != "rf-knots-ordinary-graph-panel-v1"
        or panel["partition"] != "test"
        or not panel["construction_complete"]
        or panel["size_shift"] != []
        or panel["contract"]["config"] != CONFIG
        or progress["contract"] != panel["contract"]
        or progress["status"] != "completed"
        or progress["panel_sha256"] != sha(args.panel)
        or panel["training_groups"]
        != {k: [list(g) for g in sorted(v)] for k, v in training_by_source.items()}
        or panel["historical_query_groups"] != [list(g) for g in sorted(excluded)]
    ):
        raise ValueError("construction/partition contract differs")
    source = sqlite3.connect(SOURCE.as_uri() + "?mode=ro", uri=True)
    edges, paths = independent_paths(source)
    if (
        progress["source_edges"] != edges
        or progress["total_paths"] != len(paths)
        or progress["next_index"] != len(paths)
    ):
        raise ValueError("path enumeration coverage differs")

    @cache
    def fingerprint(word, strands):
        return invariants(word, strands)

    @cache
    def independent_det(word, strands):
        return determinant(word, strands)

    for index, (row, expected) in enumerate(zip(progress["rows"], paths, strict=True)):
        if row["path_index"] != index or row["source"] != expected:
            raise ValueError("fixed source permutation differs")
        for rep, physical in zip(expected["representations"], row["states"], strict=True):
            encoded, split = source.execute(
                "SELECT encoding,split FROM representations WHERE representation_id=?", (rep,)
            ).fetchone()
            if split not in {"validation", "test"} or ordinary(bytes(encoded)) != (
                tuple(physical["word"]),
                physical["strands"],
            ):
                raise ValueError("source representation/split differs")
        trace_bounds(row, fingerprint, independent_det)
        if (index + 1) % 250 == 0:
            print(json.dumps({"paths_checked": index + 1, "total": len(paths)}), flush=True)
    records, decisions, classes, group_count, adequate = select(
        progress["rows"], training | excluded, CONFIG
    )
    if (
        panel["validation"] != records
        or panel["selection_decisions"] != decisions
        or panel["selection_counts"] != dict(Counter(decisions))
        or panel["classes"] != classes
        or panel["groups"] != group_count
        or panel["complete"] != adequate
        or progress["adequate"] != adequate
        or progress["classes"] != classes
        or progress["groups"] != group_count
    ):
        raise ValueError("first-eligible selection/adequacy differs")
    queries = Counter()
    for record in records:
        a, b = record["left_fingerprint"], record["right_fingerprint"]
        g = a[0], abs(a[1])
        if owner(a) != "test" or owner(b) == "quarantine" or g in training | excluded:
            raise ValueError("held-out query collision or forbidden reference")
        queries[g] += 1
        p = record["interval"]
        if p["left"] != record["states"][0] or p["right"] != record["states"][-1]:
            raise ValueError("oriented endpoint mismatch")
        trace_bounds(
            {**record, "lower": p["lower"], "upper": p["upper"]}, fingerprint, independent_det
        )
        shift = record["cyclic_rotation"]
        if shift is not None:
            w, target = p["left"]["word"], p["right"]["word"]
            if (
                type(shift) is not int
                or not 0 <= shift < max(1, len(w))
                or w[shift:] + w[:shift] != target
                or p["upper"] != 0
            ):
                raise ValueError("cyclic-zero certificate failed")
    if any(v > CONFIG["per_query_group"] for v in queries.values()):
        raise ValueError("global query group cap exceeded")
    source.close()
    check_pins(pins)
    result = {
        "status": "passed",
        "evaluation_ready": adequate,
        "inputs": pins,
        "source_edges": edges,
        "source_paths_checked": len(paths),
        "selected_pairs": len(records),
        "query_groups": sorted(queries),
        "classes": classes,
        "query_train_intersection": [],
        "query_historical_intersection": [],
        "scope": "Every source path and deterministic selection checked; ordinary signed CC "
        "and cyclic-zero replay, signature and independent determinant bounds. "
        "Passing integrity does not override the fixed test-supply gate. "
        "Optimizer lineage uses hash-verified prior full replay certificates.",
    }
    atomic(args.out, result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"inputs", "query_groups"}}),
        flush=True,
    )


if __name__ == "__main__":
    main()
