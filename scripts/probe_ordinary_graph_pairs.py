"""Read-only feasibility probe: replay ordinary one/two-CC graph-local paths.

Not training-ready data: reports owner support without a complete historical
identity exclusion audit. Source graph-edge/mirror-quotient bounds are not used.
"""

import argparse
import fcntl
import json
import os
import random
import sqlite3
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins
from build_mcts_training_supervision import atomic, sha
from burau_integer_check import determinant

from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE, ActionSpec
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.embedding_supervision import decode_ukb0

SOURCE = Path("../unknotdb/outputs/unknotdb-embedding-pairs-v4.sqlite").resolve()
SOURCE_SHA = "297e420f7a97aefb15615419176102ddd31ac492acc130218c8c2e236a4bceea"
CONFIG = {"seed": 20261030, "single_paths": 256, "double_paths": 256}


def ordinary(encoded):
    if len(encoded) < 12 or encoded[5] != 0:
        raise ValueError("require ordinary Artin codec without flags")
    state = decode_ukb0(0, encoded, None, "train")
    if state.strands < 1 or any(g == 0 or abs(g) >= state.strands for g in state.word):
        raise ValueError("invalid ordinary Artin word")
    return state.word, state.strands


def replay_cc(before, after):
    word, n = before
    target, target_n = after
    if n != target_n or len(word) != len(target):
        raise ValueError("CC endpoints differ in size")
    positions = [p for p, (a, b) in enumerate(zip(word, target, strict=True)) if a != b]
    if len(positions) != 1 or word[positions[0]] != -target[positions[0]]:
        raise ValueError("endpoints are not one literal CC apart")
    spec = ActionSpec(max(1, len(word)), max(2, n))
    action = spec.encode(CROSSING_CHANGE, positions[0])
    if not reference.is_legal(spec, word, n, action, True):
        raise ValueError("illegal reconstructed CC")
    if reference.apply(spec, word, n, action) != after:
        raise ValueError("reconstructed CC did not replay")
    return positions[0]


def select_paths(edges, config):
    if len(edges) < config["single_paths"]:
        raise ValueError("not enough single-CC rows for fixed probe")
    adjacency = defaultdict(list)
    for pair, a, b in edges:
        adjacency[a].append((b, pair))
        adjacency[b].append((a, pair))
    middles = sorted(k for k, v in adjacency.items() if len({r[0] for r in v}) >= 2)
    if len(middles) < config["double_paths"]:
        raise ValueError("not enough distinct two-CC midpoints for fixed probe")
    rng = random.Random(config["seed"])
    paths = [
        {"representations": [a, b], "pairs": [p]}
        for p, a, b in rng.sample(edges, config["single_paths"])
    ]
    for middle in rng.sample(middles, config["double_paths"]):
        # Sidecar pairs are unique at unordered endpoints for this metric.
        neighbors = adjacency[middle]
        if len({r[0] for r in neighbors}) != len(neighbors):
            raise ValueError("duplicate CC neighbor")
        (a, first), (b, second) = rng.sample(neighbors, 2)
        paths.append({"representations": [a, middle, b], "pairs": [first, second]})
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("new output required, or explicit resume")
    paths = [
        Path(__file__).resolve(),
        Path("research/65-ordinary-graph-pair-probe.md"),
        Path("scripts/burau_integer_check.py"),
        Path("scripts/build_mcts_training_supervision.py"),
        Path("scripts/audit_joint_policy_metric_panel.py"),
        *Path("src/rf_knots").glob("*.py"),
    ]
    pins = {str(SOURCE): SOURCE_SHA, **{str(p): sha(p) for p in paths}}
    check_pins(pins)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report_path = args.out / "report.json"
    previous = json.loads(report_path.read_text()) if args.resume else None
    if previous and (previous["inputs"] != pins or previous["config"] != CONFIG):
        raise ValueError("resume inputs/config differ")
    if previous and previous["status"] == "completed":
        print("already completed, pinned inputs verified")
        return
    db = sqlite3.connect(SOURCE.as_uri() + "?mode=ro", uri=True)
    metadata = dict(db.execute("SELECT key,value FROM meta"))
    if metadata["schema"] != "unknotdb-embedding-pairs-v3":
        raise ValueError("unexpected sidecar schema")
    # Only exact instruction-local one-CC provenance; never graph-path bounds.
    edges = list(
        db.execute("""
        SELECT p.pair_id,p.left_representation,p.right_representation
        FROM pairs p JOIN representations l ON l.representation_id=p.left_representation
        JOIN representations r ON r.representation_id=p.right_representation
        WHERE p.metric='cc' AND p.split='train' AND l.split='train' AND r.split='train'
        AND p.witness_semantic_length=1 AND p.provenance IN
          ('replayed_cc_upper_bound','replayed_cc_distinct_attested_knots')
        ORDER BY p.pair_id
    """)
    )
    selection = select_paths(edges, CONFIG)
    rows = previous["rows"] if previous else []
    if len(rows) > len(selection) or any(
        r["source"] != s for r, s in zip(rows, selection[: len(rows)], strict=True)
    ):
        raise ValueError("resumed source selection differs")

    @cache
    def state(rep):
        row = db.execute(
            "SELECT encoding FROM representations WHERE representation_id=?", (rep,)
        ).fetchone()
        if row is None:
            raise ValueError("missing source representation")
        return ordinary(bytes(row[0]))

    @cache
    def fingerprint(rep):
        word, strands = state(rep)
        value = invariants(word, strands)
        if value[0] != determinant(word, strands):
            raise ValueError("independent determinant mismatch")
        return value

    def save(status):
        summary = Counter()
        groups = set()
        for row in rows:
            label = f"{row['upper']}-CC/" + (
                f"exact-{row['lower']}"
                if row["lower"] == row["upper"]
                else f"interval-{row['lower']}-{row['upper']}"
            )
            summary[label] += 1
            if row["all_owner_train"]:
                summary["owner_train/" + label] += 1
                groups.update((f[0], abs(f[1])) for f in row["fingerprints"])
        atomic(
            report_path,
            {
                "status": status,
                "pid": os.getpid(),
                "config": CONFIG,
                "inputs": pins,
                "train_ready": False,
                "source_cc_edges": len(edges),
                "next_index": len(rows),
                "rows": rows,
                "summary": dict(summary),
                "owner_train_groups": sorted(groups),
                "scope": "Ordinary CC replay and independent determinant checks; "
                "no complete historical exclusion audit, no training or model scores.",
            },
        )
        print(
            json.dumps({"status": status, "next_index": len(rows), "summary": dict(summary)}),
            flush=True,
        )

    save("running")
    for selection_row in selection[len(rows) :]:
        reps = selection_row["representations"]
        states = [state(r) for r in reps]
        positions = [replay_cc(a, b) for a, b in zip(states[:-1], states[1:], strict=True)]
        fps = [fingerprint(r) for r in reps]
        lower = max(abs(fps[0][1] - fps[-1][1]) // 2, int(fps[0][0] != fps[-1][0]))
        if lower > len(positions):
            raise ValueError("certified lower bound exceeds reconstructed path")
        rows.append(
            {
                "source": selection_row,
                "cc_positions": positions,
                "states": [{"word": list(w), "strands": n} for w, n in states],
                "fingerprints": fps,
                "owners": [owner(f) for f in fps],
                "all_owner_train": all(owner(f) == "train" for f in fps),
                "lower": lower,
                "upper": len(positions),
            }
        )
        if len(rows) % 16 == 0:
            save("running")
    check_pins(pins)
    save("completed")
    db.close()


if __name__ == "__main__":
    main()
