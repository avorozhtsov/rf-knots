"""Score-blind ordinary graph TEST pairs from non-TRAIN source edges, protocol70."""

import argparse
import fcntl
import json
import os
import random
import sqlite3
from collections import Counter
from functools import cache
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, digest, sha
from build_ordinary_graph_metric_corpus import enumerate_paths
from burau_integer_check import determinant
from graph_evaluation_partition import partition_inputs
from probe_ordinary_graph_pairs import SOURCE, ordinary, replay_cc

from rf_knots.embedding_replay_corpus import canonical, invariants, owner

CONFIG = {"seed": 20261033, "per_category": 32, "per_query_group": 4}
EDGE_SQL = """SELECT p.pair_id,p.left_representation,p.right_representation
FROM pairs p JOIN representations l ON l.representation_id=p.left_representation
JOIN representations r ON r.representation_id=p.right_representation
WHERE p.metric='cc' AND p.split IN ('validation','test')
AND l.split=p.split AND r.split=p.split AND p.witness_semantic_length=1
AND p.provenance IN ('replayed_cc_upper_bound','replayed_cc_distinct_attested_knots')
ORDER BY p.pair_id"""


def cyclic_rotation(a, b):
    if a["strands"] != b["strands"] or len(a["word"]) != len(b["word"]):
        return None
    word, target = a["word"], b["word"]
    return next((k for k in range(max(1, len(word))) if word[k:] + word[:k] == target), None)


def orient(states, fingerprints, excluded):
    for query, reference in ((0, -1), (-1, 0)):
        a, b = fingerprints[query], fingerprints[reference]
        if owner(a) == "test" and (a[0], abs(a[1])) not in excluded and owner(b) != "quarantine":
            return query == -1
    return None


def select(rows, excluded, config):
    """Membership orientation precedes global group/category caps and pair dedup."""
    records, decisions, groups, classes, seen = [], [], Counter(), Counter(), set()
    for row in rows:
        states, fps = row["states"], row["fingerprints"]
        reverse = orient(states, fps, excluded)
        if reverse is None:
            decisions.append("no_fresh_query")
            continue
        if reverse:
            states, fps = list(reversed(states)), list(reversed(fps))
        a, b = states[0], states[-1]
        key = digest(sorted([canonical(tuple(s["word"]), s["strands"]) for s in (a, b)]))
        group = fps[0][0], abs(fps[0][1])
        lo, hi = row["lower"], row["upper"]
        category = str(lo) if lo == hi else "interval"
        reason = (
            "duplicate_pair"
            if key in seen
            else "group_cap"
            if groups[group] >= config["per_query_group"]
            else "class_cap"
            if classes[category] >= config["per_category"]
            else "accepted"
        )
        decisions.append(reason)
        if reason != "accepted":
            continue
        records.append(
            {
                "split": "test",
                "path_index": row["path_index"],
                "reversed": reverse,
                "source": row["source"],
                "states": states,
                "cc_positions": list(reversed(row["cc_positions"]))
                if reverse
                else row["cc_positions"],
                "fingerprints": fps,
                "left_fingerprint": fps[0],
                "right_fingerprint": fps[-1],
                "cyclic_rotation": cyclic_rotation(a, b),
                "interval": {
                    "metric": "cc",
                    "left": a,
                    "right": b,
                    "lower": lo,
                    "upper": hi,
                    "witness_semantic_length": len(row["cc_positions"]),
                },
            }
        )
        seen.add(key)
        groups[group] += 1
        classes[category] += 1
    adequate = len(records) >= 32 and len(groups) >= 8 and classes["2"] >= 8
    return records, decisions, dict(classes), len(groups), adequate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("new output required, or explicit resume")
    pins, groups, training, old_queries = partition_inputs()
    for path in (
        Path(__file__).resolve(),
        Path("research/70-graph-mixture-prospective-evaluation.md"),
    ):
        merge_pins(pins, {str(path): sha(path)})
    contract = {"config": CONFIG, "inputs": pins}
    check_pins(pins)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    progress = args.out / "progress.json"
    old = json.loads(progress.read_text()) if args.resume else None
    if old and old["contract"] != contract:
        raise ValueError("resume contract differs")
    if old and old["status"] == "completed":
        if sha(args.out / "panel.json") != old["panel_sha256"]:
            raise ValueError("completed panel changed")
        print("already completed, frozen inputs and panel verified")
        return
    source = sqlite3.connect(SOURCE.as_uri() + "?mode=ro", uri=True)
    if (
        dict(source.execute("SELECT key,value FROM meta")).get("schema")
        != "unknotdb-embedding-pairs-v3"
    ):
        raise ValueError("unsupported source schema")
    edges = list(source.execute(EDGE_SQL))
    paths = enumerate_paths(edges)
    random.Random(CONFIG["seed"]).shuffle(paths)
    rows = old["rows"] if old else []
    if len(rows) > len(paths) or any(
        r["source"] != p or r["path_index"] != i
        for i, (r, p) in enumerate(zip(rows, paths[: len(rows)], strict=True))
    ):
        raise ValueError("saved path prefix differs")

    @cache
    def state(rep):
        encoded, split = source.execute(
            "SELECT encoding,split FROM representations WHERE representation_id=?", (rep,)
        ).fetchone()
        if split not in {"validation", "test"}:
            raise ValueError("TRAIN state in held-out source paths")
        word, n = ordinary(bytes(encoded))
        fp = invariants(word, n)
        if fp[0] != determinant(word, n):
            raise ValueError("independent determinant mismatch")
        return {"word": list(word), "strands": n}, list(fp)

    def save(status, **extra):
        atomic(
            progress,
            {
                "status": status,
                "pid": os.getpid(),
                "contract": contract,
                "next_index": len(rows),
                "total_paths": len(paths),
                "source_edges": len(edges),
                "rows": rows,
                **extra,
            },
        )
        print(
            json.dumps({"status": status, "paths": len(rows), "total": len(paths), **extra}),
            flush=True,
        )

    save("building")
    for i in range(len(rows), len(paths)):
        selected = paths[i]
        values = [state(rep) for rep in selected["representations"]]
        states, fps = [v[0] for v in values], [v[1] for v in values]
        positions = [
            replay_cc((tuple(a["word"]), a["strands"]), (tuple(b["word"]), b["strands"]))
            for a, b in zip(states[:-1], states[1:], strict=True)
        ]
        lower = max(abs(fps[0][1] - fps[-1][1]) // 2, int(fps[0][0] != fps[-1][0]))
        upper = 0 if cyclic_rotation(states[0], states[-1]) is not None else len(positions)
        if lower > upper:
            raise ValueError("lower bound contradicts ordinary path")
        rows.append(
            {
                "path_index": i,
                "source": selected,
                "states": states,
                "fingerprints": fps,
                "cc_positions": positions,
                "lower": lower,
                "upper": upper,
            }
        )
        if len(rows) % 100 == 0:
            save("building")
    records, decisions, classes, count, adequate = select(rows, training | old_queries, CONFIG)
    panel = {
        "schema": "rf-knots-ordinary-graph-panel-v1",
        "partition": "test",
        "construction_complete": True,
        "complete": adequate,
        "contract": contract,
        "validation": records,
        "size_shift": [],
        "classes": classes,
        "groups": count,
        "training_groups": {k: sorted(v) for k, v in groups.items()},
        "historical_query_groups": sorted(old_queries),
        "selection_decisions": decisions,
        "selection_counts": dict(Counter(decisions)),
        "scope": "Ordinary general Gordian pairs, not necessarily unknotting-number pairs. "
        "No mirrored endpoints/quotient bounds. No model scores or synthetic top-up. "
        "Witness CC count can exceed zero when a separate signed-cyclic equality certifies d=0.",
    }
    check_pins(pins)
    atomic(args.out / "panel.json", panel)
    save(
        "completed",
        panel_sha256=sha(args.out / "panel.json"),
        adequate=adequate,
        classes=classes,
        groups=count,
    )
    source.close()


if __name__ == "__main__":
    main()
