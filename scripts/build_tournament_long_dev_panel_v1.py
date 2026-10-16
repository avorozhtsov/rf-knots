"""Freeze score-blind, disjoint long DEV queries for protocol262."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from build_mcts_training_supervision import atomic, sha
from evaluate_embedding_cost_search import query_instance
from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_replay_corpus import invariants, owner


PROTOCOL = Path("research/262-tournament-long-dev-replication-v1.md")
LONG = Path("artifacts/long-general-cc-supply-v1/corpus.json")
SHORT = Path("artifacts/mcts-tournament-dev-panel-v1/panel.json")
TEST150 = Path("artifacts/long-mixture-fresh-short-u-test-v1/panel.json")
SEED = 20261117
ATTEMPTS = 512
TARGET = 16


def group(fingerprint):
    return fingerprint[0], abs(fingerprint[1])


def excluded_groups():
    long = json.loads(LONG.read_text())
    short = json.loads(SHORT.read_text())
    test = json.loads(TEST150.read_text())
    excluded = {tuple(x) for x in long["excluded_query_groups"]}
    for groups in long["training_groups"].values():
        excluded.update(map(tuple, groups))
    for row in long["records"]["train"]:
        excluded.update(map(group, row["fingerprints"]))
    excluded.update(tuple(row["query_group"]) for row in short["rows"])
    excluded.update(group(row["left_fingerprint"]) for row in test["validation"])
    return excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("long DEV panel exists; never overwrite")
    excluded = excluded_groups()
    rows, seen, counts = [], set(), {"owner": 0, "excluded": 0, "length": 0, "duplicate": 0}
    ranked = sorted(range(ATTEMPTS), key=lambda i: hashlib.sha256(f"{SEED}:{i}".encode()).hexdigest())
    cc2 = 0
    for index in ranked:
        witness = conjugate_witness(SEED, index, max_strands=20, min_depth=20, max_depth=60)
        witness.verify()
        fingerprint = invariants(witness.start.word, witness.start.strands)
        query_group = group(fingerprint)
        if owner(fingerprint) != "validation":
            counts["owner"] += 1
            continue
        if query_group in excluded:
            counts["excluded"] += 1
            continue
        if query_group in seen:
            counts["duplicate"] += 1
            continue
        if not 60 <= len(witness.start.word) <= 150:
            counts["length"] += 1
            continue
        base, fits = query_instance(index, witness, witness_caps=True)
        if not fits:
            raise ValueError("constructive witness exceeds common resource cap")
        rows.append({
            "generator_index": index,
            "query_group": list(query_group),
            "word_length": len(witness.start.word),
            "reference_cc": witness.crossing_changes,
            "reference_moves": witness.moves,
            "witness": witness.to_dict(),
            "instances": [
                replace(base, cc_weight=weight, uid=f"tournament-long-dev-{index}-{weight}").to_dict()
                for weight in (10, 1000)
            ],
        })
        seen.add(query_group)
        cc2 += witness.crossing_changes == 2
        if len(rows) == TARGET:
            break
    if len(rows) != TARGET or cc2 < 4:
        raise ValueError(f"fixed candidate procedure insufficient: roots={len(rows)}, CC2={cc2}, rejects={counts}")
    payload = {
        "schema": "rf-knots-tournament-long-dev-panel-v1",
        "role": "fresh validation-owned long DEV; no search result used for selection",
        "selection": {"seed": SEED, "attempts": ATTEMPTS, "target": TARGET,
                      "length": [60, 150], "weights": [10, 1000], "rejects": counts},
        "inputs": {str(p): sha(p) for p in (PROTOCOL, LONG, SHORT, TEST150, Path(__file__))},
        "rows": rows,
    }
    args.out.mkdir(parents=True)
    atomic(args.out / "panel.json", payload)
    print(json.dumps({"rows": len(rows), "cc2": cc2,
                      "lengths": [min(r["word_length"] for r in rows), max(r["word_length"] for r in rows)],
                      "panel_sha256": sha(args.out / "panel.json")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
