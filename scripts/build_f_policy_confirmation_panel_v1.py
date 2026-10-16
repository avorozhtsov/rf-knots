"""Build the score-blind, disjoint DEV panel frozen by protocol272."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from build_mcts_training_supervision import atomic, sha
from build_tournament_long_dev_panel_v1 import excluded_groups, group
from evaluate_embedding_cost_search import query_instance
from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_replay_corpus import invariants, owner


PROTOCOL = Path("research/272-f-policy-independent-dev-confirmation-v1.md")
OLD_LONG = Path("artifacts/mcts-tournament-long-dev-panel-v1/panel.json")
SEED = 20261119
ATTEMPTS = 2048
TARGET = 32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("confirmation panel exists; never overwrite")
    prior = json.loads(OLD_LONG.read_text())
    excluded = excluded_groups() | {tuple(row["query_group"]) for row in prior["rows"]}
    rows, seen = [], set()
    rejects = {"owner": 0, "excluded": 0, "duplicate": 0, "length": 0}
    ranked = sorted(range(ATTEMPTS), key=lambda i: hashlib.sha256(f"{SEED}:{i}".encode()).hexdigest())
    for index in ranked:
        witness = conjugate_witness(SEED, index, max_strands=20, min_depth=20, max_depth=60)
        witness.verify()
        fingerprint = invariants(witness.start.word, witness.start.strands)
        query_group = group(fingerprint)
        if owner(fingerprint) != "validation":
            rejects["owner"] += 1
            continue
        if query_group in excluded:
            rejects["excluded"] += 1
            continue
        if query_group in seen:
            rejects["duplicate"] += 1
            continue
        if not 60 <= len(witness.start.word) <= 150:
            rejects["length"] += 1
            continue
        base, fits = query_instance(index, witness, witness_caps=True)
        if not fits:
            raise ValueError("constructive witness exceeds shared resource cap")
        rows.append({
            "generator_index": index,
            "query_group": list(query_group),
            "word_length": len(witness.start.word),
            "reference_cc": witness.crossing_changes,
            "reference_moves": witness.moves,
            "witness": witness.to_dict(),
            "instances": [
                replace(base, cc_weight=weight, uid=f"f-confirmation-{index}-{weight}").to_dict()
                for weight in (10, 1000)
            ],
        })
        seen.add(query_group)
        if len(rows) == TARGET:
            break
    cc2 = sum(row["reference_cc"] == 2 for row in rows)
    if len(rows) != TARGET or cc2 < 8:
        raise ValueError(f"fixed panel insufficient: roots={len(rows)}, CC2={cc2}, rejects={rejects}")
    inputs = [PROTOCOL, OLD_LONG, Path("scripts/build_tournament_long_dev_panel_v1.py"), Path(__file__)]
    payload = {
        "schema": "rf-knots-f-policy-confirmation-panel-v1",
        "role": "fresh score-blind validation-only A/F confirmation",
        "selection": {"seed": SEED, "attempts": ATTEMPTS, "target": TARGET,
                      "length": [60, 150], "weights": [10, 1000], "rejects": rejects},
        "inputs": {str(path): sha(path) for path in inputs},
        "rows": rows,
    }
    args.out.mkdir(parents=True)
    atomic(args.out / "panel.json", payload)
    print(json.dumps({"roots": len(rows), "cc2": cc2,
                      "panel_sha256": sha(args.out / "panel.json")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
