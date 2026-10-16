"""Audit raw root-search outcomes for protocol264's fixed-budget head pilot."""

import argparse
import json
from collections import Counter
from pathlib import Path

from bootstrap_cost_value_teacher_v1 import ACTOR, ACTOR_SHA, ROOTS, SOURCE
from build_mcts_training_supervision import atomic, sha
from rf_knots import mcts_loop as M
from rf_knots.embedding_replay_corpus import owner
from rf_knots.evidence import UnknotWitness


TEACHER = Path("artifacts/raster-budgeted-value-teacher-unselected-v1")
PROTOCOL = Path("research/264-root-budget16-multioutput-value-bootstrap-v1.md")
SHORT_DEV = Path("artifacts/mcts-tournament-dev-panel-v1/panel.json")
LONG_DEV = Path("artifacts/mcts-tournament-long-dev-panel-v1/panel.json")
TEST150 = Path("artifacts/long-mixture-fresh-short-u-test-v1/panel.json")


def read(path):
    return json.loads(path.read_text())


def known_excluded_starts():
    starts = set()
    for panel in (read(SHORT_DEV), read(LONG_DEV)):
        for row in panel["rows"]:
            instance = row["instances"][0]
            starts.add((tuple(instance["word"]), instance["strands"]))
    for row in read(TEST150)["validation"]:
        start = row["witness"]["start"]
        starts.add((tuple(start["word"]), start["strands"]))
    return starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("derived value labels already exist; never overwrite")
    manifest = read(TEACHER / "manifest.json")
    roots, source = read(ROOTS), read(SOURCE)
    if (manifest["contract"]["schema"] != "rf-knots-budgeted-value-teacher-unselected-v1"
            or manifest["next_task"] < 2560 or len(roots) != 256
            or source["status"] != "completed" or sha(ROOTS) != source["roots_sha256"]
            or sha(ACTOR) != ACTOR_SHA):
        raise ValueError("frozen actor, TRAIN roster, or S10 teacher identity differs")
    if any(sha(Path(p)) != h for p, h in manifest["contract"]["inputs"].items()):
        raise ValueError("teacher input pin changed")
    heldout = {tuple(x) for x in source["heldout_query_groups"]}
    excluded_starts = known_excluded_starts()
    expected = [(first + offset, seed) for seed in manifest["contract"]["search_seeds"]
                for first in manifest["contract"]["root_order"] for offset in (0, 1)]
    if len(expected) != 2560 or manifest["contract"]["budgets"][0] != 16:
        raise ValueError("expected complete first S10 budget boundary")
    labels, stats = [], Counter()
    for position, (root_index, seed) in enumerate(expected):
        entry = manifest["shards"][position]
        path = TEACHER / entry["file"]
        if entry["file"] != f"attempt-{position:05d}.json" or sha(path) != entry["sha256"]:
            raise ValueError(f"teacher shard {position} changed")
        payload = read(path)
        root = roots[root_index]
        instance = M.Instance(**{**root["instance"], "word": tuple(root["instance"]["word"])})
        fingerprint = tuple(root["fingerprint"])
        query_group = (fingerprint[0], abs(fingerprint[1]))
        if (owner(fingerprint) != "train" or query_group in heldout
                or (instance.word, instance.strands) in excluded_starts):
            raise ValueError("TRAIN root overlaps an excluded DEV/TEST query")
        row = payload["row"]
        if (payload["root_index"], payload["budget"], payload["seed"]) != (root_index, 16, seed):
            raise ValueError("teacher task order differs")
        if (row["condition"] != root_index or row["cc_weight"] != instance.cc_weight
                or row["simulations"] > 16 or bool(row["witness"]) != bool(row["solved"])):
            raise ValueError("teacher result differs from the budget-16 contract")
        cc = moves = None
        if row["solved"]:
            witness = UnknotWitness.from_dict(row["witness"])
            witness.verify()
            if ((witness.start.word, witness.start.strands) != (instance.word, instance.strands)
                    or witness.crossing_changes > instance.cap_crossing_changes
                    or witness.moves > instance.cap_moves
                    or row["L10"] != 10 * witness.crossing_changes + witness.moves
                    or row["L1000"] != 1000 * witness.crossing_changes + witness.moves):
                raise ValueError("best attempt witness, resource, or cost differs")
            cc, moves = witness.crossing_changes, witness.moves
        labels.append({
            "root_index": root_index,
            "base_root_index": root["base_root_index"],
            "seed": seed,
            "lambda": instance.cc_weight,
            "local_simulations": 16,
            "cc_left": instance.cap_crossing_changes,
            "moves_left": instance.cap_moves,
            "solved": bool(row["solved"]),
            "cc_remaining": cc,
            "moves_remaining": moves,
        })
        stats["attempts"] += 1
        stats["solved"] += bool(row["solved"])
        stats[f"lambda_{instance.cc_weight}_solved"] += bool(row["solved"])
        stats["historically_admitted_solved"] += row["eligible_solved_episodes"] > 0
    inputs = {str(p): sha(p) for p in (
        TEACHER / "manifest.json", ROOTS, SOURCE, ACTOR, PROTOCOL,
        SHORT_DEV, LONG_DEV, TEST150, Path(__file__),
    )}
    payload = {
        "schema": "rf-knots-root-budget16-multioutput-labels-v1",
        "role": "root-query-group TRAIN bootstrap at frozen b=16; not internal-node supervision",
        "inputs": inputs,
        "teacher_source_contract_schema": manifest["contract"]["schema"],
        "stats": dict(stats),
        "labels": labels,
    }
    args.out.mkdir(parents=True)
    atomic(args.out / "labels.json", payload)
    print(json.dumps({"labels": len(labels), "stats": dict(stats),
                      "sha256": sha(args.out / "labels.json")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
