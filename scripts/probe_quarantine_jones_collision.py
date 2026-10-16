"""Replay the protocol53 quarantine hit and test an exact Jones distinction.

Diagnostic only: a different Jones polynomial excludes identity with the B5
target knot, not membership in every protected near-neighbour set.
"""

import argparse
import hashlib
import json
from pathlib import Path

from rf_knots import mcts_loop as M
from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.embedding_replay_corpus import invariants
from rf_knots.interleaved_generator import BRITTENHAM_HERMILLER_B5
from rf_knots.invariants import jones_polynomial


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite Jones diagnostic")
    manifest = json.loads((args.data / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("collection is incomplete")
    roots_path = args.data / "roots.json"
    if sha(roots_path) != manifest["roots_sha256"]:
        raise ValueError("root identity differs")
    roots = json.loads(roots_path.read_text())
    condition = 51
    matching = []
    shard_pin = None
    for shard in manifest["shards"]:
        path = args.data / shard["file"]
        if sha(path) != shard["sha256"]:
            raise ValueError("collection shard differs")
        for row in json.loads(path.read_text()):
            if row["condition"] == condition:
                matching.append(row)
                shard_pin = {"file": shard["file"], "sha256": shard["sha256"]}
    if len(matching) != 1:
        raise ValueError("expected exactly one condition51 record")
    instance = M.Instance(**{
        **roots[condition]["instance"],
        "word": tuple(roots[condition]["instance"]["word"]),
    })
    row = matching[0]
    hits = []
    for episode in row["episodes"]:
        if [49, 0] not in episode["excluded_fingerprints"]:
            continue
        word, strands = instance.word, instance.strands
        for depth, action in enumerate(
            episode["tree_prefix"] + episode["continuation"]["actions"], start=1
        ):
            kind = instance.spec.decode(action)[0]
            word, strands = reference.apply(instance.spec, word, strands, action)
            if kind == CROSSING_CHANGE and invariants(word, strands) == (49, 0):
                hits.append({
                    "simulation": episode["simulation"],
                    "full_path_move": depth,
                    "word": list(word),
                    "strands": strands,
                    "jones": sorted([exponent, coefficient]
                                    for exponent, coefficient in jones_polynomial(
                                        word, strands
                                    ).items()),
                })
    if len(hits) != 1 or hits[0]["simulation"] != 8:
        raise ValueError("unexpected quarantine-hit coverage")
    target = sorted([exponent, coefficient] for exponent, coefficient in jones_polynomial(
        BRITTENHAM_HERMILLER_B5, 5
    ).items())
    report = {
        "schema": "rf-knots-quarantine-jones-collision-v1",
        "status": "passed" if hits[0]["jones"] != target else "same_jones_not_distinguished",
        "collection_manifest_sha256": sha(args.data / "manifest.json"),
        "roots_sha256": sha(roots_path),
        "shard": shard_pin,
        "script_sha256": sha(Path(__file__)),
        "condition": condition,
        "hit": hits[0],
        "protected_b5_jones": target,
        "scope": "not identical to B5 target knot; no clearance of near-neighbour quarantine",
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "move": hits[0]["full_path_move"],
        "word_length": len(hits[0]["word"]),
        "strands": hits[0]["strands"],
    }), flush=True)


if __name__ == "__main__":
    main()
