"""Correct JSON word normalization and build immutable root-only F bank v3."""

import argparse
import fcntl
import json
import shutil
from collections import Counter
from pathlib import Path

import build_additive_root_only_f_bank_v2 as previous
from build_mcts_training_supervision import atomic, digest, sha
from rf_knots.embedding_replay_corpus import invariants, owner

PROTOCOL = Path("research/344-root-only-bank-word-normalization-correction-v3.md")
PREVIOUS_SOURCE = Path("scripts/build_additive_root_only_f_bank_v2.py")
PREVIOUS_SOURCE_SHA = "dd7e631e47c68021b24ca8f131982bfca809a39e4b33b6bde0d6f3bf1e71dbcb"


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("output exists; never overwrite an immutable artifact")
    pinned = ((previous.OLD / "manifest.json", previous.OLD_SHA),
              (previous.OLD_AUDIT, previous.OLD_AUDIT_SHA),
              (previous.ROSTER, previous.ROSTER_SHA),
              (previous.ROSTER_AUDIT, previous.ROSTER_AUDIT_SHA),
              (previous.SOURCE / "manifest.json", previous.SOURCE_SHA),
              (previous.SOURCE_AUDIT, previous.SOURCE_AUDIT_SHA),
              (previous.SCALEUP_AUDIT, previous.SCALEUP_AUDIT_SHA),
              (PREVIOUS_SOURCE, PREVIOUS_SOURCE_SHA))
    for path, expected in pinned:
        if sha(path) != expected:
            raise ValueError(f"pinned input SHA differs: {path}")
    old_manifest = read(previous.OLD / "manifest.json")
    source_manifest = read(previous.SOURCE / "manifest.json")
    source_audit = read(previous.SOURCE_AUDIT)
    if (old_manifest["status"] != "completed"
            or len(old_manifest["roots"]) != 1007
            or source_manifest["status"] != "completed"
            or source_manifest["next_task"] != 32
            or source_audit["F_root_indices"] != list(previous.ROOT_INDICES)
            or read(previous.SCALEUP_AUDIT)["distinct_new_F_usable_roots"] != 0):
        raise ValueError("old bank/search/audit coverage differs")
    roster = read(previous.ROSTER)
    for path, expected in roster["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"roster pin differs: {path}")
    for path, expected in source_manifest["contract"]["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"source search pin differs: {path}")
    for position, entry in enumerate(source_manifest["shards"]):
        if (entry["file"] != f"attempt-{position:04d}.json"
                or sha(previous.SOURCE / entry["file"]) != entry["sha256"]):
            raise ValueError(f"source search shard SHA differs: {position}")
    pins = tuple(path for path, _ in pinned) + (
        PROTOCOL, Path(__file__), PREVIOUS_SOURCE,
        Path("scripts/audit_f_short_positive_r2_yield_v1.py"),
        Path("scripts/mcts_tournament_signals_v1.py"),
        Path("scripts/mcts_training_supervision.py"))
    inputs = {str(path): sha(path) for path in pins}
    contract = {"schema": "rf-knots-additive-root-only-F-bank-v3",
                "inputs": inputs, "old_root_shards": 1007,
                "new_source_root_indices": list(previous.ROOT_INDICES),
                "new_gradient_scope": "exact_zero_spent_root_only",
                "F": {"seed": 20261006, "k": 4,
                      "near_margin": 2, "gap_cap": 4}}
    args.out.mkdir(parents=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entries, seen, tally = [], set(), Counter()
        for index, entry in enumerate(old_manifest["roots"]):
            source = previous.OLD / f"root-{index:04d}.json"
            target = args.out / source.name
            if entry["file"] != source.name or sha(source) != entry["sha256"]:
                raise ValueError(f"old bank shard SHA/order differs: {index}")
            row = read(source)
            key = tuple(row["query_group"])
            if key in seen or row["root_index"] != index:
                raise ValueError(f"old bank query group/index differs: {index}")
            seen.add(key)
            shutil.copyfile(source, target)
            if sha(target) != entry["sha256"]:
                raise ValueError(f"old bank shard copy differs: {index}")
            entries.append(entry)
            tally["retained_F_roots"] += bool(row["groups"])
            tally["retained_F_groups"] += len(row["groups"])
        roots = previous.selected_roots(roster)
        for offset, root_index in enumerate(previous.ROOT_INDICES):
            root = roots[root_index]
            key = tuple(root["query_group"])
            start = root["witness"]["start"]
            fingerprint = invariants(tuple(start["word"]), start["strands"])
            if key in seen or owner(fingerprint) != "train":
                raise ValueError("new query group duplicated or not TRAIN-owned")
            seen.add(key)
            index = 1007 + offset
            row = previous.new_row(index, root_index, root, source_manifest)
            target = args.out / f"root-{index:04d}.json"
            atomic(target, row)
            entries.append({"file": target.name, "sha256": sha(target)})
            tally["new_F_roots"] += 1
            tally["new_F_groups"] += len(row["groups"])
            tally["new_root_F_terms"] += sum(group["signal_terms"]
                                               for group in row["groups"])
        if ((tally["retained_F_roots"], tally["retained_F_groups"],
             tally["new_F_roots"], tally["new_F_groups"]) != (399, 701, 3, 4)):
            raise ValueError(f"bank accounting differs: {dict(tally)}")
        manifest = {"contract": contract, "contract_sha256": digest(contract),
                    "status": "completed", "next_root": len(entries),
                    "roots": entries, "summary": dict(tally)}
        atomic(args.out / "manifest.json", manifest)
        print(json.dumps({"manifest_sha256": sha(args.out / "manifest.json"),
                          "summary": dict(tally)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
