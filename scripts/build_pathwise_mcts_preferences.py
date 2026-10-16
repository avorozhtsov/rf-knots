"""Recover TRAIN-only complete-success comparisons without a whole-condition veto.

This is a new preference overlay. It never changes frozen protocol55 targets or
turns a censored continuation into a negative action label.
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from build_mcts_training_supervision import condition_witnesses
from mcts_training_supervision import success_supervision

from rf_knots import mcts_loop as M


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new preference overlay must not overwrite an existing artifact")
    collection = json.loads((args.data / "manifest.json").read_text())
    audit = json.loads(args.audit.read_text())
    original = json.loads((args.original / "manifest.json").read_text())
    if (collection["status"], audit["status"], original["status"]) != (
        "completed", "passed", "completed"
    ):
        raise ValueError("require completed collection, passed audit and completed old targets")
    if audit["inputs"].get(str(args.data / "manifest.json")) != sha(args.data / "manifest.json"):
        raise ValueError("audit does not bind collection")
    if original["inputs"].get(str(args.audit)) != sha(args.audit):
        raise ValueError("old targets do not bind audit")
    pins = {
        **audit["inputs"],
        str(args.audit): sha(args.audit),
        str(args.original / "manifest.json"): sha(args.original / "manifest.json"),
        "scripts/build_pathwise_mcts_preferences.py": sha(Path(__file__)),
        "scripts/build_mcts_training_supervision.py": sha(
            Path("scripts/build_mcts_training_supervision.py")
        ),
        "scripts/mcts_training_supervision.py": sha(Path("scripts/mcts_training_supervision.py")),
    }
    if any(sha(Path(path)) != checksum for path, checksum in pins.items()):
        raise ValueError("frozen input or source hash differs")
    roots = json.loads((args.data / "roots.json").read_text())
    heldout = set(map(tuple, collection["heldout_query_groups"]))
    gates = {row["condition"]: row for row in audit["conditions"]}
    old_entries = {row["condition"]: row for row in original["conditions"]}
    if set(gates) != set(old_entries) or set(gates) != set(range(len(roots))):
        raise ValueError("condition coverage differs")
    args.out.mkdir(parents=True)
    counts, entries, seen = Counter(), [], set()
    for shard in collection["shards"]:
        shard_path = args.data / shard["file"]
        if sha(shard_path) != shard["sha256"]:
            raise ValueError("collection shard differs")
        for row in json.loads(shard_path.read_text()):
            index = row["condition"]
            if index in seen:
                raise ValueError("duplicate condition")
            seen.add(index)
            old_entry = old_entries[index]
            old_path = args.original / old_entry["file"]
            if sha(old_path) != old_entry["sha256"]:
                raise ValueError("old target shard differs")
            old = json.loads(old_path.read_text())
            root, gate = roots[index], gates[index]
            instance = M.Instance(**{**root["instance"], "word": tuple(root["instance"]["word"])})
            witnesses = condition_witnesses(instance, root, row, gate, heldout)
            target = success_supervision(instance, witnesses)
            target["positives"] = [
                {**positive, "state": asdict(positive["state"])}
                for positive in target["positives"]
            ]
            target["positives"] = json.loads(json.dumps(target["positives"]))
            if (
                target["positives"] != old["positives"]
                or target["unique_success_paths"] != old["unique_success_paths"]
                or target["distinct_success_costs"] != old["distinct_success_costs"]
            ):
                raise ValueError(
                    f"condition {index}: new overlay changed old eligible witness set or positives"
                )
            preferences = [
                {**preference, "state": asdict(preference["state"])}
                for preference in target["preferences"]
            ]
            preferences = json.loads(json.dumps(preferences))
            if old["preferences"] and preferences != old["preferences"]:
                raise ValueError("new overlay changed an old admitted preference")
            counts["conditions"] += 1
            counts["alternative_conditions"] += int(target["unique_success_paths"] > 1)
            counts["cost_varying_conditions"] += int(target["distinct_success_costs"] > 1)
            counts["preference_conditions"] += bool(preferences)
            counts["preference_terms"] += len(preferences)
            if preferences and not old["preferences"]:
                counts["recovered_conditions"] += 1
                counts["recovered_terms"] += len(preferences)
            if not preferences:
                continue
            payload = {
                "condition": index,
                "root": root["instance"],
                "old_target_sha256": old_entry["sha256"],
                "unique_success_paths": target["unique_success_paths"],
                "distinct_success_costs": target["distinct_success_costs"],
                "witness_sha256": sorted({
                    hashlib.sha256(json.dumps(w.to_dict(), sort_keys=True).encode()).hexdigest()
                    for w in witnesses
                }),
                "preferences": preferences,
            }
            path = args.out / f"preferences-{index:05d}.json"
            save(path, payload)
            entries.append({"condition": index, "file": path.name, "sha256": sha(path)})
    if seen != set(range(len(roots))):
        raise ValueError("incomplete condition coverage")
    if any(sha(Path(path)) != checksum for path, checksum in pins.items()):
        raise ValueError("frozen input changed during preparation")
    save(args.out / "manifest.json", {
        "schema": "rf-knots-mcts-pathwise-preferences-v1",
        "status": "prepared_unverified",
        "inputs": pins,
        "conditions": entries,
        "counts": dict(counts),
        "scope": "TRAIN-only complete successes; no censored negatives; no optimizer steps",
    })
    print(json.dumps(dict(counts), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
