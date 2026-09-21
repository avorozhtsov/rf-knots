"""Deterministic audited-supply conversion from research/41; no model operations."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

from rf_knots.embedding_replay_corpus import invariants, verify_record
from rf_knots.evidence import UnknotWitness


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n"); temp.replace(path)


def pair_key(row):
    pair = row["interval"]
    return tuple((pair[s]["strands"], tuple(pair[s]["word"])) for s in ("left", "right"))


def record(w, prefix, attempt, kind):
    endpoint = w.steps[prefix - 1].after if prefix else w.start
    left, right = invariants(w.start.word, w.start.strands), invariants(endpoint.word, endpoint.strands)
    lower = abs(left[1] - right[1]) // 2
    upper = sum(s.action.kind == "CROSSING_CHANGE" for s in w.steps[:prefix])
    stronger = max(lower, int(left[0] != right[0]))
    if stronger > upper:
        raise ValueError("invariant lower bound exceeds prefix CC count")
    row = {"split": "train", "source_attempt": attempt, "derivation": kind,
           "pair_prefix_length": prefix, "left_fingerprint": list(left), "right_fingerprint": list(right),
           "interval": {"metric": "cc", "left": w.start.to_dict(), "right": endpoint.to_dict(),
                        "lower": lower, "upper": upper, "witness_semantic_length": prefix},
           "determinant_difference_lower_bound": stronger, "witness": w.to_dict()}
    verify_record(row)
    return row


def derive(candidate):
    w = UnknotWitness.from_dict(candidate["witness"])
    result = [record(w, w.moves, candidate["index"], "root-anchor")]
    if w.crossing_changes == 2:
        prefix = next(i + 1 for i, s in enumerate(w.steps) if s.action.kind == "CROSSING_CHANGE")
        result.append(record(w, prefix, candidate["index"], "first-CC-pair"))
        suffix = UnknotWitness(w.steps[prefix - 1].after, w.max_len, w.max_strands,
                              w.steps[prefix:], w.cyclic_band_generators)
        result.append(record(suffix, suffix.moves, candidate["index"], "post-first-CC-anchor"))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    source, audit = read(args.data / "manifest.json"), read(args.data / "independence-audit.json")
    if source["status"] != "completed" or audit["status"] != "passed":
        raise ValueError("require complete supply and independent audit")
    if audit["inputs"][str(args.data / "manifest.json")] != sha(args.data / "manifest.json"):
        raise ValueError("audited supply changed")
    for name, digest in audit["inputs"].items():
        if sha(Path(name)) != digest:
            raise ValueError(f"audited input changed: {name}")
    if source["contract"]["inputs"][str(args.base)] != sha(args.base):
        raise ValueError("base differs from supply contract")
    paths = {Path(__file__), Path("research/41-diverse-short-u-corpus.md"), args.base,
             args.data / "manifest.json", args.data / "independence-audit.json",
             *Path("src/rf_knots").rglob("*.py")}
    inputs = {str(path): sha(path) for path in sorted(paths)}
    args.out.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.out / "manifest.json"
    if args.resume:
        old = read(manifest_path)
        if old["inputs"] != inputs:
            raise ValueError("resume inputs differ")
        if old["status"] == "completed":
            if sha(args.out / "corpus.json") != old["corpus_sha256"]:
                raise ValueError("completed corpus changed")
            print("already completed")
            return
    else:
        for path in paths:
            if path.suffix not in (".py", ".md"):
                continue
            relative = path.resolve().relative_to(Path.cwd())
            target = args.out / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    state = {"status": "building", "pid": os.getpid(), "inputs": inputs, "processed_candidates": 0}
    save(manifest_path, state)
    result = read(args.base)
    old_count = len(result["train"])
    holdouts = {s: json.dumps(result[s], sort_keys=True) for s in ("validation", "size_shift")}
    seen = {pair_key(r) for r in result["train"]}
    attempted = duplicate = 0
    for shard in source["shards"]:
        path = args.data / shard["file"]
        if sha(path) != shard["sha256"]:
            raise ValueError("candidate shard changed")
        for candidate in read(path):
            for row in derive(candidate):
                attempted += 1
                key = pair_key(row)
                if key in seen:
                    duplicate += 1
                    continue
                seen.add(key)
                row["index"] = len(result["train"])
                result["train"].append(row)
            state["processed_candidates"] += 1
            if state["processed_candidates"] % 32 == 0:
                state.update(added=len(result["train"]) - old_count, duplicate_pairs=duplicate)
                save(manifest_path, state)
                print(json.dumps({k: state[k] for k in ("processed_candidates", "added", "duplicate_pairs")}), flush=True)
    if state["processed_candidates"] != audit["accepted"] or attempted != 706:
        raise ValueError("unexpected complete conversion coverage")
    if not all(json.dumps(result[s], sort_keys=True) == text for s, text in holdouts.items()):
        raise ValueError("holdout objects changed during conversion")
    result["counts"]["train"] = len(result["train"])
    result["categories"]["train"] = dict(Counter(
        str(r["interval"]["lower"]) if r["interval"]["lower"] == r["interval"]["upper"] else "interval"
        for r in result["train"]))
    result["distinct_query_fingerprints"]["train"] = len({
        (r["left_fingerprint"][0], abs(r["left_fingerprint"][1])) for r in result["train"]})
    result["diverse_supply_addition"] = {"source_manifest_sha256": sha(args.data / "manifest.json"),
        "audit_sha256": sha(args.data / "independence-audit.json"), "base_sha256": sha(args.base),
        "attempted_additions": attempted, "added": len(result["train"]) - old_count,
        "duplicate_pairs": duplicate, "contract": "research/41; original signature bounds retained; "
        "determinant-difference bound stored separately; holdout objects unchanged"}
    if any(sha(Path(name)) != digest for name, digest in inputs.items()):
        raise ValueError("source changed during conversion")
    target = args.out / "corpus.json"
    if target.exists():
        if read(target) != result:
            raise ValueError("uncommitted corpus differs on deterministic replay")
    else:
        save(target, result)
    state.update(status="completed", corpus_sha256=sha(target), **result["diverse_supply_addition"],
                 train_records=len(result["train"]), train_query_groups=result["distinct_query_fingerprints"]["train"])
    save(manifest_path, state)
    print(json.dumps({k: state[k] for k in ("status", "train_records", "train_query_groups", "added")}), flush=True)


if __name__ == "__main__":
    main()
