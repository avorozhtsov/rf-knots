"""Fixed-effort, score-blind fresh-group panel; protocol in research note 28."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from rf_knots.embedding_replay_corpus import canonical, invariants, lift, owner, verify_record
from rf_knots.interleaved_generator import InterleavedConfig, generate


EXCLUSIONS = (
    "embedding-short-u-independent-test-v1",
    "embedding-multisize-clean-test-v1",
    "embedding-catalogue-independent-test-v1",
    "embedding-short-u-heldout-panel-v1",
    "embedding-external-independent-test-v1",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(fp):
    return fp[0], abs(fp[1])


def excluded_groups(panels):
    return {group(r[endpoint]) for p in panels for split in ("validation", "size_shift")
            for r in p[split] for endpoint in ("left_fingerprint", "right_fingerprint")}


def adequate(records):
    return len(records) >= 16 and len({group(r["left_fingerprint"]) for r in records}) >= 8


def atomic_json(path, data):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.out / ".runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = [Path("artifacts") / name / "panel.json" for name in EXCLUSIONS]
    code = [Path(__file__), Path("research/28-fresh-group-fixed-budget-test.md"), *[
        Path("src/rf_knots") / name for name in (
            "embedding_replay_corpus.py", "interleaved_generator.py", "reference.py",
            "evidence.py", "seifert.py", "actions.py")]]
    contract = {"attempts": 40000, "seed": 20261008, "cap_per_group": 4,
                "exclusion_sha256": {str(p): sha(p) for p in sources},
                "code_sha256": {str(p): sha(p) for p in code}}
    excluded = excluded_groups([json.loads(p.read_text()) for p in sources])
    old = json.loads((args.out / "progress.json").read_text()) if args.resume else None
    if old and old["contract"] != contract:
        raise ValueError("resume contract differs")
    if old and old["status"] == "completed":
        print("already completed; no duplicate generation")
        return
    if not old:
        snapshot = args.out / "source_snapshot"
        snapshot.mkdir()
        for i, source in enumerate(sources + code):
            shutil.copyfile(source, snapshot / f"{i:02d}-{source.name}")
        atomic_json(args.out / "excluded_groups.json", sorted(excluded))
    records = old["records"] if old else []
    counts = Counter(old["counts"] if old else {})
    seen, groups = set(), Counter()
    for r in records:
        w = verify_record(r)
        assert group(r["left_fingerprint"]) not in excluded
        seen.add(canonical(w.start.word, w.start.strands))
        groups[group(r["left_fingerprint"])] += 1
    index = old["next_index"] if old else 0
    started = time.time()

    def save(status):
        result = {"status": status, "pid": os.getpid(), "contract": contract,
                  "next_index": index, "records": records, "counts": dict(counts),
                  "groups": len(groups), "seconds_this_run": time.time() - started}
        atomic_json(args.out / "progress.json", result)
        print(json.dumps({k: result[k] for k in ("status", "next_index", "counts", "groups")}), flush=True)

    save("building")
    while index < contract["attempts"]:
        cfg = InterleavedConfig(crossing_changes=1 + index % 2,
            walk_length=24 + index % 25, settle_walk_length=12,
            target_length=32 + index % 33, target_strands=3 + (index // 2) % 7,
            max_len=80, max_strands=10, strand_band=1)
        w = generate(cfg, contract["seed"], index).witness
        assert w is not None
        w.verify()
        fp = invariants(w.start.word, w.start.strands)
        key, g = canonical(w.start.word, w.start.strands), group(fp)
        reason = ("other_partition" if owner(fp) != "test" else "previous_group" if g in excluded
                  else "duplicate" if key in seen else "group_cap" if groups[g] >= 4 else "accepted")
        if reason == "accepted":
            row = {"split": "test", "seed": contract["seed"], "index": index,
                   "config": cfg.provenance(), "left_fingerprint": list(fp),
                   "right_fingerprint": [1, 0], "pair_prefix_length": w.moves,
                   "witness": w.to_dict(), "interval": {"metric": "cc",
                       "left": w.start.to_dict(), "right": w.steps[-1].after.to_dict(),
                       "lower": abs(fp[1]) // 2, "upper": w.crossing_changes,
                       "witness_semantic_length": w.moves}}
            verify_record(row)
            records.append(row)
            seen.add(key)
            groups[g] += 1
        counts[reason] += 1
        index += 1
        if index % 500 == 0:
            save("building")
    save("writing_panel")
    large = []
    for r in records:
        w, _ = lift(verify_record(r), 33, 256)
        row = {**r, "evaluation_view": "size_shift", "witness": w.to_dict(),
               "pair_prefix_length": w.moves, "interval": {**r["interval"],
                   "left": w.start.to_dict(), "witness_semantic_length": w.moves}}
        verify_record(row)
        large.append(row)
    panel = {"schema": "rf-knots-short-u-panel-v1", "partition": "test",
             "complete": adequate(records), "construction_complete": True,
             "adequacy": "at least 8 groups and 16 representations", "contract": contract,
             "validation": records, "size_shift": large, "groups": len(groups),
             "selection": "fixed 40000 attempts, test owner, excludes five previous panels; first four per group",
             "caveat": "No model scoring; fingerprint groups are not identities. Novel relative to listed panels only."}
    atomic_json(args.out / "panel.json", panel)
    save("completed")


if __name__ == "__main__":
    main()
