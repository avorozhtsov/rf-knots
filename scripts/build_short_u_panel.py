"""Held-out u<=2 queries generated from replayed walks from the unknot.

This panel is evaluation-only. Equal fingerprint groups, including mirrors, are
assigned by the SAME partition as the training corpus. No search/model scores
are used to select it. Partial construction is persisted and resumable.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

from rf_knots.embedding_replay_corpus import canonical, invariants, lift, owner, verify_record
from rf_knots.interleaved_generator import InterleavedConfig, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--attempts", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-per-group", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_per_group < 1:
        raise SystemExit("max-per-group must be positive")
    if args.out.exists() and not args.resume:
        raise SystemExit("output exists; explicitly resume instead of overwriting")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"
    }
    previous = json.loads((args.out / "progress.json").read_text()) if args.resume else None
    code = [
        Path(__file__),
        Path("src/rf_knots/embedding_replay_corpus.py"),
        Path("src/rf_knots/interleaved_generator.py"),
        Path("src/rf_knots/reference.py"),
        Path("src/rf_knots/evidence.py"),
        Path("src/rf_knots/seifert.py"),
    ]
    code_sha = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    if previous and previous["config"] != config:
        raise SystemExit("resume configuration differs")
    if previous and previous.get("code_sha256") != code_sha:
        raise SystemExit("resume code differs; use the original frozen builder")
    if previous and previous["status"] == "completed" and (args.out / "panel.json").exists():
        print("completed panel already exists; no duplicate generation")
        return
    if not previous:
        snapshot = args.out / "source_snapshot"
        snapshot.mkdir()
        for i, source in enumerate(code):
            shutil.copyfile(source, snapshot / f"{i:02d}-{source.name}")
    records = previous["validation"] if previous else []
    index = previous["next_index"] if previous else 0
    seen, groups = set(), Counter()
    for row in records:
        w = verify_record(row)
        seen.add(canonical(w.start.word, w.start.strands))
        det, sig = row["left_fingerprint"]
        groups[(det, abs(sig))] += 1
    started = time.time()

    def save(status, next_index):
        result = {
            "schema": "rf-knots-short-u-panel-v1",
            "config": config,
            "partition": args.partition,
            "pid": os.getpid(),
            "code_sha256": code_sha,
            "status": status,
            "next_index": next_index,
            "validation": records,
            "distinct_fingerprint_groups": len(groups),
            "seconds_this_run": time.time() - started,
            "interval_counts": dict(
                Counter(f"{r['interval']['lower']}:{r['interval']['upper']}" for r in records)
            ),
        }
        temp = args.out / ".progress.tmp"
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(args.out / "progress.json")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("config", "validation", "code_sha256")
                }
            ),
            flush=True,
        )

    save("building", index)
    while index < args.attempts and len(records) < args.count:
        cfg = InterleavedConfig(
            crossing_changes=1 + index % 2,
            walk_length=12 + index % 13,
            settle_walk_length=6,
            target_length=18 + index % 12,
            target_strands=3 + (index // 2) % 3,
            max_len=40,
            max_strands=7,
            strand_band=1,
        )
        instance = generate(cfg, args.seed, index)
        index += 1
        w = instance.witness
        assert w is not None
        w.verify()
        det, sig = invariants(w.start.word, w.start.strands)
        key = canonical(w.start.word, w.start.strands)
        fp = (det, abs(sig))
        if owner(fp) == args.partition and key not in seen and groups[fp] < args.max_per_group:
            row = {
                "split": args.partition,
                "config": cfg.provenance(),
                "seed": args.seed,
                "index": index - 1,
                "pair_prefix_length": w.moves,
                "left_fingerprint": [det, sig],
                "right_fingerprint": [1, 0],
                "interval": {
                    "metric": "cc",
                    "left": w.start.to_dict(),
                    "right": w.steps[-1].after.to_dict(),
                    "lower": abs(sig) // 2,
                    "upper": w.crossing_changes,
                    "witness_semantic_length": w.moves,
                },
                "witness": w.to_dict(),
            }
            verify_record(row)
            records.append(row)
            seen.add(key)
            groups[fp] += 1
            if len(records) % 8 == 0:
                save("building", index)
        if index % 1000 == 0:
            save("building", index)
    save("writing_panel" if len(records) == args.count else "partial", index)
    large = []
    for row in records:
        w, prefix = lift(verify_record(row), 9, 60)
        lifted = {
            **row,
            "split": "test" if args.partition == "test" else "size_shift",
            "evaluation_view": "size_shift",
            "pair_prefix_length": w.moves,
            "witness": w.to_dict(),
            "interval": {
                **row["interval"],
                "left": w.start.to_dict(),
                "witness_semantic_length": w.moves,
            },
        }
        verify_record(lifted)
        large.append(lifted)
    panel = {
        "schema": "rf-knots-short-u-panel-v1",
        "partition": args.partition,
        "field_contract": (
            "validation/size_shift are evaluation-view names; "
            "record split and partition determine ownership"
        ),
        "validation": records,
        "size_shift": large,
        "selection": "same invariant-group holdout partition; no model-based selection",
        "complete": len(records) == args.count,
    }
    temp = args.out / ".panel.tmp"
    temp.write_text(json.dumps(panel, indent=2) + "\n")
    temp.replace(args.out / "panel.json")
    save("completed" if len(records) == args.count else "partial", index)


if __name__ == "__main__":
    main()
