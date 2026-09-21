"""Extend frozen pair training with balanced short-u anchors; holdouts unchanged.

Every full path stays inside the original train invariant partition. No model
scores, test targets or successful validation searches influence generation.
Immutable output, atomic progress, a single-writer lock and hash-guarded resume.
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

from rf_knots.embedding_replay_corpus import (
    canonical,
    invariants,
    mirror,
    owner,
    path_fingerprints,
    verify_record,
)
from rf_knots.interleaved_generator import InterleavedConfig, generate, positive_braid_seed


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def anchor_record(witness, **provenance):
    witness.verify()
    left = invariants(witness.start.word, witness.start.strands)
    right = witness.steps[-1].after if witness.steps else witness.start
    if right.word or right.strands != 1:
        raise ValueError("anchor endpoint must be the actual empty one-braid")
    row = {
        **provenance,
        "split": "train",
        "task": "short-u-anchor",
        "pair_prefix_length": witness.moves,
        "left_fingerprint": list(left),
        "right_fingerprint": [1, 0],
        "interval": {
            "metric": "cc",
            "left": witness.start.to_dict(),
            "right": right.to_dict(),
            "lower": abs(left[1]) // 2,
            "upper": witness.crossing_changes,
            "witness_semantic_length": witness.moves,
        },
        "witness": witness.to_dict(),
    }
    verify_record(row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--attempts", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.count < 4 or args.count % 4:
        raise SystemExit("count must be a positive multiple of four")
    args.out.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.out / ".runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code = [
        Path(__file__),
        Path("src/rf_knots/embedding_replay_corpus.py"),
        Path("src/rf_knots/interleaved_generator.py"),
        Path("src/rf_knots/reference.py"),
        Path("src/rf_knots/evidence.py"),
        Path("src/rf_knots/seifert.py"),
        Path("src/rf_knots/actions.py"),
    ]
    contract = {
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"
        },
        "base_sha256": sha(args.base),
        "code_sha256": {str(p): sha(p) for p in code},
    }
    previous = json.loads((args.out / "progress.json").read_text()) if args.resume else None
    if previous and previous["contract"] != contract:
        raise SystemExit("resume source/config mismatch")
    if previous and previous["status"] == "completed":
        print("already completed; no duplicate writer")
        return
    if not args.resume:
        snapshot = args.out / "source_snapshot"
        snapshot.mkdir()
        for i, source in enumerate(code):
            shutil.copyfile(source, snapshot / f"{i:02d}-{source.name}")
    base = json.loads(args.base.read_text())
    if base["schema"] != "rf-knots-replay-fingerprint-corpus-v1":
        raise ValueError("unsupported base corpus")
    rows = previous["anchors"] if previous else []
    index = previous["next_index"] if previous else 0
    totals = Counter(
        str(r["interval"]["lower"])
        if r["interval"]["lower"] == r["interval"]["upper"]
        else "interval"
        for r in rows
    )
    fp_counts = Counter(
        (
            str(r["interval"]["lower"])
            if r["interval"]["lower"] == r["interval"]["upper"]
            else "interval",
            r["left_fingerprint"][0],
            abs(r["left_fingerprint"][1]),
        )
        for r in rows
    )
    seen = set()
    for r in [*base["train"], *rows]:
        w = verify_record(r)
        seen.add(canonical(w.start.word, w.start.strands))
    rejects = Counter(previous["rejects"] if previous else {})
    target = args.count // 4
    started = time.time()

    def save(status):
        payload = {
            "contract": contract,
            "status": status,
            "pid": os.getpid(),
            "next_index": index,
            "anchors": rows,
            "categories": dict(totals),
            "rejects": dict(rejects),
            "seconds_this_run": time.time() - started,
        }
        atomic(args.out / "progress.json", payload)
        print(
            json.dumps({k: v for k, v in payload.items() if k not in ("anchors", "contract")}),
            flush=True,
        )

    save("building")
    while index < args.attempts and len(rows) < args.count:
        if index and index % 1000 == 0:
            save("building")
        category = min(
            (k for k in ("0", "1", "2", "interval") if totals[k] < target), key=lambda k: totals[k]
        )
        # Directed positive seeds efficiently supply exact small-u examples;
        # independent mixed-sign unknot walks prevent a positive-only corpus.
        directed = category != "interval" and index % 2 == 0
        changes = 0 if directed or category == "0" else (1 if category == "1" else 2)
        cfg = InterleavedConfig(
            crossing_changes=changes,
            walk_length=12 + index % 13,
            settle_walk_length=6,
            target_length=18 + index % 9,
            target_strands=3 + (index // 2) % 3,
            max_len=40,
            max_strands=7,
            strand_band=1,
        )
        seed_rep = (
            positive_braid_seed(
                cfg.spec(),
                strands=2 + index % 4,
                genus=int(category),
                seed=args.seed + index,
                churn=6,
            )
            if directed
            else None
        )
        instance = generate(cfg, args.seed, index, seed_rep)
        w = instance.witness
        assert w is not None
        if (index // 2) % 2:
            w = mirror(w)
        index += 1
        fp = invariants(w.start.word, w.start.strands)
        if owner(fp) != "train":
            rejects["heldout_root"] += 1
            continue
        lo, hi = abs(fp[1]) // 2, w.crossing_changes
        actual_category = str(lo) if lo == hi else "interval"
        if actual_category != category:
            rejects["wrong_category"] += 1
            continue
        fpkey = (category, fp[0], abs(fp[1]))
        if category != "0" and fp_counts[fpkey] >= max(4, target // 2):
            rejects["category_fingerprint_cap"] += 1
            continue
        key = canonical(w.start.word, w.start.strands)
        if key in seen or len(w.start.word) > 28 or w.start.strands > 6:
            rejects["duplicate_or_size"] += 1
            continue
        if any(owner(f) != "train" for f in path_fingerprints(w)):
            rejects["heldout_path"] += 1
            continue
        row = anchor_record(
            w,
            seed=args.seed,
            index=index - 1,
            config=cfg.provenance(),
            seed_family=seed_rep.provenance() if seed_rep else "unknot-walk",
            mirrored=bool(((index - 1) // 2) % 2),
        )
        rows.append(row)
        totals[category] += 1
        fp_counts[fpkey] += 1
        seen.add(key)
        if len(rows) % 32 == 0:
            save("building")
    if len(rows) != args.count:
        save("partial")
        return
    corpus = {
        **base,
        "train": base["train"] + rows,
        "counts": {**base["counts"], "train": len(base["train"]) + len(rows)},
        "parent_corpus_sha256": contract["base_sha256"],
        "short_u_addition": {
            "count": len(rows),
            "categories": dict(totals),
            "fingerprint_category_counts": [[list(k), v] for k, v in sorted(fp_counts.items())],
        },
        "provenance_note": (
            "Attempt/rejection counts inherited from base; "
            "short-u generation has separate progress."
        ),
    }
    all_train = corpus["train"]
    corpus["categories"] = {
        **base["categories"],
        "train": dict(
            Counter(
                str(r["interval"]["lower"])
                if r["interval"]["lower"] == r["interval"]["upper"]
                else "interval"
                for r in all_train
            )
        ),
    }
    corpus["distinct_query_fingerprints"] = {
        **base["distinct_query_fingerprints"],
        "train": len(
            {(r["left_fingerprint"][0], abs(r["left_fingerprint"][1])) for r in all_train}
        ),
    }
    assert corpus["validation"] == base["validation"] and corpus["size_shift"] == base["size_shift"]
    atomic(args.out / "corpus.json", corpus)
    atomic(
        args.out / "manifest.json",
        {
            "status": "completed",
            "sha256": sha(args.out / "corpus.json"),
            "contract": contract,
            "counts": corpus["counts"],
            "short_u_addition": corpus["short_u_addition"],
        },
    )
    save("completed")


if __name__ == "__main__":
    main()
