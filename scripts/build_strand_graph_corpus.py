"""Build an immutable replay-verified corpus for joint distance/policy training."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from rf_knots.embedding_replay_corpus import build_corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--validation-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()

    def write(name, value):
        temp = args.out / f".{name}.tmp"
        temp.write_text(json.dumps(value, indent=2) + "\n")
        temp.replace(args.out / name)

    def progress(value):
        row = {"status": "building", "pid": os.getpid(), "seconds": time.time() - started, **value}
        write("manifest.json", row)
        print(json.dumps(row), flush=True)

    progress({"seed": args.seed})
    corpus = build_corpus(args.train_count, args.validation_count, args.seed, progress)
    write("corpus.json", corpus)
    write(
        "manifest.json",
        {
            "status": "completed",
            "seed": args.seed,
            "seconds": time.time() - started,
            "sha256": hashlib.sha256((args.out / "corpus.json").read_bytes()).hexdigest(),
            **{
                k: corpus[k]
                for k in (
                    "counts",
                    "categories",
                    "distinct_query_fingerprints",
                    "attempts",
                    "rejects",
                    "split_contract",
                )
            },
        },
    )


if __name__ == "__main__":
    main()
