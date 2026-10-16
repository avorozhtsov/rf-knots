"""Score-blind continuing TRAIN roster from legacy and corpus witnesses."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import screen_legacy_unselected_root_f_v1 as legacy
import build_f_curriculum_roster_v1 as historical
from screen_legacy_unselected_root_f_v2 import external_exclusions
from build_mcts_training_supervision import atomic, sha
from evaluate_embedding_cost_search import query_instance
from rf_knots.embedding_replay_corpus import invariants, owner, verify_record
from rf_knots.evidence import UnknotWitness

CORPUS = Path("artifacts/embedding-diverse-short-u-corpus-v1/corpus.json")
CORPUS_SHA = "f3655e7dbfcc5b6ec7448b1f747dc2a4e1235fe036ab07fec61cc18df05c8f14"
PROTOCOL = Path("research/376-continuous-train-mcts-data-stream-v1.md")
MAX_ROOTS = 256


def read(path):
    return json.loads(path.read_text())


def exclusions():
    groups = external_exclusions()
    states = set()
    for path in legacy.VIEWED:
        for row in read(path)["rows"]:
            groups.add(tuple(row["query_group"]))
            raw = row.get("witness")
            if raw:
                w = UnknotWitness.from_dict(raw)
                states.update((tuple(s.word), s.strands) for s in
                              (w.start, *(step.after for step in w.steps))
                              if len(s.word) >= 20)
    for path in legacy.ROSTERS:
        groups.update(tuple(row["query_group"])
                      for stage in read(path)["stages"] for row in stage)
    bank = read(legacy.BANK / "manifest.json")
    for entry in bank["roots"]:
        path = legacy.BANK / entry["file"]
        if sha(path) != entry["sha256"]:
            raise ValueError("old bank row SHA differs")
        groups.add(tuple(read(path)["query_group"]))
    return groups, states


def band(n):
    return 0 if n <= 64 else 1 if n <= 96 else 2 if n <= 128 else 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or sha(CORPUS) != CORPUS_SHA or sha(legacy.ROOTS) != legacy.ROOTS_SHA:
        raise ValueError("output exists or source SHA differs")
    excluded, states = exclusions()
    seen, buckets, rejected = set(), [[] for _ in range(4)], Counter()
    raw = [("legacy", index, row) for index, row in
           enumerate(read(legacy.ROOTS)) if index % 2 == 0]
    raw += [("corpus", index, row) for index, row in enumerate(read(CORPUS)["train"])]
    raw.sort(key=lambda item: hashlib.sha256(
        f"20261028:{item[0]}:{item[1]}".encode()).hexdigest())
    for origin, index, row in raw:
        fp = row["fingerprint"] if origin == "legacy" else row["left_fingerprint"]
        key = (fp[0], abs(fp[1]))
        if key in excluded or key in seen or owner(tuple(fp)) != "train":
            rejected["excluded_duplicate_or_owner"] += 1
            continue
        witness = (UnknotWitness.from_dict(row["reference_witness"])
                   if origin == "legacy" else verify_record(row))
        witness.verify()
        if (tuple(witness.start.word), witness.start.strands) in states:
            rejected["viewed_exact_state"] += 1
            continue
        checked = invariants(tuple(witness.start.word), witness.start.strands)
        if (checked[0], abs(checked[1])) != key or owner(checked) != "train":
            raise ValueError("root invariant ownership differs")
        base, fits = query_instance(index, witness, witness_caps=True)
        if not fits:
            rejected["reference_caps"] += 1
            continue
        stage = band(len(witness.start.word))
        candidate = {"origin": origin, "source_index": index,
                     "generator_index": index, "query_group": key,
                     "stage": stage, "word_length": len(witness.start.word),
                     "witness": witness.to_dict(),
                     "instances": [replace(base, cc_weight=weight,
                         uid=f"continuous-train-v1-{origin}-{index}-{weight}").to_dict()
                         for weight in (10, 1000)]}
        buckets[stage].append(candidate)
        seen.add(key)
        if len(seen) >= MAX_ROOTS:
            break
    rows = []
    while any(buckets):
        for bucket in buckets:
            if bucket:
                rows.append(bucket.pop(0))
    if len(rows) < 8 or len({tuple(r["query_group"]) for r in rows}) != len(rows):
        raise ValueError("insufficient distinct admissible TRAIN supply")
    pins = (CORPUS, legacy.ROOTS, legacy.SOURCE, PROTOCOL, Path(__file__),
            Path("scripts/screen_legacy_unselected_root_f_v2.py"),
            historical.LONG, historical.F_DEV, historical.TEST150,
            Path("scripts/build_f_curriculum_roster_v1.py"),
            Path("scripts/evaluate_embedding_cost_search.py"),
            Path("src/rf_knots/embedding_replay_corpus.py"),
            legacy.BANK / "manifest.json", *legacy.ROSTERS, *legacy.VIEWED)
    roster = {"schema": "rf-knots-continuous-TRAIN-roster-v1", "rows": rows,
              "inputs": {str(path): sha(path) for path in pins},
              "rejections": dict(rejected),
              "stage_counts": dict(Counter(r["stage"] for r in rows)),
              "role": "candidate roots; no F labels yet"}
    args.out.mkdir(parents=True)
    atomic(args.out / "roster.json", roster)
    print(json.dumps({"roster_sha256": sha(args.out / "roster.json"),
                      "roots": len(rows), "stages": roster["stage_counts"]}), flush=True)


if __name__ == "__main__":
    main()
