"""Revised fixed-quota TRAIN candidate roster after documented v1 shortage."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import build_f_curriculum_roster_v1 as old
from build_mcts_training_supervision import atomic, sha
from evaluate_embedding_cost_search import query_instance
from rf_knots.actions import CROSSING_CHANGE, KIND_NAMES
from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_replay_corpus import invariants, owner


PROTOCOL = Path("research/274-f-curriculum-roster-stage1-supply-revision.md")
QUOTAS = (150, 283, 283, 284)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("roster output exists; never overwrite")
    excluded = old.excluded_groups()
    bands = [[] for _ in QUOTAS]
    seen = set(excluded)
    counts = Counter()
    order = sorted(range(old.MAX_INDEX),
                   key=lambda i: hashlib.sha256(f"{old.SEED}:{i}".encode()).hexdigest())
    for index in order:
        witness = conjugate_witness(old.SEED, index, max_strands=20, min_depth=8, max_depth=60)
        witness.verify()
        band = old.stage(len(witness.start.word), witness.crossing_changes)
        if band is None or len(bands[band]) == QUOTAS[band]:
            counts["size_or_filled"] += 1
            continue
        fingerprint = invariants(witness.start.word, witness.start.strands)
        query_group = old.group(fingerprint)
        if query_group in seen:
            counts["duplicate_or_excluded"] += 1
            continue
        cc_states = [witness.start] + [s.after for s in witness.steps
                                       if s.action.kind == KIND_NAMES[CROSSING_CHANGE]]
        if any(owner(fp := invariants(state.word, state.strands)) != "train"
               or old.group(fp) in excluded for state in cc_states):
            counts["ownership"] += 1
            continue
        base, fits = query_instance(index, witness, witness_caps=True)
        if not fits:
            raise ValueError("constructive witness exceeds fixed caps")
        bands[band].append({
            "generator_index": index,
            "stage": band,
            "query_group": list(query_group),
            "word_length": len(witness.start.word),
            "reference_cc": witness.crossing_changes,
            "reference_moves": witness.moves,
            "witness": witness.to_dict(),
            "instances": [
                replace(base, cc_weight=weight, uid=f"f-curriculum-{index}-{weight}").to_dict()
                for weight in (10, 1000)
            ],
        })
        seen.add(query_group)
        if all(len(rows) == quota for rows, quota in zip(bands, QUOTAS, strict=True)):
            break
    actual = [len(rows) for rows in bands]
    if actual != list(QUOTAS):
        raise ValueError(f"fixed v2 roster insufficient: {actual}, rejects={dict(counts)}")
    inputs = [
        PROTOCOL, old.PROTOCOL, old.LONG, old.SOURCE, old.ROOTS,
        old.SHORT_DEV, old.LONG_DEV, old.F_DEV, old.TEST150,
        Path("scripts/build_f_curriculum_roster_v1.py"), Path(__file__),
    ]
    payload = {
        "schema": "rf-knots-f-curriculum-candidate-roster-v2",
        "role": "TRAIN-only constructive candidates, not F-usable groups or network training",
        "selection": {"seed": old.SEED, "max_index": old.MAX_INDEX,
                      "quotas": list(QUOTAS), "rejections": dict(counts)},
        "inputs": {str(path): sha(path) for path in inputs},
        "stages": bands,
    }
    args.out.mkdir(parents=True)
    atomic(args.out / "roster.json", payload)
    print(json.dumps({"candidates": sum(actual), "per_stage": actual,
                      "roster_sha256": sha(args.out / "roster.json"),
                      "rejections": dict(counts)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
