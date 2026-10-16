"""Select 1,000 distinct TRAIN-owned constructive roots for protocol273.

This builds an input roster. It does not collect F paths or train a network.
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from build_mcts_training_supervision import atomic, sha
from evaluate_embedding_cost_search import query_instance
from rf_knots.actions import CROSSING_CHANGE, KIND_NAMES
from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_replay_corpus import invariants, owner


PROTOCOL = Path("research/273-f-dominant-curriculum-two-architecture-plan-v1.md")
LONG = Path("artifacts/long-general-cc-supply-v1/corpus.json")
SOURCE = Path("artifacts/diverse-cost-mcts-training-v1/manifest.json")
ROOTS = Path("artifacts/diverse-cost-mcts-training-v1/roots.json")
SHORT_DEV = Path("artifacts/mcts-tournament-dev-panel-v1/panel.json")
LONG_DEV = Path("artifacts/mcts-tournament-long-dev-panel-v1/panel.json")
F_DEV = Path("artifacts/f-policy-confirmation-panel-v1/panel.json")
TEST150 = Path("artifacts/long-mixture-fresh-short-u-test-v1/panel.json")
SEED = 20261125
MAX_INDEX = 16000
PER_STAGE = 250


def group(fingerprint):
    return fingerprint[0], abs(fingerprint[1])


def excluded_groups():
    long, source = json.loads(LONG.read_text()), json.loads(SOURCE.read_text())
    excluded = {tuple(row) for row in long["excluded_query_groups"]}
    excluded.update(tuple(row) for row in source["heldout_query_groups"])
    excluded.update(group(row["fingerprint"]) for row in json.loads(ROOTS.read_text()))
    for panel in (SHORT_DEV, LONG_DEV, F_DEV):
        excluded.update(tuple(row["query_group"])
                        for row in json.loads(panel.read_text())["rows"])
    excluded.update(group(row["left_fingerprint"])
                    for row in json.loads(TEST150.read_text())["validation"])
    return excluded


def stage(length, cc):
    if length <= 64 and cc == 1:
        return 0
    if 65 <= length <= 96:
        return 1
    if 97 <= length <= 128:
        return 2
    if 129 <= length <= 192:
        return 3
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("roster output exists; never overwrite")
    excluded = excluded_groups()
    bands = [[] for _ in range(4)]
    seen = set(excluded)
    counts = Counter()
    order = sorted(range(MAX_INDEX), key=lambda i: hashlib.sha256(f"{SEED}:{i}".encode()).hexdigest())
    for index in order:
        witness = conjugate_witness(SEED, index, max_strands=20, min_depth=8, max_depth=60)
        witness.verify()
        band = stage(len(witness.start.word), witness.crossing_changes)
        if band is None or len(bands[band]) == PER_STAGE:
            counts["size_or_filled"] += 1
            continue
        fingerprint = invariants(witness.start.word, witness.start.strands)
        query_group = group(fingerprint)
        if query_group in seen:
            counts["duplicate_or_excluded"] += 1
            continue
        cc_states = [witness.start] + [s.after for s in witness.steps
                                       if s.action.kind == KIND_NAMES[CROSSING_CHANGE]]
        if any(owner(fp := invariants(state.word, state.strands)) != "train"
               or group(fp) in excluded for state in cc_states):
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
        if all(len(rows) == PER_STAGE for rows in bands):
            break
    if any(len(rows) != PER_STAGE for rows in bands):
        raise ValueError(f"fixed roster insufficient: {[len(rows) for rows in bands]}, rejects={dict(counts)}")
    inputs = [PROTOCOL, LONG, SOURCE, ROOTS, SHORT_DEV, LONG_DEV, F_DEV, TEST150, Path(__file__)]
    payload = {
        "schema": "rf-knots-f-curriculum-candidate-roster-v1",
        "role": "TRAIN-only constructive candidates, not F-usable groups or network training",
        "selection": {"seed": SEED, "max_index": MAX_INDEX, "per_stage": PER_STAGE,
                      "rejections": dict(counts)},
        "inputs": {str(path): sha(path) for path in inputs},
        "stages": bands,
    }
    args.out.mkdir(parents=True)
    atomic(args.out / "roster.json", payload)
    print(json.dumps({"candidates": sum(map(len, bands)),
                      "roster_sha256": sha(args.out / "roster.json"),
                      "rejections": dict(counts)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
