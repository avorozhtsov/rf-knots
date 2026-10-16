"""Score-blind validation roots and exact-state-disjoint reference paths."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from build_f_curriculum_roster_v1 import excluded_groups, group
from build_mcts_training_supervision import atomic, sha
from evaluate_embedding_cost_search import query_instance
from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

ROSTER = Path("artifacts/f-curriculum-roster-v3/roster.json")
PROTOCOL = Path("research/282-qrun-bootstrap-dev-panel-state-disjoint-v2.md")
SEED, MAX_INDEX, QUOTAS = 20261204, 16000, (16, 16)


def state_keys(witness):
    # The terminal empty one-braid is common to all valid unknot witnesses.
    return {(tuple(s.word), s.strands) for s in
            [witness.start, *(step.after for step in witness.steps)]
            if s.word}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("v2 output exists; never overwrite")
    roster = json.loads(ROSTER.read_text())
    train_states = set()
    train_groups = set()
    for stage in roster["stages"]:
        for row in stage:
            witness = UnknotWitness.from_dict(row["witness"])
            witness.verify()
            train_states.update(state_keys(witness))
            train_groups.add(tuple(row["query_group"]))
    excluded = excluded_groups() | train_groups
    bands = [[], []]
    seen = set(excluded)
    rejects = {"length": 0, "root_owner": 0, "excluded_or_duplicate": 0,
               "exact_state_overlap": 0}
    ranked = sorted(range(MAX_INDEX),
                    key=lambda i: hashlib.sha256(f"{SEED}:{i}".encode()).hexdigest())
    for index in ranked:
        witness = conjugate_witness(SEED, index, max_strands=20,
                                    min_depth=8, max_depth=60)
        witness.verify()
        length = len(witness.start.word)
        band = 0 if 20 <= length <= 64 else 1 if 129 <= length <= 192 else None
        if band is None or len(bands[band]) == QUOTAS[band]:
            rejects["length"] += 1
            continue
        fingerprint = invariants(witness.start.word, witness.start.strands)
        identity = group(fingerprint)
        if owner(fingerprint) != "validation":
            rejects["root_owner"] += 1
            continue
        if identity in seen:
            rejects["excluded_or_duplicate"] += 1
            continue
        if state_keys(witness) & train_states:
            rejects["exact_state_overlap"] += 1
            continue
        base, fits = query_instance(index, witness, witness_caps=True)
        if not fits:
            raise ValueError("reference exceeds shared caps")
        bands[band].append({"generator_index": index, "band": band,
                            "query_group": list(identity), "word_length": length,
                            "reference_cc": witness.crossing_changes,
                            "reference_moves": witness.moves,
                            "witness": witness.to_dict(),
                            "instances": [replace(base, cc_weight=weight,
                                uid=f"qrun-bootstrap-dev-v2-{index}-{weight}").to_dict()
                                          for weight in (10, 1000)]})
        seen.add(identity)
        if list(map(len, bands)) == list(QUOTAS):
            break
    if list(map(len, bands)) != list(QUOTAS):
        raise ValueError(f"v2 fixed supply insufficient: {[len(b) for b in bands]}, {rejects}")
    pins = (ROSTER, PROTOCOL, Path(__file__),
            Path("scripts/build_f_curriculum_roster_v1.py"))
    payload = {"schema": "rf-knots-qrun-bootstrap-dev-panel-v2",
               "role": "root-validation and exact-state-disjoint DEV; invariant transitions may cross buckets",
               "selection": {"seed": SEED, "max_index": MAX_INDEX,
                             "quotas": QUOTAS, "rejects": rejects,
                             "train_exact_state_count": len(train_states)},
               "inputs": {str(path): sha(path) for path in pins},
               "rows": bands[0] + bands[1]}
    args.out.mkdir(parents=True)
    atomic(args.out / "panel.json", payload)
    print(json.dumps({"panel_sha256": sha(args.out / "panel.json"),
                      "bands": list(map(len, bands)), "rejects": rejects,
                      "train_exact_state_count": len(train_states)}, sort_keys=True))


if __name__ == "__main__":
    main()
