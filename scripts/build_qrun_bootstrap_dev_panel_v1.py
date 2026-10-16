"""Fresh, score-blind short/long validation panel for Q-run-like bootstrap."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from build_f_curriculum_roster_v1 import excluded_groups, group
from build_mcts_training_supervision import atomic, sha
from evaluate_embedding_cost_search import query_instance
from rf_knots.actions import CROSSING_CHANGE, KIND_NAMES
from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_replay_corpus import invariants, owner

SEED = 20261204
MAX_INDEX = 16000
QUOTAS = (16, 16)
ROSTER = Path("artifacts/f-curriculum-roster-v3/roster.json")
PROTOCOL = Path("research/281-qrun-bootstrap-independent-dev-panel-v1.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("panel exists; never overwrite")
    roster = json.loads(ROSTER.read_text())
    excluded = excluded_groups() | {
        tuple(row["query_group"]) for stage in roster["stages"] for row in stage
    }
    bands = [[], []]
    seen = set(excluded)
    rejects = {"length": 0, "duplicate_or_excluded": 0, "ownership": 0}
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
        identity = group(invariants(witness.start.word, witness.start.strands))
        if identity in seen:
            rejects["duplicate_or_excluded"] += 1
            continue
        states = [witness.start] + [step.after for step in witness.steps
                                    if step.action.kind == KIND_NAMES[CROSSING_CHANGE]]
        if any(owner(fp := invariants(state.word, state.strands)) != "validation"
               or group(fp) in excluded for state in states):
            rejects["ownership"] += 1
            continue
        base, fits = query_instance(index, witness, witness_caps=True)
        if not fits:
            raise ValueError("reference path exceeds common caps")
        bands[band].append({"generator_index": index, "band": band,
                            "query_group": list(identity), "word_length": length,
                            "reference_cc": witness.crossing_changes,
                            "reference_moves": witness.moves,
                            "witness": witness.to_dict(),
                            "instances": [replace(base, cc_weight=weight,
                                uid=f"qrun-bootstrap-dev-{index}-{weight}").to_dict()
                                          for weight in (10, 1000)]})
        seen.add(identity)
        if list(map(len, bands)) == list(QUOTAS):
            break
    if list(map(len, bands)) != list(QUOTAS):
        raise ValueError(f"fixed validation supply insufficient: {[len(b) for b in bands]}, {rejects}")
    pins = (ROSTER, PROTOCOL, Path(__file__),
            Path("scripts/build_f_curriculum_roster_v1.py"))
    payload = {"schema": "rf-knots-qrun-bootstrap-dev-panel-v1",
               "role": "fresh validation only, no training or checkpoint selection before freeze",
               "selection": {"seed": SEED, "max_index": MAX_INDEX,
                             "quotas": QUOTAS, "rejects": rejects},
               "inputs": {str(path): sha(path) for path in pins},
               "rows": bands[0] + bands[1]}
    args.out.mkdir(parents=True)
    atomic(args.out / "panel.json", payload)
    print(json.dumps({"panel_sha256": sha(args.out / "panel.json"),
                      "bands": list(map(len, bands)), "rejects": rejects}, sort_keys=True))


if __name__ == "__main__":
    main()
