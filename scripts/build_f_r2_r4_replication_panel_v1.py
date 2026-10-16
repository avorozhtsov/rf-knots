"""Fresh score-blind validation roots for the bounded R2/R4 replication."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from build_f_curriculum_roster_v1 import excluded_groups, group
from build_mcts_training_supervision import atomic, sha
from build_f_r2_r4_dev_panel_v1 import panel_groups, substantial_states
from evaluate_embedding_cost_search import query_instance
from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness


ROSTER = Path("artifacts/f-curriculum-roster-v3/roster.json")
VIEWED = tuple(Path(f"artifacts/{name}/panel.json") for name in (
    "mcts-tournament-dev-panel-v1",
    "mcts-tournament-long-dev-panel-v1",
    "qrun-bootstrap-dev-panel-v3",
    "qrun-rung1-dev-panel-v1",
    "qrun-equal-cost-dev-panel-v1",
    "f-r2-r4-fresh-dev-panel-v1",
))
PROTOCOL = Path("research/312-f-r2-r4-bounded-dev-replication-v1.md")
SEED, MAX_INDEX, QUOTAS = 20261010, 32000, (16, 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new panel output must not exist")
    roster = json.loads(ROSTER.read_text())
    train_states, train_groups = set(), set()
    for stage in roster["stages"]:
        for row in stage:
            witness = UnknotWitness.from_dict(row["witness"])
            witness.verify()
            train_states.update(substantial_states(witness))
            train_groups.add(tuple(row["query_group"]))
    viewed_groups = set().union(*(panel_groups(path) for path in VIEWED))
    seen = excluded_groups() | train_groups | viewed_groups
    bands = [[], []]
    rejects = {"length": 0, "root_owner": 0, "excluded_or_duplicate": 0,
               "substantial_state_overlap": 0}
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
        if substantial_states(witness) & train_states:
            rejects["substantial_state_overlap"] += 1
            continue
        base, fits = query_instance(index, witness, witness_caps=True)
        if not fits:
            raise ValueError("reference exceeds shared resource caps")
        bands[band].append({"generator_index": index, "band": band,
                            "query_group": list(identity), "word_length": length,
                            "reference_cc": witness.crossing_changes,
                            "reference_moves": witness.moves,
                            "witness": witness.to_dict(),
                            "instances": [replace(base, cc_weight=weight,
                                uid=f"f-r2-r4-replication-v1-{index}-{weight}").to_dict()
                                          for weight in (10, 1000)]})
        seen.add(identity)
        if list(map(len, bands)) == list(QUOTAS):
            break
    if list(map(len, bands)) != list(QUOTAS):
        raise ValueError(f"replication DEV supply insufficient: {[len(b) for b in bands]}, {rejects}")
    pins = (ROSTER, *VIEWED, PROTOCOL, Path(__file__),
            Path("scripts/build_f_r2_r4_dev_panel_v1.py"),
            Path("scripts/build_f_curriculum_roster_v1.py"))
    payload = {"schema": "rf-knots-f-r2-r4-replication-panel-v1",
               "role": "score-blind validation roots with TRAIN substantial-state exclusion",
               "selection": {"seed": SEED, "max_index": MAX_INDEX,
                             "quotas": list(QUOTAS), "min_state_length": 20,
                             "rejects": rejects,
                             "train_substantial_state_count": len(train_states),
                             "viewed_excluded_group_count": len(viewed_groups)},
               "inputs": {str(path): sha(path) for path in pins},
               "rows": bands[0] + bands[1]}
    args.out.mkdir(parents=True)
    atomic(args.out / "panel.json", payload)
    print(json.dumps({"panel_sha256": sha(args.out / "panel.json"),
                      "bands": list(map(len, bands)), "rejects": rejects},
                     sort_keys=True))


if __name__ == "__main__":
    main()
