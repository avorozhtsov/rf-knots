"""Independent replay, ownership, exclusion and SHA audit of rung-1 DEV."""

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from build_f_curriculum_roster_v1 import excluded_groups, group
from evaluate_embedding_cost_search import query_instance
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

PANEL = Path("artifacts/qrun-rung1-dev-panel-v1/panel.json")
OLD_PANEL = Path("artifacts/qrun-bootstrap-dev-panel-v3/panel.json")
ROSTER = Path("artifacts/f-curriculum-roster-v3/roster.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def states(witness):
    return {(tuple(s.word), s.strands) for s in
            (witness.start, *(step.after for step in witness.steps))
            if len(s.word) >= 20}


def main():
    panel = json.loads(PANEL.read_text())
    roster = json.loads(ROSTER.read_text())
    old = json.loads(OLD_PANEL.read_text())
    if (panel["schema"] != "rf-knots-qrun-rung1-dev-panel-v1"
            or len(panel["rows"]) != 32
            or [r["band"] for r in panel["rows"]] != [0] * 16 + [1] * 16
            or panel["selection"]["seed"] != 20261205
            or panel["selection"]["min_state_length"] != 20):
        raise ValueError("panel identity differs")
    for path, expected in panel["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"input SHA differs: {path}")
    train_states, train_groups = set(), set()
    for stage in roster["stages"]:
        for row in stage:
            witness = UnknotWitness.from_dict(row["witness"])
            witness.verify()
            train_states.update(states(witness))
            train_groups.add(tuple(row["query_group"]))
    if len(train_states) != panel["selection"]["train_substantial_state_count"]:
        raise ValueError("TRAIN state count differs")
    excluded = (excluded_groups() | train_groups
                | {tuple(row["query_group"]) for row in old["rows"]})
    seen = set()
    counts = Counter()
    for row in panel["rows"]:
        witness = UnknotWitness.from_dict(row["witness"])
        witness.verify()
        fingerprint = invariants(witness.start.word, witness.start.strands)
        identity = group(fingerprint)
        length = len(witness.start.word)
        if (owner(fingerprint) != "validation"
                or tuple(row["query_group"]) != identity
                or identity in excluded or identity in seen
                or row["word_length"] != length
                or not (20 <= length <= 64 if row["band"] == 0 else 129 <= length <= 192)
                or (row["reference_cc"], row["reference_moves"])
                != (witness.crossing_changes, witness.moves)
                or states(witness) & train_states):
            raise ValueError("replay, ownership, exclusions or state overlap differ")
        seen.add(identity)
        base, fits = query_instance(row["generator_index"], witness, witness_caps=True)
        if not fits or len(row["instances"]) != 2:
            raise ValueError("reference outside shared caps")
        for weight, stored in zip((10, 1000), row["instances"], strict=True):
            expected = replace(base, cc_weight=weight,
                               uid=f"qrun-rung1-dev-v1-{row['generator_index']}-{weight}").to_dict()
            if stored != expected:
                raise ValueError("paired resource instance differs")
        counts[f"band{row['band']}_cc{witness.crossing_changes}"] += 1
    print(json.dumps({"panel_sha256": sha(PANEL), "roots": len(seen),
                      "train_substantial_states": len(train_states),
                      "by_band_cc": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
