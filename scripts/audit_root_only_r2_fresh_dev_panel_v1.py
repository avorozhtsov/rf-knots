"""Independently replay and check the new root-only R2 DEV panel."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from build_f_curriculum_roster_v1 import excluded_groups, group
from build_mcts_training_supervision import atomic
from evaluate_embedding_cost_search import query_instance
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

PANEL = Path("artifacts/root-only-r2-fresh-dev-panel-v1/panel.json")
STATUS = Path("artifacts/root-only-r2-fresh-dev-panel-v1/status.json")
ROSTERS = tuple(Path(f"artifacts/{name}/roster.json") for name in (
    "f-curriculum-roster-v3", "f-incremental-roster-v2",
    "f-root-only-roster-v1", "f-short-positive-roster-v1"))
VIEWED = tuple(Path(f"artifacts/{name}/panel.json") for name in (
    "mcts-tournament-dev-panel-v1", "mcts-tournament-long-dev-panel-v1",
    "qrun-bootstrap-dev-panel-v3", "qrun-rung1-dev-panel-v1",
    "qrun-equal-cost-dev-panel-v1", "f-r2-r4-fresh-dev-panel-v1",
    "f-r2-r4-replication-panel-v1", "additive-r2-fresh-dev-panel-v1"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def substantial(witness):
    return {(tuple(state.word), state.strands) for state in
            (witness.start, *(step.after for step in witness.steps))
            if len(state.word) >= 20}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("audit output exists")
    panel, status = read(PANEL), read(STATUS)
    if (status["status"] != "completed"
            or status["panel_sha256"] != sha(PANEL)
            or panel["schema"] != "rf-knots-root-only-R2-fresh-DEV-panel-v1"
            or len(panel["rows"]) != 32
            or [row["band"] for row in panel["rows"]] != [0] * 16 + [1] * 16
            or panel["selection"]["seed"] != 20261020
            or panel["selection"]["max_index"] != 32000
            or panel["selection"]["quotas"] != [16, 16]
            or panel["selection"]["min_state_length"] != 20):
        raise ValueError("panel/status identity differs")
    for name, expected in panel["inputs"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"input SHA differs: {name}")
    train_states, train_groups = set(), set()
    for path in ROSTERS:
        for stage in read(path)["stages"]:
            for row in stage:
                witness = UnknotWitness.from_dict(row["witness"])
                witness.verify()
                train_states.update(substantial(witness))
                train_groups.add(tuple(row["query_group"]))
    if len(train_states) != panel["selection"]["train_substantial_state_count"]:
        raise ValueError("TRAIN state count differs")
    viewed_groups = {tuple(row["query_group"]) for path in VIEWED
                     for row in read(path)["rows"]}
    if len(viewed_groups) != panel["selection"]["viewed_excluded_group_count"]:
        raise ValueError("viewed DEV group count differs")
    excluded = excluded_groups() | train_groups | viewed_groups
    seen = set()
    counts = Counter()
    for row in panel["rows"]:
        witness = UnknotWitness.from_dict(row["witness"])
        witness.verify()
        fp = invariants(witness.start.word, witness.start.strands)
        identity = group(fp)
        length = len(witness.start.word)
        if (owner(fp) != "validation"
                or tuple(row["query_group"]) != identity
                or identity in excluded or identity in seen
                or row["word_length"] != length
                or not (20 <= length <= 64 if row["band"] == 0
                        else 129 <= length <= 192)
                or (row["reference_cc"], row["reference_moves"])
                != (witness.crossing_changes, witness.moves)
                or substantial(witness) & train_states):
            raise ValueError("ownership, reference or state exclusion differs")
        seen.add(identity)
        base, fits = query_instance(row["generator_index"], witness,
                                    witness_caps=True)
        if not fits or len(row["instances"]) != 2:
            raise ValueError("paired caps fail")
        for weight, stored in zip((10, 1000), row["instances"], strict=True):
            expected = replace(
                base, cc_weight=weight,
                uid=f"root-only-r2-fresh-dev-v1-{row['generator_index']}-{weight}").to_dict()
            if stored != json.loads(json.dumps(expected)):
                raise ValueError("paired instance differs")
        counts[f"band{row['band']}_cc{witness.crossing_changes}"] += 1
    report = {
        "schema": "rf-knots-root-only-R2-fresh-DEV-panel-audit-v1",
        "panel_sha256": sha(PANEL), "status_sha256": sha(STATUS),
        "source_sha256": sha(Path(__file__)),
        "reference_witnesses_replayed": 32,
        "distinct_validation_root_groups": len(seen),
        "train_substantial_state_count": len(train_states),
        "by_band_cc": dict(counts),
        "limitation": "common exact witness tails shorter than 20 may remain",
    }
    args.out.mkdir(parents=True)
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": sha(args.out / "report.json"),
                      "roots": len(seen)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
