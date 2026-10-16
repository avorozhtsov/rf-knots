"""Independent replay, ownership, exclusion and SHA audit for protocol321."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import build_f_curriculum_roster_v1 as old
import build_f_incremental_roster_v2 as incremental
from build_mcts_training_supervision import atomic, sha
from evaluate_embedding_cost_search import query_instance
from rf_knots.actions import CROSSING_CHANGE, KIND_NAMES
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

OLD = Path("artifacts/f-curriculum-roster-v3/roster.json")
INCREMENTAL = Path("artifacts/f-incremental-roster-v2/roster.json")
EXPECTED_SEED = 20261012
EXPECTED_MAX_INDEX = 4096
EXPECTED_QUOTAS = [16, 16]


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    status = read(args.roster.parent / "status.json")
    payload = read(args.roster)
    if (status["status"] != "completed"
            or status["roster_sha256"] != sha(args.roster)
            or payload["schema"] != "rf-knots-interleaved-positive-seed-roster-v1"
            or payload["selection"]["seed"] != EXPECTED_SEED
            or payload["selection"]["max_index"] != EXPECTED_MAX_INDEX
            or payload["selection"]["quotas"] != EXPECTED_QUOTAS
            or [len(rows) for rows in payload["stages"]] != EXPECTED_QUOTAS):
        raise ValueError("roster schema/status/selection differs")
    for path, expected in payload["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"pinned source input differs: {path}")
    forbidden = old.excluded_groups()
    for panel in incremental.PANELS:
        forbidden.update(tuple(row["query_group"]) for row in read(panel)["rows"])
    prior = set(forbidden)
    for source in (OLD, INCREMENTAL):
        prior.update(tuple(row["query_group"])
                     for stage in read(source)["stages"] for row in stage)
    seen = set()
    indices = set()
    stats = Counter()
    for offset, rows in enumerate(payload["stages"]):
        stage = offset + 2
        hashes = []
        for row in rows:
            index = row["generator_index"]
            if (not 0 <= index < EXPECTED_MAX_INDEX or index in indices
                    or row["generator_seed"] != EXPECTED_SEED
                    or row["source_family"] != "positive-seed-interleaved-v1"):
                raise ValueError(f"source index/provenance differs: {index}")
            indices.add(index)
            hashes.append(hashlib.sha256(f"{EXPECTED_SEED}:{index}".encode()).hexdigest())
            witness = UnknotWitness.from_dict(row["witness"])
            witness.verify()
            fp = invariants(witness.start.word, witness.start.strands)
            group = old.group(fp)
            if (owner(fp) != "train" or row["stage"] != stage
                    or old.stage(len(witness.start.word), witness.crossing_changes) != stage
                    or row["word_length"] != len(witness.start.word)
                    or row["reference_cc"] != witness.crossing_changes
                    or row["reference_moves"] != witness.moves
                    or row["query_group"] != list(group)
                    or group in prior or group in seen):
                raise ValueError(f"root identity/stage/exclusion differs: {index}")
            seen.add(group)
            cc_states = [witness.start] + [step.after for step in witness.steps
                if step.action.kind == KIND_NAMES[CROSSING_CHANGE]]
            for state in cc_states:
                fingerprint = invariants(state.word, state.strands)
                if owner(fingerprint) != "train" or old.group(fingerprint) in forbidden:
                    raise ValueError(f"witness path leaves TRAIN: {index}")
            instance, fits = query_instance(index, witness, witness_caps=True)
            if not fits or len(row["instances"]) != 2:
                raise ValueError(f"reference/caps differ: {index}")
            for weight, stored in zip((10, 1000), row["instances"], strict=True):
                expected = replace(instance, cc_weight=weight,
                                   uid=f"f-interleaved-{EXPECTED_SEED}-{index}-{weight}").to_dict()
                if stored != expected:
                    raise ValueError(f"paired instance differs: {index}, {weight}")
            stats[f"stage{stage}_CC{witness.crossing_changes}"] += 1
        if hashes != sorted(hashes):
            raise ValueError(f"SHA-ranked order differs: stage{stage}")
    report = {"schema": "rf-knots-interleaved-roster-audit-v1",
              "roster_sha256": sha(args.roster), "distinct_TRAIN_roots": len(seen),
              "by_stage_cc": dict(stats)}
    args.out.mkdir(parents=True, exist_ok=True)
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": sha(args.out / "report.json"),
                      **report}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
