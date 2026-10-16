"""Independent replay, ownership and SHA audit of the v3 candidate roster."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import build_f_curriculum_roster_v1 as old
from evaluate_embedding_cost_search import query_instance
from rf_knots.actions import CROSSING_CHANGE, KIND_NAMES
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness


QUOTAS = (140, 286, 287, 287)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.roster.read_text())
    if (payload["schema"] != "rf-knots-f-curriculum-candidate-roster-v3"
            or payload["selection"]["quotas"] != list(QUOTAS)
            or payload["selection"]["seed"] != old.SEED
            or payload["selection"]["max_index"] != old.MAX_INDEX
            or list(map(len, payload["stages"])) != list(QUOTAS)):
        raise ValueError("roster schema or fixed selection differs")
    for path, expected in payload["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"pinned input changed: {path}")
    excluded = old.excluded_groups()
    groups, indices = set(), set()
    stats = Counter()
    for stage, rows in enumerate(payload["stages"]):
        hashes = []
        for row in rows:
            index = row["generator_index"]
            witness = UnknotWitness.from_dict(row["witness"])
            witness.verify()
            if not 0 <= index < old.MAX_INDEX or index in indices:
                raise ValueError("generator index duplicated or out of bounds")
            indices.add(index)
            hashes.append(hashlib.sha256(f"{old.SEED}:{index}".encode()).hexdigest())
            fingerprint = invariants(witness.start.word, witness.start.strands)
            identity = old.group(fingerprint)
            if (row["stage"] != stage
                    or old.stage(len(witness.start.word), witness.crossing_changes) != stage
                    or row["word_length"] != len(witness.start.word)
                    or row["reference_cc"] != witness.crossing_changes
                    or row["reference_moves"] != witness.moves
                    or row["query_group"] != list(identity)
                    or identity in groups or identity in excluded):
                raise ValueError(f"root identity/stage differs at index {index}")
            groups.add(identity)
            cc_states = [witness.start] + [step.after for step in witness.steps
                                           if step.action.kind == KIND_NAMES[CROSSING_CHANGE]]
            if any(owner(fp := invariants(state.word, state.strands)) != "train"
                   or old.group(fp) in excluded for state in cc_states):
                raise ValueError(f"TRAIN witness ownership differs at index {index}")
            base, fits = query_instance(index, witness, witness_caps=True)
            if not fits or len(row["instances"]) != 2:
                raise ValueError("reference solution/caps missing")
            for weight, stored in zip((10, 1000), row["instances"], strict=True):
                expected = replace(base, cc_weight=weight,
                                   uid=f"f-curriculum-{index}-{weight}").to_dict()
                if stored != expected:
                    raise ValueError(f"instance differs at index {index}, lambda {weight}")
            stats[f"stage{stage}_cc{witness.crossing_changes}"] += 1
        if hashes != sorted(hashes):
            raise ValueError(f"SHA-ranked order differs in stage {stage}")
    print(json.dumps({"roster_sha256": sha(args.roster), "roots": len(groups),
                      "stages": [len(rows) for rows in payload["stages"]],
                      "by_stage_cc": dict(stats)}, sort_keys=True))


if __name__ == "__main__":
    main()
