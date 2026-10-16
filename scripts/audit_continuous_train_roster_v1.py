"""Independent replay and exclusion audit of continuing TRAIN source."""

import argparse
import json
from pathlib import Path

import build_continuous_train_roster_v1 as source
from build_mcts_training_supervision import atomic, sha
from rf_knots import mcts_loop as M
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("immutable audit exists")
    roster = source.read(args.roster)
    for path, expected in roster["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"roster input SHA differs: {path}")
    excluded, states = source.exclusions()
    seen = set()
    for row in roster["rows"]:
        witness = UnknotWitness.from_dict(row["witness"])
        witness.verify()
        fp = invariants(witness.start.word, witness.start.strands)
        key = (fp[0], abs(fp[1]))
        if (owner(fp) != "train" or key != tuple(row["query_group"])
                or key in excluded or key in seen
                or (witness.start.word, witness.start.strands) in states):
            raise ValueError("root ownership or exclusion differs")
        seen.add(key)
        for weight, data in zip((10, 1000), row["instances"], strict=True):
            instance = M.Instance(**{**data, "word": tuple(data["word"])})
            if ((instance.word, instance.strands) != (witness.start.word, witness.start.strands)
                    or instance.cc_weight != weight
                    or witness.crossing_changes > instance.cap_crossing_changes
                    or witness.moves > instance.cap_moves):
                raise ValueError("paired starts/caps differ")
    report = {"schema": "rf-knots-continuous-TRAIN-roster-audit-v1",
              "roster_sha256": sha(args.roster), "roots_replayed": len(seen),
              "auditor_sha256": sha(Path(__file__)), "status": "passed",
              "gradient_scope": "exact_zero_spent_root_only"}
    args.out.mkdir(parents=True)
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": sha(args.out / "report.json"),
                      "roots": len(seen)}), flush=True)


if __name__ == "__main__":
    main()
