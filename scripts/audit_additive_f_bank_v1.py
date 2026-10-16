"""Independent SHA, replay, ownership and F-signal audit of additive bank v1."""

import argparse
import json
from collections import Counter
from pathlib import Path

import build_f_incremental_roster_v2 as roster_source
from build_mcts_training_supervision import atomic, sha
from mcts_tournament_signals_v1 import _groups
from mcts_training_supervision import verified_success_traces
from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE, KIND_NAMES
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

BANK = Path("artifacts/additive-f-bank-v1")
OLD = Path("artifacts/joint-f-bank-v2")
MANIFEST_SHA = "3a65df6530d8fe62d9e297b204072db4c1a7ec437f131338c640dc28c23a182e"
OLD_SHA = "6a61a0ec0bf47c59cdb601b633c962dc4b2d4df3b46ba9a2ca429fab49f19a12"


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = BANK / "manifest.json"
    if sha(manifest_path) != MANIFEST_SHA or sha(OLD / "manifest.json") != OLD_SHA:
        raise ValueError("new/old bank manifest SHA differs")
    manifest, old_manifest = read(manifest_path), read(OLD / "manifest.json")
    if (manifest["status"] != "completed" or len(manifest["roots"]) != 1007
            or old_manifest["status"] != "completed"
            or manifest["next_root"] != 1007):
        raise ValueError("bank coverage differs")
    for path, expected in manifest["contract"]["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"bank input SHA differs: {path}")
    excluded = roster_source.old.excluded_groups()
    for panel in roster_source.PANELS:
        excluded.update(tuple(row["query_group"]) for row in read(panel)["rows"])
    seen = set()
    tally = Counter()
    for index, entry in enumerate(manifest["roots"]):
        path = BANK / f"root-{index:04d}.json"
        if entry["file"] != path.name or sha(path) != entry["sha256"]:
            raise ValueError(f"root SHA/order differs: {index}")
        row = read(path)
        key = tuple(row["query_group"])
        if row["root_index"] != index or key in seen:
            raise ValueError(f"duplicate root group or index differs: {index}")
        seen.add(key)
        if index < 1000:
            old_entry = old_manifest["roots"][index]
            if entry != old_entry or path.read_bytes() != (OLD / old_entry["file"]).read_bytes():
                raise ValueError(f"old root was not preserved byte-for-byte: {index}")
            tally["old_F_roots"] += bool(row["groups"])
            tally["old_F_groups"] += len(row["groups"])
            continue
        if key in excluded or owner((key[0], key[1])) != "train":
            raise ValueError(f"new root not TRAIN-owned or excluded: {index}")
        tally["new_F_roots"] += 1
        tally[f"new_stage{row['stage']}_roots"] += 1
        if not row["groups"]:
            raise ValueError(f"new root has no F signal: {index}")
        for group in row["groups"]:
            if (group["root_index"] != index or group["stage"] != row["stage"]
                    or group["weight"] not in (10, 1000)):
                raise ValueError(f"new group identity differs: {index}")
            instance_dict = group["instance"]
            instance = M.Instance(**{**instance_dict,
                                     "word": tuple(instance_dict["word"])})
            if instance.cc_weight != group["weight"]:
                raise ValueError(f"new objective differs: {index}")
            for source in group["source_shards"]:
                if sha(Path(source["file"])) != source["sha256"]:
                    raise ValueError(f"new source shard SHA differs: {index}")
            paths = group["paths"]
            if len({tuple(item["actions"]) for item in paths}) != len(paths):
                raise ValueError(f"duplicate path: {index}")
            witnesses = []
            for item in paths:
                witness = UnknotWitness.from_actions(instance.word, instance.strands,
                                                     instance.spec, item["actions"])
                witness.verify()
                if ((witness.crossing_changes, witness.moves)
                        != (item["cc"], item["moves"])
                        or witness.crossing_changes > instance.cap_crossing_changes
                        or witness.moves > instance.cap_moves):
                    raise ValueError(f"path replay/caps differ: {index}")
                cc_states = [witness.start] + [step.after for step in witness.steps
                    if step.action.kind == KIND_NAMES[CROSSING_CHANGE]]
                for state in cc_states:
                    fingerprint = invariants(state.word, state.strands)
                    if (owner(fingerprint) != "train"
                            or (fingerprint[0], abs(fingerprint[1])) in excluded):
                        raise ValueError(f"path leaves TRAIN split: {index}")
                witnesses.append(witness)
            signals = _groups(verified_success_traces(instance, witnesses), "F",
                              seed=20261006, k=4, near_margin=2,
                              gap_scale=group["weight"], gap_cap=4)
            if (len(signals) != 1 or not signals[0]["coefficients"]
                    or len(signals[0]["coefficients"]) != group["signal_terms"]
                    or signals[0]["positive_paths"] != group["positive_paths"]
                    or signals[0]["negative_paths"] != group["negative_paths"]):
                raise ValueError(f"F signal differs: {index}")
            tally["new_F_groups"] += 1
            tally["new_paths"] += len(paths)
            tally["new_F_terms"] += group["signal_terms"]
    if (tally["old_F_roots"], tally["old_F_groups"], tally["new_F_roots"],
            tally["new_F_groups"]) != (392, 690, 7, 11):
        raise ValueError("additive bank tally differs")
    report = {"schema": "rf-knots-additive-F-bank-audit-v1",
              "bank_manifest_sha256": MANIFEST_SHA, "root_shards": 1007,
              "distinct_query_groups": len(seen), "metrics": dict(tally)}
    args.out.mkdir(parents=True, exist_ok=True)
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": sha(args.out / "report.json"),
                      "metrics": dict(tally)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
