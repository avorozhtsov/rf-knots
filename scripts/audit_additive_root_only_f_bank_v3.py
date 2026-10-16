"""Independently audit v3 bank retention, source paths and root-only labels."""

import argparse
import json
from collections import Counter
from pathlib import Path

import audit_additive_root_only_f_bank_v2 as common
import build_f_short_positive_roster_v1 as roster_source
from build_mcts_training_supervision import atomic, sha
from rf_knots import mcts_loop as M
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

BANK = Path("artifacts/additive-root-only-f-bank-v3")
OLD = Path("artifacts/additive-f-bank-v1")
OLD_SHA = "3a65df6530d8fe62d9e297b204072db4c1a7ec437f131338c640dc28c23a182e"
ROSTER = Path("artifacts/f-short-positive-roster-v1/roster.json")
ROSTER_SHA = "53d6a26828acb966747d17699a3b74f5ab13e0121fa055edbbaa2f57fd242f42"
ROSTER_AUDIT = Path("artifacts/f-short-positive-roster-audit-v1/report.json")
ROSTER_AUDIT_SHA = "86257d258ca2f121e924d4343fe39f67410aa51b762bf435aae6af9ae23cc61e"
SOURCE = Path("artifacts/f-short-positive-r2-yield-v1")
SOURCE_SHA = "ca61603ca547361662beacbdbbea584b961f3024a5e9cef319eaf69a1b6413a8"
SOURCE_AUDIT = Path("artifacts/f-short-positive-r2-yield-audit-v1/report.json")
SOURCE_AUDIT_SHA = "e94db1cf89a364e301dc540302c061211204ae9cdc5de2c4383893a152667cbb"
ROOT_INDICES = (1, 3, 4)


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("audit output exists")
    for path, expected in ((OLD / "manifest.json", OLD_SHA),
                           (ROSTER, ROSTER_SHA),
                           (ROSTER_AUDIT, ROSTER_AUDIT_SHA),
                           (SOURCE / "manifest.json", SOURCE_SHA),
                           (SOURCE_AUDIT, SOURCE_AUDIT_SHA)):
        if sha(path) != expected:
            raise ValueError(f"source SHA differs: {path}")
    manifest_path = BANK / "manifest.json"
    manifest = read(manifest_path)
    old_manifest = read(OLD / "manifest.json")
    source_manifest = read(SOURCE / "manifest.json")
    contract = manifest["contract"]
    if (manifest["status"] != "completed" or manifest["next_root"] != 1010
            or len(manifest["roots"]) != 1010
            or contract["schema"] != "rf-knots-additive-root-only-F-bank-v3"
            or contract["old_root_shards"] != 1007
            or contract["new_source_root_indices"] != list(ROOT_INDICES)
            or contract["new_gradient_scope"] != "exact_zero_spent_root_only"):
        raise ValueError("bank coverage/contract differs")
    for path, expected in contract["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"bank input SHA differs: {path}")
    roster = read(ROSTER)
    for path, expected in roster["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"roster input SHA differs: {path}")
    panels = tuple(dict.fromkeys((*roster_source.root_only.viewed_panel_paths(),
                                  roster_source.EXTRA_PANEL)))
    heldout_exact = roster_source.root_only.viewed_states(panels)
    roots = roster["stages"][0][:4] + roster["stages"][1][:4]
    seen, tally = set(), Counter()
    for index, entry in enumerate(manifest["roots"]):
        path = BANK / f"root-{index:04d}.json"
        if entry["file"] != path.name or sha(path) != entry["sha256"]:
            raise ValueError(f"bank shard SHA/order differs: {index}")
        row = read(path)
        key = tuple(row["query_group"])
        if row["root_index"] != index or key in seen:
            raise ValueError(f"bank query group/index differs: {index}")
        seen.add(key)
        if index < 1007:
            old_entry = old_manifest["roots"][index]
            old_path = OLD / old_entry["file"]
            if (entry != old_entry or sha(old_path) != old_entry["sha256"]
                    or path.read_bytes() != old_path.read_bytes()):
                raise ValueError(f"old row changed: {index}")
            tally["retained_F_roots"] += bool(row["groups"])
            tally["retained_F_groups"] += len(row["groups"])
            continue
        root_index = ROOT_INDICES[index - 1007]
        root = roots[root_index]
        witness = UnknotWitness.from_dict(root["witness"])
        witness.verify()
        if (row["stage"] != 1 or row["query_group"] != root["query_group"]
                or row["gradient_scope"] != "exact_zero_spent_root_only"
                or (tuple(witness.start.word), witness.start.strands) in heldout_exact
                or owner(invariants(witness.start.word,
                                    witness.start.strands)) != "train"):
            raise ValueError(f"new root owner/exclusion/scope differs: {index}")
        groups = {group["weight"]: group for group in row["groups"]}
        if len(groups) != len(row["groups"]):
            raise ValueError("duplicate objective group")
        expected_count = 0
        for weight_index, weight in enumerate((10, 1000)):
            instance_dict = root["instances"][weight_index]
            instance = M.Instance(**{**instance_dict,
                                     "word": tuple(instance_dict["word"])})
            if (witness.start.word, witness.start.strands) != (instance.word,
                                                                instance.strands):
                raise ValueError("reference/instance start differs")
            paths, pins = common.source_paths(root_index, weight_index, root,
                                              instance, source_manifest)
            expected = common.expected_root_terms(instance, paths)
            if expected is None:
                if weight in groups:
                    raise ValueError("non-F group admitted")
                continue
            expected_count += 1
            if weight not in groups:
                raise ValueError("F group omitted")
            group = groups[weight]
            signal, terms = expected
            if (group["root_index"] != index or group["stage"] != 1
                    or group["gradient_scope"] != "exact_zero_spent_root_only"
                    or group["instance"] != instance_dict
                    or group["source_shards"] != pins
                    or group["root_coefficients"] != terms
                    or group["signal_terms"] != len(terms)
                    or group["positive_paths"] != signal["positive_paths"]
                    or group["negative_paths"] != signal["negative_paths"]):
                raise ValueError(f"exact-root F metadata differs: {index}")
            if len(group["paths"]) != len(paths):
                raise ValueError("completed path count differs")
            stored_paths = {}
            for item in group["paths"]:
                actions = tuple(item["actions"])
                replayed = UnknotWitness.from_actions(
                    instance.word, instance.strands, instance.spec, actions)
                replayed.verify()
                if (replayed.crossing_changes > instance.cap_crossing_changes
                        or replayed.moves > instance.cap_moves
                        or (item["cc"], item["moves"]) !=
                        (replayed.crossing_changes, replayed.moves)):
                    raise ValueError("stored path replay/caps/cost differs")
                stored_paths[actions] = replayed
            if stored_paths.keys() != paths.keys():
                raise ValueError("bank paths not identical to source")
            tally["new_F_groups"] += 1
            tally["new_root_F_terms"] += len(terms)
            tally["replayed_new_paths"] += len(stored_paths)
        if expected_count != len(row["groups"]) or not expected_count:
            raise ValueError("new root group accounting differs")
        tally["new_F_roots"] += 1
    if (len(seen), tally["retained_F_roots"], tally["retained_F_groups"],
            tally["new_F_roots"], tally["new_F_groups"]) != (1010, 399, 701, 3, 4):
        raise ValueError(f"final tally differs: {dict(tally)}")
    if any(manifest["summary"].get(key) != tally[key]
           for key in manifest["summary"]):
        raise ValueError("manifest summary differs")
    report = {"schema": "rf-knots-additive-root-only-F-bank-audit-v3",
              "bank_manifest_sha256": sha(manifest_path),
              "auditor_sha256": sha(Path(__file__)),
              "audit_helper_sha256": sha(Path(common.__file__)),
              "root_shards": 1010, "distinct_query_groups": len(seen),
              "F_usable_roots": 402, "F_groups": 705,
              "metrics": dict(tally),
              "boundary": "new F coefficients only at exact zero-spent TRAIN root"}
    args.out.mkdir(parents=True)
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": sha(args.out / "report.json"),
                      "metrics": dict(tally)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
