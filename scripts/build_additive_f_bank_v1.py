"""Immutable old-plus-new F bank from independently audited TRAIN searches."""

import argparse
import fcntl
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import audit_f_incremental_hard_b256_v1 as hard
import audit_f_incremental_r2_yield_pilot_v1 as easy
from build_mcts_training_supervision import atomic, digest, sha
from mcts_tournament_signals_v1 import _groups
from mcts_training_supervision import verified_success_traces
from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness

OLD = Path("artifacts/joint-f-bank-v2")
OLD_SHA = "6a61a0ec0bf47c59cdb601b633c962dc4b2d4df3b46ba9a2ca429fab49f19a12"
ROSTER = Path("artifacts/f-incremental-roster-v2/roster.json")
ROSTER_SHA = "93bbaa282ec79bd017b5c7adfbd4f55041adddebfe5505650d258d1f17ce58f8"
SOURCES = (
    (easy, Path("artifacts/f-incremental-r2-yield-pilot-audit-v1"),
     "a42ae6b1b78123aa71e2f4ba9cb2e52fd8e881a83ad37176949e85af83ed60bb"),
    (hard, Path("artifacts/f-incremental-hard-b256-audit-v1"),
     "908a4f3c0c175aafbf5453244dafbb635cd0a4e92b0756dc60abfbd6535d5000"),
)
PROTOCOL = Path("research/320-additive-f-bank-and-new-source-gate-v1.md")


def read(path):
    return json.loads(path.read_text())


def collect_new(roster):
    by_group = {}
    source_pins = {}
    for module, audit_dir, report_sha in SOURCES:
        report_path = audit_dir / "report.json"
        if sha(report_path) != report_sha:
            raise ValueError(f"source audit report SHA differs: {report_path}")
        report = read(report_path)
        source_manifest = module.DATA / "manifest.json"
        if sha(source_manifest) != report["manifest_sha256"]:
            raise ValueError("search/audit manifest SHA differs")
        manifest = read(source_manifest)
        if manifest["status"] != "completed":
            raise ValueError("search did not complete")
        selected = module.selected_roots(roster)
        indices = report["metrics"]["F_usable_root_indices"]
        source_pins[str(report_path)] = report_sha
        source_pins[str(source_manifest)] = report["manifest_sha256"]
        for root_index in indices:
            root = selected[root_index]
            key = tuple(root["query_group"])
            slot = by_group.setdefault(key, {"root": root, "paths": defaultdict(set),
                                              "source_shards": defaultdict(list)})
            if slot["root"]["generator_index"] != root["generator_index"]:
                raise ValueError("same query group has different new roots")
            for weight_index, weight in enumerate((10, 1000)):
                for seed_index in range(2):
                    position = root_index * 4 + weight_index * 2 + seed_index
                    saved = read(audit_dir / f"audit-{position:04d}.json")
                    if (saved["manifest_sha256"] != report["manifest_sha256"]
                            or saved["position"] != position
                            or saved["record"]["root_index"] != root_index
                            or saved["record"]["weight"] != weight):
                        raise ValueError("saved audit identity differs")
                    slot["paths"][weight].update(tuple(path) for path
                                                 in saved["record"]["admitted_paths"])
                    entry = manifest["shards"][position]
                    shard = module.DATA / entry["file"]
                    if sha(shard) != entry["sha256"]:
                        raise ValueError(f"raw search shard SHA differs: {shard}")
                    slot["source_shards"][weight].append({"file": str(shard),
                                                           "sha256": entry["sha256"]})
    return by_group, source_pins


def new_root_row(root_index, entry):
    root, paths, source_shards = entry["root"], entry["paths"], entry["source_shards"]
    reference = UnknotWitness.from_dict(root["witness"])
    reference.verify()
    groups = []
    for weight_index, weight in enumerate((10, 1000)):
        instance_dict = root["instances"][weight_index]
        instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
        if ((reference.start.word, reference.start.strands)
                != (instance.word, instance.strands)
                or reference.crossing_changes > instance.cap_crossing_changes
                or reference.moves > instance.cap_moves):
            raise ValueError("reference start/caps differs")
        ref_actions = tuple(step.action.to_flat(instance.spec) for step in reference.steps)
        witnesses = {ref_actions: reference}
        for actions in sorted(paths[weight]):
            if actions not in witnesses:
                witness = UnknotWitness.from_actions(instance.word, instance.strands,
                                                      instance.spec, actions)
                witness.verify()
                if (witness.crossing_changes > instance.cap_crossing_changes
                        or witness.moves > instance.cap_moves):
                    raise ValueError("new path cap differs")
                witnesses[actions] = witness
        signals = _groups(verified_success_traces(instance, list(witnesses.values())),
                          "F", seed=20261006, k=4, near_margin=2,
                          gap_scale=weight, gap_cap=4)
        if signals and signals[0]["coefficients"]:
            groups.append({"root_index": root_index, "stage": root["stage"],
                           "weight": weight, "source": "incremental-b64-b256",
                           "instance": instance_dict,
                           "source_shards": source_shards[weight],
                           "signal_terms": len(signals[0]["coefficients"]),
                           "positive_paths": signals[0]["positive_paths"],
                           "negative_paths": signals[0]["negative_paths"],
                           "paths": [{"actions": list(actions),
                                      "cc": witness.crossing_changes,
                                      "moves": witness.moves}
                                     for actions, witness in witnesses.items()]})
    if not groups:
        raise ValueError("audited F-usable new root lost its signal")
    return {"root_index": root_index, "stage": root["stage"],
            "query_group": root["query_group"], "groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new bank output exists; never overwrite")
    if sha(OLD / "manifest.json") != OLD_SHA or sha(ROSTER) != ROSTER_SHA:
        raise ValueError("old bank or new roster SHA differs")
    old_manifest = read(OLD / "manifest.json")
    if old_manifest["status"] != "completed" or len(old_manifest["roots"]) != 1000:
        raise ValueError("old bank coverage differs")
    roster = read(ROSTER)
    new, source_pins = collect_new(roster)
    if len(new) != 7:
        raise ValueError(f"expected seven audited new root groups, found {len(new)}")
    inputs = {str(OLD / "manifest.json"): OLD_SHA, str(ROSTER): ROSTER_SHA,
              **source_pins, str(PROTOCOL): sha(PROTOCOL),
              str(Path(__file__).resolve()): sha(Path(__file__)),
              "scripts/mcts_tournament_signals_v1.py":
              sha(Path("scripts/mcts_tournament_signals_v1.py"))}
    contract = {"schema": "rf-knots-additive-F-bank-v1", "inputs": inputs,
                "old_roots": 1000, "new_root_query_groups": [list(k) for k in sorted(new)],
                "F": {"seed": 20261006, "k": 4, "near_margin": 2, "gap_cap": 4}}
    args.out.mkdir(parents=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old_groups = set()
        roots = []
        summary = Counter()
        for index, item in enumerate(old_manifest["roots"]):
            source = OLD / item["file"]
            target = args.out / f"root-{index:04d}.json"
            if item["file"] != target.name or sha(source) != item["sha256"]:
                raise ValueError(f"old bank root SHA differs: {index}")
            row = read(source)
            old_groups.add(tuple(row["query_group"]))
            shutil.copyfile(source, target)
            if sha(target) != item["sha256"]:
                raise ValueError("old root copy SHA differs")
            roots.append({"file": target.name, "sha256": item["sha256"]})
            summary["old_F_roots"] += bool(row["groups"])
            summary["old_F_groups"] += len(row["groups"])
        if len(old_groups) != 1000 or old_groups.intersection(new):
            raise ValueError("old/new query groups overlap or old groups repeat")
        for offset, key in enumerate(sorted(new)):
            index = 1000 + offset
            row = new_root_row(index, new[key])
            target = args.out / f"root-{index:04d}.json"
            atomic(target, row)
            roots.append({"file": target.name, "sha256": sha(target)})
            summary["new_F_roots"] += 1
            summary["new_F_groups"] += len(row["groups"])
            summary[f"new_stage{row['stage']}_roots"] += 1
        if summary["old_F_roots"] != 392 or summary["new_F_roots"] != 7:
            raise ValueError("additive root accounting differs")
        manifest = {"contract": contract, "contract_sha256": digest(contract),
                    "status": "completed", "next_root": len(roots), "roots": roots,
                    "summary": dict(summary)}
        atomic(args.out / "manifest.json", manifest)
        print(json.dumps({"manifest_sha256": sha(args.out / "manifest.json"),
                          "summary": dict(summary)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
