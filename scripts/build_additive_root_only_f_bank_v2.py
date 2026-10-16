"""Immutable 399-old-plus-three-new exact-root-only F bank."""

import argparse
import fcntl
import json
import shutil
from collections import Counter
from pathlib import Path

from audit_f_short_positive_r2_yield_v1 import replay_episode
from build_mcts_training_supervision import atomic, digest, sha
from mcts_tournament_signals_v1 import _groups
from mcts_training_supervision import verified_success_traces
from rf_knots import mcts_loop as M, reference
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

OLD = Path("artifacts/additive-f-bank-v1")
OLD_SHA = "3a65df6530d8fe62d9e297b204072db4c1a7ec437f131338c640dc28c23a182e"
OLD_AUDIT = Path("artifacts/additive-f-bank-audit-v1/report.json")
OLD_AUDIT_SHA = "47487fb089336eb424a34e79603fb510d61a0285868cea6a18d54b31ac7a2b57"
ROSTER = Path("artifacts/f-short-positive-roster-v1/roster.json")
ROSTER_SHA = "53d6a26828acb966747d17699a3b74f5ab13e0121fa055edbbaa2f57fd242f42"
ROSTER_AUDIT = Path("artifacts/f-short-positive-roster-audit-v1/report.json")
ROSTER_AUDIT_SHA = "86257d258ca2f121e924d4343fe39f67410aa51b762bf435aae6af9ae23cc61e"
SOURCE = Path("artifacts/f-short-positive-r2-yield-v1")
SOURCE_SHA = "ca61603ca547361662beacbdbbea584b961f3024a5e9cef319eaf69a1b6413a8"
SOURCE_AUDIT = Path("artifacts/f-short-positive-r2-yield-audit-v1/report.json")
SOURCE_AUDIT_SHA = "e94db1cf89a364e301dc540302c061211204ae9cdc5de2c4383893a152667cbb"
SCALEUP_AUDIT = Path("artifacts/f-short-positive-r2-scaleup-audit-v2/report.json")
SCALEUP_AUDIT_SHA = "da067e7436fbecb30324b1c2661b24d56161565dd09299b875a0c000b8c0aee9"
PROTOCOL = Path("research/343-additive-root-only-f-bank-v2.md")
ROOT_INDICES = (1, 3, 4)
SEEDS = (20271340, 20271341)


def read(path):
    return json.loads(path.read_text())


def root_coefficients(instance, witnesses):
    signals = _groups(verified_success_traces(instance, list(witnesses.values())),
                      "F", seed=20261006, k=4, near_margin=2,
                      gap_scale=instance.cc_weight, gap_cap=4)
    if len(signals) > 1:
        raise ValueError("unexpected multiple F contexts")
    if not signals:
        return None
    root = M.SearchState(instance.word, instance.strands, 0, 0)
    coefficients = {action: float(value)
                    for (state, action), value in signals[0]["coefficients"].items()
                    if state == root and abs(value) > 1e-12}
    positive = [action for action, value in coefficients.items() if value > 0]
    negative = [action for action, value in coefficients.items() if value < 0]
    successors = {}
    for action in coefficients:
        if not reference.is_legal(instance.spec, instance.word,
                                  instance.strands, action, True):
            raise ValueError("illegal root F action")
        successors[action] = reference.apply(
            instance.spec, instance.word, instance.strands, action)
    if not any(successors[a] != successors[b]
               for a in positive for b in negative):
        return None
    return signals[0], [
        {"action": action, "value": coefficients[action]}
        for action in sorted(coefficients)]


def selected_roots(roster):
    roots = roster["stages"][0][:4] + roster["stages"][1][:4]
    if len(roots) != 8:
        raise ValueError("pilot root order differs")
    return roots


def new_row(index, root_index, root, source_manifest):
    reference_witness = UnknotWitness.from_dict(root["witness"])
    reference_witness.verify()
    groups = []
    for weight_index, weight in enumerate((10, 1000)):
        instance_dict = root["instances"][weight_index]
        instance = M.Instance(**{**instance_dict,
                                 "word": tuple(instance_dict["word"])})
        if ((reference_witness.start.word, reference_witness.start.strands)
                != (instance.word, instance.strands)
                or reference_witness.crossing_changes > instance.cap_crossing_changes
                or reference_witness.moves > instance.cap_moves):
            raise ValueError("reference start/caps differs")
        actions = tuple(step.action.to_flat(instance.spec)
                        for step in reference_witness.steps)
        witnesses = {actions: reference_witness}
        shard_pins = []
        for seed_index, seed in enumerate(SEEDS):
            position = root_index * 4 + weight_index * 2 + seed_index
            entry = source_manifest["shards"][position]
            shard = SOURCE / entry["file"]
            if entry["file"] != f"attempt-{position:04d}.json" or sha(shard) != entry["sha256"]:
                raise ValueError("source shard SHA/order differs")
            saved = read(shard)
            if ((saved["position"], saved["root_index"], saved["seed"],
                 saved["weight"], saved["generator_index"])
                    != (position, root_index, seed, weight,
                        root["generator_index"])):
                raise ValueError("source task identity differs")
            shard_pins.append({"file": str(shard), "sha256": entry["sha256"]})
            if saved["cpu_censored"]:
                if saved["row"] is not None or saved["counts"]:
                    raise ValueError("CPU-censored shard contains outcomes")
                continue
            row = saved["row"]
            if (row["cc_weight"] != weight or row["search_seed"] != seed
                    or row["simulations"] != 64):
                raise ValueError("source search identity differs")
            for episode in row["episodes"]:
                witness = replay_episode(instance, episode)
                if witness is not None:
                    path = tuple(step.action.to_flat(instance.spec)
                                 for step in witness.steps)
                    witnesses[path] = witness
        result = root_coefficients(instance, witnesses)
        if result is None:
            continue
        signal, root_terms = result
        groups.append({
            "root_index": index, "stage": root["stage"], "weight": weight,
            "source": "short-positive-R2-pilot-v1",
            "gradient_scope": "exact_zero_spent_root_only",
            "instance": instance_dict, "source_shards": shard_pins,
            "signal_terms": len(root_terms),
            "root_coefficients": root_terms,
            "positive_paths": signal["positive_paths"],
            "negative_paths": signal["negative_paths"],
            "paths": [{"actions": list(path), "cc": witness.crossing_changes,
                       "moves": witness.moves}
                      for path, witness in sorted(witnesses.items())],
        })
    if not groups:
        raise ValueError(f"audited F root lost all exact-root groups: {root_index}")
    return {"root_index": index, "stage": root["stage"],
            "query_group": root["query_group"],
            "gradient_scope": "exact_zero_spent_root_only",
            "groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("output exists; immutable bank must not be overwritten")
    for path, expected in ((OLD / "manifest.json", OLD_SHA),
                           (OLD_AUDIT, OLD_AUDIT_SHA), (ROSTER, ROSTER_SHA),
                           (ROSTER_AUDIT, ROSTER_AUDIT_SHA),
                           (SOURCE / "manifest.json", SOURCE_SHA),
                           (SOURCE_AUDIT, SOURCE_AUDIT_SHA),
                           (SCALEUP_AUDIT, SCALEUP_AUDIT_SHA)):
        if sha(path) != expected:
            raise ValueError(f"pinned input SHA differs: {path}")
    old_manifest, source_manifest = read(OLD / "manifest.json"), read(SOURCE / "manifest.json")
    if (old_manifest["status"] != "completed" or len(old_manifest["roots"]) != 1007
            or source_manifest["status"] != "completed"
            or source_manifest["next_task"] != 32
            or read(SOURCE_AUDIT)["F_root_indices"] != list(ROOT_INDICES)
            or read(SCALEUP_AUDIT)["distinct_new_F_usable_roots"] != 0):
        raise ValueError("source bank/search/audit coverage differs")
    roster = read(ROSTER)
    for path, expected in roster["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"roster input SHA differs: {path}")
    for path, expected in source_manifest["contract"]["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"search input SHA differs: {path}")
    for position, entry in enumerate(source_manifest["shards"]):
        if (entry["file"] != f"attempt-{position:04d}.json"
                or sha(SOURCE / entry["file"]) != entry["sha256"]):
            raise ValueError(f"source shard SHA differs: {position}")
    pins = (OLD / "manifest.json", OLD_AUDIT, ROSTER, ROSTER_AUDIT,
            SOURCE / "manifest.json", SOURCE_AUDIT, SCALEUP_AUDIT,
            PROTOCOL, Path(__file__),
            Path("scripts/audit_f_short_positive_r2_yield_v1.py"),
            Path("scripts/mcts_tournament_signals_v1.py"),
            Path("scripts/mcts_training_supervision.py"))
    inputs = {str(path): sha(path) for path in pins}
    contract = {"schema": "rf-knots-additive-root-only-F-bank-v2",
                "inputs": inputs, "old_root_shards": 1007,
                "new_source_root_indices": list(ROOT_INDICES),
                "new_gradient_scope": "exact_zero_spent_root_only",
                "F": {"seed": 20261006, "k": 4,
                      "near_margin": 2, "gap_cap": 4}}
    args.out.mkdir(parents=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        roots, seen, tally = [], set(), Counter()
        for index, entry in enumerate(old_manifest["roots"]):
            source = OLD / f"root-{index:04d}.json"
            target = args.out / source.name
            if entry["file"] != source.name or sha(source) != entry["sha256"]:
                raise ValueError(f"old bank shard SHA/order differs: {index}")
            row = read(source)
            key = tuple(row["query_group"])
            if key in seen or row["root_index"] != index:
                raise ValueError("old bank query group duplicated")
            seen.add(key)
            shutil.copyfile(source, target)
            if sha(target) != entry["sha256"]:
                raise ValueError("old bank shard copy differs")
            roots.append(entry)
            tally["retained_F_roots"] += bool(row["groups"])
            tally["retained_F_groups"] += len(row["groups"])
        selected = selected_roots(roster)
        for offset, root_index in enumerate(ROOT_INDICES):
            root = selected[root_index]
            key = tuple(root["query_group"])
            if key in seen or owner(invariants(root["witness"]["start"]["word"],
                                                root["witness"]["start"]["strands"])) != "train":
                raise ValueError("new query group duplicated or not TRAIN-owned")
            seen.add(key)
            index = 1007 + offset
            row = new_row(index, root_index, root, source_manifest)
            target = args.out / f"root-{index:04d}.json"
            atomic(target, row)
            roots.append({"file": target.name, "sha256": sha(target)})
            tally["new_F_roots"] += 1
            tally["new_F_groups"] += len(row["groups"])
            tally["new_root_F_terms"] += sum(group["signal_terms"] for group in row["groups"])
        if ((tally["retained_F_roots"], tally["retained_F_groups"],
             tally["new_F_roots"], tally["new_F_groups"]) != (399, 701, 3, 4)):
            raise ValueError(f"bank F accounting differs: {dict(tally)}")
        manifest = {"contract": contract, "contract_sha256": digest(contract),
                    "status": "completed", "next_root": len(roots),
                    "roots": roots, "summary": dict(tally)}
        atomic(args.out / "manifest.json", manifest)
        print(json.dumps({"manifest_sha256": sha(args.out / "manifest.json"),
                          "summary": dict(tally)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
