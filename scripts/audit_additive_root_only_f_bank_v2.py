"""Independent byte-retention, SHA, path replay and root-only F bank audit."""

import argparse
import json
from collections import Counter
from pathlib import Path

import build_f_short_positive_roster_v1 as roster_source
from audit_f_short_positive_r2_yield_v1 import replay_episode
from build_mcts_training_supervision import atomic, sha
from mcts_tournament_signals_v1 import _groups
from mcts_training_supervision import verified_success_traces
from rf_knots import mcts_loop as M, reference
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

BANK = Path("artifacts/additive-root-only-f-bank-v2")
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


def source_paths(root_index, weight_index, root, instance, manifest):
    reference_witness = UnknotWitness.from_dict(root["witness"])
    reference_witness.verify()
    actions = tuple(step.action.to_flat(instance.spec)
                    for step in reference_witness.steps)
    paths = {actions: reference_witness}
    pins = []
    for seed_index, seed in enumerate((20271340, 20271341)):
        position = root_index * 4 + weight_index * 2 + seed_index
        entry = manifest["shards"][position]
        shard = SOURCE / f"attempt-{position:04d}.json"
        if entry["file"] != shard.name or sha(shard) != entry["sha256"]:
            raise ValueError(f"source shard SHA/order differs: {position}")
        saved = read(shard)
        if ((saved["position"], saved["root_index"], saved["seed"],
             saved["weight"], saved["generator_index"])
                != (position, root_index, seed, instance.cc_weight,
                    root["generator_index"])):
            raise ValueError("source shard task identity differs")
        pins.append({"file": str(shard), "sha256": entry["sha256"]})
        if saved["cpu_censored"]:
            if saved["row"] is not None or saved["counts"]:
                raise ValueError("CPU-censored source contains outcomes")
            continue
        row = saved["row"]
        if (row["search_seed"] != seed or row["cc_weight"] != instance.cc_weight
                or row["simulations"] != 64):
            raise ValueError("source row identity differs")
        for episode in row["episodes"]:
            witness = replay_episode(instance, episode)
            if witness is not None:
                actions = tuple(step.action.to_flat(instance.spec)
                                for step in witness.steps)
                paths[actions] = witness
    return paths, pins


def expected_root_terms(instance, paths):
    signals = _groups(verified_success_traces(instance, list(paths.values())),
                      "F", seed=20261006, k=4, near_margin=2,
                      gap_scale=instance.cc_weight, gap_cap=4)
    if len(signals) > 1:
        raise ValueError("unexpected multiple F contexts")
    if not signals:
        return None
    state = M.SearchState(instance.word, instance.strands, 0, 0)
    terms = {action: float(value)
             for (candidate, action), value in signals[0]["coefficients"].items()
             if candidate == state and abs(value) > 1e-12}
    pos = [action for action, value in terms.items() if value > 0]
    neg = [action for action, value in terms.items() if value < 0]
    successors = {}
    for action in terms:
        if not reference.is_legal(instance.spec, instance.word,
                                  instance.strands, action, True):
            raise ValueError("illegal root term action")
        successors[action] = reference.apply(
            instance.spec, instance.word, instance.strands, action)
    if not any(successors[a] != successors[b] for a in pos for b in neg):
        return None
    return signals[0], [{"action": action, "value": terms[action]}
                        for action in sorted(terms)]


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
            raise ValueError(f"pinned source SHA differs: {path}")
    manifest_path = BANK / "manifest.json"
    manifest = read(manifest_path)
    old_manifest = read(OLD / "manifest.json")
    source_manifest = read(SOURCE / "manifest.json")
    if (manifest["status"] != "completed" or manifest["next_root"] != 1010
            or len(manifest["roots"]) != 1010
            or manifest["contract"]["schema"] !=
            "rf-knots-additive-root-only-F-bank-v2"
            or manifest["contract"]["old_root_shards"] != 1007):
        raise ValueError("bank completion/contract differs")
    if (manifest["contract"]["new_source_root_indices"] != list(ROOT_INDICES)
            or manifest["contract"]["new_gradient_scope"] !=
            "exact_zero_spent_root_only"):
        raise ValueError("bank new-root contract differs")
    for path, expected in manifest["contract"]["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"bank pinned input SHA differs: {path}")
    roster = read(ROSTER)
    for path, expected in roster["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"roster pinned input SHA differs: {path}")
    panels = tuple(dict.fromkeys((*roster_source.root_only.viewed_panel_paths(),
                                  roster_source.EXTRA_PANEL)))
    heldout_exact = roster_source.root_only.viewed_states(panels)
    roots = roster["stages"][0][:4] + roster["stages"][1][:4]
    old_groups, tally = set(), Counter()
    for index, entry in enumerate(manifest["roots"]):
        path = BANK / f"root-{index:04d}.json"
        if entry["file"] != path.name or sha(path) != entry["sha256"]:
            raise ValueError(f"bank shard SHA/order differs: {index}")
        row = read(path)
        group_key = tuple(row["query_group"])
        if row["root_index"] != index or group_key in old_groups:
            raise ValueError(f"bank index/query group differs: {index}")
        old_groups.add(group_key)
        if index < 1007:
            old_entry = old_manifest["roots"][index]
            old_path = OLD / old_entry["file"]
            if (entry != old_entry or sha(old_path) != old_entry["sha256"]
                    or path.read_bytes() != old_path.read_bytes()):
                raise ValueError(f"old row not byte-identical: {index}")
            tally["retained_F_roots"] += bool(row["groups"])
            tally["retained_F_groups"] += len(row["groups"])
            continue
        root_index = ROOT_INDICES[index - 1007]
        root = roots[root_index]
        witness = UnknotWitness.from_dict(root["witness"])
        witness.verify()
        if (row["stage"] != 1 or row["query_group"] != root["query_group"]
                or row["gradient_scope"] != "exact_zero_spent_root_only"
                or (tuple(witness.start.word), witness.start.strands)
                in heldout_exact or owner(invariants(witness.start.word,
                                                     witness.start.strands)) != "train"):
            raise ValueError(f"new root ownership/exclusion/scope differs: {index}")
        by_weight = {group["weight"]: group for group in row["groups"]}
        if len(by_weight) != len(row["groups"]):
            raise ValueError("duplicate objective group")
        expected_count = 0
        for weight_index, weight in enumerate((10, 1000)):
            instance_dict = root["instances"][weight_index]
            instance = M.Instance(**{**instance_dict,
                                     "word": tuple(instance_dict["word"])})
            if (witness.start.word, witness.start.strands) != (instance.word,
                                                                instance.strands):
                raise ValueError("reference/instance start differs")
            paths, pins = source_paths(root_index, weight_index, root,
                                       instance, source_manifest)
            expected = expected_root_terms(instance, paths)
            if expected is None:
                if weight in by_weight:
                    raise ValueError("non-F objective admitted to bank")
                continue
            expected_count += 1
            if weight not in by_weight:
                raise ValueError("F objective missing from bank")
            group = by_weight[weight]
            signal, terms = expected
            if (group["root_index"] != index or group["stage"] != 1
                    or group["gradient_scope"] != "exact_zero_spent_root_only"
                    or group["instance"] != instance_dict
                    or group["source_shards"] != pins
                    or group["root_coefficients"] != terms
                    or group["signal_terms"] != len(terms)
                    or group["positive_paths"] != signal["positive_paths"]
                    or group["negative_paths"] != signal["negative_paths"]):
                raise ValueError(f"new root-only F terms/metadata differ: {index}")
            stored_paths = group["paths"]
            if len(stored_paths) != len(paths):
                raise ValueError("new completed path count differs")
            observed = {}
            for item in stored_paths:
                actions = tuple(item["actions"])
                path_witness = UnknotWitness.from_actions(
                    instance.word, instance.strands, instance.spec, actions)
                path_witness.verify()
                if (path_witness.crossing_changes > instance.cap_crossing_changes
                        or path_witness.moves > instance.cap_moves
                        or (item["cc"], item["moves"]) !=
                        (path_witness.crossing_changes, path_witness.moves)):
                    raise ValueError("stored path replay/caps/cost differs")
                observed[actions] = path_witness
            if observed.keys() != paths.keys():
                raise ValueError("bank path set not identical to source")
            tally["new_F_groups"] += 1
            tally["new_root_F_terms"] += len(terms)
            tally["replayed_new_paths"] += len(observed)
        if expected_count != len(row["groups"]) or not expected_count:
            raise ValueError("new root group accounting differs")
        tally["new_F_roots"] += 1
    if (len(old_groups), tally["retained_F_roots"], tally["retained_F_groups"],
            tally["new_F_roots"], tally["new_F_groups"]) != (1010, 399, 701, 3, 4):
        raise ValueError(f"final bank tally differs: {dict(tally)}")
    if manifest["summary"] != {key: tally[key] for key in manifest["summary"]}:
        raise ValueError("manifest summary differs")
    report = {"schema": "rf-knots-additive-root-only-F-bank-audit-v2",
              "bank_manifest_sha256": sha(manifest_path),
              "auditor_sha256": sha(Path(__file__)),
              "root_shards": 1010, "distinct_query_groups": len(old_groups),
              "F_usable_roots": 402, "F_groups": 705,
              "metrics": dict(tally),
              "boundary": "new F coefficients only at exact zero-spent TRAIN root"}
    args.out.mkdir(parents=True)
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": sha(args.out / "report.json"),
                      "metrics": dict(tally)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
