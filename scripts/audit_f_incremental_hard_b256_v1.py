"""Independent SHA, replay, ownership and exact F audit for protocol319."""

import argparse
import json
from pathlib import Path

import audit_f_incremental_r2_yield_pilot_v1 as base
import build_f_incremental_roster_v2 as roster_source
from audit_f_curriculum_yield_pilot_v1 import path_owned
from build_mcts_training_supervision import atomic
from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness

DATA = Path("artifacts/f-incremental-hard-b256-v1")
ROSTER = Path("artifacts/f-incremental-roster-v2/roster.json")
ROSTER_SHA = "93bbaa282ec79bd017b5c7adfbd4f55041adddebfe5505650d258d1f17ce58f8"
CONTRACT_SHA = "70a37c1a6997da9e4a97ebe77dba9ff9ec2911010d39968ace55904fc791b1f7"
SOURCE_SHA = "8067ab722d6710b8d65d42d3b1d2e28e3ac9de4e5f2432192bb468d0e4c07a82"


def selected_roots(roster):
    chosen = []
    for stage in (2, 3):
        rows = roster["stages"][stage]
        chosen += [row for row in rows if row["reference_cc"] == 1][:4]
        chosen += [row for row in rows if row["reference_cc"] == 2][:4]
    return chosen


def audit_one(position, manifest, roots, excluded):
    entry = manifest["shards"][position]
    path = DATA / f"attempt-{position:04d}.json"
    if entry["file"] != path.name or base.sha(path) != entry["sha256"]:
        raise ValueError(f"shard SHA/order differs: {position}")
    payload = base.read(path)
    root_index, within = divmod(position, 4)
    weight_index, seed_index = divmod(within, 2)
    root = roots[root_index]
    weight, seed = (10, 1000)[weight_index], (20271321, 20271322)[seed_index]
    instance_dict = root["instances"][weight_index]
    instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
    row = payload["row"]
    if ((payload["root_index"], payload["generator_index"], payload["weight"],
         payload["seed"]) != (root_index, root["generator_index"], weight, seed)
            or row["condition"] != root_index
            or row["panel_index"] != root["generator_index"]
            or row["cc_weight"] != weight
            or row["search_seed"] != seed
            or not 0 <= row["simulations"] <= 256
            or bool(row["witness"]) != bool(row["solved"])):
        raise ValueError(f"task/search identity differs: {position}")
    if row["witness"]:
        best = UnknotWitness.from_dict(row["witness"])
        best.verify()
        if ((best.start.word, best.start.strands) != (instance.word, instance.strands)
                or best.crossing_changes > instance.cap_crossing_changes
                or best.moves > instance.cap_moves
                or row["L10"] != 10 * best.crossing_changes + best.moves
                or row["L1000"] != 1000 * best.crossing_changes + best.moves):
            raise ValueError(f"best witness replay/cost/caps differs: {position}")
    elif row["L10"] is not None or row["L1000"] is not None:
        raise ValueError(f"unsolved cost differs: {position}")
    if len(row["episodes"]) != len(row["episodes_train_owned"]):
        raise ValueError(f"ownership flags differ: {position}")
    admitted = []
    solved_episodes = 0
    for episode, recorded in zip(row["episodes"], row["episodes_train_owned"], strict=True):
        if path_owned(instance, episode, excluded) != recorded:
            raise ValueError(f"episode ownership differs: {position}")
        if episode["continuation"]["stop_reason"] != "solved":
            continue
        actions = tuple(episode["tree_prefix"] + episode["continuation"]["actions"])
        witness = UnknotWitness.from_actions(instance.word, instance.strands,
                                             instance.spec, actions)
        witness.verify()
        end = episode["continuation"]["end"]
        if ((witness.crossing_changes, witness.moves)
                != (end["used_cc"], end["used_moves"])
                or witness.crossing_changes > instance.cap_crossing_changes
                or witness.moves > instance.cap_moves):
            raise ValueError(f"episode replay/caps differs: {position}")
        solved_episodes += 1
        if recorded:
            admitted.append(list(actions))
    if len(admitted) != row["eligible_solved_episodes"]:
        raise ValueError(f"admitted solved count differs: {position}")
    return {"position": position, "root_index": root_index, "stage": root["stage"],
            "weight": weight, "seed": seed, "reference_cc": root["reference_cc"],
            "raw_solved": bool(row["solved"]), "episodes": len(row["episodes"]),
            "solved_episodes": solved_episodes, "admitted_paths": admitted,
            "cpu_seconds": payload["cpu_seconds"],
            "wall_seconds": payload["wall_seconds"],
            "neural_evaluations": payload["counts"]["policy_neural_evaluations"]
                                  + payload["counts"]["tree_embedding_evaluations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = DATA / "manifest.json"
    manifest = base.read(manifest_path)
    manifest_sha = base.sha(manifest_path)
    if (base.sha(ROSTER) != ROSTER_SHA or manifest["status"] != "completed"
            or manifest["next_task"] != 64 or len(manifest["shards"]) != 64
            or manifest["contract_sha256"] != CONTRACT_SHA
            or [value for path, value in manifest["contract"]["inputs"].items()
                if path.endswith("scripts/collect_f_incremental_hard_b256_v1.py")]
            != [SOURCE_SHA]):
        raise ValueError("completed pilot contract/coverage differs")
    for path, expected in manifest["contract"]["inputs"].items():
        if base.sha(Path(path)) != expected:
            raise ValueError(f"pinned input changed: {path}")
    roots = selected_roots(base.read(ROSTER))
    if ([r["generator_index"] for r in roots]
            != manifest["contract"]["root_generator_indices"]):
        raise ValueError("frozen root selection differs")
    excluded = roster_source.old.excluded_groups()
    for panel in roster_source.PANELS:
        excluded.update(tuple(row["query_group"]) for row in base.read(panel)["rows"])
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for position in range(64):
        record_path = args.out / f"audit-{position:04d}.json"
        if record_path.exists():
            saved = base.read(record_path)
            if saved["manifest_sha256"] != manifest_sha or saved["position"] != position:
                raise ValueError(f"saved audit position differs: {position}")
            record = saved["record"]
        else:
            record = audit_one(position, manifest, roots, excluded)
            atomic(record_path, {"manifest_sha256": manifest_sha,
                                 "position": position, "record": record})
        records.append(record)
        if (position + 1) % 8 == 0:
            print(json.dumps({"audited": position + 1, "of": 64}), flush=True)
    report = {"schema": "rf-knots-incremental-hard-b256-audit-v1",
              "manifest_sha256": manifest_sha,
              "audited_shards": len(records), "metrics": base.summarize(records, roots)}
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": base.sha(args.out / "report.json"),
                      "metrics": report["metrics"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
