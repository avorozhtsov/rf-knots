"""Resumable independent replay/ownership/F-yield audit for protocol295."""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import audit_qrun_rung1_onpolicy_teacher_v1 as teacher_audit
from build_f_curriculum_roster_v1 import excluded_groups
from mcts_tournament_signals_v1 import _groups
from mcts_training_supervision import verified_success_traces
from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness

ROOT = Path("artifacts/f-hard-stage-move-slack-pilot-v1")
ROSTER = Path("artifacts/f-curriculum-roster-v3/roster.json")
ROSTER_SHA = "268f1c4d2e8731bcfd4cc743bed767c9864fb74b2a2022ed679fbe36af134506"
MANIFEST_SHA = "e4423cd705cc5c351e83b886fe941d63b652dd24666cd69feff8c0e92a6be926"
CONTRACT_SHA = "a528d2872fb45a6a1f773c2b4b545d9dc751a354c125b5abc800c2bc53f9387a"
SOURCE_SHA = "8e498c91a0b8b358d6b6c6859b4b3c48fb458ef24aec4474888f5a5bae08228b"
teacher_audit.invariants = lru_cache(maxsize=20_000)(teacher_audit.invariants)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def audit_one(position, manifest, roots, excluded):
    entry = manifest["shards"][position]
    path = ROOT / f"condition-{position:04d}.json"
    if entry["file"] != path.name or sha(path) != entry["sha256"]:
        raise ValueError(f"shard SHA/order differs: {position}")
    row = json.loads(path.read_text())
    root_index, weight_index, seed, slack = manifest["contract"]["tasks"][position]
    root = roots[root_index]
    base_instance = root["instances"][weight_index]
    instance_dict = {**base_instance, "cap_moves": base_instance["cap_moves"] + slack}
    instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
    weight = (10, 1000)[weight_index]
    result = row["result"]
    if (row["task"], row["root_index"], row["stage"], row["generator_index"],
        row["weight"], row["seed"], row["simulations"], row["move_slack"]) != (
            position, root_index, 3, root["generator_index"], weight, seed, 128, slack):
        raise ValueError(f"task identity differs: {position}")
    if (result["condition"], result["panel_index"], result["cc_weight"],
        result["search_seed"], result["simulations"]) != (
            root_index, root["generator_index"], weight, seed + position, 128):
        raise ValueError(f"search identity differs: {position}")
    if bool(result["witness"]) != bool(result["solved"]):
        raise ValueError(f"solved/witness mismatch: {position}")
    if result["witness"]:
        best = UnknotWitness.from_dict(result["witness"])
        best.verify()
        if ((best.start.word, best.start.strands) != (instance.word, instance.strands)
                or best.crossing_changes > instance.cap_crossing_changes
                or best.moves > instance.cap_moves
                or result["L10"] != 10 * best.crossing_changes + best.moves
                or result["L1000"] != 1000 * best.crossing_changes + best.moves):
            raise ValueError(f"best witness replay/cost/cap differs: {position}")
    elif result["L10"] is not None or result["L1000"] is not None:
        raise ValueError(f"unsolved cost differs: {position}")
    if len(result["episodes"]) != len(result["episodes_train_owned"]):
        raise ValueError(f"ownership flags differ: {position}")
    solved_paths, admitted_paths = [], []
    for episode, recorded in zip(result["episodes"],
                                 result["episodes_train_owned"], strict=True):
        allowed = teacher_audit.owned(instance, episode, excluded)
        if allowed != recorded:
            raise ValueError(f"independent TRAIN ownership differs: {position}")
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
            raise ValueError(f"episode witness replay/cap differs: {position}")
        solved_paths.append(list(actions))
        if allowed:
            admitted_paths.append(list(actions))
    if len(admitted_paths) != result["eligible_solved_episodes"]:
        raise ValueError(f"eligible solved count differs: {position}")
    return {"position": position, "root_index": root_index,
            "weight": weight, "seed": seed, "slack": slack,
            "reference_cc": root["instances"][0]["cap_crossing_changes"],
            "raw_solved": result["solved"],
            "episodes": len(result["episodes"]),
            "solved_episodes": len(solved_paths),
            "admitted_solved_episodes": len(admitted_paths),
            "admitted_paths": admitted_paths,
            "cpu_seconds": row["cpu_seconds"],
            "wall_seconds": row["wall_seconds"],
            "neural_evaluations": (row["counts"]["policy_neural_evaluations"]
                                   + row["counts"]["tree_embedding_evaluations"])}


def summarize(records, roots):
    by_cell = defaultdict(Counter)
    paths = defaultdict(set)
    for record in records:
        key = (record["slack"], record["weight"], record["reference_cc"])
        counter = by_cell[key]
        counter["attempts"] += 1
        counter["raw_solved_attempts"] += int(record["raw_solved"])
        counter["admitted_solved_attempts"] += int(bool(record["admitted_paths"]))
        counter["episodes"] += record["episodes"]
        counter["solved_episodes"] += record["solved_episodes"]
        counter["admitted_solved_episodes"] += record["admitted_solved_episodes"]
        counter["cpu_ms"] += round(record["cpu_seconds"] * 1000)
        counter["wall_ms"] += round(record["wall_seconds"] * 1000)
        counter["neural_evaluations"] += record["neural_evaluations"]
        for path in record["admitted_paths"]:
            paths[(record["root_index"], record["weight"], record["slack"])].add(tuple(path))
    f_roots_by_slack = defaultdict(set)
    for (root_index, weight, slack), alternatives in sorted(paths.items()):
        root = roots[root_index]
        base_instance = root["instances"][0 if weight == 10 else 1]
        instance_dict = {**base_instance, "cap_moves": base_instance["cap_moves"] + slack}
        instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
        reference = UnknotWitness.from_dict(root["witness"])
        reference.verify()
        witnesses = [reference]
        seen = {tuple(step.action for step in reference.steps)}
        for actions in sorted(alternatives):
            if actions not in seen:
                seen.add(actions)
                witness = UnknotWitness.from_actions(instance.word, instance.strands,
                                                     instance.spec, actions)
                witness.verify()
                witnesses.append(witness)
        traces = verified_success_traces(instance, witnesses)
        key = (slack, weight, instance.cap_crossing_changes)
        counter = by_cell[key]
        counter["unique_admitted_paths"] += len(alternatives)
        if len({trace["cost"] for trace in traces}) > 1:
            counter["different_cost_groups"] += 1
        groups = _groups(traces, "F", seed=20261006, k=4,
                         near_margin=2, gap_scale=weight, gap_cap=4)
        if groups and any(group["coefficients"] for group in groups):
            counter["F_usable_groups"] += 1
            counter["F_terms"] += sum(len(group["coefficients"]) for group in groups)
            f_roots_by_slack[slack].add(root_index)
    cells = {}
    for (slack, weight, cap), counter in sorted(by_cell.items()):
        counter["F_usable_groups"] += 0
        counter["different_cost_groups"] += 0
        counter["unique_admitted_paths"] += 0
        cells[f"slack{slack}_L{weight}_CC{cap}"] = dict(counter)
    return {"by_slack_objective_cc": cells,
            "distinct_F_roots_by_slack": {str(s): len(f_roots_by_slack[s])
                                          for s in (16, 32)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = ROOT / "manifest.json"
    if sha(manifest_path) != MANIFEST_SHA or sha(ROSTER) != ROSTER_SHA:
        raise ValueError("pinned manifest or roster changed")
    manifest = json.loads(manifest_path.read_text())
    if (manifest["status"] != "completed" or manifest["next_task"] != 64
            or len(manifest["shards"]) != 64
            or manifest["contract_sha256"] != CONTRACT_SHA):
        raise ValueError("pilot contract/coverage differs")
    for path, expected in manifest["contract"]["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"pinned input changed: {path}")
    source = next(v for k, v in manifest["contract"]["inputs"].items()
                  if k.endswith("collect_f_hard_stage_move_slack_pilot_v1.py"))
    if source != SOURCE_SHA:
        raise ValueError("pilot source differs")
    roots = json.loads(ROSTER.read_text())["stages"][3]
    excluded = excluded_groups()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for position in range(64):
        path = args.out / f"audit-{position:04d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if (saved["manifest_sha256"] != MANIFEST_SHA
                    or saved["source_sha256"] != sha(Path(__file__))
                    or saved["record"]["position"] != position):
                raise ValueError(f"saved audit record differs: {position}")
            record = saved["record"]
        else:
            record = audit_one(position, manifest, roots, excluded)
            atomic(path, {"manifest_sha256": MANIFEST_SHA,
                          "source_sha256": sha(Path(__file__)),
                          "record": record})
        records.append(record)
        print(json.dumps({"audited": position + 1, "of": 64}), flush=True)
    report = {"schema": "rf-knots-f-hard-stage-move-slack-pilot-audit-v1",
              "manifest_sha256": MANIFEST_SHA, "audit_source_sha256": sha(Path(__file__)),
              "audited_shards": len(records), "metrics": summarize(records, roots)}
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": sha(args.out / "report.json"),
                      "metrics": report["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
