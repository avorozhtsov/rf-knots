"""Independent shard SHA, full replay and exact F-yield audit for protocol318."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import build_f_incremental_roster_v2 as roster_source
from audit_f_curriculum_yield_pilot_v1 import path_owned
from build_mcts_training_supervision import atomic
from mcts_tournament_signals_v1 import _groups
from mcts_training_supervision import verified_success_traces
from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness

DATA = Path("artifacts/f-incremental-r2-yield-pilot-v1")
ROSTER = Path("artifacts/f-incremental-roster-v2/roster.json")
MANIFEST_SHA = "5955d30ecc30d092087b63ef864c0fe6fab4a9296e0417bf82d2d73df682c2d2"
ROSTER_SHA = "93bbaa282ec79bd017b5c7adfbd4f55041adddebfe5505650d258d1f17ce58f8"
CONTRACT_SHA = "a1230d748cfd908ea15d642f3186928d3a14c79f114ddd1b81d525e277b1a1e4"
SOURCE_SHA = "eba76950e3fd543a42e3e037d1ee6b00acc6365f8b4726939e705c27ad903a6c"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def selected_roots(roster):
    chosen = list(roster["stages"][0][:8])
    for rows in roster["stages"][1:]:
        chosen += [r for r in rows if r["reference_cc"] == 1][:4]
        chosen += [r for r in rows if r["reference_cc"] == 2][:4]
    return chosen


def audit_one(position, manifest, roots, excluded):
    entry = manifest["shards"][position]
    path = DATA / f"attempt-{position:04d}.json"
    if entry["file"] != path.name or sha(path) != entry["sha256"]:
        raise ValueError(f"shard SHA/order differs: {position}")
    payload = read(path)
    root_index, within = divmod(position, 4)
    weight_index, seed_index = divmod(within, 2)
    root = roots[root_index]
    weight, seed = (10, 1000)[weight_index], (20271317, 20271318)[seed_index]
    instance_dict = root["instances"][weight_index]
    instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
    row = payload["row"]
    if ((payload["root_index"], payload["generator_index"], payload["weight"],
         payload["seed"]) != (root_index, root["generator_index"], weight, seed)
            or row["condition"] != root_index
            or row["panel_index"] != root["generator_index"]
            or row["cc_weight"] != weight
            or row["search_seed"] != seed
            or not 0 <= row["simulations"] <= 64
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


def summarize(records, roots):
    cells = defaultdict(Counter)
    paths = defaultdict(set)
    total_cpu = 0.0
    for record in records:
        key = (record["stage"], record["weight"], record["reference_cc"])
        counter = cells[key]
        counter["attempts"] += 1
        counter["raw_solved_attempts"] += int(record["raw_solved"])
        counter["admitted_solved_attempts"] += int(bool(record["admitted_paths"]))
        counter["episodes"] += record["episodes"]
        counter["solved_episodes"] += record["solved_episodes"]
        counter["admitted_solved_episodes"] += len(record["admitted_paths"])
        counter["cpu_ms"] += round(record["cpu_seconds"] * 1000)
        counter["wall_ms"] += round(record["wall_seconds"] * 1000)
        counter["neural_evaluations"] += record["neural_evaluations"]
        total_cpu += record["cpu_seconds"]
        for actions in record["admitted_paths"]:
            paths[(record["root_index"], record["weight"])].add(tuple(actions))
    usable_roots = set()
    improved_roots = set()
    exact_contexts = set()
    for (root_index, weight), alternatives in sorted(paths.items()):
        root = roots[root_index]
        instance_dict = root["instances"][0 if weight == 10 else 1]
        instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
        reference = UnknotWitness.from_dict(root["witness"])
        reference.verify()
        ref_actions = tuple(step.action.to_flat(instance.spec) for step in reference.steps)
        witnesses = [reference]
        for actions in sorted(alternatives):
            if actions != ref_actions:
                witness = UnknotWitness.from_actions(instance.word, instance.strands,
                                                      instance.spec, actions)
                witness.verify()
                witnesses.append(witness)
        reference_cost = weight * reference.crossing_changes + reference.moves
        best_cost = min(weight * w.crossing_changes + w.moves for w in witnesses)
        key = (root["stage"], weight, root["reference_cc"])
        counter = cells[key]
        counter["unique_admitted_paths"] += len(alternatives)
        if best_cost < reference_cost:
            counter["better_than_reference_groups"] += 1
            improved_roots.add(root_index)
        traces = verified_success_traces(instance, witnesses)
        if len({trace["cost"] for trace in traces}) > 1:
            counter["different_cost_groups"] += 1
        signals = _groups(traces, "F", seed=20261006, k=4,
                          near_margin=2, gap_scale=weight, gap_cap=4)
        nonzero = [group for group in signals if group["coefficients"]]
        if nonzero:
            counter["F_usable_groups"] += 1
            counter["F_terms"] += sum(len(group["coefficients"]) for group in nonzero)
            usable_roots.add(root_index)
            for group in nonzero:
                for state, _action in group["coefficients"]:
                    exact_contexts.add((root_index, weight, state.word, state.strands,
                                        state.used_cc, state.used_moves))
    for counter in cells.values():
        for field in ("unique_admitted_paths", "better_than_reference_groups",
                      "different_cost_groups", "F_usable_groups", "F_terms"):
            counter[field] += 0
    return {"attempts": len(records), "distinct_new_F_usable_roots": len(usable_roots),
            "F_usable_root_indices": sorted(usable_roots),
            "F_usable_roots_by_stage": {str(stage): sum(roots[i]["stage"] == stage
                                                        for i in usable_roots)
                                        for stage in range(4)},
            "exact_F_contexts": len(exact_contexts),
            "improved_reference_roots": len(improved_roots),
            "cpu_seconds": round(total_cpu, 3),
            "cpu_seconds_per_F_root": round(total_cpu / len(usable_roots), 3)
                                      if usable_roots else None,
            "by_stage_objective_cc": {f"stage{s}_L{w}_CC{c}": dict(row)
                                      for (s, w, c), row in sorted(cells.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sha(DATA / "manifest.json") != MANIFEST_SHA or sha(ROSTER) != ROSTER_SHA:
        raise ValueError("frozen manifest or roster differs")
    manifest = read(DATA / "manifest.json")
    if (manifest["status"] != "completed" or manifest["next_task"] != 128
            or len(manifest["shards"]) != 128
            or manifest["contract_sha256"] != CONTRACT_SHA
            or [value for path, value in manifest["contract"]["inputs"].items()
                if path.endswith("scripts/collect_f_incremental_r2_yield_pilot_v1.py")]
            != [SOURCE_SHA]):
        raise ValueError("completed pilot contract/coverage differs")
    for path, expected in manifest["contract"]["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"pinned input changed: {path}")
    roots = selected_roots(read(ROSTER))
    if ([r["generator_index"] for r in roots]
            != manifest["contract"]["root_generator_indices"]):
        raise ValueError("frozen root selection differs")
    excluded = roster_source.old.excluded_groups()
    for panel in roster_source.PANELS:
        excluded.update(tuple(row["query_group"]) for row in read(panel)["rows"])
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for position in range(128):
        record_path = args.out / f"audit-{position:04d}.json"
        if record_path.exists():
            saved = read(record_path)
            if (saved["manifest_sha256"] != MANIFEST_SHA
                    or saved["position"] != position):
                raise ValueError(f"saved audit position differs: {position}")
            record = saved["record"]
        else:
            record = audit_one(position, manifest, roots, excluded)
            atomic(record_path, {"manifest_sha256": MANIFEST_SHA,
                                 "position": position, "record": record})
        records.append(record)
        if (position + 1) % 8 == 0:
            print(json.dumps({"audited": position + 1, "of": 128}), flush=True)
    report = {"schema": "rf-knots-incremental-R2-F-yield-audit-v1",
              "manifest_sha256": MANIFEST_SHA,
              "audited_shards": len(records), "metrics": summarize(records, roots)}
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": sha(args.out / "report.json"),
                      "metrics": report["metrics"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
