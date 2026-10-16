"""Independent SHA, replay, TRAIN ownership and yield audit of Q-run teacher."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_f_curriculum_roster_v1 import excluded_groups
from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.cost_mcts_rollout import advance
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

ROOT = Path("artifacts/qrun-rung1-onpolicy-teacher-v1")
ROSTER = Path("artifacts/f-curriculum-roster-v3/roster.json")
SOURCE_SHA = "d45b6c20a4a4b956eeaf490093754fb2e6460e4ba1c4812c69d14abcfcc965bd"
CONTRACT_SHA = "7445b1708a5ea91811b115ed781773c7420b714f05642fba61a081b84bee6931"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def owned(instance, episode, excluded):
    state = M.SearchState(instance.word, instance.strands, 0, 0)
    states = [state]
    actions = episode["tree_prefix"] + episode["continuation"]["actions"]
    for action in actions:
        state = advance(instance, state, action)
        if instance.spec.decode(action)[0] == CROSSING_CHANGE:
            states.append(state)
    return all(owner(fp := invariants(s.word, s.strands)) == "train"
               and (fp[0], abs(fp[1])) not in excluded for s in states)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    count = manifest["next_task"]
    if (manifest["contract_sha256"] != CONTRACT_SHA
            or manifest["contract"]["schema"] != "rf-knots-qrun-rung1-onpolicy-teacher-v1"
            or manifest["contract"]["config"]["simulations"] != 64
            or manifest["contract"]["config"]["seed"] != 20271300
            or count != len(manifest["shards"])):
        raise ValueError("teacher contract or coverage differs")
    if not args.allow_partial and (manifest["status"] != "completed" or count != 2000):
        raise ValueError("full audit requires completed 2000-condition collection")
    for name, expected in manifest["contract"]["inputs"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"pinned teacher input changed: {name}")
    if next(value for name, value in manifest["contract"]["inputs"].items()
            if name.endswith("collect_qrun_rung1_onpolicy_teacher_v1.py")) != SOURCE_SHA:
        raise ValueError("teacher source identity differs")
    roster = json.loads(ROSTER.read_text())
    roots = [row for stage in roster["stages"] for row in stage]
    if (len(roots) != 1000
            or [row["generator_index"] for row in roots]
            != manifest["contract"]["root_generator_indices"]):
        raise ValueError("roster order differs")
    excluded = excluded_groups()
    stats = defaultdict(Counter)
    unique_paths = defaultdict(set)
    if args.start < 0 or args.start > count or args.limit < 0:
        raise ValueError("audit shard range differs")
    audit_end = min(count, args.start + args.limit) if args.limit else count
    for position in range(args.start, audit_end):
        entry = manifest["shards"][position]
        name = f"condition-{position:04d}.json"
        if entry["file"] != name or sha(ROOT / name) != entry["sha256"]:
            raise ValueError(f"shard SHA/order differs: {position}")
        row = json.loads((ROOT / name).read_text())
        root_index, weight_index = divmod(position, 2)
        root = roots[root_index]
        instance_dict = root["instances"][weight_index]
        instance = M.Instance(**{**instance_dict,
                                  "word": tuple(instance_dict["word"])})
        weight = (10, 1000)[weight_index]
        result = row["result"]
        if ((row["task"], row["root_index"], row["stage"],
             row["generator_index"], row["weight"], row["seed"])
            != (position, root_index, root["stage"], root["generator_index"],
                weight, 20271300)
                or (result["condition"], result["panel_index"],
                    result["cc_weight"], result["search_seed"],
                    result["simulations"])
                != (root_index, root["generator_index"], weight, 20271300, 64)
                or bool(result["witness"]) != bool(result["solved"])):
            raise ValueError(f"task/search identity differs: {position}")
        if result["witness"] is not None:
            best_raw = UnknotWitness.from_dict(result["witness"])
            best_raw.verify()
            if ((best_raw.start.word, best_raw.start.strands)
                    != (instance.word, instance.strands)
                    or best_raw.crossing_changes > instance.cap_crossing_changes
                    or best_raw.moves > instance.cap_moves):
                raise ValueError(f"raw best start/cap differs: {position}")
        if len(result["episodes"]) != len(result["episodes_train_owned"]):
            raise ValueError(f"ownership flag count differs: {position}")
        key = (root["stage"], weight)
        counter = stats[key]
        counter["attempts"] += 1
        counter["raw_solved_attempts"] += int(result["solved"])
        counter["cpu_ms"] += round(row["cpu_seconds"] * 1000)
        counter["wall_ms"] += round(row["wall_seconds"] * 1000)
        counter["neural_evaluations"] += (
            row["counts"]["policy_neural_evaluations"]
            + row["counts"]["tree_embedding_evaluations"])
        reference = UnknotWitness.from_dict(root["witness"])
        reference.verify()
        reference_cost = weight * reference.crossing_changes + reference.moves
        best_admitted = None
        admitted_count = 0
        for episode, recorded in zip(result["episodes"],
                                     result["episodes_train_owned"], strict=True):
            allowed = owned(instance, episode, excluded)
            if allowed != recorded:
                raise ValueError(f"TRAIN ownership differs: {position}")
            counter["episodes"] += 1
            counter["admitted_episodes"] += int(allowed)
            if not allowed or episode["continuation"]["stop_reason"] != "solved":
                continue
            actions = tuple(episode["tree_prefix"]
                            + episode["continuation"]["actions"])
            witness = UnknotWitness.from_actions(instance.word,
                                                  instance.strands,
                                                  instance.spec, actions)
            witness.verify()
            end = episode["continuation"]["end"]
            if ((witness.crossing_changes, witness.moves)
                    != (end["used_cc"], end["used_moves"])
                    or witness.crossing_changes > instance.cap_crossing_changes
                    or witness.moves > instance.cap_moves):
                raise ValueError(f"admitted solution cost/cap differs: {position}")
            cost = weight * witness.crossing_changes + witness.moves
            best_admitted = cost if best_admitted is None else min(best_admitted, cost)
            unique_paths[key].add((root_index, actions))
            counter["admitted_solved_episodes"] += 1
            admitted_count += 1
        if result["eligible_solved_episodes"] != admitted_count:
            raise ValueError(f"admitted solution count differs: {position}")
        if best_admitted is not None:
            counter["admitted_solved_attempts"] += 1
            counter["improved_over_reference"] += int(best_admitted < reference_cost)
            counter["equal_to_reference"] += int(best_admitted == reference_cost)
            counter["dearer_than_reference"] += int(best_admitted > reference_cost)
    output = {}
    for (stage, weight), counter in sorted(stats.items()):
        counter["unique_admitted_paths"] = len(unique_paths[(stage, weight)])
        output[f"stage{stage}_L{weight}"] = dict(counter)
    print(json.dumps({"manifest_sha256": sha(ROOT / "manifest.json"),
                      "contract_sha256": CONTRACT_SHA,
                      "audited_shards": audit_end - args.start,
                      "start_shard": args.start,
                      "completed_collection": manifest["status"] == "completed",
                      "by_stage_lambda": output}, sort_keys=True))


if __name__ == "__main__":
    main()
