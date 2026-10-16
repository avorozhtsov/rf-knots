"""Independent full-shard, replay and TRAIN-ownership audit of the S10 pilot."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import build_f_curriculum_roster_v1 as roster_source
from mcts_tournament_signals_v1 import _groups
from mcts_training_supervision import verified_success_traces
from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.cost_mcts_rollout import advance
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

ROSTER = Path("artifacts/f-curriculum-roster-v3/roster.json")
EXPECTED_SEEDS = tuple(range(20270000, 20270010))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def chosen_roots(roster):
    chosen = []
    for stage, rows in enumerate(roster["stages"]):
        if stage == 0:
            chosen.extend(rows[:8])
        else:
            chosen.extend([r for r in rows if r["reference_cc"] == 1][:4])
            chosen.extend([r for r in rows if r["reference_cc"] == 2][:4])
    return chosen


def path_owned(instance, episode, excluded):
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
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.data / "manifest.json")
    contract = manifest["contract"]
    if (manifest["status"] != "completed" or manifest["next_task"] != 640
            or len(manifest["shards"]) != 640
            or contract["schema"] != "rf-knots-f-curriculum-s10-yield-pilot-v1"
            or tuple(contract["search_seeds"]) != EXPECTED_SEEDS
            or contract["config"]["simulations"] != 16):
        raise ValueError("completed fixed search contract differs")
    for path, expected in contract["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"contract input changed: {path}")
    roster = read(ROSTER)
    chosen = chosen_roots(roster)
    if ([r["generator_index"] for r in chosen] != contract["root_generator_indices"]
            or len({tuple(r["query_group"]) for r in chosen}) != 32):
        raise ValueError("balanced root selection differs")
    excluded = roster_source.excluded_groups()
    paths = defaultdict(dict)
    counts = defaultdict(Counter)
    total = Counter()
    for position, entry in enumerate(manifest["shards"]):
        name = f"attempt-{position:04d}.json"
        if entry["file"] != name or sha(args.data / name) != entry["sha256"]:
            raise ValueError(f"shard SHA/order differs: {position}")
        root_index, within = divmod(position, 20)
        weight_index, seed_index = divmod(within, 10)
        root = chosen[root_index]
        instance_dict = root["instances"][weight_index]
        instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
        payload = read(args.data / name)
        row = payload["row"]
        weight, seed = (10, 1000)[weight_index], EXPECTED_SEEDS[seed_index]
        if (
            (payload["root_index"], payload["generator_index"], payload["weight"], payload["seed"])
            != (root_index, root["generator_index"], weight, seed)
            or row["condition"] != root_index or row["panel_index"] != root["generator_index"]
            or row["cc_weight"] != weight or row["simulations"] > 16
            or row["search_seed"] != seed or bool(row["witness"]) != bool(row["solved"])
        ):
            raise ValueError(f"attempt identity differs: {position}")
        if row["witness"] is not None:
            best = UnknotWitness.from_dict(row["witness"])
            best.verify()
            if (best.start.word, best.start.strands) != (instance.word, instance.strands):
                raise ValueError(f"best path starts elsewhere: {position}")
        if len(row["episodes"]) != len(row["episodes_train_owned"]):
            raise ValueError("ownership list differs")
        stage = root["stage"]
        key = (stage, weight)
        counts[key]["attempts"] += 1
        counts[key]["raw_solved_attempts"] += int(row["solved"])
        counts[key]["cpu_ms"] += round(payload["cpu_seconds"] * 1000)
        counts[key]["wall_ms"] += round(payload["wall_seconds"] * 1000)
        counts[key]["neural_evaluations"] += (
            payload["counts"]["policy_neural_evaluations"]
            + payload["counts"]["tree_embedding_evaluations"])
        admitted_solved = 0
        for ep, recorded in zip(row["episodes"], row["episodes_train_owned"], strict=True):
            allowed = path_owned(instance, ep, excluded)
            if allowed != recorded:
                raise ValueError(f"episode ownership differs: {position}")
            counts[key]["episodes"] += 1
            counts[key]["admitted_episodes"] += int(allowed)
            if not allowed or ep["continuation"]["stop_reason"] != "solved":
                continue
            actions = tuple(ep["tree_prefix"] + ep["continuation"]["actions"])
            witness = UnknotWitness.from_actions(instance.word, instance.strands,
                                                  instance.spec, actions)
            witness.verify()
            end = ep["continuation"]["end"]
            if ((witness.crossing_changes, witness.moves)
                    != (end["used_cc"], end["used_moves"])
                    or witness.crossing_changes > instance.cap_crossing_changes
                    or witness.moves > instance.cap_moves):
                raise ValueError(f"solution cost/cap differs: {position}")
            paths[(root_index, weight)][actions] = witness
            admitted_solved += 1
        if admitted_solved != row["eligible_solved_episodes"]:
            raise ValueError(f"admitted solved count differs: {position}")
        counts[key]["admitted_solved_attempts"] += int(admitted_solved > 0)
        counts[key]["admitted_solved_episodes"] += admitted_solved
    usable_roots = set()
    contexts = set()
    for (root_index, weight), witnesses in paths.items():
        root = chosen[root_index]
        stage = root["stage"]
        instance_dict = root["instances"][0 if weight == 10 else 1]
        instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
        traces = verified_success_traces(instance, list(witnesses.values()))
        counts[(stage, weight)]["solved_root_objective_groups"] += 1
        counts[(stage, weight)]["distinct_paths"] += len(traces)
        if len({trace["cost"] for trace in traces}) < 2:
            continue
        signals = _groups(traces, "F", seed=0, k=4, near_margin=2,
                          gap_scale=1, gap_cap=10)
        nonzero = [g for g in signals if g["coefficients"]]
        if not nonzero:
            continue
        usable_roots.add(root_index)
        counts[(stage, weight)]["f_usable_root_objective_groups"] += 1
        for signal in nonzero:
            for state, _action in signal["coefficients"]:
                contexts.add((root_index, weight, state.word, state.strands,
                              state.used_cc, state.used_moves))
    total["audited_shards"] = len(manifest["shards"])
    total["distinct_f_usable_roots"] = len(usable_roots)
    total["distinct_f_contexts"] = len(contexts)
    total["cpu_seconds"] = round(sum(c["cpu_ms"] for c in counts.values()) / 1000, 3)
    total["f_usable_roots_by_stage"] = {
        f"stage{stage}": sum(chosen[index]["stage"] == stage for index in usable_roots)
        for stage in range(4)
    }
    total["f_usable_root_indices"] = sorted(usable_roots)
    print(json.dumps({"manifest_sha256": sha(args.data / "manifest.json"),
                      "total": total,
                      "by_stage_lambda": {f"stage{stage}_L{weight}": dict(c)
                                          for (stage, weight), c in sorted(counts.items())}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
