"""Independent SHA, path replay, cap and exact-root F audit of short pilot."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_mcts_training_supervision import atomic, sha
from mcts_tournament_signals_v1 import _groups
from mcts_training_supervision import verified_success_traces
from rf_knots import mcts_loop as M, reference
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

DATA = Path("artifacts/f-short-positive-r2-yield-v1")
ROSTER = Path("artifacts/f-short-positive-roster-v1/roster.json")
ROSTER_SHA = "53d6a26828acb966747d17699a3b74f5ab13e0121fa055edbbaa2f57fd242f42"
ROSTER_AUDIT = Path("artifacts/f-short-positive-roster-audit-v1/report.json")
ROSTER_AUDIT_SHA = "86257d258ca2f121e924d4343fe39f67410aa51b762bf435aae6af9ae23cc61e"
SEEDS = (20271340, 20271341)


def read(path):
    return json.loads(path.read_text())


def replay_episode(instance, episode):
    word, strands = instance.word, instance.strands
    cc = moves = 0
    actions = tuple(episode["tree_prefix"] + episode["continuation"]["actions"])
    for action in actions:
        if not reference.is_legal(instance.spec, word, strands, action, True):
            raise ValueError("illegal saved episode action")
        cc += instance.spec.decode(action)[0] == CROSSING_CHANGE
        moves += 1
        word, strands = reference.apply(instance.spec, word, strands, action)
        if cc > instance.cap_crossing_changes or moves > instance.cap_moves:
            raise ValueError("saved episode exceeds caps")
    end = episode["continuation"]["end"]
    if ((tuple(end["word"]), end["strands"], end["used_cc"], end["used_moves"])
            != (word, strands, cc, moves)):
        raise ValueError("saved episode end/replay differs")
    if episode["continuation"]["stop_reason"] == "solved":
        if (word, strands) != ((), 1):
            raise ValueError("saved solved episode is not terminal")
        witness = UnknotWitness.from_actions(
            instance.word, instance.strands, instance.spec, actions)
        witness.verify()
        return witness
    return None


def root_f_signal(instance, witnesses):
    traces = verified_success_traces(instance, witnesses)
    signals = _groups(traces, "F", seed=20261006, k=4,
                      near_margin=2, gap_scale=instance.cc_weight, gap_cap=4)
    if not signals:
        return False, 0
    root = M.SearchState(instance.word, instance.strands, 0, 0)
    coefficients = signals[0]["coefficients"]
    positive = [action for (state, action), value in coefficients.items()
                if state == root and value > 1e-12]
    negative = [action for (state, action), value in coefficients.items()
                if state == root and value < -1e-12]
    if not positive or not negative:
        return False, 0
    successors = {}
    for action in set(positive + negative):
        if not reference.is_legal(instance.spec, instance.word,
                                  instance.strands, action, True):
            raise ValueError("F coefficient uses illegal root action")
        successors[action] = reference.apply(
            instance.spec, instance.word, instance.strands, action)
    good = any(successors[a] != successors[b]
               for a in positive for b in negative)
    return good, len(positive) + len(negative) if good else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("audit output exists")
    if sha(ROSTER) != ROSTER_SHA or sha(ROSTER_AUDIT) != ROSTER_AUDIT_SHA:
        raise ValueError("pinned roster/audit differs")
    manifest_path = DATA / "manifest.json"
    manifest = read(manifest_path)
    n = manifest["next_task"]
    if (n != len(manifest["shards"]) or n > 32
            or manifest["contract"]["schema"] !=
            "rf-knots-short-positive-R2-F-yield-pilot-v1"
            or manifest["contract"]["task_cpu_ceiling"] != 120
            or manifest["contract"]["total_cpu_ceiling"] != 2400
            or manifest["contract"]["gradient_boundary"] !=
            "root only; no intermediate targets"):
        raise ValueError("manifest contract/progress differs")
    if not args.allow_partial and not (manifest["status"] in
            ("completed", "cpu_ceiling") and
            (n == 32 or manifest["status"] == "cpu_ceiling")):
        raise ValueError("full audit requires terminal pilot")
    for path, expected in manifest["contract"]["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"pinned input SHA differs: {path}")
    roster = read(ROSTER)
    roots = roster["stages"][0][:4] + roster["stages"][1][:4]
    instances = {}
    references = {}
    paths = defaultdict(dict)
    for root_index, root in enumerate(roots):
        for weight_index, weight in enumerate((10, 1000)):
            data = root["instances"][weight_index]
            instance = M.Instance(**{**data, "word": tuple(data["word"])})
            reference_witness = UnknotWitness.from_dict(root["witness"])
            reference_witness.verify()
            if ((reference_witness.start.word, reference_witness.start.strands)
                    != (instance.word, instance.strands)
                    or owner(invariants(instance.word, instance.strands)) != "train"
                    or reference_witness.crossing_changes > instance.cap_crossing_changes
                    or reference_witness.moves > instance.cap_moves):
                raise ValueError("root/reference ownership/start/cap differs")
            key = root_index, weight_index
            instances[key] = instance
            references[key] = reference_witness
            actions = tuple(step.action.to_flat(instance.spec)
                            for step in reference_witness.steps)
            paths[key][actions] = reference_witness
    cells = defaultdict(Counter)
    cpu = wall = 0.0
    neural = episodes = solved_episodes = 0
    for position, entry in enumerate(manifest["shards"]):
        shard = DATA / f"attempt-{position:04d}.json"
        if entry["file"] != shard.name or sha(shard) != entry["sha256"]:
            raise ValueError(f"shard order/SHA differs: {position}")
        saved = read(shard)
        root_index, rem = divmod(position, 4)
        weight_index, seed_index = divmod(rem, 2)
        root = roots[root_index]
        instance = instances[(root_index, weight_index)]
        seed = SEEDS[seed_index]
        if ((saved["position"], saved["root_index"],
             saved["generator_index"], saved["weight"], saved["seed"])
                != (position, root_index, root["generator_index"],
                    instance.cc_weight, seed)):
            raise ValueError("shard task identity differs")
        cell = cells[(root["bucket"], instance.cc_weight)]
        cell["conditions"] += 1
        cpu += float(saved["cpu_seconds"])
        wall += float(saved["wall_seconds"])
        if saved["cpu_censored"]:
            if saved["row"] is not None or saved["counts"]:
                raise ValueError("CPU-censored task fabricated outcomes")
            cell["cpu_censored"] += 1
            continue
        row = saved["row"]
        if (row["condition"] != root_index
                or row["panel_index"] != root["generator_index"]
                or row["cc_weight"] != instance.cc_weight
                or row["search_seed"] != seed
                or row["simulations"] != 64
                or bool(row["witness"]) != bool(row["solved"])):
            raise ValueError("search identity/result differs")
        if row["witness"]:
            best = UnknotWitness.from_dict(row["witness"])
            best.verify()
            if ((best.start.word, best.start.strands)
                    != (instance.word, instance.strands)
                    or best.crossing_changes > instance.cap_crossing_changes
                    or best.moves > instance.cap_moves
                    or row["L10"] != 10 * best.crossing_changes + best.moves
                    or row["L1000"] != 1000 * best.crossing_changes + best.moves):
                raise ValueError("best witness replay/cost/caps differs")
        elif row["L10"] is not None or row["L1000"] is not None:
            raise ValueError("unsolved task has saved solution costs")
        cell["solved_searches"] += bool(row["solved"])
        for episode in row["episodes"]:
            witness = replay_episode(instance, episode)
            episodes += 1
            if witness is not None:
                actions = tuple(step.action.to_flat(instance.spec)
                                for step in witness.steps)
                paths[(root_index, weight_index)][actions] = witness
                cell["solved_episodes"] += 1
                solved_episodes += 1
        cell["neural_evaluations"] += (
            saved["counts"]["policy_neural_evaluations"]
            + saved["counts"]["tree_embedding_evaluations"])
        neural += (saved["counts"]["policy_neural_evaluations"]
                   + saved["counts"]["tree_embedding_evaluations"])
    if abs(cpu - manifest["cpu_seconds"]) > .02:
        raise ValueError("manifest CPU sum differs")
    usable = set()
    for (root_index, weight_index), witnesses in paths.items():
        instance = instances[(root_index, weight_index)]
        good, terms = root_f_signal(instance, list(witnesses.values()))
        cell = cells[(roots[root_index]["bucket"], instance.cc_weight)]
        cell["unique_completed_paths_including_reference"] += len(witnesses)
        if good:
            usable.add(root_index)
            cell["exact_root_F_groups"] += 1
            cell["root_F_terms"] += terms
    report = {
        "schema": "rf-knots-short-positive-R2-F-yield-audit-v1",
        "manifest_sha256": sha(manifest_path),
        "auditor_sha256": sha(Path(__file__)),
        "conditions": n, "episodes_replayed": episodes,
        "solved_episodes_replayed": solved_episodes,
        "distinct_new_F_usable_roots": len(usable),
        "F_root_indices": sorted(usable),
        "cpu_seconds": cpu, "wall_seconds": wall,
        "neural_evaluations": neural,
        "by_bucket_objective": {
            f"bucket{bucket}/L{weight}": dict(values)
            for (bucket, weight), values in sorted(cells.items())},
        "decision": ("partial_no_scale" if n < 32 else
                     "stop_recipe" if len(usable) == 0 else
                     "replicate_before_scaling" if len(usable) == 1 else
                     "small_scale_up_eligible"),
        "boundary": "F terms only at zero-spent TRAIN roots; no intermediate labels",
    }
    args.out.mkdir(parents=True)
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": sha(args.out / "report.json"),
                      "conditions": n, "episodes_replayed": episodes,
                      "distinct_new_F_usable_roots": len(usable),
                      "decision": report["decision"]}, sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
