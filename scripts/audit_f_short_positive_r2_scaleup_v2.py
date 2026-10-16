"""Independent full-replay and exact-root F audit of the small scale-up."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_f_short_positive_r2_yield_v1 import replay_episode, root_f_signal
from build_mcts_training_supervision import atomic, sha
from rf_knots import mcts_loop as M, reference
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

DATA = Path("artifacts/f-short-positive-r2-scaleup-v2")
ROSTER = Path("artifacts/f-short-positive-roster-v1/roster.json")
ROSTER_SHA = "53d6a26828acb966747d17699a3b74f5ab13e0121fa055edbbaa2f57fd242f42"
ROSTER_AUDIT = Path("artifacts/f-short-positive-roster-audit-v1/report.json")
ROSTER_AUDIT_SHA = "86257d258ca2f121e924d4343fe39f67410aa51b762bf435aae6af9ae23cc61e"
PILOT_AUDIT = Path("artifacts/f-short-positive-r2-yield-audit-v1/report.json")
PILOT_AUDIT_SHA = "e94db1cf89a364e301dc540302c061211204ae9cdc5de2c4383893a152667cbb"
SEEDS = (20271342, 20271343)


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("audit output exists")
    for path, expected in ((ROSTER, ROSTER_SHA),
                           (ROSTER_AUDIT, ROSTER_AUDIT_SHA),
                           (PILOT_AUDIT, PILOT_AUDIT_SHA)):
        if sha(path) != expected:
            raise ValueError(f"pinned input SHA differs: {path}")
    manifest_path = DATA / "manifest.json"
    manifest = read(manifest_path)
    n = manifest["next_task"]
    contract = manifest["contract"]
    if (n != len(manifest["shards"]) or n > 32
            or contract["schema"] != "rf-knots-short-positive-R2-F-scaleup-v2"
            or contract["task_cpu_ceiling"] != 120
            or contract["total_cpu_ceiling"] != 2400
            or contract["search_seeds"] != list(SEEDS)
            or contract["gradient_boundary"] !=
            "root only; no intermediate targets"):
        raise ValueError("manifest contract/progress differs")
    if not args.allow_partial and not (manifest["status"] in
            ("completed", "cpu_ceiling") and
            (n == 32 or manifest["status"] == "cpu_ceiling")):
        raise ValueError("full audit requires terminal pilot")
    for path, expected in contract["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"pinned input SHA differs: {path}")
    roster = read(ROSTER)
    roots = roster["stages"][0][4:8] + roster["stages"][1][4:8]
    first = roster["stages"][0][:4] + roster["stages"][1][:4]
    if (len(roots) != 8 or
            len({tuple(root["query_group"]) for root in roots}) != 8 or
            {tuple(root["query_group"]) for root in roots} &
            {tuple(root["query_group"]) for root in first} or
            contract["root_generator_indices"] !=
            [root["generator_index"] for root in roots]):
        raise ValueError("frozen disjoint root panel differs")
    instances, paths = {}, defaultdict(dict)
    for root_index, root in enumerate(roots):
        for weight_index, weight in enumerate((10, 1000)):
            data = root["instances"][weight_index]
            instance = M.Instance(**{**data, "word": tuple(data["word"])})
            witness = UnknotWitness.from_dict(root["witness"])
            witness.verify()
            if ((witness.start.word, witness.start.strands)
                    != (instance.word, instance.strands)
                    or owner(invariants(instance.word, instance.strands)) != "train"
                    or witness.crossing_changes > instance.cap_crossing_changes
                    or witness.moves > instance.cap_moves):
                raise ValueError("root/reference ownership/start/cap differs")
            key = root_index, weight_index
            instances[key] = instance
            actions = tuple(step.action.to_flat(instance.spec)
                            for step in witness.steps)
            paths[key][actions] = witness
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
        if ((saved["position"], saved["root_index"],
             saved["generator_index"], saved["weight"], saved["seed"])
                != (position, root_index, root["generator_index"],
                    instance.cc_weight, SEEDS[seed_index])):
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
                or row["search_seed"] != SEEDS[seed_index]
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
        count = (saved["counts"]["policy_neural_evaluations"] +
                 saved["counts"]["tree_embedding_evaluations"])
        cell["neural_evaluations"] += count
        neural += count
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
        "schema": "rf-knots-short-positive-R2-F-scaleup-audit-v2",
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
                     "stop_source" if len(usable) == 0 else
                     "replicate_before_scaling" if len(usable) == 1 else
                     "supply_cost_review_only"),
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
