"""Independent full replay and root-fork cost audit of the eight-condition micro-pilot."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_mcts_training_supervision import atomic, sha
from rf_knots import mcts_loop as M
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

DATA = Path("artifacts/f-root-only-hard-micro-v1")
ROSTER = Path("artifacts/f-root-only-roster-v1/roster.json")
ROSTER_SHA = "3c4a8eb62a63d9f4e22a4314aa94434164bb5a4c57fa70282783b9c602e23a81"
AUDIT = Path("artifacts/f-root-only-roster-audit-v1/report.json")
AUDIT_SHA = "755d48594e69c76c16041fce7279a258532b82e9ebc65254d64666d47511dd66"


def read(path):
    return json.loads(path.read_text())


def root_action(witness, instance):
    return witness.steps[0].action.to_flat(instance.spec) if witness.steps else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("audit output exists")
    if sha(ROSTER) != ROSTER_SHA or sha(AUDIT) != AUDIT_SHA:
        raise ValueError("roster/audit SHA differs")
    roster = read(ROSTER)
    roots = roster["stages"][0][:2] + roster["stages"][1][:2]
    manifest_path = DATA / "manifest.json"
    manifest = read(manifest_path)
    if (manifest["status"] != "completed" or manifest["next_task"] != 8
            or len(manifest["shards"]) != 8
            or manifest["contract"]["gradient_boundary"] !=
            "root state only; no intermediate targets"):
        raise ValueError("incomplete or changed micro contract")
    for path, expected in manifest["contract"]["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"frozen input differs: {path}")
    cells = defaultdict(Counter)
    root_forks = set()
    total_cpu = total_wall = 0.0
    neural = 0
    for position, entry in enumerate(manifest["shards"]):
        shard = DATA / f"attempt-{position:04d}.json"
        if entry["file"] != shard.name or sha(shard) != entry["sha256"]:
            raise ValueError(f"shard SHA/order differs: {position}")
        saved = read(shard)
        root_index, weight_index = divmod(position, 2)
        root = roots[root_index]
        weight = (10, 1000)[weight_index]
        instance_dict = root["instances"][weight_index]
        instance = M.Instance(**{**instance_dict,
                                 "word": tuple(instance_dict["word"])})
        row = saved["row"]
        if ((saved["position"], saved["root_index"],
             saved["generator_index"], saved["weight"], saved["seed"])
                != (position, root_index, root["generator_index"], weight,
                    20271330)
                or row["condition"] != root_index
                or row["panel_index"] != root["generator_index"]
                or row["cc_weight"] != weight
                or row["search_seed"] != 20271330
                or row["simulations"] != 64
                or bool(row["witness"]) != bool(row["solved"])):
            raise ValueError(f"task/search identity differs: {position}")
        reference = UnknotWitness.from_dict(root["witness"])
        reference.verify()
        if ((reference.start.word, reference.start.strands)
                != (instance.word, instance.strands)
                or owner(invariants(instance.word, instance.strands)) != "train"
                or reference.crossing_changes > instance.cap_crossing_changes
                or reference.moves > instance.cap_moves):
            raise ValueError(f"root ownership/start/reference caps differ: {position}")
        paths = {tuple(step.action.to_flat(instance.spec)
                       for step in reference.steps): reference}
        if row["witness"]:
            best = UnknotWitness.from_dict(row["witness"])
            best.verify()
            if ((best.start.word, best.start.strands)
                    != (instance.word, instance.strands)
                    or best.crossing_changes > instance.cap_crossing_changes
                    or best.moves > instance.cap_moves
                    or row["L10"] != 10 * best.crossing_changes + best.moves
                    or row["L1000"] != 1000 * best.crossing_changes + best.moves):
                raise ValueError(f"best witness/cost/caps differ: {position}")
        elif row["L10"] is not None or row["L1000"] is not None:
            raise ValueError(f"unsolved costs differ: {position}")
        solved_episodes = 0
        for episode in row["episodes"]:
            if episode["continuation"]["stop_reason"] != "solved":
                continue
            actions = tuple(episode["tree_prefix"] +
                            episode["continuation"]["actions"])
            witness = UnknotWitness.from_actions(instance.word,
                                                 instance.strands,
                                                 instance.spec, actions)
            witness.verify()
            end = episode["continuation"]["end"]
            if ((witness.crossing_changes, witness.moves)
                    != (end["used_cc"], end["used_moves"])
                    or witness.crossing_changes > instance.cap_crossing_changes
                    or witness.moves > instance.cap_moves):
                raise ValueError(f"episode replay/caps differ: {position}")
            paths[actions] = witness
            solved_episodes += 1
        cell = cells[(root["stage"], weight)]
        cell["attempts"] += 1
        cell["solved_episodes"] += solved_episodes
        cell["solved_searches"] += bool(row["solved"])
        cell["unique_paths_including_reference"] += len(paths)
        reference_cost = weight * reference.crossing_changes + reference.moves
        ref_root_action = root_action(reference, instance)
        if any((weight * w.crossing_changes + w.moves != reference_cost
                and root_action(w, instance) != ref_root_action)
               for w in paths.values()):
            root_forks.add(root_index)
            cell["root_fork_cost_diverse"] += 1
        cpu = float(saved["cpu_seconds"])
        wall = float(saved["wall_seconds"])
        evals = (saved["counts"]["policy_neural_evaluations"]
                 + saved["counts"]["tree_embedding_evaluations"])
        total_cpu += cpu
        total_wall += wall
        neural += evals
        cell["cpu_milliseconds"] += round(cpu * 1000)
        cell["neural_evaluations"] += evals
    report = {
        "schema": "rf-knots-root-only-hard-F-yield-micro-audit-v1",
        "manifest_sha256": sha(manifest_path),
        "roster_sha256": ROSTER_SHA,
        "roster_audit_sha256": AUDIT_SHA,
        "auditor_sha256": sha(Path(__file__)),
        "verified_shards": 8,
        "cells": {f"stage{stage}-L{weight}": dict(values)
                  for (stage, weight), values in sorted(cells.items())},
        "root_fork_cost_diverse_indices": sorted(root_forks),
        "root_fork_cost_diverse_distinct": len(root_forks),
        "cpu_seconds": total_cpu,
        "wall_seconds": total_wall,
        "neural_evaluations": neural,
        "censoring": "no-solve is not a negative action",
    }
    args.out.mkdir(parents=True)
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": sha(args.out / "report.json"),
                      "root_fork_cost_diverse_distinct": len(root_forks)},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
