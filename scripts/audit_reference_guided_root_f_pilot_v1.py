"""Independent SHA, trajectory, bridge, ownership and cost audit of TRAIN pilot."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_cost_mcts_continuations import audit_episode
from build_mcts_training_supervision import atomic, sha
from reference_guided_bridge_v1 import completed_bridges, reference_actions_states
from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness


ROOT = Path("artifacts/f-reference-guided-root-pilot-v1")
ROSTER = Path("artifacts/f-root-only-roster-v1/roster.json")
GENERATORS = (0, 90, 35, 63)
SEEDS = (20271336, 20271337)


def read(path):
    return json.loads(path.read_text())


def check(predicate, message):
    if not predicate:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("audit output exists")
    manifest_path = ROOT / "manifest.json"
    manifest = read(manifest_path)
    contract = manifest["contract"]
    check(contract["generator_indices"] == list(GENERATORS)
          and contract["seeds"] == list(SEEDS), "pilot selection differs")
    check(len(manifest["shards"]) == manifest["next_task"] <= 16,
          "pilot progress differs")
    if not args.allow_partial:
        check(manifest["status"] in ("completed", "cpu_ceiling")
              and (manifest["next_task"] == 16 or manifest["status"] == "cpu_ceiling"),
              "pilot still running")
    for name, expected in contract["inputs"].items():
        check(sha(Path(name)) == expected, f"pinned input changed: {name}")
    roster = read(ROSTER)
    selected = []
    for stage in roster["stages"]:
        ordered = sorted(stage, key=lambda row:
                         (row["instances"][0]["cap_crossing_changes"],
                          row["generator_index"]))
        selected.extend((ordered[0], ordered[-1]))
    check(tuple(row["generator_index"] for row in selected) == GENERATORS,
          "independent root selection differs")
    episodes_replayed = bridges_replayed = 0
    useful_roots = set()
    groups = defaultdict(Counter)
    cpu = 0.0
    for index, entry in enumerate(manifest["shards"]):
        shard = ROOT / f"attempt-{index:04d}.json"
        check(entry["file"] == shard.name and sha(shard) == entry["sha256"],
              f"shard SHA/order differs: {index}")
        row = read(shard)
        root_index, within_root = divmod(index, 4)
        weight_index, seed_index = divmod(within_root, 2)
        root = selected[root_index]
        instance_dict = root["instances"][weight_index]
        instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
        check((row["task"], row["root_index"], row["generator_index"],
               row["stage"], row["weight"], row["seed"])
              == (index, root_index, root["generator_index"], root["stage"],
                  instance.cc_weight, SEEDS[seed_index]), "task identity differs")
        check(row["cpu_seconds"] >= 0 and row["wall_seconds"] >= 0,
              "negative compute duration")
        cpu += row["cpu_seconds"]
        group = groups[f"stage{root['stage']}/L{int(instance.cc_weight)}"]
        group["conditions"] += 1
        if row["status"] == "cpu_censored":
            check(row["result"] is None, "censored task has a result")
            group["cpu_censored"] += 1
            continue
        check(row["status"] == "searched", "unknown task status")
        result = row["result"]
        check(0 <= result["simulations"] <= contract["config"]["simulations"],
              "simulation count differs")
        ref = UnknotWitness.from_dict(root["witness"])
        actions, states = reference_actions_states(instance, ref)
        local_counts = Counter()
        strict = []
        for episode in result["episodes"]:
            audit_episode(instance, episode, {
                "horizon": instance.cap_moves,
                "insertion_candidates": contract["config"]["insertion_candidates"]})
            episodes_replayed += 1
            counts, bridges = completed_bridges(instance, actions, states, episode)
            local_counts.update(counts)
            for bridge in bridges:
                witness = UnknotWitness.from_dict(bridge["witness"])
                witness.verify()
                check((witness.start.word, witness.start.strands)
                      == (instance.word, instance.strands), "bridge wrong start")
                check(witness.crossing_changes <= instance.cap_crossing_changes
                      and witness.moves <= instance.cap_moves, "bridge resource cap")
                check(bridge["cost"] == instance.cc_weight * witness.crossing_changes
                      + witness.moves, "bridge cost differs")
                bridges_replayed += 1
                if bridge["strict_cost_pair"]:
                    strict.append(bridge)
        check(dict(local_counts) == result["bridge_counts"],
              "bridge count differs from independent replay")
        expected_best = min(strict, key=lambda r: r["cost"], default=None)
        check((expected_best is None) == (result["best_strict_bridge"] is None),
              "best strict bridge presence differs")
        if expected_best is not None:
            check(result["best_strict_bridge"]["cost"] == expected_best["cost"],
                  "best strict bridge cost differs")
            useful_roots.add(tuple(root["query_group"]))
        if result["natural_witness"] is not None:
            natural = UnknotWitness.from_dict(result["natural_witness"])
            natural.verify()
            check((natural.start.word, natural.start.strands)
                  == (instance.word, instance.strands), "natural witness wrong start")
        group["episodes"] += len(result["episodes"])
        group["strict_pairs"] += local_counts["strict_cost_pair"]
        group["policy_neural_evaluations"] += result["policy_neural_evaluations"]
    check(abs(cpu - manifest["cpu_seconds"]) < .01,
          "total CPU duration differs")
    report = {"schema": "rf-knots-reference-guided-root-F-pilot-audit-v1",
              "manifest_sha256": sha(manifest_path),
              "auditor_sha256": sha(Path(__file__)),
              "conditions": manifest["next_task"],
              "episodes_replayed": episodes_replayed,
              "bridges_replayed": bridges_replayed,
              "distinct_new_F_usable_roots": len(useful_roots),
              "cpu_seconds": cpu,
              "by_stage_objective": {name: dict(counts) for name, counts in groups.items()},
              "decision": ("stop_recipe" if manifest["next_task"] == 16
                           and len(useful_roots) == 0 else
                           "replicate_before_scaling" if manifest["next_task"] == 16
                           and len(useful_roots) == 1 else
                           "small_scale_up_eligible" if manifest["next_task"] == 16
                           and len(useful_roots) >= 2 else "incomplete")}
    args.out.mkdir(parents=True)
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": sha(args.out / "report.json"),
                      "conditions": report["conditions"],
                      "distinct_new_F_usable_roots": len(useful_roots),
                      "decision": report["decision"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
