"""Read-only diagnostic: exact reference joins in the finished hard micro search."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from rf_knots import mcts_loop as M
from rf_knots.cost_mcts_rollout import advance
from rf_knots.evidence import SemanticAction, UnknotWitness


ROSTER = Path("artifacts/f-root-only-roster-v1/roster.json")
PILOT = Path("artifacts/f-root-only-hard-micro-v1")
REPORT = Path("artifacts/f-reference-bridge-micro-diagnostic-v1/report.json")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_states(instance, witness):
    """Re-encode semantic reference actions under the actual search ActionSpec."""
    witness.verify()
    if (witness.start.word, witness.start.strands) != (instance.word, instance.strands):
        raise ValueError("reference start differs")
    flats = tuple(step.action.to_flat(instance.spec) for step in witness.steps)
    state = M.SearchState(instance.word, instance.strands, 0, 0)
    states = [state]
    for action in flats:
        state = advance(instance, state, action)
        states.append(state)
    if state.key != M.SOLVED:
        raise ValueError("reference terminal differs")
    return flats, states


def scan_episode(instance, flats, reference, episode):
    """Replay the observed prefix, then splice only exact-state reference suffixes."""
    actions = tuple(episode["tree_prefix"] + episode["continuation"]["actions"])
    state = M.SearchState(instance.word, instance.strands, 0, 0)
    visited = {state.key}
    joins = Counter()
    best = None
    by_key = {}
    for index, ref_state in enumerate(reference):
        if index >= 2:
            by_key.setdefault(ref_state.key, []).append(index)
    for depth, action in enumerate(actions, 1):
        state = advance(instance, state, action)
        if state.key in visited:
            joins["repeated_prefix_state"] += 1
            break
        visited.add(state.key)
        for ref_index in by_key.get(state.key, ()):
            joins["exact_state"] += 1
            if actions[0] == flats[0]:
                joins["same_first_action"] += 1
                continue
            suffix = flats[ref_index:]
            if (state.used_cc + reference[-1].used_cc - reference[ref_index].used_cc
                    > instance.cap_crossing_changes or
                    state.used_moves + len(suffix) > instance.cap_moves):
                joins["resource_cap"] += 1
                continue
            try:
                completed = UnknotWitness.from_actions(
                    instance.word, instance.strands, instance.spec,
                    actions[:depth] + suffix)
                completed.verify()
            except ValueError:
                joins["invalid_splice"] += 1
                continue
            if (completed.crossing_changes > instance.cap_crossing_changes
                    or completed.moves > instance.cap_moves):
                joins["resource_cap"] += 1
                continue
            joins["valid_root_fork"] += 1
            ref_cost = (instance.cc_weight * reference[-1].used_cc
                        + reference[-1].used_moves)
            cost = instance.cc_weight * completed.crossing_changes + completed.moves
            if cost != ref_cost:
                joins["strict_cost_pair"] += 1
            candidate = {"depth": depth, "reference_index": ref_index,
                         "cc": completed.crossing_changes, "moves": completed.moves,
                         "cost": cost, "reference_cost": ref_cost,
                         "strict_cost_pair": cost != ref_cost}
            if best is None or (cost, depth) < (best["cost"], best["depth"]):
                best = candidate
    return joins, best


def main():
    if REPORT.exists():
        raise ValueError("diagnostic output already exists")
    manifest_path = PILOT / "manifest.json"
    manifest = read(manifest_path)
    if manifest["status"] != "completed" or len(manifest["shards"]) != 8:
        raise ValueError("micro pilot incomplete")
    for path, expected in manifest["contract"]["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"pinned input changed: {path}")
    roster = read(ROSTER)
    roots = roster["stages"][0][:2] + roster["stages"][1][:2]
    output = []
    for task, entry in enumerate(manifest["shards"]):
        shard = PILOT / f"attempt-{task:04d}.json"
        if entry["file"] != shard.name or sha(shard) != entry["sha256"]:
            raise ValueError("micro shard SHA/order differs")
        row = read(shard)
        root = roots[task // 2]
        instance_dict = root["instances"][task % 2]
        instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
        witness = UnknotWitness.from_dict(root["witness"])
        flats, states = reference_states(instance, witness)
        counts, best = Counter(), None
        for episode in row["row"]["episodes"]:
            found, candidate = scan_episode(instance, flats, states, episode)
            counts.update(found)
            if candidate is not None and (best is None or
                    (candidate["cost"], candidate["depth"]) <
                    (best["cost"], best["depth"])):
                best = candidate
        output.append({"task": task, "root_index": task // 2,
                       "stage": root["stage"], "weight": instance.cc_weight,
                       "episodes": len(row["row"]["episodes"]),
                       "counts": dict(counts), "best": best,
                       "cpu_seconds": row["cpu_seconds"]})
    report = {"schema": "rf-knots-reference-bridge-micro-diagnostic-v1",
              "role": "posthoc TRAIN diagnostic, not a frozen yield pilot or training",
              "manifest_sha256": sha(manifest_path),
              "source_sha256": sha(Path(__file__)),
              "tasks": output,
              "totals": dict(sum((Counter(row["counts"]) for row in output), Counter())),
              "roots_with_strict_cost_pair": len({row["root_index"] for row in output
                                                  if row["counts"].get("strict_cost_pair", 0)}),
              "cpu_seconds": sum(row["cpu_seconds"] for row in output)}
    REPORT.parent.mkdir(parents=True, exist_ok=False)
    REPORT.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"report_sha256": sha(REPORT),
                      "totals": report["totals"],
                      "roots_with_strict_cost_pair": report["roots_with_strict_cost_pair"]},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
