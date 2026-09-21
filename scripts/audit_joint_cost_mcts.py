"""Final-only independent episode, witness, resource and accounting audit of TEST MCTS."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from audit_cost_mcts_continuations import audit_episode
from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, sha
from evaluate_joint_cost_mcts import CONFIG, PANEL, RUNS, build_roots, committed_rows

from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness


def audit_condition(instance, row, config):
    if row["cc_weight"] != instance.cc_weight:
        raise ValueError("condition cost weight differs")
    if not 0 <= row["simulations"] <= config["simulations"]:
        raise ValueError("simulation count outside contract")
    if not 0 <= row["maximum_tree_depth"] <= row["maximum_observed_depth"] <= instance.cap_moves:
        raise ValueError("observed depth outside resource caps")
    counts, last_sim = Counter(conditions=1), 0
    for episode in row["episodes"]:
        if not last_sim < episode["simulation"] <= row["simulations"]:
            raise ValueError("episode simulations are not strictly ordered")
        last_sim = episode["simulation"]
        steps, scores, depth = audit_episode(instance, episode, config)
        if (
            depth > row["maximum_observed_depth"]
            or len(episode["tree_prefix"]) > row["maximum_tree_depth"]
        ):
            raise ValueError("episode exceeds declared observed depth")
        counts.update(
            episodes=1,
            continuation_moves=steps,
            full_path_moves=steps + len(episode["tree_prefix"]),
            policy_action_scores=scores,
        )
    if row["solved"] != (row["witness"] is not None):
        raise ValueError("solution status differs from witness presence")
    if not row["solved"]:
        if row["L10"] is not None or row["L1000"] is not None or row["incumbents"]:
            raise ValueError("unsolved condition has a cost/incumbent")
        return counts
    witness = UnknotWitness.from_dict(row["witness"])
    witness.verify()
    if (witness.start.word, witness.start.strands) != (instance.word, instance.strands):
        raise ValueError("witness starts at another root")
    if (witness.max_len, witness.max_strands) != (instance.spec.max_len, instance.spec.max_strands):
        raise ValueError("witness alphabet contract differs")
    if (
        witness.moves > instance.cap_moves
        or witness.crossing_changes > instance.cap_crossing_changes
    ):
        raise ValueError("witness exceeds resource caps")
    if (row["L10"], row["L1000"]) != (
        10 * witness.crossing_changes + witness.moves,
        1000 * witness.crossing_changes + witness.moves,
    ):
        raise ValueError("cost differs from replayed witness")
    best, last_sim = float("inf"), 0
    for incumbent in row["incumbents"]:
        if not (
            0 <= incumbent["cc"] <= instance.cap_crossing_changes
            and 0 <= incumbent["moves"] <= instance.cap_moves
            and last_sim <= incumbent["simulation"] <= row["simulations"]
        ):
            raise ValueError("incumbent exceeds resources or is out of order")
        cost = instance.cc_weight * incumbent["cc"] + incumbent["moves"]
        if incumbent["cost"] != cost or not cost < best:
            raise ValueError("incumbent costs are not strictly improving")
        best, last_sim = cost, incumbent["simulation"]
    if best != instance.cc_weight * witness.crossing_changes + witness.moves:
        raise ValueError("last incumbent differs from final witness")
    counts["solved_conditions"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite search audit")
    manifest_path = args.run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["status"] != "completed"
        or manifest["config"] != CONFIG
        or manifest["value_head_used"] is not False
    ):
        raise ValueError("require final protocol-58 evaluation without value-head returns")
    model = RUNS["parent" if manifest["arm"] == "uniform" else manifest["arm"]]
    model_manifest = json.loads((model / "manifest.json").read_text())
    if (
        model_manifest["status"] != "completed"
        or manifest["inputs"].get(str(model / "checkpoint.pt"))
        != model_manifest["checkpoint_sha256"]
    ):
        raise ValueError("evaluation model differs from completed checkpoint")
    pins = dict(manifest["inputs"])
    sources = [
        Path(__file__).resolve(),
        Path("scripts/evaluate_joint_cost_mcts.py"),
        Path("scripts/audit_cost_mcts_continuations.py"),
    ]
    merge_pins(pins, {str(p): sha(p) for p in [manifest_path, args.run / "roots.json", *sources]})
    check_pins(pins)
    roots = json.loads((args.run / "roots.json").read_text())
    if sha(args.run / "roots.json") != manifest["roots_sha256"] or roots != build_roots(
        json.loads(PANEL.read_text())
    ):
        raise ValueError("query roots differ from frozen panel")
    rows = committed_rows(manifest, args.run)
    if len(rows) != len(roots):
        raise ValueError("incomplete query coverage")
    counts = Counter()
    for index, (row, root) in enumerate(zip(rows, roots, strict=True)):
        begin = index // CONFIG["batch_conditions"] * CONFIG["batch_conditions"]
        if (
            row["condition"] != root["condition"]
            or row["panel_index"] != root["panel_index"]
            or row["search_seed"] != CONFIG["seed"] + begin
            or row["tree_seed"] != (CONFIG["seed"] + begin) * 9973 + index - begin
        ):
            raise ValueError("query identity or RNG context differs")
        instance = M.Instance(**{**root["instance"], "word": tuple(root["instance"]["word"])})
        counts.update(audit_condition(instance, row, CONFIG))
    for key in (
        "conditions",
        "episodes",
        "continuation_moves",
        "full_path_moves",
        "policy_action_scores",
        "solved_conditions",
    ):
        if counts[key] != manifest["counts"].get(key, 0):
            raise ValueError(f"replayed count differs: {key}")
    neural = counts["continuation_moves"] if manifest["arm"] != "uniform" else 0
    if manifest["counts"]["policy_neural_evaluations"] != neural:
        raise ValueError("neural-state evaluation count differs from replay")
    if any(not isinstance(v, int) or v < 0 for v in manifest["counts"].values()):
        raise ValueError("invalid compute counters")
    seconds = sum(s["seconds"] for s in manifest["shards"])
    if (
        not math.isfinite(seconds)
        or seconds < 0
        or not math.isclose(seconds, manifest["seconds_committed_batches"])
    ):
        raise ValueError("committed-batch timing differs")
    for shard in manifest["shards"]:
        pins[str(args.run / shard["file"])] = shard["sha256"]
    check_pins(pins)
    result = {
        "status": "passed",
        "arm": manifest["arm"],
        "inputs": pins,
        "counts_replayed": dict(counts),
        "seconds_committed_batches": seconds,
        "caveat": "Every episode and final witness replayed; all roots/resources/seeds checked. "
        "Recorded timing and tree neural-work counters are not independently rerun.",
    }
    atomic(args.out, result)
    print(json.dumps({"status": "passed", "arm": manifest["arm"], "counts_replayed": dict(counts)}))


if __name__ == "__main__":
    main()
