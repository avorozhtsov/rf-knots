"""Replay all139 search paths and certify native costs; no new neural evaluation."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from audit_joint_cost_mcts import audit_condition
from audit_joint_policy_metric_panel import check_pins, merge_pins
from check_qrun_common_witnesses import sha, write_new
from evaluate_graph_mixture_cost_mcts import build_roots
from evaluate_joint_cost_mcts import committed_rows
from evaluate_resource_search import CONFIG, require_cache_gate, validate_resume
from resource_evaluation_inputs import load_panel, model_entry
from summarize_graph_mixture_experiment import certify_costs

from rf_knots import mcts_loop as M


def inspect_rows(roots, rows, state):
    if len(roots) != len(rows) or state["next_condition"] != len(roots):
        raise ValueError("incomplete search coverage")
    counts = Counter()
    for index, (root, row) in enumerate(zip(roots, rows, strict=True)):
        begin = index // CONFIG["batch_conditions"] * CONFIG["batch_conditions"]
        if (
            row["condition"] != root["condition"]
            or root["condition"] != index
            or row["panel_index"] != root["panel_index"]
            or row["search_seed"] != CONFIG["seed"] + begin
            or row["tree_seed"] != (CONFIG["seed"] + begin) * 9973 + index - begin
        ):
            raise ValueError("query/RNG identity differs")
        instance = M.Instance(**{**root["instance"], "word": tuple(root["instance"]["word"])})
        if root["horizon"] != instance.cap_moves:
            raise ValueError("per-root horizon differs")
        counts.update(audit_condition(instance, row, {**CONFIG, "horizon": instance.cap_moves}))
    stored = state["counts"]
    for key in (
        "conditions",
        "episodes",
        "continuation_moves",
        "full_path_moves",
        "policy_action_scores",
        "solved_conditions",
    ):
        if counts[key] != stored.get(key, 0):
            raise ValueError(f"replayed count differs: {key}")
    if stored["policy_neural_evaluations"] != counts["continuation_moves"]:
        raise ValueError("policy neural-state count differs from replay")
    if state["contract"]["prior"] == "uniform" and (
        stored["tree_embedding_evaluations"] or stored["tree_embedding_batches"]
    ):
        raise ValueError("uniform tree prior reports learned-distance calls")
    if any(type(v) is not int or v < 0 for v in stored.values()):
        raise ValueError("invalid work counters")
    if stored["cache_logical_tiles"] != stored["cache_computed_tiles"] + stored["cache_hits"]:
        raise ValueError("cache accounting differs")
    durations = [shard["seconds"] for shard in state["shards"]]
    if any(not math.isfinite(t) or t < 0 for t in durations) or not math.isclose(
        sum(durations), state["seconds_committed_batches"]
    ):
        raise ValueError("invalid committed timing")
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest_path, roots_path = args.search / "manifest.json", args.search / "roots.json"
    state = json.loads(manifest_path.read_text())
    contract = state["contract"]
    if (
        state["status"] != "completed"
        or contract["config"] != CONFIG
        or contract["arm"] not in ("units", "raw")
        or contract["stage"] not in ("initial", "final")
        or contract["prior"] not in ("distance", "uniform")
        or contract["stage"] == "initial"
        and contract["prior"] != "distance"
    ):
        raise ValueError("require completed declared139 search")
    pins = dict(contract["inputs"])
    pair_path = Path(contract["pair_audit"])
    if pins.get(str(pair_path)) != sha(pair_path):
        raise ValueError("paired training audit changed")
    pair = json.loads(pair_path.read_text())
    _, checkpoint = model_entry(
        Path(contract["training_run"]), pair, contract["arm"], contract["stage"]
    )
    if (
        str(checkpoint) != contract["checkpoint"]
        or sha(checkpoint) != contract["checkpoint_sha256"]
    ):
        raise ValueError("search model differs from audited model")
    require_cache_gate(
        Path(contract["cache_gate"]), checkpoint, contract["arm"], contract["stage"], pins
    )
    panel, panel_pins, panel_path = load_panel("short-u")
    merge_pins(pins, panel_pins)
    if contract["panel"] != str(panel_path):
        raise ValueError("search panel differs")
    roots = build_roots(panel)
    if len(roots) != 128:
        raise ValueError("require128 conditions")
    validate_resume(state, contract, roots, args.search)
    rows = committed_rows(state, args.search)
    counts = inspect_rows(roots, rows, state)
    label = "-".join(contract[k] for k in ("arm", "stage", "prior"))
    costs = certify_costs(panel, roots, {label: rows})
    for path in (
        manifest_path,
        roots_path,
        Path(__file__).resolve(),
        Path("scripts/audit_joint_cost_mcts.py"),
        Path("scripts/summarize_graph_mixture_experiment.py"),
        Path("scripts/audit_semantic_cost_floor.py"),
        Path("tests/test_resource_search_audit.py"),
    ):
        merge_pins(pins, {str(path): sha(path)})
    merge_pins(pins, {str(args.search / s["file"]): s["sha256"] for s in state["shards"]})
    check_pins(pins)
    write_new(
        args.out,
        dict(
            status="passed",
            label=label,
            inputs=pins,
            checkpoint_sha256=sha(checkpoint),
            counts_replayed=counts,
            certified_costs=costs,
            seconds_committed_batches=state["seconds_committed_batches"],
            scope="All successes and censored paths replayed; only certified "
            "cost-floor equality proves optimality. Timing/tree/cache "
            "counters are recorded, not independently rerun compute.",
        ),
    )
    print(json.dumps(dict(status="passed", label=label, counts=counts)), flush=True)


if __name__ == "__main__":
    main()
