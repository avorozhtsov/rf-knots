"""Post-exit149 replay of every successful/censored path and both native costs."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from audit_joint_cost_mcts import audit_condition
from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, sha
from evaluate_graph_mixture_cost_mcts import build_roots
from evaluate_joint_cost_mcts import committed_rows
from evaluate_long_general_search import CONFIG, declared, read, require_cache_gate, validate_resume
from long_general_evaluation_inputs import load_inputs, load_panel, model_entry
from summarize_graph_mixture_experiment import certify_costs

from rf_knots import mcts_loop as M


def inspect_rows(roots, rows, state):
    if len(roots) != len(rows) or state["next_condition"] != len(roots):
        raise ValueError("incomplete149 search coverage")
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
            raise ValueError("149 query/RNG identity differs")
        instance = M.Instance(**{**root["instance"], "word": tuple(root["instance"]["word"])})
        if root["horizon"] != instance.cap_moves:
            raise ValueError("native per-root horizon differs")
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
            raise ValueError(f"replayed149 count differs: {key}")
    if stored["policy_neural_evaluations"] != counts["continuation_moves"]:
        raise ValueError("policy neural-state count differs from replay")
    if state["contract"]["prior"] == "uniform" and (
        stored["tree_embedding_evaluations"] or stored["tree_embedding_batches"]
    ):
        raise ValueError("uniform tree prior reports learned-distance forwards")
    if any(type(value) is not int or value < 0 for value in stored.values()):
        raise ValueError("invalid work counter")
    if stored["cache_logical_tiles"] != stored["cache_computed_tiles"] + stored["cache_hits"]:
        raise ValueError("cache accounting differs")
    seconds = [s["seconds"] for s in state["shards"]]
    if any(not math.isfinite(t) or t < 0 for t in seconds) or not math.isclose(
        sum(seconds), state["seconds_committed_batches"]
    ):
        raise ValueError("invalid committed timing")
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".partial").exists():
        raise ValueError("new149 search audit output required")
    manifest_path = args.search / "manifest.json"
    state = read(manifest_path)
    contract = state["contract"]
    declared(contract["arm"], contract["stage"], contract["prior"])
    if state["status"] != "completed" or contract["config"] != CONFIG:
        raise ValueError("require completed149 search")
    run, pair_path = Path(contract["training_run"]), Path(contract["pair_audit"])
    parent, audit, pins = load_inputs(run, pair_path)
    merge_pins(pins, contract["inputs"])
    if contract["training_contract_sha256"] != parent["contract_sha256"]:
        raise ValueError("search training contract differs")
    _, checkpoint = model_entry(run, audit, contract["arm"], contract["stage"])
    if (
        str(checkpoint) != contract["checkpoint"]
        or sha(checkpoint) != contract["checkpoint_sha256"]
    ):
        raise ValueError("search checkpoint differs from audited149 pair")
    require_cache_gate(
        Path(contract["cache_gate"]),
        checkpoint,
        contract["arm"],
        contract["stage"],
        pins,
        parent["contract_sha256"],
    )
    _, panel_path = load_panel("short-u", parent, pins)
    if contract["panel"] != str(panel_path):
        raise ValueError("search query panel differs")
    panel = read(panel_path)
    roots = build_roots(panel)
    if len(roots) != 128:
        raise ValueError("require64 roots at both L10/L1000")
    validate_resume(state, contract, roots, args.search)
    for path in (
        manifest_path,
        args.search / "roots.json",
        Path(__file__).resolve(),
        Path("scripts/audit_joint_cost_mcts.py"),
        Path("scripts/summarize_graph_mixture_experiment.py"),
        Path("scripts/audit_semantic_cost_floor.py"),
        Path("tests/test_long_general_search_audit.py"),
    ):
        merge_pins(pins, {str(path): sha(path)})
    merge_pins(pins, {str(args.search / s["file"]): s["sha256"] for s in state["shards"]})
    check_pins(pins)
    rows = committed_rows(state, args.search)
    counts = inspect_rows(roots, rows, state)
    label = "-".join(contract[k] for k in ("arm", "stage", "prior"))
    costs = certify_costs(panel, roots, {label: rows})
    check_pins(pins)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic(
        args.out,
        dict(
            schema="rf-knots-long-general-search-audit-v1",
            status="passed",
            label=label,
            inputs=pins,
            search=str(args.search),
            training_contract_sha256=parent["contract_sha256"],
            checkpoint_sha256=sha(checkpoint),
            counts_replayed=counts,
            certified_costs=costs,
            seconds_committed_batches=state["seconds_committed_batches"],
            scope="All success/censored paths and reference witnesses replayed. Bounds are not "
            "automatically optima. Hardware/cache/timing counters are recorded, "
            "not recomputed forwards.",
        ),
    )
    print(json.dumps(dict(status="passed", label=label, counts=counts)), flush=True)


if __name__ == "__main__":
    main()
