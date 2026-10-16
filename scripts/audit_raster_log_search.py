"""Final97 search audit: every path, resource, count and semantic cost certificate."""

import argparse
import json
from pathlib import Path

from audit_graph_mixture_cost_mcts import inspect_rows
from audit_joint_policy_metric_panel import check_pins, merge_pins
from check_qrun_common_witnesses import sha, write_new
from evaluate_graph_mixture_cost_mcts import CONFIG, PANEL, build_roots
from evaluate_joint_cost_mcts import committed_rows
from summarize_graph_mixture_experiment import certify_costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    mp, rp = args.run / "manifest.json", args.run / "roots.json"
    manifest = json.loads(mp.read_text())
    if (
        manifest["status"] != "completed"
        or manifest["config"] != CONFIG
        or manifest["schema"] != "rf-knots-raster-log-cost-mcts-v1"
        or manifest["arm"] not in ("q254", "random")
        or manifest["value_head_used"] is not False
    ):
        raise ValueError("require final97 raw-raster search without value returns")
    pair_path = Path(manifest["training_pair_audit"])
    pair = json.loads(pair_path.read_text())
    model = Path(manifest["model_checkpoint"])
    expected = manifest["checkpoint_sha256"]
    if (
        pair["status"] != "passed"
        or set(pair["arms"]) != {"q254", "random"}
        or pair["arms"][manifest["arm"]]["checkpoint_sha256"] != expected
        or sha(model) != expected
        or pair["inputs"].get(str(model)) != expected
        or manifest["inputs"].get(str(model)) != expected
        or manifest["inputs"].get(str(pair_path)) != sha(pair_path)
    ):
        raise ValueError("search differs from audited trained model")
    pins = dict(manifest["inputs"])
    merge_pins(pins, pair["inputs"])
    sources = [
        Path(__file__),
        Path("scripts/audit_graph_mixture_cost_mcts.py"),
        Path("scripts/audit_joint_cost_mcts.py"),
        Path("scripts/summarize_graph_mixture_experiment.py"),
        Path("scripts/audit_semantic_cost_floor.py"),
        Path("scripts/burau_integer_check.py"),
        Path("tests/test_raster_log_search.py"),
    ]
    merge_pins(pins, {str(p): sha(p) for p in [mp, rp, *sources]})
    merge_pins(pins, {str(args.run / s["file"]): s["sha256"] for s in manifest["shards"]})
    check_pins(pins)
    panel = json.loads(PANEL.read_text())
    roots = json.loads(rp.read_text())
    if len(roots) != 128 or sha(rp) != manifest["roots_sha256"] or roots != build_roots(panel):
        raise ValueError("search roots/resources differ from complete panel")
    rows = committed_rows(manifest, args.run)
    counts = inspect_rows(roots, rows, manifest)
    costs = certify_costs(panel, roots, {manifest["arm"]: rows})
    check_pins(pins)
    write_new(
        args.out,
        {
            "status": "passed",
            "arm": manifest["arm"],
            "inputs": pins,
            "checkpoint_sha256": expected,
            "counts_replayed": counts,
            "certified_costs": costs,
            "seconds_committed_batches": manifest["seconds_committed_batches"],
            "scope": "All successes and censored paths replayed. Only equality with a certified "
            "semantic cost floor proves optimality. Tree encoding and timing counters "
            "are recorded measurements, not independently rerun compute.",
        },
    )
    print(json.dumps({"status": "passed", "arm": manifest["arm"], "counts": counts}))


if __name__ == "__main__":
    main()
