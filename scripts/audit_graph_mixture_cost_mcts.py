"""Final protocol70 audit: full episode replay, cost and per-root-cap accounting."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from audit_joint_cost_mcts import audit_condition
from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, sha
from evaluate_graph_mixture_cost_mcts import CONFIG, PANEL, RUNS, build_roots, committed_rows

from rf_knots import mcts_loop as M


def inspect_rows(roots, rows, manifest):
    if len(roots) != len(rows) or manifest["next_condition"] != len(roots):
        raise ValueError("incomplete query coverage")
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
            raise ValueError("query/RNG context differs")
        instance = M.Instance(**{**root["instance"], "word": tuple(root["instance"]["word"])})
        if root["horizon"] != instance.cap_moves:
            raise ValueError("per-root horizon differs")
        counts.update(audit_condition(instance, row, {**CONFIG, "horizon": instance.cap_moves}))
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
        raise ValueError("neural-state count differs from replay")
    if any(type(v) is not int or v < 0 for v in manifest["counts"].values()):
        raise ValueError("invalid compute counters")
    seconds = [s["seconds"] for s in manifest["shards"]]
    if any(not math.isfinite(s) or s < 0 for s in seconds) or not math.isclose(
        sum(seconds), manifest["seconds_committed_batches"]
    ):
        raise ValueError("invalid committed batch timing")
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite search audit")
    mp, rp = args.run / "manifest.json", args.run / "roots.json"
    manifest = json.loads(mp.read_text())
    if (
        manifest["status"] != "completed"
        or manifest["config"] != CONFIG
        or manifest["value_head_used"] is not False
    ):
        raise ValueError("require final protocol70 search without value returns")
    model = RUNS["initial" if manifest["arm"] == "uniform" else manifest["arm"]]
    model_manifest = json.loads((model / "manifest.json").read_text())
    if (
        model_manifest["status"] != "completed"
        or manifest["inputs"].get(str(model / "checkpoint.pt"))
        != model_manifest["checkpoint_sha256"]
    ):
        raise ValueError("search used a different model")
    pins = dict(manifest["inputs"])
    sources = [
        Path(__file__).resolve(),
        Path("scripts/audit_joint_cost_mcts.py"),
        Path("scripts/evaluate_graph_mixture_cost_mcts.py"),
    ]
    merge_pins(pins, {str(p): sha(p) for p in [mp, rp, *sources]})
    merge_pins(pins, {str(args.run / s["file"]): s["sha256"] for s in manifest["shards"]})
    check_pins(pins)
    roots = json.loads(rp.read_text())
    if sha(rp) != manifest["roots_sha256"] or roots != build_roots(json.loads(PANEL.read_text())):
        raise ValueError("search roots/resources differ from panel")
    rows = committed_rows(manifest, args.run)
    counts = inspect_rows(roots, rows, manifest)
    check_pins(pins)
    atomic(
        args.out,
        {
            "status": "passed",
            "arm": manifest["arm"],
            "inputs": pins,
            "counts_replayed": counts,
            "seconds_committed_batches": manifest["seconds_committed_batches"],
            "scope": "Every success and censored episode replayed; per-root resources, "
            "RNG context and recorded cost checked. No failure-is-bad labels. "
            "Timing and tree work counters not independently rerun.",
        },
    )
    print(
        json.dumps({"status": "passed", "arm": manifest["arm"], "counts_replayed": counts}),
        flush=True,
    )


if __name__ == "__main__":
    main()
