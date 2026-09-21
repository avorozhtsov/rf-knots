"""Final-checkpoint-only protocol74 distances; requires independent data and pair audits."""

import argparse
import json
from pathlib import Path

import torch
from audit_graph_mixture_short_u_panel import stratum
from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, sha
from build_pure_braid_distance_panel import SCHEMA
from evaluate_graph_mixture_cost_mcts import RUNS
from evaluate_strand_graph_panel import group_report
from run_strand_graph_pilot import metric_report, pair_row
from summarize_clean_paired_evaluation import metric_stats
from train_graph_metric_mixture import INITIAL_SHA

from rf_knots import mcts_loop as M
from rf_knots.embedding_metric import MetricEmbeddingObjective

PANEL = Path("artifacts/pure-braid-short-distance-test-v1/panel.json")
PAIR_AUDIT = Path("artifacts/graph-mixture-evaluation-v1/training-pair-audit.json")


def summaries(predictions, records):
    if not predictions or len(predictions) != len(records):
        raise ValueError("incomplete distance coverage")
    for i, (p, r) in enumerate(zip(predictions, records, strict=True)):
        interval = r["interval"]
        g = f"{r['left_fingerprint'][0]}:{abs(r['left_fingerprint'][1])}"
        if (p["panel_index"], p["lower"], p["upper"], p["fingerprint_group"]) != (
            i,
            interval["lower"],
            interval["upper"],
            g,
        ):
            raise ValueError("prediction labels/identity differ")

    def stats(rows):
        if not rows:
            return {"rows": 0, "statistics": None}
        return {
            **metric_stats(rows),
            "constants": {
                str(c): metric_stats(
                    [
                        {**r, "prediction": c, "violation": max(r["lower"] - c, c - r["upper"], 0)}
                        for r in rows
                    ]
                )
                for c in (0, 1, 2)
            },
        }

    result = {
        "all": stats(predictions),
        "positive_lower_bound": stats([p for p in predictions if p["lower"] >= 1]),
        "exact_zero_coordinate": stats([p for p in predictions if p["lower"] == p["upper"] == 0]),
        "exact_one": stats([p for p in predictions if p["lower"] == p["upper"] == 1]),
        "exact_two": stats([p for p in predictions if p["lower"] == p["upper"] == 2]),
    }
    result["raw_query_strata"] = {
        s: stats(
            [
                p
                for p, r in zip(predictions, records, strict=True)
                if stratum(len(r["interval"]["left"]["word"]), r["interval"]["left"]["strands"])
                == s
            ]
        )
        for s in ("within", "length_only", "strands_only", "both")
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(RUNS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite distance evaluation")
    panel = json.loads(PANEL.read_text())
    ap = PANEL.parent / "independence-audit.json"
    audit = json.loads(ap.read_text())
    pair = json.loads(PAIR_AUDIT.read_text())
    if (
        panel["schema"] != SCHEMA
        or not panel["complete"]
        or not panel["construction_complete"]
        or audit["status"] != "passed"
        or audit["evaluation_ready"] is not True
        or audit["inputs"].get(str(PANEL)) != sha(PANEL)
        or pair["status"] != "passed"
        or pair["same_initial_checkpoint"] != INITIAL_SHA
        or set(pair["arms"]) != {"graph", "control"}
    ):
        raise ValueError(
            "require adequate independently audited panel and final matched training pair"
        )
    run = RUNS[args.arm]
    model_path, manifest_path = run / "checkpoint.pt", run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = INITIAL_SHA if args.arm == "initial" else pair["arms"][args.arm]["checkpoint_sha256"]
    if (
        manifest["status"] != "completed"
        or manifest["checkpoint_sha256"] != expected
        or sha(model_path) != expected
        or pair["inputs"].get(str(model_path)) != expected
    ):
        raise ValueError("model differs from declared final checkpoint")
    pins = dict(audit["inputs"])
    merge_pins(pins, pair["inputs"])
    sources = [
        Path(__file__).resolve(),
        Path("scripts/evaluate_strand_graph_panel.py"),
        Path("scripts/run_strand_graph_pilot.py"),
        Path("scripts/summarize_clean_paired_evaluation.py"),
        Path("scripts/evaluate_graph_mixture_cost_mcts.py"),
    ]
    merge_pins(
        pins, {str(p): sha(p) for p in [PAIR_AUDIT, ap, model_path, manifest_path, *sources]}
    )
    check_pins(pins)
    torch.set_num_threads(1)
    torch.manual_seed(manifest["config"]["seed"])
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    net = M.SitePolicyValueNet(**checkpoint["architecture"])
    net.load_state_dict(checkpoint["model"], strict=True)
    objective = MetricEmbeddingObjective(cc_rank_weight=0.1, witness_priority=False)
    objective.load_state_dict(checkpoint["objective"], strict=True)
    if any(
        not torch.isfinite(t).all()
        for t in [*net.state_dict().values(), *objective.state_dict().values()]
    ):
        raise ValueError("nonfinite model/objective")
    records = panel["validation"]
    rows = []
    for i, r in enumerate(records):
        p = r["interval"]
        a, b = p["left"], p["right"]
        rows.append(
            pair_row(
                i,
                "pure-braid-general",
                (tuple(a["word"]), a["strands"]),
                (tuple(b["word"]), b["strands"]),
                p["lower"],
                p["upper"],
                p["witness_semantic_length"],
            )
        )
    metrics = metric_report(net, objective, rows, "cpu")
    by_group = group_report(metrics["predictions"], records)
    result = {
        "status": "completed",
        "arm": args.arm,
        "run": str(run),
        "inputs": pins,
        "checkpoint_sha256": expected,
        "panel_sha256": sha(PANEL),
        "architecture": checkpoint["architecture"],
        "normalization": "none beyond checkpoint architecture",
        "calibration": "frozen checkpoint objective",
        "bounds": "signed-signature plus determinant; ordinary path/cyclic upper certificate",
        "validation": metrics,
        "by_fingerprint": by_group,
        "summary": summaries(metrics["predictions"], records),
        "raw_sizes": audit["raw_sizes"],
        "scope": "General d(K1,K2) on the declared synthetic pure-block family. "
        "No u(K) or search result; exact-zero pairs are coordinate-invariance checks.",
    }
    check_pins(pins)
    atomic(args.out, result)
    print(
        json.dumps({"status": "completed", "arm": args.arm, "summary": result["summary"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
