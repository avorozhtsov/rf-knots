"""Protocol97 final-pair-gated initial/final metrics on frozen70/74 panels."""

import argparse
import json
import sys
from pathlib import Path

import torch
from audit_joint_policy_metric_panel import check_pins, merge_pins
from check_qrun_common_witnesses import sha, write_new
from evaluate_strand_graph_panel import certified_lower, constant_baseline, group_report
from raster_log_training import matched_initializations
from summarize_clean_paired_evaluation import metric_stats

from rf_knots.embedding_metric import MetricEmbeddingObjective

PANELS = {
    "short-u": Path("artifacts/graph-mixture-fresh-short-u-test-v1/panel.json"),
    "general": Path("artifacts/pure-braid-short-distance-test-v1/panel.json"),
}


def interval_ranking(predictions):
    """Only disjoint certified intervals define a strict target ordering."""
    correct = ties = total = 0
    for a in predictions:
        for b in predictions:
            if a["upper"] < b["lower"]:
                total += 1
                correct += a["prediction"] < b["prediction"]
                ties += a["prediction"] == b["prediction"]
    return {
        "comparable_pairs": total,
        "correct": correct,
        "ties": ties,
        "accuracy_half_credit_ties": (correct + 0.5 * ties) / total if total else None,
        "scope": "Descriptive pair comparisons, not independent statistical samples",
    }


@torch.no_grad()
def score_records(net, objective, records):
    if not records:
        raise ValueError("empty panel split")
    was_training = net.training
    net.eval()
    predictions = []
    try:
        for record in records:
            pair = record["interval"]
            vectors = []
            # Fixed microbatch1: no padding dependence or large activation batch.
            for endpoint in (pair["left"], pair["right"]):
                word, n = endpoint["word"], endpoint["strands"]
                vector = net.encoder(
                    torch.tensor([word or [0]], dtype=torch.long),
                    torch.tensor([len(word)]),
                    torch.tensor([n]),
                )["knot"][0]
                if not torch.isfinite(vector).all():
                    raise ValueError("nonfinite embedding")
                vectors.append(vector)
            cosine = (vectors[0] * vectors[1]).sum().clamp(-1, 1)
            distance = objective.scale("cc") * objective.cc_distance(cosine)
            if not torch.isfinite(distance) or distance < 0:
                raise ValueError("invalid distance")
            prediction = float(distance)
            lower = certified_lower(
                pair, record["left_fingerprint"], record["right_fingerprint"], determinant=True
            )
            upper = pair["upper"]
            predictions.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "prediction": prediction,
                    "violation": max(lower - prediction, prediction - upper, 0.0),
                }
            )
    finally:
        net.train(was_training)
    grouped = group_report(predictions, records)
    return {
        "predictions": predictions,
        "statistics": metric_stats(predictions),
        "by_fingerprint": grouped,
        "ranking": interval_ranking(predictions),
        "constant_baselines": {str(c): constant_baseline(predictions, c) for c in (0, 1, 2)},
        "encoded_endpoints": 2 * len(records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--arm", choices=("q254", "random"), required=True)
    parser.add_argument("--state", choices=("initial", "final"), required=True)
    parser.add_argument("--panel", choices=tuple(PANELS), required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    audit = json.loads(args.pair_audit.read_text())
    if audit["status"] != "passed" or set(audit["arms"]) != {"q254", "random"}:
        raise ValueError("require final paired training audit")
    pins = dict(audit["inputs"])
    path = args.run / args.arm / ("initial.pt" if args.state == "initial" else "checkpoint.pt")
    key = "initial_sha256" if args.state == "initial" else "checkpoint_sha256"
    if sha(path) != audit["arms"][args.arm][key] or pins.get(str(path)) != sha(path):
        raise ValueError("checkpoint differs from audited pair")
    panel_path = PANELS[args.panel]
    panel_audit_path = panel_path.parent / "independence-audit.json"
    panel_audit = json.loads(panel_audit_path.read_text())
    panel = json.loads(panel_path.read_text())
    if (
        panel_audit["status"] != "passed"
        or not panel_audit["evaluation_ready"]
        or panel_audit["inputs"].get(str(panel_path)) != sha(panel_path)
        or not panel["complete"]
        or not panel["construction_complete"]
    ):
        raise ValueError("require complete audited test panel")
    merge_pins(pins, panel_audit["inputs"])
    sources = [
        Path(__file__),
        Path("scripts/evaluate_strand_graph_panel.py"),
        Path("scripts/summarize_clean_paired_evaluation.py"),
        Path("tests/test_raster_log_distances.py"),
    ]
    merge_pins(pins, {str(p): sha(p) for p in [args.pair_audit, panel_audit_path, *sources]})
    check_pins(pins)
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    net, _, architecture, _ = matched_initializations(pins)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["architecture"] != architecture:
        raise ValueError("checkpoint architecture differs")
    net.load_state_dict(checkpoint["model"], strict=True)
    objective = MetricEmbeddingObjective()
    objective.load_state_dict(checkpoint["objective"], strict=True)
    splits = ("validation", "size_shift") if args.panel == "short-u" else ("validation",)
    expected_counts = (
        {"validation": 64, "size_shift": 64} if args.panel == "short-u" else {"validation": 126}
    )
    if any(len(panel[s]) != expected_counts[s] for s in splits):
        raise ValueError("unexpected panel coverage")
    results = {s: score_records(net, objective, panel[s]) for s in splits}
    check_pins(pins)
    write_new(
        args.out,
        {
            "status": "completed",
            "arm": args.arm,
            "state": args.state,
            "panel": args.panel,
            "inputs": pins,
            "checkpoint_sha256": sha(path),
            "panel_sha256": sha(panel_path),
            "normalization": "raw diagrams; cosine clamped to [-1,1] for floating-point safety",
            "calibration": "saved objective, no TEST fitting",
            "results": results,
            "scope": "Shared frozen70/74 panels; Q254 pretraining exposure not fully audited. "
            "Metric evidence only, not search success or established invariance.",
        },
    )
    print(
        json.dumps(
            {"status": "completed", "arm": args.arm, "state": args.state, "panel": args.panel}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
