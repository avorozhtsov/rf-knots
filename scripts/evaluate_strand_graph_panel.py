"""Score an evaluation-only replay panel using a pinned joint/policy checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from run_strand_graph_pilot import metric_report, pair_row

from rf_knots import mcts_loop as M
from rf_knots.embedding_checkpoint_io import load_initial_checkpoint
from rf_knots.embedding_evaluation import verify_evaluation_pair
from rf_knots.embedding_metric import MetricEmbeddingObjective
from rf_knots.embedding_normalization import normalize_global_view
from rf_knots.embedding_replay_corpus import verify_record


def certified_lower(pair, left_fingerprint, right_fingerprint, *, determinant=False):
    lower = pair["lower"]
    if determinant:
        lower = max(lower, int(left_fingerprint[0] != right_fingerprint[0]))
    if lower > pair["upper"]:
        raise ValueError("certified lower bound contradicts replayed upper bound")
    return lower


def constant_baseline(predictions, value):
    exact = [r for r in predictions if r["lower"] == r["upper"]]
    return {
        "count": len(predictions),
        "constant": value,
        "mean_interval_violation": sum(
            max(r["lower"] - value, value - r["upper"], 0.0) for r in predictions
        )
        / len(predictions),
        "exact_groups": {
            str(target): {
                "count": sum(r["lower"] == target for r in exact),
                "mae": abs(value - target),
            }
            for target in sorted({r["lower"] for r in exact})
        },
    }


def group_report(predictions, records):
    groups = {}
    for i, (prediction, record) in enumerate(zip(predictions, records, strict=True)):
        fp = record["left_fingerprint"]
        key = f"{fp[0]}:{abs(fp[1])}"
        prediction.update(panel_index=i, left_fingerprint=record["left_fingerprint"],
                          right_fingerprint=record["right_fingerprint"], fingerprint_group=key)
        groups.setdefault(key, []).append(prediction)
    result = {}
    for key, rows in groups.items():
        exact = {}
        for row in rows:
            if row["lower"] == row["upper"]:
                exact.setdefault(str(row["lower"]), []).append(row["violation"])
        result[key] = {
            "count": len(rows), "mean_interval_violation": sum(r["violation"] for r in rows)/len(rows),
            "exact_groups": {k: {"count": len(v), "mae": sum(v)/len(v)} for k,v in exact.items()},
            "constant_baselines": {str(v): constant_baseline(rows, v) for v in (0,1,2)},
        }
    return {"groups": result, "group_count": len(result),
            "mean_group_interval_violation": sum(r["mean_interval_violation"] for r in result.values())/len(result)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial", action="store_true")
    parser.add_argument("--bounds", choices=("original", "determinant"), default="original")
    parser.add_argument(
        "--normalize", action="store_true", help="diagnostic global-view preprocessing"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output exists; refusing to overwrite an evaluation")
    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise SystemExit("require a completed run for immutable evaluation")
    sha = hashlib.sha256((args.run / "checkpoint.pt").read_bytes()).hexdigest()
    if sha != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA mismatch")
    checkpoint = torch.load(args.run / "checkpoint.pt", map_location="cpu", weights_only=False)
    torch.set_num_threads(1)
    torch.manual_seed(manifest["config"]["seed"])
    net = M.SitePolicyValueNet(**checkpoint["architecture"])
    objective = MetricEmbeddingObjective(cc_rank_weight=0.1, witness_priority=False)
    objective.calibrate_cc_scale(reference_cosine=0.4)
    if not args.initial:
        net.load_state_dict(checkpoint["model"], strict=True)
        objective.load_state_dict(checkpoint["objective"], strict=True)
    else:
        initial = load_initial_checkpoint(args.run, manifest, checkpoint["architecture"])
        if initial is not None:
            net.load_state_dict(initial["model"], strict=True)
            objective.load_state_dict(initial["objective"], strict=True)
    panel = json.loads(args.panel.read_text())
    if panel.get("complete") is False:
        raise ValueError("refusing to evaluate a partial panel")
    report = {
        "status": "completed",
        "run": str(args.run),
        "initial": args.initial,
        "checkpoint_sha256": sha,
        "initial_checkpoint_sha256": manifest.get("initial_checkpoint_sha256") if args.initial else None,
        "initial_state_source": (
            "saved_tensors" if manifest.get("initial_checkpoint_sha256") else "legacy_seed_reconstruction"
        ) if args.initial else None,
        "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
        "partition": panel.get("partition", "validation"),
        "bounds": args.bounds,
        "constant_baselines": {},
        "architecture": checkpoint["architecture"],
        "normalization": "replayed REDUCE/DESTABILIZE" if args.normalize else "none",
        "code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(__file__), Path("scripts/run_strand_graph_pilot.py"),
            Path("src/rf_knots/embedding_checkpoint_io.py"), Path("src/rf_knots/embedding_evaluation.py"),
            Path("src/rf_knots/embedding_replay_corpus.py"), Path("src/rf_knots/embedding_metric.py"),
            Path("src/rf_knots/strand_graph_encoder.py"), Path("src/rf_knots/mcts_loop.py"),
            Path("src/rf_knots/site_head.py"), Path("src/rf_knots/embedding_encoder.py"),
            Path("src/rf_knots/embedding_normalization.py")]},
    }
    for split in ("validation", "size_shift"):
        rows = []
        for i, record in enumerate(panel[split]):
            if panel.get("schema") == "rf-knots-gordian-pair-panel-v1":
                verify_evaluation_pair(record)
            else:
                verify_record(record)
            p = record["interval"]
            left = (tuple(p["left"]["word"]), p["left"]["strands"])
            right = (tuple(p["right"]["word"]), p["right"]["strands"])
            if args.normalize:
                a, b = normalize_global_view(*left), normalize_global_view(*right)
                left, right = (a.word, a.strands), (b.word, b.strands)
            rows.append(
                pair_row(
                    i,
                    split,
                    left,
                    right,
                    certified_lower(
                        p,
                        record["left_fingerprint"],
                        record["right_fingerprint"],
                        determinant=args.bounds == "determinant",
                    ),
                    p["upper"],
                    p["witness_semantic_length"],
                )
            )
        report[split] = metric_report(net, objective, rows, "cpu")
        report[split]["by_fingerprint"] = group_report(report[split]["predictions"], panel[split])
        report["constant_baselines"][split] = {
            str(value): constant_baseline(report[split]["predictions"], value)
            for value in (0, 1, 2)
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != value
           for p, value in report["code_sha256"].items()):
        raise ValueError("evaluation sources changed during scoring")
    temporary = args.output.with_suffix(".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                split: {k: v for k, v in report[split].items() if k != "predictions"}
                for split in ("validation", "size_shift")
            }
        )
    )


if __name__ == "__main__":
    main()
