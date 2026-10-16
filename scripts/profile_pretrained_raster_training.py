"""Protocol90: bounded CPU training-cost probe on audited graph TRAIN pairs.

Disposable optimizer updates are not a scientific training run or promotion.
Each case runs in a fresh process so peak RSS includes that case's imports,
corpus, model, optimizer and activations rather than previous cases' peaks.
"""

import argparse
import gc
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch
from check_qrun_common_witnesses import check_pins, sha, write_new
from evaluate_common_greedy_policies import load_model
from graph_metric_sampling import load_graph_sampler
from pretrained_raster_graph import RasterGraphPolicy
from pretrained_raster_tiles import PretrainedRasterTiles
from run_strand_graph_pilot import metric_loss

from rf_knots.embedding_metric import MetricEmbeddingObjective, witness_priority_weight

INITIAL = Path("artifacts/pretrained-raster-graph-initialization-v1")
GRAPH = Path("artifacts/ordinary-graph-metric-corpus-v1")
CASES = [(0.5, 1, 1), (0.5, 2, 2), (0.5, 4, 4), (0.9, 2, 2), (1.0, 1, 1), (1.0, 4, 1)]


def selected_rows(sampler, quantile, count):
    """Score-blind order by total active crossings; select adjacent actual pairs."""
    if not 0 <= quantile <= 1 or count < 1:
        raise ValueError("invalid quantile/count")
    rows = [row for category in sampler.bins for group in category for row in group]
    rows.sort(key=lambda r: (len(r.left_word) + len(r.right_word), r.spec.index))
    if count > len(rows):
        raise ValueError("not enough unique corpus rows")
    start = min(round(quantile * (len(rows) - 1)), len(rows) - count)
    return rows[start : start + count], len(rows)


def backward_metric(net, objective, rows, microbatch):
    """Exact weighted-mean accumulation for CC-only, rank-free interval loss."""
    if microbatch < 1 or objective.cc_rank_weight != 0 or any(r.metric != "cc" for r in rows):
        raise ValueError("requires CC-only rank-free loss and positive microbatch")
    weights = [
        witness_priority_weight("cc", r.spec.distance_upper, r.witness_semantic_length)
        if objective.witness_priority
        else 1.0
        for r in rows
    ]
    total = sum(weights)
    value = 0.0
    for start in range(0, len(rows), microbatch):
        chunk = rows[start : start + microbatch]
        loss, _ = metric_loss(net, objective, chunk, "cpu")
        scaled = loss * (sum(weights[start : start + microbatch]) / total)
        if not torch.isfinite(scaled):
            raise ValueError("nonfinite metric loss")
        scaled.backward()
        value += float(scaled.detach())
    return value


def provenance():
    report = json.loads((INITIAL / "report.json").read_text())
    if report["status"] != "passed" or report["optimizer_steps"] != 0:
        raise ValueError("require verified untrained initialization")
    if sha(INITIAL / "initialization.pt") != report["checkpoint_sha256"]:
        raise ValueError("initial checkpoint changed")
    audit = json.loads((GRAPH / "partition-replay-audit.json").read_text())
    if audit["status"] != "passed" or not audit["train_ready"]:
        raise ValueError("require audited TRAIN corpus")
    pins = dict(report["inputs"])
    for path, digest in audit["inputs"].items():
        if path in pins and pins[path] != digest:
            raise ValueError("conflicting input lineage")
        pins[path] = digest
    for path in (
        INITIAL / "initialization.pt",
        INITIAL / "report.json",
        GRAPH / "partition-replay-audit.json",
        Path(__file__),
        Path("scripts/graph_metric_sampling.py"),
        Path("scripts/run_strand_graph_pilot.py"),
        Path("research/90-pretrained-raster-training-cost-protocol.md"),
        Path("tests/test_pretrained_raster_training_profile.py"),
    ):
        pins[str(path.resolve())] = sha(path)
    check_pins(pins)
    return pins


def run_case(args):
    torch.set_num_threads(1)
    pins = provenance()
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    sampler = load_graph_sampler(GRAPH)
    quantile, count, microbatch = CASES[args.case]
    rows, population = selected_rows(sampler, quantile, count)
    del sampler
    gc.collect()
    baseline = json.loads(Path("artifacts/qrun-baseline-input-check-v2/report.json").read_text())
    q = load_model("q254", baseline, pins)
    net = RasterGraphPolicy(PretrainedRasterTiles(q.network.representation)).train()
    payload = torch.load(INITIAL / "initialization.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(payload["model"], strict=True)
    objective = MetricEmbeddingObjective()
    params = list(net.parameters()) + list(objective.parameters())
    optimizer = torch.optim.AdamW(params, lr=2.5e-5, weight_decay=0.01)
    measurements = []
    for iteration in range(3):
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss = backward_metric(net, objective, rows, microbatch)
        backward_seconds = time.perf_counter() - started
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        if not any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in net.encoder.tiles.parameters()
        ):
            raise ValueError("pretrained trunk received no nonzero metric gradient")
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in params):
            raise ValueError("nonfinite updated parameters")
        measurements.append(
            {
                "iteration": iteration,
                "warmup": iteration == 0,
                "loss": loss,
                "gradient_norm": float(norm),
                "forward_backward_seconds": backward_seconds,
                "step_seconds": time.perf_counter() - started,
            }
        )
    check_pins(pins)
    result = {
        "status": "passed",
        "case": args.case,
        "quantile_total_crossings": quantile,
        "batch_pairs": count,
        "microbatch_pairs": microbatch,
        "population_excluding_exact_zero": population,
        "rows": [
            {
                "index": r.spec.index,
                "left_length": len(r.left_word),
                "right_length": len(r.right_word),
                "left_strands": r.left_strands,
                "right_strands": r.right_strands,
                "lower": r.spec.distance_lower,
                "upper": r.spec.distance_upper,
            }
            for r in rows
        ],
        "measurements": measurements,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "torch_version": torch.__version__,
        "threads": torch.get_num_threads(),
        "mps_available": torch.backends.mps.is_available(),
        "device": "cpu",
        "source_and_inputs": pins,
        "saved_trained_checkpoint": False,
    }
    write_new(args.out / f"case-{args.case}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "source_and_inputs"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", type=int, choices=range(len(CASES)))
    args = parser.parse_args()
    if args.case is not None:
        run_case(args)
        return
    pins = provenance()
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(
        args.out / "protocol.json",
        {"cases": CASES, "inputs": pins, "scope": "CPU metric-only disposable cost probe"},
    )
    for index in range(len(CASES)):
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--pgx-root",
                str(args.pgx_root),
                "--out",
                str(args.out),
                "--case",
                str(index),
            ],
            check=True,
        )
    reports = [json.loads((args.out / f"case-{i}.json").read_text()) for i in range(len(CASES))]
    check_pins(pins)
    write_new(
        args.out / "report.json",
        {
            "status": "passed",
            "inputs": pins,
            "cases": [
                {
                    "path": str(args.out / f"case-{i}.json"),
                    "sha256": sha(args.out / f"case-{i}.json"),
                    "batch_pairs": r["batch_pairs"],
                    "peak_process_rss_bytes": r["peak_process_rss_bytes"],
                    "mean_step_seconds": sum(x["step_seconds"] for x in r["measurements"][1:]) / 2,
                }
                for i, r in enumerate(reports)
            ],
            "scope": "Not model selection, policy training, MCTS, or GPU throughput evidence",
        },
    )


if __name__ == "__main__":
    main()
