"""Independent CPU evaluation of149 endpoints on frozen147/150 TEST views."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, sha
from evaluate_raster_log_distances import interval_ranking, score_records
from evaluate_strand_graph_panel import constant_baseline
from long_general_evaluation_inputs import load_inputs, load_model, load_panel
from probe_raster_cache_full_network import state_digest
from summarize_clean_paired_evaluation import metric_stats


def describe(rows):
    if not rows:
        return dict(rows=0, statistics=None, constant1=None, ranking=interval_ranking([]))
    return dict(
        rows=len(rows),
        statistics=metric_stats(rows),
        constant1=constant_baseline(rows, 1),
        ranking=interval_ranking(rows),
    )


def enrich(result, records):
    if len(result["predictions"]) != len(records):
        raise ValueError("prediction/TEST record coverage differs")
    lengths, strands = defaultdict(list), defaultdict(list)
    for i, (p, r) in enumerate(zip(result["predictions"], records, strict=True)):
        if p["panel_index"] != i:
            raise ValueError("prediction order differs")
        endpoints = [r["interval"][k] for k in ("left", "right")]
        length, strand = (
            max(len(s["word"]) for s in endpoints),
            max(s["strands"] for s in endpoints),
        )
        p.update(source_index=r["index"], maximum_length=length, maximum_strands=strand)
        lengths[
            next(
                label
                for upper, label in (
                    (48, "0..48"),
                    (96, "49..96"),
                    (192, "97..192"),
                    (384, "193..384"),
                    (float("inf"), "385+"),
                )
                if length <= upper
            )
        ].append(p)
        strands[
            next(
                label
                for upper, label in (
                    (12, "1..12"),
                    (24, "13..24"),
                    (33, "25..33"),
                    (float("inf"), "34+"),
                )
                if strand <= upper
            )
        ].append(p)
    result.update(
        certified_positive=describe([p for p in result["predictions"] if p["lower"] >= 1]),
        by_maximum_length={k: describe(v) for k, v in sorted(lengths.items())},
        by_maximum_strands={k: describe(v) for k, v in sorted(strands.items())},
        scope="Positive subset requires certified lower>=1. Disjoint-interval comparisons "
        "are dependent; fingerprint groups are not identified knot types.",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "long"), required=True)
    parser.add_argument("--stage", choices=("initial", "final"), required=True)
    parser.add_argument("--panel", choices=("general", "short-u"), required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".partial").exists():
        raise ValueError("new metric output required")
    if args.stage == "initial" and args.arm != "control":
        raise ValueError("frozen matrix uses only the verified shared control initialization")
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    parent, audit, pins = load_inputs(args.run, args.pair_audit)
    views, path = load_panel(args.panel, parent, pins)
    for source in (
        Path(__file__).resolve(),
        Path("scripts/long_general_evaluation_inputs.py").resolve(),
        Path("scripts/evaluate_raster_log_distances.py"),
        Path("scripts/evaluate_strand_graph_panel.py"),
        Path("scripts/summarize_clean_paired_evaluation.py"),
        Path("scripts/probe_raster_cache_full_network.py"),
        Path("tests/test_long_general_evaluation.py"),
    ):
        merge_pins(pins, {str(source): sha(source)})
    check_pins(pins)
    net, objective, checkpoint, shared = load_model(
        args.run, parent, audit, pins, args.arm, args.stage
    )
    before, start = state_digest(net, objective), time.monotonic()
    results = {
        view: enrich(score_records(net, objective, rows), rows) for view, rows in views.items()
    }
    if state_digest(net, objective) != before:
        raise ValueError("independent evaluation changed weights")
    check_pins(pins)
    report = dict(
        schema="rf-knots-long-general-distances-v1",
        status="completed",
        arm=args.arm,
        stage=args.stage,
        panel=str(path),
        panel_sha256=sha(path),
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),
        training_contract_sha256=parent["contract_sha256"],
        inputs=pins,
        results=results,
        shared_initial_identity=shared,
        state_sha256=before,
        weights_unchanged=True,
        device="cpu",
        torch_version=torch.__version__,
        threads=1,
        microbatch=1,
        seconds=time.monotonic() - start,
        optimizer_steps=0,
        observed_training_sizes={a: row["size_counters"] for a, row in audit["arms"].items()},
        scope="Complete frozen TEST views, no selection or optimizer. Size counters describe "
        "actual149 draws, not all historical pool support or Q254 pretraining. "
        "Distance results do not establish MCTS utility or public bound improvements.",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic(args.out, report)
    print(
        json.dumps(
            dict(
                status="completed",
                arm=args.arm,
                stage=args.stage,
                counts={k: len(v) for k, v in views.items()},
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
