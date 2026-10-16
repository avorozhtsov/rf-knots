"""Protocol145: fixed actual-TRAIN draw fit, no optimizer or TEST scoring."""

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from audit_joint_policy_metric_panel import check_pins, merge_pins
from audit_raster_train_draws import source_slices
from check_qrun_common_witnesses import sha, write_new
from graph_metric_sampling import load_graph_sampler
from mcts_joint_training import load_metric_samplers, load_policy_targets
from probe_raster_cache_full_network import state_digest
from raster_log_training import CORPUS, GRAPH, SUPERVISION, digest, draw_batch, relative_due
from resource_evaluation_inputs import load_model
from run_strand_graph_pilot import metric_loss
from train_raster_resource_pair import CONFIG

from rf_knots.embedding_metric import witness_priority_weight

UPDATES = tuple(range(0, 2048, 128))
MODELS = (("units", "initial"), ("units", "final"), ("raw", "final"))
PROTOCOL = Path("research/145-resource-train-fit-diagnostic-protocol.md")


def interval_error(prediction, lower, upper):
    if not all(math.isfinite(x) for x in (prediction, lower, upper)) or not 0 <= lower <= upper:
        raise ValueError("invalid distance/interval")
    return max(lower - prediction, prediction - upper, 0.0)


def make_batch(step, sources):
    weighted = [
        (
            name,
            row,
            witness_priority_weight("cc", row.spec.distance_upper, row.witness_semantic_length),
        )
        for name, rows in sources.items()
        for row in rows
    ]
    total = sum(weight for _, _, weight in weighted)
    if not weighted or total <= 0:
        raise ValueError("empty training metric batch")
    result = []
    for offset, (source, row, weight) in enumerate(weighted):
        if row.metric != "cc":
            raise ValueError("require CC TRAIN pairs")
        left = (row.left_strands, row.left_word)
        right = (row.right_strands, row.right_word)
        result.append(
            dict(
                update_zero_based=step,
                offset=offset,
                source=source,
                pair=asdict(row),
                pair_sha256=digest(sorted((left, right))),
                lower=row.spec.distance_lower,
                upper=row.spec.distance_upper,
                max_length=max(len(row.left_word), len(row.right_word)),
                max_strands=max(row.left_strands, row.right_strands),
                batch_weight=weight / total,
            )
        )
    return result


def summarize(rows):
    if not rows:
        raise ValueError("empty predictions")
    return dict(
        draws=len(rows),
        unique_raw_pairs=len({r["pair_sha256"] for r in rows}),
        mean_interval_violation=sum(r["violation"] for r in rows) / len(rows),
        mean_native_loss=sum(r["native_loss"] for r in rows) / len(rows),
        mean_prediction=sum(r["prediction"] for r in rows) / len(rows),
        constants={
            str(c): sum(interval_error(c, r["lower"], r["upper"]) for r in rows) / len(rows)
            for c in (0, 1, 2)
        },
        lengths=[min(r["max_length"] for r in rows), max(r["max_length"] for r in rows)],
        strands=[min(r["max_strands"] for r in rows), max(r["max_strands"] for r in rows)],
        clamp_count=sum(r["native_prediction"] != r["prediction"] for r in rows),
        max_clamp=max(abs(r["native_prediction"] - r["prediction"]) for r in rows),
    )


@torch.no_grad()
def score(net, objective, selected, materialized):
    if objective.cc_geometry != "chord" or objective.cc_rank_weight != 0:
        raise ValueError("diagnostic requires the unchanged135 distance objective")
    before = state_digest(net, objective)
    scale = float(objective.scale("cc"))
    rows = []
    for metadata, row in zip(selected, materialized, strict=True):
        loss, prediction = metric_loss(net, objective, [row], "cpu")
        native, value = float(prediction[0]), float(loss)
        error = interval_error(native, metadata["lower"], metadata["upper"])
        if not math.isfinite(value) or not math.isclose(
            value, error**2, rel_tol=1e-5, abs_tol=1e-6
        ):
            raise ValueError("native single-row loss differs from squared interval violation")
        bounded = min(max(native, 0.0), 2 * scale)
        rows.append(
            dict(
                **metadata,
                native_prediction=native,
                prediction=bounded,
                native_loss=value,
                violation=interval_error(bounded, metadata["lower"], metadata["upper"]),
            )
        )
    if state_digest(net, objective) != before:
        raise ValueError("diagnostic changed model/objective state")
    batches = sorted({r["update_zero_based"] for r in rows})
    if batches != list(UPDATES) or len(rows) != 512:
        raise ValueError("diagnostic batch coverage differs")
    for step in batches:
        group = [r for r in rows if r["update_zero_based"] == step]
        if len(group) != 32 or not math.isclose(sum(r["batch_weight"] for r in group), 1.0):
            raise ValueError("diagnostic batch weights differ")
    sources = {}
    for name in sorted({r["source"] for r in rows}):
        group = [r for r in rows if r["source"] == name]
        sources[name] = dict(
            **summarize(group),
            classes={
                f"{lo:g}..{hi:g}": summarize(
                    [r for r in group if (r["lower"], r["upper"]) == (lo, hi)]
                )
                for lo, hi in sorted({(r["lower"], r["upper"]) for r in group})
            },
            mean_batch_weight_mass=sum(r["batch_weight"] for r in group) / len(batches),
            weighted_loss_contribution=sum(r["native_loss"] * r["batch_weight"] for r in group)
            / len(batches),
        )
    return dict(
        rows=rows,
        overall=summarize(rows),
        sources=sources,
        classes={
            f"{lo:g}..{hi:g}": summarize([r for r in rows if (r["lower"], r["upper"]) == (lo, hi)])
            for lo, hi in sorted({(r["lower"], r["upper"]) for r in rows})
        },
        mean_weighted_batch_metric_loss=sum(r["native_loss"] * r["batch_weight"] for r in rows)
        / len(batches),
        cc_scale=scale,
        weights_unchanged=True,
        state_sha256=before,
        encoder_calls=2 * len(rows),
        optimizer_steps=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    audit = json.loads(args.pair_audit.read_text())
    if audit["status"] != "passed" or set(audit["arms"]) != {"units", "raw"}:
        raise ValueError("require completed paired training audit")
    pins = dict(audit["inputs"])
    for path in (
        args.pair_audit,
        Path(__file__).resolve(),
        PROTOCOL,
        Path("scripts/audit_raster_train_draws.py"),
        Path("tests/test_resource_train_fit.py"),
    ):
        merge_pins(pins, {str(path): sha(path)})
    check_pins(pins)
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(
        args.out / "inputs.json", dict(inputs=pins, updates=UPDATES, models=MODELS, pid=os.getpid())
    )
    print("reconstructing actual TRAIN draws; no model scoring yet", flush=True)
    positives, groups = load_policy_targets(SUPERVISION)
    references = [(g, torch.zeros(len(g))) for g in groups]
    original, suffix = load_metric_samplers(CORPUS, SUPERVISION)
    graph = load_graph_sampler(GRAPH)
    rng, trace, counts = random.Random(CONFIG["seed"]), digest([]), [0] * len(groups)
    selected, materialized, hashes = [], [], []
    for step in range(CONFIG["steps"]):
        rows, _, _, hashed, group = draw_batch(rng, original, suffix, graph, positives, references)
        trace = digest([trace, hashed])
        hashes.append(hashed)
        if relative_due(step):
            counts[group] += 1
        if step in UPDATES:
            selected.extend(make_batch(step, source_slices(rows)))
            materialized.extend(rows)
    for arm in audit["arms"].values():
        if arm["steps"] != 2048 or arm["draw_digest"] != trace or arm["relative_counts"] != counts:
            raise ValueError("reconstructed sampling differs from completed training")
    if len(selected) != 512 or Counter(r["source"] for r in selected) != {
        "original": 112,
        "suffix": 128,
        "graph": 256,
        "sentinel": 16,
    }:
        raise ValueError("fixed TRAIN selection coverage differs")
    check_pins(pins)
    write_new(
        args.out / "selection.json", dict(rows=selected, draw_digest=trace, draw_hashes=hashes)
    )
    merge_pins(pins, {str(args.out / "selection.json"): sha(args.out / "selection.json")})
    completed = []
    for arm, stage in MODELS:
        net, objective, added, checkpoint = load_model(args.run, args.pair_audit, arm, stage)
        merge_pins(pins, added)
        report = score(net, objective, selected, materialized)
        report.update(
            status="completed",
            arm=arm,
            stage=stage,
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha(checkpoint),
            inputs=dict(pins),
        )
        check_pins(pins)
        path = args.out / f"{arm}-{stage}.json"
        write_new(path, report)
        completed.append(dict(file=str(path), sha256=sha(path)))
        print(json.dumps(dict(arm=arm, stage=stage, **report["overall"])), flush=True)
    check_pins(pins)
    write_new(
        args.out / "report.json",
        dict(
            status="completed",
            inputs=pins,
            reports=completed,
            draw_digest=trace,
            selected_draws=512,
            models=3,
            encoder_calls=3072,
            optimizer_steps=0,
            scope="Post-TEST exploratory TRAIN fit; repeated draws, not held-out generalization. "
            "No TEST scoring/tuning or causal diagnosis. "
            "Concurrent one-thread139 search unchanged.",
        ),
    )


if __name__ == "__main__":
    main()
