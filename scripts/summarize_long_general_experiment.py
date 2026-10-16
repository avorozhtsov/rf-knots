"""Complete149 comparison: all distance views, five searches and full replays."""

from collections import defaultdict
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import sha
from evaluate_graph_mixture_cost_mcts import build_roots
from evaluate_joint_cost_mcts import committed_rows
from evaluate_raster_log_distances import interval_ranking
from evaluate_strand_graph_panel import constant_baseline
from run_long_general_evaluation import MODELS, SEARCH_MATRIX, jobs, panel_views, read, verify_job
from summarize_clean_paired_evaluation import metric_stats
from summarize_graph_mixture_experiment import certify_costs
from summarize_joint_policy_metric_evaluation import paired_metric, paired_search, search_behavior
from summarize_resource_experiment import validate_metric_report


def require_complete_matrix(state, planned):
    if (
        state["status"] not in ("scored", "completed")
        or state["jobs"] != planned
        or len(planned) != 18
        or len({j["name"] for j in planned}) != 18
        or len({j["output"] for j in planned}) != 18
        or len(state["completed"]) != 18
        or any(
            (j["name"], j["output"]) != (r["name"], r["output"])
            for j, r in zip(planned, state["completed"], strict=True)
        )
    ):
        raise ValueError("require all18 frozen149 evaluation jobs, not partial scores")


def subset(rows):
    return dict(
        rows=len(rows),
        statistics=metric_stats(rows) if rows else None,
        constant1=constant_baseline(rows, 1) if rows else None,
        ranking=interval_ranking(rows),
    )


def validate_metric_result(report, records):
    values = validate_metric_report(report, records)
    lengths, strands = defaultdict(list), defaultdict(list)
    for p, r in zip(report["predictions"], records, strict=True):
        pair = r["interval"]
        length = max(len(pair[k]["word"]) for k in ("left", "right"))
        strand = max(pair[k]["strands"] for k in ("left", "right"))
        if (p["source_index"], p["maximum_length"], p["maximum_strands"]) != (
            r["index"],
            length,
            strand,
        ):
            raise ValueError("metric source index/size metadata differs")
        length_label = (
            "0..48"
            if length <= 48
            else "49..96"
            if length <= 96
            else "97..192"
            if length <= 192
            else "193..384"
            if length <= 384
            else "385+"
        )
        strand_label = (
            "1..12"
            if strand <= 12
            else "13..24"
            if strand <= 24
            else "25..33"
            if strand <= 33
            else "34+"
        )
        lengths[length_label].append(p)
        strands[strand_label].append(p)
    extra = dict(
        certified_positive=subset([p for p in report["predictions"] if p["lower"] >= 1]),
        by_maximum_length={k: subset(v) for k, v in lengths.items()},
        by_maximum_strands={k: subset(v) for k, v in strands.items()},
    )
    if any(report[k] != v for k, v in extra.items()):
        raise ValueError("positive/size-stratum metric arithmetic differs")
    return {**values, **extra}


def compare_metrics(a, b):
    if not a and not b:
        return dict(rows=0, groups=0, mean_record_error_delta=None, mean_group_error_delta=None)
    return {
        **paired_metric(a, b),
        "mean_record_error_delta": sum(
            y["violation"] - x["violation"] for x, y in zip(a, b, strict=True)
        )
        / len(a),
    }


def summarize_verified(directory):
    state = read(directory / "manifest.json")
    run = Path(state["training_run"])
    planned = jobs(directory, run, Path(state["pgx_root"]))
    require_complete_matrix(state, planned)
    pins, reports = dict(state["inputs"]), {}
    for job, stored in zip(planned, state["completed"], strict=True):
        if verify_job(job, directory, run) != stored:
            raise ValueError("completed149 child changed")
        report = read(job["output"])
        reports[job["name"]] = report
        merge_pins(
            pins, report["contract"]["inputs"] if job["kind"] == "search" else report["inputs"]
        )
        merge_pins(pins, {job["output"]: stored["sha256"]})
    check_pins(pins)
    parent = read(run / "manifest.json")
    panels = {kind: panel_views(parent, kind) for kind in ("general", "short-u")}
    metrics, predictions = {}, {}
    for arm, stage in MODELS:
        label = f"{arm}-{stage}"
        metrics[label] = {}
        for kind, (_, views) in panels.items():
            result = reports[f"metric-{label}-{kind}"]["results"]
            metrics[label][kind] = {}
            for view, records in views.items():
                metrics[label][kind][view] = validate_metric_result(result[view], records)
                predictions[label, kind, view] = result[view]["predictions"]
    metric_comparisons = {}
    contrasts = (
        ("control-initial", "control-final"),
        ("control-initial", "long-final"),
        ("control-final", "long-final"),
    )
    for kind, (_, views) in panels.items():
        for view in views:
            comparisons = {}
            for baseline, candidate in contrasts:
                a, b = predictions[baseline, kind, view], predictions[candidate, kind, view]
                comparisons[f"{candidate}-vs-{baseline}"] = dict(
                    all=compare_metrics(a, b),
                    certified_positive=compare_metrics(
                        [p for p in a if p["lower"] >= 1], [p for p in b if p["lower"] >= 1]
                    ),
                )
            metric_comparisons[f"{kind}/{view}"] = comparisons
    short_panel = read(panels["short-u"][0])
    roots = build_roots(short_panel)
    costs, compute, audits = {}, {}, {}
    for row in SEARCH_MATRIX:
        label = "-".join(row[k] for k in ("arm", "stage", "prior"))
        folder = directory / "mcts" / label
        manifest = reports[f"search-{label}"]
        if read(folder / "roots.json") != roots:
            raise ValueError("149 search arms have different roots/resources")
        costs[label] = committed_rows(manifest, folder)
        compute[label] = dict(
            counts=manifest["counts"], seconds=manifest["seconds_committed_batches"]
        )
        audits[label] = reports[f"search-audit-{label}"]
    certified = certify_costs(short_panel, roots, costs)
    for label, audit in audits.items():
        if audit["certified_costs"] != {
            **certified,
            "solutions": {label: certified["solutions"][label]},
        }:
            raise ValueError("replay certificate differs from149 comparison")
    comparisons = [
        ("control-initial-distance", f"{arm}-final-distance") for arm in ("control", "long")
    ]
    comparisons += [("control-final-distance", "long-final-distance")]
    comparisons += [
        (f"{arm}-final-uniform", f"{arm}-final-distance") for arm in ("control", "long")
    ]
    search_comparisons = {
        f"{candidate}-vs-{baseline}": {
            str(w): paired_search(roots, costs[baseline], costs[candidate], w) for w in (10, 1000)
        }
        for baseline, candidate in comparisons
    }
    check_pins(pins)
    return dict(
        schema="rf-knots-long-general-comparison-v1",
        status="completed",
        inputs=pins,
        controller_contract_sha256=state["contract_sha256"],
        checkpoints=reports["training-pair-audit"]["arms"],
        panels={
            kind: dict(
                path=str(path), sha256=sha(path), views={v: len(r) for v, r in views.items()}
            )
            for kind, (path, views) in panels.items()
        },
        metric=metrics,
        paired_metric=metric_comparisons,
        search_comparisons=search_comparisons,
        search_behavior={label: search_behavior(roots, rows) for label, rows in costs.items()},
        compute=compute,
        certified_costs=certified,
        scope=dict(
            promotion="not_decided",
            intervention="Same initialization, architecture, raw features and policy data; "
            "eight historical graph slots replaced by long-to-long metric pairs. "
            "This changes length, family and class mixture, not length alone. "
            "Equal updates are not equal hardware work.",
            embedding="Uniform versus distance TREE prior retains global embeddings "
            "in the rollout policy.",
            split="Frozen synthetic RF-lineage query groups; Q254 exposure remains unknown. "
            "Invariant groups and lifted views are not independent identified knot types.",
            failure="Unsuccessful bounded searches are censored, "
            "not impossible-state or bad-action labels.",
            cost="New/lost successes separate from common-success costs. Floor equality alone "
            "certifies representation-cost optimality; not all-presentation optimality.",
            scale="Matched static training, not million-game online RL. This synthetic experiment "
            "does not establish public bound improvement or arbitrary-size solution quality.",
        ),
    )
