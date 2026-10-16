"""Complete139 paired comparison; no automatic scientific promotion."""

from pathlib import Path

from audit_graph_mixture_short_u_panel import stratum
from audit_joint_policy_metric_panel import check_pins, merge_pins
from check_qrun_common_witnesses import sha
from evaluate_graph_mixture_cost_mcts import build_roots
from evaluate_joint_cost_mcts import committed_rows
from evaluate_strand_graph_panel import constant_baseline, group_report
from run_resource_evaluation import jobs, panel_path, read, search_matrix, verify_job
from summarize_graph_mixture_experiment import certify_costs
from summarize_joint_policy_metric_evaluation import (
    metric_with_constants,
    paired_metric,
    paired_search,
    search_behavior,
)
from summarize_raster_log_experiment import validate_predictions


def require_complete_matrix(state, planned):
    if (
        state["status"] not in ("scored", "completed")
        or state["jobs"] != planned
        or len(planned) != 24
        or len(state["completed"]) != 24
        or any(
            (j["name"], j["output"]) != (r["name"], r["output"])
            for j, r in zip(planned, state["completed"], strict=True)
        )
    ):
        raise ValueError("require the complete24-job matrix, not partial scores")


def compare_metrics(baseline, candidate):
    result = paired_metric(baseline, candidate)
    result["mean_record_error_delta"] = sum(
        b["violation"] - a["violation"] for a, b in zip(baseline, candidate, strict=True)
    ) / len(baseline)
    return result


def metric_strata(predictions, records):
    groups = {}
    for prediction, record in zip(predictions, records, strict=True):
        pair = record["interval"]
        label = stratum(
            max(len(pair[s]["word"]) for s in ("left", "right")),
            max(pair[s]["strands"] for s in ("left", "right")),
        )
        groups.setdefault(label, []).append(prediction)
    return {label: metric_with_constants(rows) for label, rows in groups.items()}


def validate_metric_report(report, records):
    rows = report["predictions"]
    values = validate_predictions(rows, records)
    statistics = {k: v for k, v in values.items() if k not in ("constants", "ranking")}
    if report["statistics"] != statistics:
        raise ValueError("stored distance statistics differ")
    if report["ranking"] != values["ranking"]:
        raise ValueError("stored ranking differs")
    # group_report annotates rows; use copies after identity verification so the
    # consumer cannot repair malformed evidence as a side effect of checking it.
    grouped = group_report([dict(row) for row in rows], records)
    constants = {str(c): constant_baseline(rows, c) for c in (0, 1, 2)}
    if report["by_fingerprint"] != grouped or report["constant_baselines"] != constants:
        raise ValueError("stored grouping or constant controls differ")
    if report["encoded_endpoints"] != 2 * len(records):
        raise ValueError("metric endpoint coverage differs")
    return values


def summarize_verified(directory):
    state = read(directory / "manifest.json")
    planned = jobs(directory, Path(state["training_run"]), Path(state["pgx_root"]))
    require_complete_matrix(state, planned)
    pins, reports = dict(state["inputs"]), {}
    for job, stored in zip(planned, state["completed"], strict=True):
        if verify_job(job, directory) != stored:
            raise ValueError("committed child changed")
        report = read(job["output"])
        reports[job["name"]] = report
        merge_pins(
            pins, report["contract"]["inputs"] if job["kind"] == "search" else report["inputs"]
        )
        merge_pins(pins, {job["output"]: stored["sha256"]})
    check_pins(pins)
    panels = {kind: read(panel_path(kind)) for kind in ("short-u", "general")}
    metrics, predictions = {}, {}
    for arm in ("units", "raw"):
        metrics[arm] = {}
        for stage in ("initial", "final"):
            metrics[arm][stage] = {}
            for kind, panel in panels.items():
                result = reports[f"metric-{arm}-{stage}-{kind}"]["results"]
                metrics[arm][stage][kind] = {}
                for view, report in result.items():
                    rows = report["predictions"]
                    values = validate_metric_report(report, panel[view])
                    values["endpoint_size_strata"] = metric_strata(rows, panel[view])
                    predictions[arm, stage, kind, view] = rows
                    metrics[arm][stage][kind][view] = values
    metric_comparisons = {}
    for kind in panels:
        views = ("validation", "size_shift") if kind == "short-u" else ("validation",)
        for view in views:
            if (
                predictions["raw", "initial", kind, view]
                != predictions["units", "initial", kind, view]
            ):
                raise ValueError("identical initial encoders produced different distance records")
            comparisons = {
                f"{a}-final-vs-initial": compare_metrics(
                    predictions[a, "initial", kind, view], predictions[a, "final", kind, view]
                )
                for a in ("units", "raw")
            }
            comparisons["units-vs-raw-final"] = compare_metrics(
                predictions["raw", "final", kind, view], predictions["units", "final", kind, view]
            )
            metric_comparisons[f"{kind}/{view}"] = comparisons
    roots = build_roots(panels["short-u"])
    costs, compute, audits = {}, {}, {}
    for arm, stage, prior in search_matrix():
        label = f"{arm}-{stage}-{prior}"
        folder = directory / "mcts" / label
        manifest = reports[f"search-{label}"]
        if read(folder / "roots.json") != roots:
            raise ValueError("search comparisons use different roots/resources")
        costs[label] = committed_rows(manifest, folder)
        compute[label] = dict(
            counts=manifest["counts"], seconds=manifest["seconds_committed_batches"]
        )
        audits[label] = reports[f"search-audit-{label}"]
    # References and successful solutions are re-certified; the full censored
    # paths are bound to the separate completed replay audits, not run twice.
    certified = certify_costs(panels["short-u"], roots, costs)
    for label, audit in audits.items():
        if audit["certified_costs"] != {
            **certified,
            "solutions": {label: certified["solutions"][label]},
        }:
            raise ValueError("search audit certificate differs from final comparison")
    pairs = [(f"{a}-initial-distance", f"{a}-final-distance") for a in ("units", "raw")]
    pairs += [(f"raw-{s}-distance", f"units-{s}-distance") for s in ("initial", "final")]
    pairs += [(f"{a}-final-uniform", f"{a}-final-distance") for a in ("units", "raw")]
    comparisons = {
        f"{candidate}-vs-{baseline}": {
            str(w): paired_search(roots, costs[baseline], costs[candidate], w) for w in (10, 1000)
        }
        for baseline, candidate in pairs
    }
    check_pins(pins)
    return dict(
        status="completed",
        inputs=pins,
        controller_contract_sha256=state["contract_sha256"],
        checkpoints=reports["training-pair-audit"]["arms"],
        panels={
            kind: dict(sha256=sha(panel_path(kind)), rows=len(panel["validation"]))
            for kind, panel in panels.items()
        },
        metric=metrics,
        paired_metric=metric_comparisons,
        search_comparisons=comparisons,
        search_behavior={label: search_behavior(roots, rows) for label, rows in costs.items()},
        compute=compute,
        certified_costs=certified,
        scope=dict(
            promotion="not_decided",
            resource_contrast="Same initial learned tensors, different resource feature maps; "
            "matched training draws and budgets. Initial policies differ.",
            embedding_contrast="Uniform versus learned-distance TREE prior at fixed final policy. "
            "Global embeddings remain in policy inputs in both; total channel "
            "causality is not established.",
            split="Fresh synthetic RF-lineage query-separated panels; Q254 full pretraining "
            "exposure incompletely audited, reference endpoints may be familiar.",
            failure="Budget failures are censored, not impossible-state or bad-action labels.",
            compute="Same simulations/resources, not equal neural work or wall time.",
            cost="Costs compared on common successes; certified native floor equality alone "
            "proves representation-cost optimality, not every presentation's optimum.",
            size="Endpoint maxima use length254/strands21 training-support strata. Lifted views "
            "are correlated, not new independent knots. Arbitrary-size quality unproven.",
            scale="Static matched training, not million-scale online reinforcement learning; "
            "synthetic results do not establish public upper-bound improvements.",
        ),
    )
