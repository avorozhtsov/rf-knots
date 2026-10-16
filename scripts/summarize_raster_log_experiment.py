"""Final-only97 comparison, with shared TEST and transfer-attribution limitations."""

from pathlib import Path

from audit_graph_mixture_cost_mcts import inspect_rows
from audit_joint_policy_metric_panel import check_pins, merge_pins
from check_qrun_common_witnesses import sha
from evaluate_raster_log_distances import PANELS, interval_ranking
from evaluate_strand_graph_panel import certified_lower
from run_raster_log_evaluation import jobs, read, verify_job
from summarize_graph_mixture_experiment import certify_costs
from summarize_joint_policy_metric_evaluation import (
    metric_with_constants,
    paired_metric,
    paired_search,
    search_behavior,
)


def validate_predictions(predictions, records):
    if not predictions or len(predictions) != len(records):
        raise ValueError("metric record coverage differs")
    for i, (p, r) in enumerate(zip(predictions, records, strict=True)):
        lower = certified_lower(
            r["interval"], r["left_fingerprint"], r["right_fingerprint"], determinant=True
        )
        fp = r["left_fingerprint"]
        if (
            p["panel_index"],
            p["lower"],
            p["upper"],
            p["fingerprint_group"],
            p["left_fingerprint"],
            p["right_fingerprint"],
        ) != (
            i,
            lower,
            r["interval"]["upper"],
            f"{fp[0]}:{abs(fp[1])}",
            fp,
            r["right_fingerprint"],
        ):
            raise ValueError("prediction identity/certified interval differs")
    return {**metric_with_constants(predictions), "ranking": interval_ranking(predictions)}


def summarize_verified(directory):
    state = read(directory / "manifest.json")
    planned = jobs(directory, Path(state["training_run"]), Path(state["pgx_root"]))
    if (
        state["status"] not in ("scored", "completed")
        or state["jobs"] != planned
        or len(state["completed"]) != len(planned)
        or len(planned) != 13
    ):
        raise ValueError("require complete13-job matrix, not partial scores")
    pins = dict(state["inputs"])
    reports = {}
    for job, recorded in zip(planned, state["completed"], strict=True):
        if verify_job(job, directory) != recorded:
            raise ValueError("completed evaluation artifact changed")
        report = read(Path(job["output"]))
        reports[job["name"]] = report
        merge_pins(pins, report["inputs"])
        merge_pins(pins, {job["output"]: recorded["sha256"]})
    check_pins(pins)
    panels = {name: read(path) for name, path in PANELS.items()}
    metric = {}
    predictions = {}
    for arm in ("q254", "random"):
        metric[arm] = {}
        for model_state in ("initial", "final"):
            metric[arm][model_state] = {}
            for panel in PANELS:
                result = reports[f"metric-{arm}-{model_state}-{panel}"]["results"]
                metric[arm][model_state][panel] = {}
                for view, report in result.items():
                    rows = report["predictions"]
                    predictions[arm, model_state, panel, view] = rows
                    metric[arm][model_state][panel][view] = validate_predictions(
                        rows, panels[panel][view]
                    )
    comparisons = {}
    for panel in PANELS:
        for view in ("validation", "size_shift") if panel == "short-u" else ("validation",):
            key = f"{panel}/{view}"
            comparisons[key] = {
                f"{arm}-final-vs-initial": paired_metric(
                    predictions[arm, "initial", panel, view], predictions[arm, "final", panel, view]
                )
                for arm in ("q254", "random")
            }
            comparisons[key]["q254-vs-random-final"] = paired_metric(
                predictions["random", "final", panel, view],
                predictions["q254", "final", panel, view],
            )
    from evaluate_graph_mixture_cost_mcts import PANEL, build_roots, committed_rows

    roots = build_roots(panels["short-u"])
    costs, compute = {}, {}
    for arm in ("q254", "random"):
        search = reports[f"search-{arm}"]
        folder = directory / "mcts" / arm
        if read(folder / "roots.json") != roots:
            raise ValueError("paired search roots differ")
        rows = committed_rows(search, folder)
        inspect_rows(roots, rows, search)
        costs[arm] = rows
        compute[arm] = {"counts": search["counts"], "seconds": search["seconds_committed_batches"]}
    certified = certify_costs(panels["short-u"], roots, costs)
    for arm in costs:
        audit = reports[f"search-audit-{arm}"]
        if audit["certified_costs"] != {
            **certified,
            "solutions": {arm: certified["solutions"][arm]},
        }:
            raise ValueError("cost audit differs from recomputed certificates")
    check_pins(pins)
    return {
        "status": "completed",
        "inputs": pins,
        "controller_contract_sha256": state["contract_sha256"],
        "short_u_panel_sha256": sha(PANEL),
        "metric": metric,
        "paired_metric": comparisons,
        "search_q254_vs_random": {
            str(w): paired_search(roots, costs["random"], costs["q254"], w) for w in (10, 1000)
        },
        "search_behavior": {a: search_behavior(roots, rows) for a, rows in costs.items()},
        "compute": compute,
        "certified_costs": certified,
        "checkpoints": reports["training-pair-audit"]["arms"],
        "scope": {
            "promotion": "not_decided",
            "comparison": "Q254 local pretraining versus random local initialization; "
            "same upper layers, data sequence and training budget.",
            "embedding_causality": "Both arms use embeddings. This is not an embedding-versus-"
            "no-embedding test, nor a before/after search comparison.",
            "split": "Shared frozen70/74 TEST panels, not fresh replications. "
            "Q254 historical pretraining exposure is incompletely audited.",
            "failure": "Budget failures are censored, not unsolvable or bad-action labels.",
            "cost": "Only equality with a certified semantic floor proves optimality; "
            "cost deltas on common successes are conditional summaries.",
            "scale": "Size-independent architecture; arbitrary-size solution quality "
            "and million-scale online RL are not established.",
        },
    }
