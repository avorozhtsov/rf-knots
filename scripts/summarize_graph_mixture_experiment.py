"""Final-only combined69/70/74 report: metrics, replayed search costs and limits."""

import argparse
import json
from collections import Counter
from pathlib import Path

import run_graph_mixture_evaluation as core
import run_pure_braid_distance_evaluation as general
from audit_joint_policy_metric_panel import check_pins, merge_pins
from audit_semantic_cost_floor import reference_bounds, witness_identity
from build_mcts_training_supervision import atomic, sha
from burau_integer_check import determinant
from evaluate_graph_mixture_cost_mcts import PANEL, RUNS, build_roots, committed_rows
from summarize_joint_policy_metric_evaluation import (
    metric_with_constants,
    paired_metric,
    paired_search,
    search_behavior,
)

from rf_knots.embedding_evaluation import verify_evaluation_pair
from rf_knots.evidence import UnknotWitness
from rf_knots.mcts_loop import Instance

PAIRS = (("initial", "graph"), ("initial", "control"), ("control", "graph"))


def require_final_matrix(state, planned):
    if (
        state["status"] != "completed"
        or state["jobs"] != planned
        or len(state["completed"]) != len(planned)
    ):
        raise ValueError("require the complete declared job matrix, not partial scores")


def verified_inputs(core_dir, general_dir):
    state = core.read(core_dir / "manifest.json")
    planned = core.jobs(core_dir)
    require_final_matrix(state, planned)
    pins = dict(state["inputs"])
    merge_pins(pins, state["completed_gate_inputs"])
    reports = {}
    for job, stored in zip(planned, state["completed"], strict=True):
        if core.verify_job(job) != stored:
            raise ValueError("committed core output differs")
        report = core.read(Path(job["output"]))
        reports[job["name"]] = report
        merge_pins(pins, report.get("inputs", report.get("code_sha256", {})))
        merge_pins(pins, {job["output"]: stored["sha256"]})
    pair = reports["training-pair-audit"]
    for arm, run in RUNS.items():
        expected = sha(run / "checkpoint.pt")
        if arm != "initial" and pair["arms"][arm]["checkpoint_sha256"] != expected:
            raise ValueError("training-pair and evaluated model differ")
        if reports[f"metric-{arm}-short-u"]["checkpoint_sha256"] != expected:
            raise ValueError("short-u metric model differs")
    roots, costs = None, {}
    panel = core.read(PANEL)
    expected_roots = build_roots(panel)
    for arm in (*RUNS, "uniform"):
        directory = core_dir / "mcts" / arm
        manifest = reports[f"mcts-{arm}"]
        audit = reports[f"mcts-audit-{arm}"]
        this_roots = core.read(directory / "roots.json")
        if (
            this_roots != expected_roots
            or (roots is not None and roots != this_roots)
            or sha(directory / "roots.json") != manifest["roots_sha256"]
            or audit["inputs"].get(str(directory / "manifest.json"))
            != sha(directory / "manifest.json")
        ):
            raise ValueError("search audit/roots/resources differ")
        roots = this_roots
        costs[arm] = committed_rows(manifest, directory)
        if len(costs[arm]) != len(roots):
            raise ValueError("incomplete search query coverage")
    extra = general.read(general_dir / "manifest.json")
    general_jobs = general.jobs(general_dir)
    require_final_matrix(extra, general_jobs)
    merge_pins(pins, extra["inputs"])
    merge_pins(pins, extra["gate_inputs"])
    general_panel = general.read(general.PANEL)
    general_reports = {}
    for job, stored in zip(general_jobs, extra["completed"], strict=True):
        actual, report = general.verify_report(job, general_panel, pair)
        if actual != stored:
            raise ValueError("committed general-distance output differs")
        general_reports[job["name"]] = report
        merge_pins(pins, report["inputs"])
        merge_pins(pins, {job["output"]: stored["sha256"]})
    comparison_path = general_dir / "comparison.json"
    comparison = general.read(comparison_path)
    expected = general.compare(general_reports, panel)
    if (
        sha(comparison_path) != extra["comparison_sha256"]
        or comparison["status"] != "completed"
        or {k: comparison[k] for k in expected} != json.loads(json.dumps(expected))
    ):
        raise ValueError("general comparison differs from raw reports")
    merge_pins(pins, comparison["inputs"])
    merge_pins(
        pins,
        {
            str(p): sha(p)
            for p in (comparison_path, core_dir / "manifest.json", general_dir / "manifest.json")
        },
    )
    check_pins(pins)
    return (
        pins,
        reports,
        roots,
        costs,
        general_reports,
        comparison,
        panel,
        state["general_distance"],
    )


def certify_costs(panel, roots, costs):
    """Full reference/solution replay; a lower-bound gap is not an optimality regret."""
    if len(roots) != 2 * len(panel["validation"]):
        raise ValueError("two cost conditions required per reference")
    references = []
    for i, record in enumerate(panel["validation"]):
        witness = verify_evaluation_pair(record)
        if witness.moves != record["pair_prefix_length"]:
            raise ValueError("reference is only a pair prefix")
        det = determinant(witness.start.word, witness.start.strands)
        if det != record["left_fingerprint"][0]:
            raise ValueError("independent reference determinant differs")
        lower = max(abs(record["left_fingerprint"][1]) // 2, int(det != 1))
        identity = witness_identity(witness)
        if identity["cc"] != record["interval"]["upper"]:
            raise ValueError("reference CC upper bound differs")
        bound = reference_bounds(identity, lower, roots[2 * i]["instance"]["cap_moves"])
        for root, w in zip(roots[2 * i : 2 * i + 2], (10, 1000), strict=True):
            instance = root["instance"]
            if (
                root["panel_index"] != i
                or instance["cc_weight"] != w
                or (tuple(instance["word"]), instance["strands"])
                != (witness.start.word, witness.start.strands)
                or root[f"reference_L{w}"] != bound[str(w)]["reference_cost"]
            ):
                raise ValueError("reference cost/query context differs")
        references.append(
            {
                "panel_index": i,
                "fingerprint": record["left_fingerprint"],
                "identity": identity,
                "lower_cc": lower,
                "bounds": bound,
            }
        )
    solutions = {}
    for arm, rows in costs.items():
        if len(rows) != len(roots):
            raise ValueError("search result coverage differs")
        verified = []
        for root, row in zip(roots, rows, strict=True):
            if (row["condition"], row["panel_index"], row["cc_weight"]) != (
                root["condition"],
                root["panel_index"],
                root["instance"]["cc_weight"],
            ):
                raise ValueError("solution identity differs")
            if row["solved"] != (row["witness"] is not None):
                raise ValueError("solution status differs from certificate")
            if not row["solved"]:
                if row["L10"] is not None or row["L1000"] is not None:
                    raise ValueError("unsolved condition has a numeric cost")
                continue
            witness = UnknotWitness.from_dict(row["witness"])
            identity = witness_identity(witness)
            instance = root["instance"]
            r = references[row["panel_index"]]
            spec = Instance(**{**instance, "word": tuple(instance["word"])}).spec
            if (
                (witness.start.word, witness.start.strands)
                != (tuple(instance["word"]), instance["strands"])
                or not r["lower_cc"] <= identity["cc"] <= instance["cap_crossing_changes"]
                or identity["moves"] > instance["cap_moves"]
                or (witness.max_len, witness.max_strands) != (spec.max_len, spec.max_strands)
            ):
                raise ValueError("solution resource or lower-bound violation")
            for w in (10, 1000):
                if row[f"L{w}"] != w * identity["cc"] + identity["moves"]:
                    raise ValueError("reported L differs from witness replay")
            w = row["cc_weight"]
            bound = r["bounds"][str(w)]
            value = row[f"L{w}"]
            if value < bound["certified_lower_cost"]:
                raise ValueError("solution below certified semantic floor")
            verified.append(
                {
                    "condition": row["condition"],
                    "panel_index": row["panel_index"],
                    "lambda": w,
                    "identity": identity,
                    "cost": value,
                    "gap_to_lower_bound": value - bound["certified_lower_cost"],
                    "gap_to_reference": value - bound["reference_cost"],
                    "solution_certifiably_optimal": value == bound["certified_lower_cost"],
                    "certified_regret": value - bound["reference_cost"]
                    if bound["reference_certifiably_optimal"]
                    else 0
                    if value == bound["certified_lower_cost"]
                    else None,
                }
            )
        solutions[arm] = verified
    return {
        "references": references,
        "solutions": solutions,
        "reference_summary": {
            str(w): {
                "roots": len(references),
                "certified_reference_optima": sum(
                    r["bounds"][str(w)]["reference_certifiably_optimal"] for r in references
                ),
                "necessary_cost_envelope_exceeds_move_cap": sum(
                    r["bounds"][str(w)]["envelope_beyond_current_cap"] > 0 for r in references
                ),
            }
            for w in (10, 1000)
        },
    }


def summarize(reports, roots, costs, pure, comparison, panel, unavailable):
    metric = {
        arm: {
            view: metric_with_constants(reports[f"metric-{arm}-short-u"][view]["predictions"])
            for view in ("validation", "size_shift")
        }
        for arm in RUNS
    }
    distance_pairs = {
        f"{b}-vs-{a}": paired_metric(
            reports[f"metric-{a}-short-u"]["validation"]["predictions"],
            reports[f"metric-{b}-short-u"]["validation"]["predictions"],
        )
        for a, b in PAIRS
    }
    searches = {
        f"{b}-vs-{a}": {str(w): paired_search(roots, costs[a], costs[b], w) for w in (10, 1000)}
        for a, b in (*PAIRS, ("uniform", "graph"), ("uniform", "control"), ("uniform", "initial"))
    }
    return {
        "short_u": metric,
        "paired_short_u": distance_pairs,
        "general_distance": comparison,
        "original_general_source": unavailable,
        "search_comparisons": searches,
        "search_coverage": {
            arm: {
                str(w): {
                    "conditions": sum(r["cc_weight"] == w for r in rows),
                    "solved": sum(r["cc_weight"] == w and r["solved"] for r in rows),
                }
                for w in (10, 1000)
            }
            for arm, rows in costs.items()
        },
        "compute": {
            arm: {
                "counts": reports[f"mcts-{arm}"]["counts"],
                "seconds_committed_batches": reports[f"mcts-{arm}"]["seconds_committed_batches"],
            }
            for arm in costs
        },
        "behavior": {arm: search_behavior(roots, rows) for arm, rows in costs.items()},
        "certified_costs": certify_costs(panel, roots, costs),
        "short_u_raw_metric_strata": dict(Counter(r["stratum"] for r in roots[::2])),
        "short_u_raw_policy_strata": dict(Counter(r["policy_stratum"] for r in roots[::2])),
        "general_raw_sizes": pure["initial"]["raw_sizes"],
        "checkpoints": {a: reports[f"metric-{a}-short-u"]["checkpoint_sha256"] for a in RUNS},
        "scope": {
            "promotion": "not_decided",
            "million_scale_rl": "not_performed",
            "arbitrary_size_solution_quality": "not_established",
            "censoring": "Budget-limited failures are not bad-action or impossibility labels; "
            "costs compared only on common successes.",
            "cost_optimality": "Only equality with a replayed certified semantic floor proves "
            "an optimum, for this exact start/action system. Other gaps are not regrets.",
            "budgets": "Size-scaled caps, eight simulations and finite proposals remain "
            "resource limits. A larger necessary envelope does not prove a missed better solution.",
            "families": "Constructive short-u and pure-block ordinary distance are separate "
            "synthetic families. Coordinate lifts are not new knot types.",
            "uncertainty": "One training seed; descriptive comparisons, not a significance "
            "or catalogue-distribution claim.",
            "compute": "Same conditions/simulation contracts do not mean equal actual "
            "evaluations or isolated wall times.",
        },
    }


def markdown(result):
    lines = [
        "# Graph-mixture experiment: complete descriptive results",
        "",
        "No automatic promotion. Failed searches are censored; lower-bound gaps are not "
        "always optimality regrets.",
        "",
        "## Distance errors",
        "",
        "| Model | Short-u equal-group error | Short-u exact2 MAE (count) | "
        "General equal-group error | General positive-bound error | General exact2 MAE (count) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in RUNS:
        short = result["short_u"][arm]["validation"]
        general_stats = result["general_distance"]["arms"][arm]
        exact_short = short["exact"].get("2")
        short_two = (
            f"{exact_short['mae']:.6f} ({exact_short['count']})"
            if exact_short is not None
            else "not available (0)"
        )
        exact_general = general_stats["exact_two"]
        lines.append(
            f"| {arm} | {short['equal_group_mean_interval_violation']:.6f} | "
            f"{short_two} | "
            f"{general_stats['all']['equal_group_mean_interval_violation']:.6f} | "
            f"{general_stats['positive_lower_bound']['equal_group_mean_interval_violation']:.6f} | "
            f"{exact_general['mean_interval_violation']:.6f} ({exact_general['rows']}) |"
        )
    short = result["short_u"]["initial"]["validation"]
    g = result["general_distance"]["arms"]["initial"]
    const_u = short["constants"]["1"]["equal_group_mean_interval_violation"]
    const_g = g["all"]["constants"]["1"]["equal_group_mean_interval_violation"]
    const_positive = g["positive_lower_bound"]["constants"]["1"][
        "equal_group_mean_interval_violation"
    ]
    lines += [
        "",
        f"Constant1 equal-group controls: short-u {const_u:.6f}; general {const_g:.6f}; "
        f"general positive-bound subset {const_positive:.6f}.",
        "Exact0 coordinate checks and zero-CC lifted views remain separate in the JSON report.",
        "",
        "## Replayed search coverage",
        "",
        "| Model | Lambda10 solved / attempted | Lambda1000 solved / attempted |",
        "| --- | ---: | ---: |",
    ]
    for arm, weights in result["search_coverage"].items():
        a, b = weights["10"], weights["1000"]
        lines.append(
            f"| {arm} | {a['solved']} / {a['conditions']} | {b['solved']} / {b['conditions']} |"
        )
    lines += [
        "",
        "## Paired costs on common successes",
        "",
        "| Comparison | Lambda | Common successes | Mean candidate-minus-baseline cost | "
        "New / lost solutions |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, weights in result["search_comparisons"].items():
        for weight, r in weights.items():
            d = r["mean_cost_delta_common_only"]
            display = d if d is not None else "not available"
            new, lost = r["outcomes"].get("newly_solved", 0), r["outcomes"].get("lost_solution", 0)
            lines.append(
                f"| {name} | {weight} | {r['common_solved']} | {display} | {new} / {lost} |"
            )
    lines += ["", "## Scope and limits", ""]
    lines.extend(f"- {key}: {value}" for key, value in result["scope"].items())
    lines += [
        "",
        "The original held-out sidecar test remains unavailable; the separate pure-block "
        "test does not erase that failed supply result.",
        "Per-root witnesses, certified floors, cost gaps, raw-size strata, compute counters, "
        "constant controls and all provenance are retained in report.json.",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core", type=Path, default=Path("artifacts/graph-mixture-evaluation-v1"))
    p.add_argument(
        "--general", type=Path, default=Path("artifacts/pure-braid-short-distance-evaluation-v1")
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite combined summary")
    pins, reports, roots, costs, pure, comparison, panel, unavailable = verified_inputs(
        args.core, args.general
    )
    sources = [
        Path(__file__).resolve(),
        Path("scripts/audit_semantic_cost_floor.py"),
        Path("scripts/summarize_joint_policy_metric_evaluation.py"),
        Path("scripts/summarize_clean_paired_evaluation.py"),
        Path("scripts/run_graph_mixture_evaluation.py"),
        Path("scripts/run_pure_braid_distance_evaluation.py"),
    ]
    merge_pins(pins, {str(path): sha(path) for path in sources})
    result = summarize(reports, roots, costs, pure, comparison, panel, unavailable)
    check_pins(pins)
    args.out.mkdir(parents=True)
    atomic(args.out / "report.json", {"status": "completed", "inputs": pins, **result})
    (args.out / "report.md").write_text(markdown(result))
    print(
        json.dumps({"status": "completed", "report_sha256": sha(args.out / "report.json")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
