"""Final-only descriptive comparison of the frozen joint-policy/metric experiment."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import run_joint_policy_metric_evaluation as sequence
from audit_cost_resource_envelope import envelope
from audit_joint_policy_metric_panel import check_pins, size_stratum
from audit_semantic_cost_floor import reference_bounds
from build_mcts_training_supervision import atomic, sha
from evaluate_joint_cost_mcts import committed_rows
from recover_joint_evaluation_schema import verified_recovery
from summarize_clean_paired_evaluation import metric_stats, paired_stats

from rf_knots.actions import KIND_NAMES
from rf_knots.mcts_loop import Instance

PAIRS = (("parent", "joint"), ("parent", "metric-control"), ("metric-control", "joint"))
STRATA = ("within", "length_only", "strands_only", "both")
ENVELOPE = Path("artifacts/mcts-policy-metric-resource-envelope-v1/report.json")
FLOORS = Path("artifacts/mcts-policy-metric-cost-floor-audit-v1/report.json")


def certified_costs(audit, roots, costs):
    """Read replay-certified floors; gaps are regrets only for certified optima."""
    if audit["status"] != "passed":
        raise ValueError("semantic cost audit did not pass")
    references = {r["panel_index"]: r for r in audit["references"]}
    if len(references) != len(audit["references"]) or set(references) != {
        r["panel_index"] for r in roots
    }:
        raise ValueError("semantic cost audit query coverage differs")
    for root in roots:
        reference = references[root["panel_index"]]
        identity, instance = reference["identity"], root["instance"]
        if (
            reference["fingerprint"] != root["fingerprint"]
            or reference["stratum"] != root["stratum"]
            or 2 * identity["q0"] != len(instance["word"]) + instance["strands"] - 1
            or reference["bounds"]
            != reference_bounds(identity, reference["lower_cc"], instance["cap_moves"])
            or any(
                reference["bounds"][str(w)]["reference_cost"] != root[f"reference_L{w}"]
                for w in (10, 1000)
            )
        ):
            raise ValueError("semantic cost audit root/bounds differ")
    result = {"reference_summary": {}, "arms": {}}
    for weight in (10, 1000):
        bounds = [r["bounds"][str(weight)] for r in references.values()]
        result["reference_summary"][str(weight)] = {
            "queries": len(bounds),
            "certified_reference_optima": sum(r["reference_certifiably_optimal"] for r in bounds),
            "move_envelope_beyond_cap": sum(r["envelope_beyond_current_cap"] > 0 for r in bounds),
        }
    for arm, rows in costs.items():
        if len(rows) != len(roots):
            raise ValueError("semantic cost result coverage differs")
        solutions = []
        for root, row in zip(roots, rows, strict=True):
            weight = root["instance"]["cc_weight"]
            if (row["condition"], row["panel_index"], row["cc_weight"]) != (
                root["condition"],
                root["panel_index"],
                weight,
            ):
                raise ValueError("semantic cost result identity differs")
            value = row[f"L{weight}"]
            if row["solved"] != (value is not None):
                raise ValueError("semantic cost solved/censoring differs")
            if value is None:
                continue
            bound = references[root["panel_index"]]["bounds"][str(weight)]
            if value < bound["certified_lower_cost"]:
                raise ValueError("solution violates certified cost floor")
            solutions.append(
                {
                    "condition": row["condition"],
                    "panel_index": row["panel_index"],
                    "lambda": weight,
                    "cost": value,
                    "gap_to_lower_bound": value - bound["certified_lower_cost"],
                    "certified_regret": value - bound["reference_cost"]
                    if bound["reference_certifiably_optimal"]
                    else None,
                }
            )
        result["arms"][arm] = solutions
    return result


def metric_with_constants(rows):
    stats = metric_stats(rows)
    stats["constants"] = {
        str(v): metric_stats(
            [
                {**r, "prediction": v, "violation": max(r["lower"] - v, v - r["upper"], 0)}
                for r in rows
            ]
        )
        for v in (0, 1, 2)
    }
    return stats


def paired_metric(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("paired metric coverage differs or is empty")
    groups = defaultdict(list)
    for left, right in zip(a, b, strict=True):
        if any(left[k] != right[k] for k in ("panel_index", "fingerprint_group", "lower", "upper")):
            raise ValueError("paired metric labels/queries differ")
        groups[left["fingerprint_group"]].append(right["violation"] - left["violation"])
    deltas = [sum(values) / len(values) for values in groups.values()]
    return {
        "rows": len(a),
        "groups": len(groups),
        "mean_group_error_delta": sum(deltas) / len(deltas),
        "lower_error_groups": sum(x < 0 for x in deltas),
        "higher_error_groups": sum(x > 0 for x in deltas),
        "equal_error_groups": sum(x == 0 for x in deltas),
    }


def paired_search(roots, baseline, candidate, weight):
    if len(roots) != len(baseline) or len(roots) != len(candidate):
        raise ValueError("paired search query coverage differs")
    pairs = []
    for root, left, right in zip(roots, baseline, candidate, strict=True):
        if (
            left["condition"] != root["condition"]
            or right["condition"] != root["condition"]
            or left["panel_index"] != root["panel_index"]
            or right["panel_index"] != root["panel_index"]
            or left["cc_weight"] != root["instance"]["cc_weight"]
            or right["cc_weight"] != root["instance"]["cc_weight"]
        ):
            raise ValueError("paired search identity/resources differ")
        if root["instance"]["cc_weight"] == weight:
            pairs.append(
                {
                    "panel_index": root["panel_index"],
                    "stratum": root["stratum"],
                    "fingerprint": root["fingerprint"],
                    "reference_cost": root[f"reference_L{weight}"],
                    "baseline_cost": left[f"L{weight}"],
                    "candidate_cost": right[f"L{weight}"],
                }
            )
    return {
        **paired_stats(pairs),
        "paired": pairs,
        "strata": {
            s: paired_stats([p for p in pairs if p["stratum"] == s])
            if any(p["stratum"] == s for p in pairs)
            else None
            for s in STRATA
        },
    }


def search_behavior(roots, rows):
    """Descriptive continuation behavior, not causal action-quality labels."""
    if len(roots) != len(rows):
        raise ValueError("behavior query coverage differs")
    results = {}
    for weight in (10, 1000):
        actions, stops = Counter(), Counter()
        count = start_total = end_total = move_total = conditions = 0
        for root, row in zip(roots, rows, strict=True):
            if (
                row["condition"] != root["condition"]
                or row["cc_weight"] != root["instance"]["cc_weight"]
            ):
                raise ValueError("behavior identity/resources differ")
            if row["cc_weight"] != weight:
                continue
            conditions += 1
            if not row["episodes"]:
                continue
            instance = Instance(**{**root["instance"], "word": tuple(root["instance"]["word"])})
            for episode in row["episodes"]:
                continuation = episode["continuation"]
                count += 1
                stops[continuation["stop_reason"]] += 1
                start_total += len(continuation["start"]["word"])
                end_total += len(continuation["end"]["word"])
                move_total += len(continuation["actions"])
                actions.update(
                    KIND_NAMES[instance.spec.decode(a)[0]] for a in continuation["actions"]
                )
        results[str(weight)] = {
            "conditions": conditions,
            "continuations": count,
            "stop_reasons": dict(stops),
            "continuation_action_kinds": dict(actions),
            "mean_start_length": start_total / count if count else None,
            "mean_end_length": end_total / count if count else None,
            "mean_length_change_end_minus_start": (end_total - start_total) / count
            if count
            else None,
            "mean_continuation_moves": move_total / count if count else None,
        }
    return results


def verified_inputs(path):
    state = sequence.read(path)
    planned = sequence.jobs(path.parent)
    if (
        state["status"] != "completed"
        or state["jobs"] != planned
        or len(state["completed"]) != len(planned)
    ):
        raise ValueError("require the whole final 15-job experiment, not partial model scores")
    check_pins({**state["inputs"], **state["gate_pins"]})
    reports = {}
    for job, stored in zip(planned, state["completed"], strict=True):
        if stored != sequence.verify_job(job):
            raise ValueError("completed job no longer matches controller")
        reports[job["name"]] = sequence.read(Path(job["output"]))
    pair = reports["training-pair-audit"]
    for name, run in sequence.RUNS.items():
        expected = sha(run / "checkpoint.pt")
        if (
            name != "parent"
            and pair["arms"]["control" if name == "metric-control" else name]["checkpoint_sha256"]
            != expected
        ):
            raise ValueError("training-pair audit does not cover evaluated model")
        for panel in sequence.PANELS:
            if reports[f"metric-{name}-{panel}"]["checkpoint_sha256"] != expected:
                raise ValueError("metric model differs from final checkpoint")
    roots, costs = None, {}
    for arm in (*sequence.RUNS, "uniform"):
        directory = path.parent / "mcts" / arm
        manifest = reports[f"mcts-{arm}"]
        audit = reports[f"mcts-audit-{arm}"]
        if (
            audit["arm"] != arm
            or manifest["arm"] != arm
            or audit["inputs"].get(str(directory / "manifest.json"))
            != sha(directory / "manifest.json")
        ):
            raise ValueError("search replay audit belongs to another result")
        this_roots = sequence.read(directory / "roots.json")
        if roots is not None and roots != this_roots:
            raise ValueError("search arms have different roots/resources")
        roots = this_roots
        costs[arm] = committed_rows(manifest, directory)
    diagnostic = sequence.read(ENVELOPE)
    if diagnostic["status"] != "completed" or diagnostic["rows"] != [
        envelope(root, root["instance"]["cc_weight"]) for root in roots
    ]:
        raise ValueError("cost envelope diagnostic does not cover the evaluated roots")
    if diagnostic["inputs"].get(str(sequence.PANELS["short-u"])) != sha(sequence.PANELS["short-u"]):
        raise ValueError("cost envelope diagnostic belongs to another panel")
    check_pins(diagnostic["inputs"])
    reports["cost-resource-envelope"] = diagnostic
    floor_audit = sequence.read(FLOORS)
    for dependency in (sequence.PANELS["short-u"], path.parent / "mcts/joint/manifest.json"):
        if floor_audit["inputs"].get(str(dependency)) != sha(dependency):
            raise ValueError("semantic cost audit belongs to another experiment")
    check_pins(floor_audit["inputs"])
    certified_costs(floor_audit, roots, costs)
    reports["semantic-cost-floor"] = floor_audit
    extra_pins = {
        **verified_recovery(state),
        **diagnostic["inputs"],
        str(ENVELOPE): sha(ENVELOPE),
        **floor_audit["inputs"],
        str(FLOORS): sha(FLOORS),
    }
    return state, reports, roots, costs, extra_pins


def summarize(reports, panels, roots, costs):
    result = {
        "metric": {},
        "paired_metric": {},
        "search_comparisons": {},
        "compute": {},
        "behavior": {},
    }
    for panel_name, panel in panels.items():
        for view in ("validation", "size_shift"):
            predictions = {
                arm: reports[f"metric-{arm}-{panel_name}"][view]["predictions"]
                for arm in sequence.RUNS
            }
            for arm, rows in predictions.items():
                stats = metric_with_constants(rows)
                if view == "validation":
                    labels = [
                        size_stratum(
                            r["interval"]["left"]["word"], r["interval"]["left"]["strands"]
                        )
                        for r in panel[view]
                    ]
                    stats["raw_size_strata"] = {
                        s: metric_with_constants(
                            [r for r, label in zip(rows, labels, strict=True) if label == s]
                        )
                        if s in labels
                        else None
                        for s in STRATA
                    }
                result["metric"][f"{arm}/{panel_name}/{view}"] = stats
            for baseline, candidate in PAIRS:
                result["paired_metric"][f"{candidate}-vs-{baseline}/{panel_name}/{view}"] = (
                    paired_metric(predictions[baseline], predictions[candidate])
                )
    for baseline, candidate in (*PAIRS, ("uniform", "joint")):
        result["search_comparisons"][f"{candidate}-vs-{baseline}"] = {
            str(w): paired_search(roots, costs[baseline], costs[candidate], w) for w in (10, 1000)
        }
    for arm in costs:
        m = reports[f"mcts-{arm}"]
        result["compute"][arm] = {
            "counts": m["counts"],
            "seconds_committed_batches": m["seconds_committed_batches"],
        }
        result["behavior"][arm] = search_behavior(roots, costs[arm])
    result["scope"] = {
        "promotion": "not_decided",
        "short_u": "64 synthetic roots / 64 fingerprint groups",
        "general_gordian": (
            "87 pairs / 83 groups, known common-block family; only four exact-2 rows"
        ),
        "size_shift": "zero-CC lifts, not independent or intrinsically harder knots",
        "search": "eight simulations, full policy continuations, separate lambda10/lambda1000",
        "resource_caveat": "CC cap is the reference witness count; this can exclude lower-L10 "
        "paths with extra CC. These are resource-constrained comparisons, not global cost minima.",
        "compute": "equal simulation/resource contract, not equal neural evaluations or wall time",
        "uncertainty": "descriptive single-seed comparison; small strata, no significance claim",
        "arbitrary_size_quality": "not established",
        "size_budget_confound": (
            "The move cap max(64, reference moves) gives shorter roots more relative headroom. "
            "Size-stratum differences cannot be attributed solely to architecture extrapolation."
        ),
        "million_scale_rl": "not performed",
        "behavior": (
            "Action frequencies/word-length changes describe continuations, not bad-action "
            "or impossibility labels. Tree prefixes are excluded from these behavior counters."
        ),
        "timing_observation": (
            "Parent wall times include a two-second native sample and concurrent isolated "
            "encoder benchmarking; joint timing includes concurrent TRAIN-only device probes. "
            "These are not isolated end-to-end performance measurements."
        ),
    }
    if "cost-resource-envelope" in reports:
        rows = reports["cost-resource-envelope"]["rows"]
        counts = {
            str(w): {
                "conditions": sum(r["lambda"] == w for r in rows),
                "extra_cc_improvement_not_ruled_out": sum(
                    r["lambda"] == w and r["extra_cc_improvement_not_ruled_out"] for r in rows
                ),
            }
            for w in (10, 1000)
        }
        result["resource_envelope"] = counts
        if not any(r["extra_cc_improvement_not_ruled_out"] for r in rows):
            result["scope"]["resource_caveat"] = (
                "For these audited roots, semantic move lower bounds rule out improving the "
                "reference witness cost with extra CC at both lambdas. This does not prove "
                "optimal costs: move/growth caps, horizon and candidate limits remain."
            )
    if "semantic-cost-floor" in reports:
        result["certified_costs"] = certified_costs(reports["semantic-cost-floor"], roots, costs)
        result["scope"]["certified_costs"] = (
            "Cost optima are certified only when a replayed reference attains the semantic "
            "lower bound, for that exact start and action system. Other lower-bound gaps are "
            "not known regrets. Test certificates are not training labels. A move envelope "
            "above the cap does not demonstrate a missing cheaper solution."
        )
    return result


def markdown(result):
    lines = [
        "# Joint policy/metric: complete prospective evaluation",
        "",
        "No automatic promotion. Negative error/cost deltas mean lower error/cost.",
        "",
        "## Distance calibration",
        "",
        "| Arm / panel / view | Rows / groups | Equal-group interval error "
        "| Constant 0 / 1 / 2 | Exact-class MAE (count) |",
        "|---|---:|---:|---|---|",
    ]
    for name, row in result["metric"].items():
        constants = " / ".join(
            f"{row['constants'][str(v)]['equal_group_mean_interval_violation']:.6f}"
            for v in (0, 1, 2)
        )
        exact = "; ".join(f"{k}: {v['mae']:.6f} ({v['count']})" for k, v in row["exact"].items())
        lines.append(
            f"| {name} | {row['rows']} / {row['groups']} | "
            f"{row['equal_group_mean_interval_violation']:.6f} | {constants} | {exact} |"
        )
    lines += [
        "",
        "## Full-policy cost-MCTS",
        "",
        "| Candidate vs baseline | Lambda | Solved baseline → candidate | New / lost "
        "| Lower / higher / equal cost | Common-success mean delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, weights in result["search_comparisons"].items():
        for weight, row in weights.items():
            o = row["outcomes"]
            delta = row["mean_cost_delta_common_only"]
            delta = "unavailable" if delta is None else f"{delta:.3f}"
            lines.append(
                f"| {name} | {weight} | {row['baseline_solved']} → {row['candidate_solved']} "
                f"/ {row['observations']} | "
                f"{o.get('newly_solved', 0)} / {o.get('lost_solution', 0)} | "
                f"{o.get('lower_cost', 0)} / {o.get('higher_cost', 0)} "
                f"/ {o.get('equal_cost', 0)} | {delta} |"
            )
    lines += [
        "",
        "## Continuation behavior (descriptive, not action-quality labels)",
        "",
        "| Arm | Lambda | Episodes | Stops | Mean letters: start → end | Mean moves |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for arm, weights in result["behavior"].items():
        for weight, row in weights.items():
            lengths = (
                "unavailable"
                if row["mean_start_length"] is None
                else f"{row['mean_start_length']:.2f} → {row['mean_end_length']:.2f}"
            )
            moves = row["mean_continuation_moves"]
            moves = "unavailable" if moves is None else f"{moves:.2f}"
            stops = "; ".join(f"{k}: {v}" for k, v in sorted(row["stop_reasons"].items())) or "none"
            lines.append(
                f"| {arm} | {weight} | {row['continuations']} | {stops} | {lengths} | {moves} |"
            )
    if "certified_costs" in result:
        lines += ["", "## Replay-certified semantic cost bounds", ""]
        for weight, item in result["certified_costs"]["reference_summary"].items():
            lines.append(
                f"Lambda {weight}: {item['certified_reference_optima']}/{item['queries']} "
                f"reference costs certified optimal; {item['move_envelope_beyond_cap']} "
                "queries have a necessary move envelope above the current cap."
            )
        lines += [
            "",
            "| Arm | Query | Lambda | Cost | Gap to lower bound | Certified regret |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for arm, rows in result["certified_costs"]["arms"].items():
            for row in rows:
                regret = row["certified_regret"]
                regret = "unavailable" if regret is None else str(regret)
                lines.append(
                    f"| {arm} | {row['panel_index']} | {row['lambda']} | {row['cost']} "
                    f"| {row['gap_to_lower_bound']} | {regret} |"
                )
        lines += ["", result["scope"]["certified_costs"]]
    lines += [
        "",
        "## Scope and compute",
        "",
        "Unsolved searches are censored, not assigned zero/infinite cost "
        "or impossible-state labels. "
        "Common-success deltas must be read with new/lost solutions. Raw size strata, paired query "
        "results, constant controls and compute counters are retained in report.json. "
        "The short-u test has only four strand-only extrapolation examples; general Gordian has "
        "only four exact-distance-2 rows. Lifted views reuse the same knots. "
        "Equal simulation/resource caps are not equal compute. "
        "These results do not establish optimal costs, arbitrary-size solution quality "
        "or million-scale RL.",
        "",
        result["scope"]["resource_caveat"],
        "",
        result["scope"]["size_budget_confound"],
        "",
        result["scope"]["timing_observation"],
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    digest = sha(args.manifest)
    state, reports, roots, costs, reuse_pins = verified_inputs(args.manifest)
    panels = {name: sequence.read(path) for name, path in sequence.PANELS.items()}
    result = summarize(reports, panels, roots, costs)
    sources = [
        Path(__file__).resolve(),
        Path("scripts/summarize_clean_paired_evaluation.py"),
        Path("scripts/recover_joint_evaluation_schema.py"),
        Path("scripts/audit_cost_resource_envelope.py"),
        Path("scripts/audit_semantic_cost_floor.py"),
    ]
    result.update(
        status="completed",
        controller_sha256=digest,
        completed_outputs=state["completed"],
        inputs={str(args.manifest): digest, **state["inputs"], **state["gate_pins"], **reuse_pins},
        recovery=state.get("recovery"),
        sources={str(p): sha(p) for p in sources},
    )
    check_pins(result["inputs"])
    args.out.mkdir(parents=True, exist_ok=False)
    atomic(args.out / "report.json", result)
    (args.out / "report.md").write_text(markdown(result))
    print(args.out / "report.md")


if __name__ == "__main__":
    main()
