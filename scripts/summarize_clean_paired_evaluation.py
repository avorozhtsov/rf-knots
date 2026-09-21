"""Describe the complete frozen evaluation without selecting a winning checkpoint.

This is a read-only consumer of the controller's artifacts. Incomplete or changed
inputs fail closed; in particular, partial model scores cannot produce a summary.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verified_reports(manifest_path):
    state = read(manifest_path)
    jobs, completed = state["jobs"], state["completed"]
    if state["status"] != "completed" or len(jobs) != 42 or len(completed) != len(jobs):
        raise ValueError("require all 42 frozen evaluation jobs, not partial scores")
    if len({j["name"] for j in jobs}) != 42 or len({j["output"] for j in jobs}) != 42:
        raise ValueError("duplicate job names or outputs")
    for name, digest in state["inputs"].items():
        if sha(Path(name)) != digest:
            raise ValueError(f"evaluation input changed: {name}")
    reports = {}
    for job, row in zip(jobs, completed, strict=True):
        if (job["name"], job["output"]) != (row["name"], row["output"]):
            raise ValueError("completed jobs differ from frozen order")
        path = Path(row["output"])
        if sha(path) != row["sha256"]:
            raise ValueError(f"completed output changed: {path}")
        report = read(path)
        if report.get("status") not in ("completed", "passed"):
            raise ValueError("non-final child report")
        reports[row["name"]] = report
    return state, reports


def metric_stats(rows):
    if not rows:
        raise ValueError("empty metric predictions")
    groups = defaultdict(list)
    exact = defaultdict(list)
    for row in rows:
        prediction, lower, upper = row["prediction"], row["lower"], row["upper"]
        if not all(math.isfinite(x) for x in (prediction,lower,upper,row["violation"])) or lower > upper:
            raise ValueError("invalid metric prediction or bounds")
        value = max(lower-prediction,prediction-upper,0.0)
        if not math.isclose(value,row["violation"],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError("stored metric error differs from prediction")
        groups[row["fingerprint_group"]].append(value)
        if row["lower"] == row["upper"]:
            exact[str(row["lower"])].append(value)
    return {
        "rows": len(rows), "groups": len(groups),
        "mean_interval_violation": sum(r["violation"] for r in rows) / len(rows),
        "equal_group_mean_interval_violation": sum(sum(v)/len(v) for v in groups.values()) / len(groups),
        "exact": {k: {"count": len(v), "mae": sum(v)/len(v)} for k, v in sorted(exact.items())},
    }


def paired_stats(rows):
    if not rows:
        raise ValueError("empty paired observations")
    counts, deltas = Counter(), []
    for row in rows:
        a, b = row["baseline_cost"], row["candidate_cost"]
        outcome = ("both_unsolved" if a is None and b is None else "newly_solved" if a is None
                   else "lost_solution" if b is None else "lower_cost" if b < a
                   else "higher_cost" if b > a else "equal_cost")
        counts[outcome] += 1
        if a is not None and b is not None:
            deltas.append(b-a)
    return {"observations": len(rows), "outcomes": dict(counts),
            "baseline_solved": sum(r["baseline_cost"] is not None for r in rows),
            "candidate_solved": sum(r["candidate_cost"] is not None for r in rows),
            "common_solved": len(deltas),
            "mean_cost_delta_common_only": sum(deltas)/len(deltas) if deltas else None}


def summarize(reports):
    result = {"metric": {}, "search_comparisons": {}, "portfolio_comparisons": {}}
    for name, report in reports.items():
        if name.startswith("metric-"):
            result["metric"][name] = {
                "checkpoint_sha256": report["checkpoint_sha256"],
                "bounds": report["bounds"], "constant_baselines": report["constant_baselines"],
                **{view: metric_stats(report[view]["predictions"]) for view in ("validation", "size_shift")},
            }
        elif name.startswith("cost-") and "-vs-" in name:
            result["search_comparisons"][name] = {
                str(weight): {**paired_stats([r for r in report["paired"] if r["weight"] == weight]),
                              "compute_and_audited_summary": report["summary"][str(weight)]}
                for weight in (10,1000)
            }
        elif name.startswith("mcts-") and "-vs-" in name:
            result["search_comparisons"][name] = {
                "comparison_kind": report["comparison_kind"], "cc_weight": report["cc_weight"],
                **paired_stats(report["paired"]), "compute": report["compute"],
            }
        elif name.startswith("portfolio-vs-parent-"):
            weight = int(report["cc_weight"])
            pairs = [{"baseline_cost": r[f"mcts_L{weight}"], "candidate_cost": r[f"rollout_L{weight}"]}
                     for r in report["rows"]]
            result["portfolio_comparisons"][name] = {
                **paired_stats(pairs), "cc_weight": weight, "mcts_seed": report["mcts_seed"],
                "budget_per_root": report["budget_per_root"],
                "rollout_neural_evaluations": sum(r["neural_state_evaluations"] for r in report["rows"]),
                "partial_episodes_censored": sum(r["partial_episode_censored"] for r in report["rows"]),
            }
    if tuple(map(len, result.values())) != (9,16,2):
        raise ValueError("unexpected evaluation summary coverage")
    return result


def verify_model_pins(state,reports):
    if set(state["final_checkpoints"]) != {"parent","value","control"}:
        raise ValueError("missing final model pins")
    for model,digest in state["final_checkpoints"].items():
        selected = [r for name,r in reports.items() if name.startswith(f"metric-{model}-")]
        if len(selected) != 3:
            raise ValueError("missing model metric evaluations")
        for report in selected:
            if report["checkpoint_sha256"] != digest or sha(Path(report["run"])/"checkpoint.pt") != digest:
                raise ValueError("evaluated or current checkpoint differs from final pin")


def markdown(result):
    lines = ["# Clean paired evaluation: complete descriptive results", "",
             "No automatic promotion. Original and lifted views, bounds and cost conditions remain separate.", "",
             "## Distance estimates", "",
             "| Model / panel / bounds | View | Rows / groups | Interval error | Equal-group error | Exact-class MAE (count) |",
             "|---|---|---:|---:|---:|---|"]
    for name, report in result["metric"].items():
        for view in ("validation", "size_shift"):
            r = report[view]
            exact = "; ".join(f"{k}: {v['mae']:.6f} ({v['count']})" for k,v in r["exact"].items())
            lines.append(f"| {name} | {view} | {r['rows']} / {r['groups']} | {r['mean_interval_violation']:.6f} | "
                         f"{r['equal_group_mean_interval_violation']:.6f} | {exact} |")
    lines += ["", "Constant 0/1/2 controls and all compute measurements are retained in report.json.", "",
              "## Paired search results", "",
              "| Candidate vs baseline | CC cost | Solved: baseline → candidate | New / lost | Lower / higher / equal cost | Common-only mean cost delta |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, report in {**result["search_comparisons"], **result["portfolio_comparisons"]}.items():
        rows = [(k,v) for k,v in report.items()] if name.startswith("cost-") else [(str(report["cc_weight"]),report)]
        for weight, r in rows:
            c = r["outcomes"]
            delta = r["mean_cost_delta_common_only"]
            delta = "unavailable" if delta is None else f"{delta:.3f}"
            lines.append(f"| {name} | {weight} | {r['baseline_solved']} → {r['candidate_solved']} / {r['observations']} | "
                         f"{c.get('newly_solved',0)} / {c.get('lost_solution',0)} | "
                         f"{c.get('lower_cost',0)} / {c.get('higher_cost',0)} / {c.get('equal_cost',0)} | {delta} |")
    lines += ["", "## Interpretation limits", "",
              "Unsolved searches are censored, not assigned infinite costs or impossible-state labels. "
              "Common-only cost deltas must be read alongside new/lost solutions. Repeated seeds, "
              "views and cost conditions are correlated, not independent trials. Fingerprint groups "
              "are not knot identities. Different leaf modes and changed weights are disclosed in JSON.", "",
              "These synthetic independent panels test this frozen iteration. They do not prove "
              "optimal unknotting costs, arbitrary-size solution quality or million-node RL. "
              "Any next training experiment needs a new protocol; these results must not select "
              "an intermediate checkpoint retrospectively.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_sha = sha(args.manifest)
    state, reports = verified_reports(args.manifest)
    verify_model_pins(state,reports)
    result = summarize(reports)
    result.update(status="completed", controller_sha256=manifest_sha,
                  final_checkpoints=state["final_checkpoints"], completed_outputs=state["completed"],
                  summarizer_sha256=sha(Path(__file__)), promotion="not_decided")
    if sha(args.manifest) != manifest_sha:
        raise ValueError("controller changed during summary")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(json.dumps(result, indent=2)+"\n")
    (args.out / "report.md").write_text(markdown(result))
    print(args.out / "report.md")


if __name__ == "__main__":
    main()
