"""Final-only descriptive consumer of protocols 42/44/45; never promotes a model."""

import argparse
import json
import math
from pathlib import Path

import run_diverse_metric_sequence as sequence
import summarize_clean_paired_evaluation as common

read, sha = common.read, common.sha


def verified_reports(path):
    state = read(path)
    expected = sequence.evaluation_jobs(path.parent)
    if state["status"] != "completed" or len(state["completed"]) != 17:
        raise ValueError("require all 17 final jobs; partial scores cannot be summarized")
    if state["jobs"] != expected:
        raise ValueError("job matrix differs from protocol45")
    sequence.check_completed_prefix(state["completed"], expected)
    sequence.verify_pins(state["inputs"])
    if set(state["final_checkpoints"]) != set(sequence.RUNS):
        raise ValueError("missing final models")
    sequence.verify_pins({str(sequence.RUNS[n] / "checkpoint.pt"): h
                          for n, h in state["final_checkpoints"].items()})
    required_audits = {str(p.parent / a) for p in sequence.PANELS.values()
                       for a in ("metric-independence-audit.json", "rollout-independence-audit.json")}
    if set(state["audit_pins"]) != required_audits:
        raise ValueError("missing independence certificates")
    sequence.verify_pins(state["audit_pins"])
    for name in required_audits:
        audit = read(Path(name))
        if audit["status"] != "passed":
            raise ValueError("failed independence certificate")
        sequence.verify_pins(audit["inputs"])
    reports = {}
    for row in state["completed"]:
        p = Path(row["output"])
        if sha(p) != row["sha256"]:
            raise ValueError("completed output changed")
        report = read(p)
        if report["status"] not in ("passed", "completed"):
            raise ValueError("non-final output")
        sequence.verify_pins(report.get("code_sha256", {}))
        reports[row["name"]] = report
    verify_report_contracts(state, reports)
    return state, reports


def check_metric_rows(rows, records, bounds):
    if len(rows) != len(records):
        raise ValueError("metric query coverage differs")
    for i, (row, record) in enumerate(zip(rows, records, strict=True)):
        interval = record["interval"]
        fp, other = record["left_fingerprint"], record["right_fingerprint"]
        lo = max(interval["lower"], int(fp[0] != other[0])) if bounds == "determinant" else interval["lower"]
        if (row["panel_index"] != i or row["left_fingerprint"] != fp
                or row["right_fingerprint"] != other
                or row["fingerprint_group"] != f"{fp[0]}:{abs(fp[1])}"
                or row["lower"] != lo or row["upper"] != interval["upper"]):
            raise ValueError("metric labels or query order differ from pinned panel")
    common.metric_stats(rows)


def verify_report_contracts(state, reports):
    panels = {n: read(p) for n, p in sequence.PANELS.items()}
    for model, run in sequence.RUNS.items():
        for panel, bounds in (("supplement", "original"), ("short-u", "original"), ("short-u", "determinant")):
            r = reports[f"metric-{model}-{panel}-{bounds}"]
            if (r["run"] != str(run) or r["checkpoint_sha256"] != state["final_checkpoints"][model]
                    or r["panel_sha256"] != sha(sequence.PANELS[panel])
                    or r["bounds"] != bounds or r["initial"] or r["normalization"] != "none"):
                raise ValueError("metric evaluated a different model/panel/contract")
            for view in ("validation", "size_shift"):
                check_metric_rows(r[view]["predictions"], panels[panel][view], bounds)
    expected_keys = {(f"short-u-{i}", w) for i in range(len(panels["short-u"]["validation"])) for w in (10, 1000)}
    costs = {}
    for model in (*sequence.RUNS, "length"):
        r = reports[f"cost-{model}"]
        source = "parent" if model == "length" else model
        if (r["checkpoint_sha256"] != state["final_checkpoints"][source]
                or r["run"] != str(sequence.RUNS[source])
                or r["panel_sha256"] != sha(sequence.PANELS["short-u"])
                or r["mode"] != ("length" if model == "length" else "learned")
                or r["config"] != {"expansions": 128, "cc_weights": [10, 1000], "cap_moves": 64,
                                   "layout_tiebreak": True, "all_queries": True, "witness_caps": True}):
            raise ValueError("cost evaluated a different model/panel/contract")
        keyed = {(row["instance"], row["cc_weight"]): row for row in r["rows"]}
        if set(keyed) != expected_keys or len(keyed) != len(r["rows"]):
            raise ValueError("cost query coverage differs")
        costs[model] = keyed
    for baseline, candidate in (("parent", "candidate"), ("parent", "control"), ("control", "candidate"), ("length", "candidate")):
        name = f"cost-{candidate}-vs-{baseline}"
        r = reports[name]
        job_outputs = {j["name"]: j["output"] for j in state["jobs"]}
        expected_sources = [{"path": job_outputs[f"cost-{n}"], "sha256": sha(Path(job_outputs[f"cost-{n}"]))}
                            for n in (baseline, candidate)]
        if r["reports"] != expected_sources or r["panel_sha256"] != sha(sequence.PANELS["short-u"]):
            raise ValueError("comparison source audit differs")
        keys = {(row["instance"], row["weight"]) for row in r["paired"]}
        if keys != expected_keys or len(keys) != len(r["paired"]):
            raise ValueError("comparison query coverage differs")
        for row in r["paired"]:
            key = row["instance"], row["weight"]
            for role, model in (("baseline", baseline), ("candidate", candidate)):
                cost = row[f"{role}_cost"]
                if cost != costs[model][key][f"L{row['weight']}"] or (cost is not None and (not math.isfinite(cost) or cost < 0)):
                    raise ValueError("comparison cost differs from audited search")


def metric_with_constants(rows):
    result = common.metric_stats(rows)
    # Apply the SAME equal-group weighting to the neural prediction and constants.
    result["constants"] = {}
    for value in (0, 1, 2):
        controls = [{**r, "prediction": value, "violation": max(r["lower"] - value, value - r["upper"], 0)} for r in rows]
        result["constants"][str(value)] = common.metric_stats(controls)
    return result


def summarize(reports):
    result = {"metric": {}, "search_comparisons": {}, "portfolio_comparisons": {}}
    for name, r in reports.items():
        if name.startswith("metric-"):
            result["metric"][name] = {"checkpoint_sha256": r["checkpoint_sha256"], "bounds": r["bounds"],
                **{v: metric_with_constants(r[v]["predictions"]) for v in ("validation", "size_shift")}}
        elif name.startswith("cost-") and "-vs-" in name:
            result["search_comparisons"][name] = {
                str(w): {**common.paired_stats([p for p in r["paired"] if p["weight"] == w]),
                         "compute_and_audited_summary": r["summary"][str(w)]} for w in (10, 1000)}
    if len(result["metric"]) != 9 or len(result["search_comparisons"]) != 4:
        raise ValueError("unexpected summary coverage")
    result["scope"] = {"original_general_panel": "INADEQUATE; not scored",
        "supplement": "protocol44 common-block family; does not repair the failed original requirement",
        "size_shift": "zero-CC lifted stability views, not harder-knot transfer",
        "promotion": "not_decided", "mcts_benefit": "not tested", "arbitrary_size_quality": "not established"}
    return result


def markdown(result):
    body = common.markdown(result).replace("# Clean paired evaluation: complete descriptive results",
                                          "# Diverse metric iteration: complete descriptive results")
    lines = [body, "## Same-weight constant controls", "",
             "| Evaluation | View | Neural equal-group error | Constant 0 | Constant 1 | Constant 2 |",
             "|---|---|---:|---:|---:|---:|"]
    for name, r in result["metric"].items():
        for view in ("validation", "size_shift"):
            m = r[view]
            values = [m["equal_group_mean_interval_violation"],
                      *(m["constants"][str(v)]["equal_group_mean_interval_violation"] for v in (0, 1, 2))]
            lines.append(f"| {name} | {view} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", "## Required scope disclosure", "",
              "The original general panel FAILED adequacy and remains unscored. The prospectively declared "
              "protocol44 common-block supplement is a DIFFERENT family; it does not repair that failed requirement. "
              "Its five exact-distance-2 rows are a small sample. Lifted views test representation stability, "
              "not genuinely harder long knots. No MCTS benefit, arbitrary-size solution quality or automatic "
              "scientific promotion follows from this report.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    digest = sha(args.manifest)
    state, reports = verified_reports(args.manifest)
    result = summarize(reports)
    result.update(status="completed", controller_sha256=digest, final_checkpoints=state["final_checkpoints"],
                  completed_outputs=state["completed"], audit_pins=state["audit_pins"], promotion="not_decided",
                  summarizer_sources={str(f): sha(f) for f in (Path(__file__), Path(common.__file__), Path(sequence.__file__))})
    if sha(args.manifest) != digest:
        raise ValueError("controller changed during summary")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "report.md").write_text(markdown(result))
    print(args.out / "report.md")


if __name__ == "__main__":
    main()
