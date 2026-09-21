"""Read-only cost-bound diagnostic, not a new search or an optimality certificate.

Current semantic moves remove at most two letters or one strand. Crossing
changes remove neither and are included in the total semantic move count.
Thus any solution has m >= c + q, q=max(ceil(L/2), N-1). A strict improvement
over an integer witness cost B=lambda*c_ref+m_ref must satisfy
(lambda+1)*c+q <= B-1. This necessary condition does not assert a path exists.
"""

import argparse
from collections import Counter
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins
from build_mcts_training_supervision import atomic, sha
from evaluate_joint_cost_mcts import PANEL, build_roots
from run_joint_policy_metric_evaluation import read


def envelope(root, weight):
    if weight not in (10, 1000) or root["instance"]["cc_weight"] != weight:
        raise ValueError("require matching frozen cost condition")
    instance = root["instance"]
    length, strands = len(instance["word"]), instance["strands"]
    cap = instance["cap_crossing_changes"]
    if length < 0 or strands < 1 or not isinstance(cap, int) or cap < 0:
        raise ValueError("invalid root resources")
    non_cc_lower = max((length + 1) // 2, strands - 1)
    reference_cost = root[f"reference_L{weight}"]
    if not isinstance(reference_cost, int) or reference_cost < 0:
        raise ValueError("require nonnegative integer witness cost")
    if reference_cost < (weight + 1) * cap + non_cc_lower:
        raise ValueError("reference witness violates semantic move lower bound")
    maximum_cc = (reference_cost - 1 - non_cc_lower) // (weight + 1)
    first_excluded_lower_cost = (weight + 1) * (cap + 1) + non_cc_lower
    return {
        "condition": root["condition"],
        "panel_index": root["panel_index"],
        "stratum": root["stratum"],
        "lambda": weight,
        "length": length,
        "strands": strands,
        "reference_cost": reference_cost,
        "current_cc_cap": cap,
        "non_cc_moves_lower_bound": non_cc_lower,
        "maximum_cc_for_strict_improvement_by_necessary_bound": maximum_cc,
        "first_excluded_cc_path_cost_lower_bound": first_excluded_lower_cost,
        "extra_cc_improvement_not_ruled_out": first_excluded_lower_cost < reference_cost,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit_path = PANEL.parent / "independence-audit.json"
    audit = read(audit_path)
    if audit["status"] != "passed" or audit["inputs"].get(str(PANEL)) != sha(PANEL):
        raise ValueError("require passed independence audit of the exact panel")
    check_pins(audit["inputs"])
    source_paths = [
        Path(__file__).resolve(),
        Path("scripts/evaluate_joint_cost_mcts.py"),
        Path("scripts/evaluate_embedding_cost_search.py"),
        Path("scripts/audit_joint_policy_metric_panel.py"),
        Path("scripts/build_mcts_training_supervision.py"),
        Path("scripts/run_joint_policy_metric_evaluation.py"),
        *Path("src/rf_knots").glob("*.py"),
    ]
    inputs = {
        **audit["inputs"],
        str(PANEL): sha(PANEL),
        str(audit_path): sha(audit_path),
        **{str(p): sha(p) for p in source_paths},
    }
    roots = build_roots(read(PANEL))  # Replay witnesses and recompute panel fingerprints.
    rows = [envelope(root, root["instance"]["cc_weight"]) for root in roots]
    summaries = {}
    for weight in (10, 1000):
        selected = [r for r in rows if r["lambda"] == weight]
        summaries[str(weight)] = {
            "conditions": len(selected),
            "extra_cc_improvement_not_ruled_out": sum(
                r["extra_cc_improvement_not_ruled_out"] for r in selected
            ),
            "strata_with_extra_cc_not_ruled_out": dict(
                Counter(r["stratum"] for r in selected if r["extra_cc_improvement_not_ruled_out"])
            ),
        }
    report = {
        "status": "completed",
        "inputs": inputs,
        "bound": "m >= c + max(ceil(L/2), N-1); strict improvement requires (lambda+1)c+q <= B-1",
        "summary": summaries,
        "rows": rows,
        "scope": {
            "not_ruled_out": "a necessary arithmetic condition, not existence of a cheaper path",
            "ruled_out": "extra CC cannot strictly improve this witness cost under these semantics",
            "other_caps": (
                "move, growth, horizon and candidate limits can still exclude improvements"
            ),
            "execution": "no search/training changes, no model selection, no new negative labels",
        },
    }
    check_pins(inputs)
    args.out.mkdir(parents=True, exist_ok=False)
    atomic(args.out / "report.json", report)
    print(summaries, flush=True)


if __name__ == "__main__":
    main()
