"""Replay/counter proof of semantic cost floors and current budget headroom.

For every prefix: used_moves + q(state) = q(start) + CC + excess,
q=(length+strands-1)/2, excess=2*(INSERT+STABILIZE)+BRAID+COMMUTE+PASS.
Only equality with a certified CC lower bound certifies an optimal cost.
"""

import argparse
from collections import Counter
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, sha
from burau_integer_check import determinant
from evaluate_joint_cost_mcts import PANEL, build_roots, committed_rows
from run_joint_policy_metric_evaluation import read

from rf_knots.embedding_evaluation import verify_evaluation_pair
from rf_knots.evidence import UnknotWitness

JOINT = Path("artifacts/mcts-policy-metric-evaluation-schema-v2/mcts/joint")
DELTA = {
    "REDUCE": (-2, 0),
    "DESTABILIZE": (-1, -1),
    "INSERT": (2, 0),
    "STABILIZE_POS": (1, 1),
    "STABILIZE_NEG": (1, 1),
    "BRAID": (0, 0),
    "COMMUTE": (0, 0),
    "CROSSING_CHANGE": (0, 0),
    "PASS": (0, 0),
}


def potential(state):
    numerator = len(state.word) + state.strands - 1
    if numerator < 0 or numerator % 2:
        raise ValueError("noninteger knot semantic potential")
    return numerator // 2


def witness_identity(witness):
    witness.verify()
    if witness.cyclic_band_generators:
        raise ValueError("this audit contract is for ordinary Artin generators")
    q0 = potential(witness.start)
    counts, before = Counter(), witness.start
    for index, step in enumerate(witness.steps):
        kind = step.action.kind
        change = len(step.after.word) - len(before.word), step.after.strands - before.strands
        if change != DELTA[kind]:
            raise ValueError("semantic move size delta differs")
        counts[kind] += 1
        extra = 2 * sum(counts[k] for k in ("INSERT", "STABILIZE_POS", "STABILIZE_NEG")) + sum(
            counts[k] for k in ("BRAID", "COMMUTE", "PASS")
        )
        if index + 1 + potential(step.after) != q0 + counts["CROSSING_CHANGE"] + extra:
            raise ValueError("prefix semantic cost identity failed")
        before = step.after
    extra = witness.moves - q0 - witness.crossing_changes
    if extra < 0 or potential(before) != 0:
        raise ValueError("invalid full-witness cost identity")
    return {
        "q0": q0,
        "cc": witness.crossing_changes,
        "moves": witness.moves,
        "excess": extra,
        "action_counts": dict(counts),
        "prefixes_checked": witness.moves,
    }


def reference_bounds(identity, lower_cc, cap_moves):
    if not isinstance(lower_cc, int) or not 0 <= lower_cc <= identity["cc"]:
        raise ValueError("invalid certified CC lower bound")
    if cap_moves < identity["moves"]:
        raise ValueError("reference does not fit search")
    result = {}
    for weight in (10, 1000):
        cost = weight * identity["cc"] + identity["moves"]
        floor = identity["q0"] + (weight + 1) * lower_cc
        envelope = cost - weight * lower_cc
        result[str(weight)] = {
            "reference_cost": cost,
            "certified_lower_cost": floor,
            "reference_certifiably_optimal": cost == floor,
            "necessary_move_envelope_for_cost_at_most_reference": envelope,
            "current_move_cap": cap_moves,
            "envelope_beyond_current_cap": max(0, envelope - cap_moves),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite cost audit")
    data_audit_path = PANEL.parent / "independence-audit.json"
    search_audit_path = JOINT / "replay-audit.json"
    data_audit, search_audit = read(data_audit_path), read(search_audit_path)
    if data_audit["status"] != "passed" or search_audit["status"] != "passed":
        raise ValueError("require passed panel and joint replay audits")
    if data_audit["inputs"].get(str(PANEL)) != sha(PANEL) or search_audit["inputs"].get(
        str(JOINT / "manifest.json")
    ) != sha(JOINT / "manifest.json"):
        raise ValueError("prerequisite audit belongs to different inputs")
    paths = [
        Path(__file__).resolve(),
        data_audit_path,
        search_audit_path,
        Path("research/63-semantic-cost-floor-and-budget-audit.md"),
        Path("scripts/burau_integer_check.py"),
        Path("scripts/evaluate_joint_cost_mcts.py"),
        Path("scripts/build_mcts_training_supervision.py"),
        *Path("src/rf_knots").glob("*.py"),
    ]
    pins = dict(data_audit["inputs"])
    merge_pins(pins, search_audit["inputs"])
    merge_pins(pins, {str(p): sha(p) for p in paths})
    check_pins(pins)
    panel = read(PANEL)
    roots = build_roots(panel)
    if roots != read(JOINT / "roots.json"):
        raise ValueError("search roots differ from panel")
    rows = []
    for index, record in enumerate(panel["validation"]):
        witness = verify_evaluation_pair(record)
        if record["pair_prefix_length"] != witness.moves:
            raise ValueError("require complete explicit-unknot reference, not a pair prefix")
        independent_det = determinant(witness.start.word, witness.start.strands)
        if independent_det != record["left_fingerprint"][0]:
            raise ValueError("independent determinant differs")
        lower = max(abs(record["left_fingerprint"][1]) // 2, int(independent_det != 1))
        identity = witness_identity(witness)
        if identity["cc"] != record["interval"]["upper"]:
            raise ValueError("reference upper bound differs")
        root = roots[2 * index]
        bounds = reference_bounds(identity, lower, root["instance"]["cap_moves"])
        for weight in (10, 1000):
            if bounds[str(weight)]["reference_cost"] != root[f"reference_L{weight}"]:
                raise ValueError("search reference cost differs")
        rows.append(
            {
                "panel_index": index,
                "stratum": root["stratum"],
                "fingerprint": record["left_fingerprint"],
                "lower_cc": lower,
                "independent_determinant": independent_det,
                "identity": identity,
                "bounds": bounds,
            }
        )
        if (index + 1) % 8 == 0:
            print({"references_replayed": index + 1}, flush=True)
    solutions = []
    manifest = read(JOINT / "manifest.json")
    if manifest["status"] != "completed":
        raise ValueError("require completed joint search")
    for result in committed_rows(manifest, JOINT):
        if not result["solved"]:
            continue
        witness = UnknotWitness.from_dict(result["witness"])
        identity = witness_identity(witness)
        index, weight = result["panel_index"], result["cc_weight"]
        reference = rows[index]
        root = roots[result["condition"]]
        if (witness.start.word, witness.start.strands) != (
            tuple(root["instance"]["word"]),
            root["instance"]["strands"],
        ) or identity["cc"] < reference["lower_cc"]:
            raise ValueError("solution differs from audited root/lower bound")
        cost = weight * identity["cc"] + identity["moves"]
        bound = reference["bounds"][str(weight)]
        if cost != result[f"L{weight}"] or cost < bound["certified_lower_cost"]:
            raise ValueError("solution violates certified cost bound")
        solutions.append(
            {
                "condition": result["condition"],
                "panel_index": index,
                "lambda": weight,
                "identity": identity,
                "cost": cost,
                "gap_to_lower_bound": cost - bound["certified_lower_cost"],
                "certified_regret": cost - bound["reference_cost"]
                if bound["reference_certifiably_optimal"]
                else None,
            }
        )
    summary = {
        "queries": len(rows),
        "reference_excess_zero": sum(r["identity"]["excess"] == 0 for r in rows),
        "reference_optima": {
            str(w): sum(r["bounds"][str(w)]["reference_certifiably_optimal"] for r in rows)
            for w in (10, 1000)
        },
        "move_envelope_exceeds_current_cap": {
            str(w): sum(r["bounds"][str(w)]["envelope_beyond_current_cap"] > 0 for r in rows)
            for w in (10, 1000)
        },
        "joint_solutions_checked": len(solutions),
    }
    check_pins(pins)
    args.out.mkdir(parents=True, exist_ok=False)
    atomic(
        args.out / "report.json",
        {
            "schema": "rf-knots-semantic-cost-floor-audit-v1",
            "status": "passed",
            "inputs": pins,
            "summary": summary,
            "references": rows,
            "joint_solutions": solutions,
            "scope": (
                "All reference witnesses, prefix identities, and joint solutions replayed; "
                "signature bounds recomputed and determinants independently recomputed. "
                "Optima apply only to this semantic action cost and exact representation. "
                "A larger necessary move envelope does not prove a missing cheaper path exists. "
                "No training/search changes or failure labels."
            ),
        },
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
