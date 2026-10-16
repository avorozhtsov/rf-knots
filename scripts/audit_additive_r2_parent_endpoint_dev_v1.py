"""Independent SHA, full replay, paired cost and promotion-gate audit."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from build_mcts_training_supervision import atomic
from rf_knots.evidence import UnknotWitness

ROOT = Path("artifacts/additive-r2-parent-endpoint-dev-v1")
PANEL = Path("artifacts/additive-r2-fresh-dev-panel-v1/panel.json")
ARMS = ("parent", "endpoint")
SEEDS = (20271016, 20271017)
WEIGHTS = (10, 1000)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("audit output exists")
    manifest_path = ROOT / "manifest.json"
    manifest = read(manifest_path)
    panel = read(PANEL)
    check(manifest["status"] == "completed" and manifest["next_task"] == 256
          and len(manifest["shards"]) == 256, "DEV evaluation incomplete")
    config = manifest["contract"]["config"]
    check(config["arms"] == list(ARMS) and config["seeds"] == list(SEEDS)
          and config["simulations"] == 128 and len(panel["rows"]) == 32,
          "matched DEV contract differs")
    for name, expected in manifest["contract"]["inputs"].items():
        check(sha(Path(name)) == expected, f"pinned input changed: {name}")
    rows = {}
    grouped = defaultdict(list)
    replayed = 0
    for task, entry in enumerate(manifest["shards"]):
        path = ROOT / f"condition-{task:03d}.json"
        check(entry["file"] == path.name and sha(path) == entry["sha256"],
              f"shard SHA/order differs: {task}")
        row = read(path)
        arm_index, within_arm = divmod(task, 128)
        seed_index, within_seed = divmod(within_arm, 64)
        root_index, weight_index = divmod(within_seed, 2)
        arm, seed, weight = ARMS[arm_index], SEEDS[seed_index], WEIGHTS[weight_index]
        instance = panel["rows"][root_index]["instances"][weight_index]
        result = row["result"]
        check((row["task"], row["arm"], row["seed"],
               row["row_index"], row["lambda"])
              == (task, arm, seed, root_index, weight), "task identity differs")
        check((result["condition"], result["panel_index"],
               result["cc_weight"], result["search_seed"],
               result["simulations"])
              == (task, panel["rows"][root_index]["generator_index"],
                  weight, seed, 128), "search identity differs")
        witness_dict = result["witness"]
        check(result["solved"] == (witness_dict is not None), "solve flag differs")
        if witness_dict is not None:
            witness = UnknotWitness.from_dict(witness_dict)
            witness.verify()
            replayed += 1
            check((list(witness.start.word), witness.start.strands)
                  == (instance["word"], instance["strands"]), "wrong start")
            check(witness.crossing_changes <= instance["cap_crossing_changes"]
                  and witness.moves <= instance["cap_moves"], "cap exceeded")
            check((result["L10"], result["L1000"])
                  == (10 * witness.crossing_changes + witness.moves,
                      1000 * witness.crossing_changes + witness.moves),
                  "objective cost differs")
        else:
            check(result["L10"] is None and result["L1000"] is None,
                  "unsolved condition has cost")
        check(row["cpu_seconds"] >= 0 and row["wall_seconds"] >= 0,
              "negative compute duration")
        band = ("short", "long")[panel["rows"][root_index]["band"]]
        rows[(arm, seed, root_index, weight)] = row
        grouped[(arm, seed, band, weight)].append(row)
    summary = {}
    for (arm, seed, band, weight), subset in grouped.items():
        solved = [row for row in subset if row["result"]["solved"]]
        witnesses = [UnknotWitness.from_dict(row["result"]["witness"])
                     for row in solved]
        summary[f"{arm}/{seed}/{band}/L{weight}"] = {
            "solved_attempts": len(solved), "total_attempts": len(subset),
            "cc_sum_solved": sum(w.crossing_changes for w in witnesses),
            "moves_sum_solved": sum(w.moves for w in witnesses),
            "cpu_seconds": sum(row["cpu_seconds"] for row in subset),
            "wall_seconds": sum(row["wall_seconds"] for row in subset),
            "policy_neural_evaluations": sum(
                row["counts"]["policy_neural_evaluations"] for row in subset),
            "tree_embedding_evaluations": sum(
                row["counts"]["tree_embedding_evaluations"] for row in subset),
        }
    distinct = {}
    for arm in ARMS:
        for band, indices in (("short", range(16)),
                              ("long", range(16, 32)), ("all", range(32))):
            for weight in WEIGHTS:
                distinct[f"{arm}/{band}/L{weight}"] = sum(
                    any(rows[(arm, seed, index, weight)]["result"]["solved"]
                        for seed in SEEDS) for index in indices)
    paired = {}
    any_higher_cc = False
    for seed in SEEDS:
        for band, indices in (("short", range(16)),
                              ("long", range(16, 32))):
            for weight in WEIGHTS:
                counts = {"endpoint_gained": 0, "endpoint_lost": 0,
                          "endpoint_cheaper": 0, "endpoint_dearer": 0,
                          "equal_cost": 0, "endpoint_higher_CC_joint_success": 0}
                for index in indices:
                    parent = rows[("parent", seed, index, weight)]["result"]
                    endpoint = rows[("endpoint", seed, index, weight)]["result"]
                    if endpoint["solved"] and not parent["solved"]:
                        counts["endpoint_gained"] += 1
                    elif parent["solved"] and not endpoint["solved"]:
                        counts["endpoint_lost"] += 1
                    elif parent["solved"]:
                        pw = UnknotWitness.from_dict(parent["witness"])
                        ew = UnknotWitness.from_dict(endpoint["witness"])
                        higher = ew.crossing_changes > pw.crossing_changes
                        counts["endpoint_higher_CC_joint_success"] += higher
                        any_higher_cc |= higher
                        delta = endpoint[f"L{weight}"] - parent[f"L{weight}"]
                        counts[("endpoint_cheaper" if delta < 0 else
                                "endpoint_dearer" if delta > 0 else
                                "equal_cost")] += 1
                paired[f"{seed}/{band}/L{weight}"] = counts
    totals = {}
    for arm in ARMS:
        subset = [row for (name, _, _, _), row in rows.items() if name == arm]
        totals[arm] = {
            "cpu_seconds": sum(row["cpu_seconds"] for row in subset),
            "wall_seconds": sum(row["wall_seconds"] for row in subset),
            "policy_neural_evaluations": sum(
                row["counts"]["policy_neural_evaluations"] for row in subset),
            "tree_embedding_evaluations": sum(
                row["counts"]["tree_embedding_evaluations"] for row in subset),
        }
    l1000_gain = (distinct["endpoint/all/L1000"]
                  - distinct["parent/all/L1000"])
    l10_gain = distinct["endpoint/all/L10"] - distinct["parent/all/L10"]
    compute_gap = {
        field: totals["endpoint"][field] / totals["parent"][field] - 1
        for field in ("cpu_seconds", "policy_neural_evaluations")
        if totals["parent"][field] > 0
    }
    decision = {
        "L1000_endpoint_minus_parent_distinct_roots": l1000_gain,
        "L10_endpoint_minus_parent_distinct_roots": l10_gain,
        "endpoint_higher_CC_on_joint_success": any_higher_cc,
        "compute_gap_fraction": compute_gap,
        "outcome": ("compute_match_required" if any(abs(v) > .10
                                                    for v in compute_gap.values())
                    else "promote" if l1000_gain >= 3 and l10_gain >= -2
                    and not any_higher_cc
                    else "tie_or_no_promotion"),
    }
    report = {
        "schema": "rf-knots-additive-R2-parent-endpoint-DEV-audit-v1",
        "manifest_sha256": sha(manifest_path),
        "contract_sha256": manifest["contract_sha256"],
        "auditor_sha256": sha(Path(__file__)),
        "replayed_solutions": replayed,
        "distinct_roots_solved_any_seed": distinct,
        "summary_by_seed": summary, "paired": paired,
        "totals": totals, "decision": decision,
    }
    args.out.mkdir(parents=True)
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": sha(args.out / "report.json"),
                      "replayed_solutions": replayed,
                      "decision": decision}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
