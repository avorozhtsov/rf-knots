"""Independent SHA, replay, paired-cost and decision audit of R2/R4 replication."""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from rf_knots.evidence import UnknotWitness


ROOT = Path("artifacts/f-r2-r4-bounded-replication-v1")
PANEL = Path("artifacts/f-r2-r4-replication-panel-v1/panel.json")
ARMS = ("R2", "R4")
SEEDS = (20271010, 20271011)
WEIGHTS = (10, 1000)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    panel = json.loads(PANEL.read_text())
    check(manifest["status"] == "completed" and manifest["next_task"] == 256
          and len(manifest["shards"]) == 256, "replication incomplete")
    config = manifest["contract"]["config"]
    check(config["arms"] == list(ARMS) and config["seeds"] == list(SEEDS)
          and config["simulations"] == 128 and len(panel["rows"]) == 32,
          "matched contract differs")
    for name, expected in manifest["contract"]["inputs"].items():
        check(sha(Path(name)) == expected, f"changed pinned input: {name}")
    rows = {}
    grouped = defaultdict(list)
    replayed = 0
    for task, entry in enumerate(manifest["shards"]):
        check(entry["file"] == f"condition-{task:03d}.json", "shard order differs")
        path = ROOT / entry["file"]
        check(sha(path) == entry["sha256"], f"shard SHA differs: {path}")
        row = json.loads(path.read_text())
        arm_index, within_arm = divmod(task, 128)
        seed_index, within_seed = divmod(within_arm, 64)
        root_index, weight_index = divmod(within_seed, 2)
        arm, seed, weight = ARMS[arm_index], SEEDS[seed_index], WEIGHTS[weight_index]
        instance = panel["rows"][root_index]["instances"][weight_index]
        result = row["result"]
        check((row["task"], row["arm"], row["seed"], row["row_index"], row["lambda"])
              == (task, arm, seed, root_index, weight), "task identity differs")
        check((result["condition"], result["panel_index"], result["cc_weight"],
               result["search_seed"], result["simulations"])
              == (task, panel["rows"][root_index]["generator_index"], weight,
                  seed, 128), "search identity differs")
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
                  "cost differs")
        else:
            check(result["L10"] is None and result["L1000"] is None,
                  "unsolved row has cost")
        check(row["cpu_seconds"] >= 0 and row["wall_seconds"] >= 0,
              "negative duration")
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
                row["counts"]["policy_neural_evaluations"] for row in subset)}
    distinct = {}
    for arm in ARMS:
        for band, indices in (("short", range(16)), ("long", range(16, 32)),
                              ("all", range(32))):
            for weight in WEIGHTS:
                distinct[f"{arm}/{band}/L{weight}"] = sum(
                    any(rows[(arm, seed, index, weight)]["result"]["solved"]
                        for seed in SEEDS) for index in indices)
    paired = {}
    for seed in SEEDS:
        for band, indices in (("short", range(16)), ("long", range(16, 32))):
            for weight in WEIGHTS:
                counts = {"R4_gained": 0, "R4_lost": 0, "R4_cheaper": 0,
                          "R4_dearer": 0, "equal_cost": 0,
                          "R4_higher_CC_joint_success": 0}
                for index in indices:
                    a = rows[("R2", seed, index, weight)]["result"]
                    b = rows[("R4", seed, index, weight)]["result"]
                    if b["solved"] and not a["solved"]:
                        counts["R4_gained"] += 1
                    elif a["solved"] and not b["solved"]:
                        counts["R4_lost"] += 1
                    elif a["solved"]:
                        aw = UnknotWitness.from_dict(a["witness"])
                        bw = UnknotWitness.from_dict(b["witness"])
                        counts["R4_higher_CC_joint_success"] += (
                            bw.crossing_changes > aw.crossing_changes)
                        delta = b[f"L{weight}"] - a[f"L{weight}"]
                        counts[("R4_cheaper" if delta < 0 else
                                "R4_dearer" if delta > 0 else "equal_cost")] += 1
                paired[f"{seed}/{band}/L{weight}"] = counts
    totals = {}
    for arm in ARMS:
        subset = [row for (name, _, _, _), row in rows.items() if name == arm]
        totals[arm] = {"cpu_seconds": sum(row["cpu_seconds"] for row in subset),
                       "wall_seconds": sum(row["wall_seconds"] for row in subset),
                       "policy_neural_evaluations": sum(
                           row["counts"]["policy_neural_evaluations"]
                           for row in subset)}
    l1000_gap = distinct["R4/all/L1000"] - distinct["R2/all/L1000"]
    l10_gap = distinct["R4/all/L10"] - distinct["R2/all/L10"]
    compute_gap = {key: totals["R4"][key] / totals["R2"][key] - 1
                   for key in ("cpu_seconds", "policy_neural_evaluations")}
    decision = {"L1000_R4_minus_R2_distinct_roots": l1000_gap,
                "L10_R4_minus_R2_distinct_roots": l10_gap,
                "compute_gap_R4_fraction": compute_gap,
                "outcome": "tie" if abs(l1000_gap) < 3 else "guardrails_needed"}
    report = {"manifest_sha256": sha(ROOT / "manifest.json"),
              "contract_sha256": manifest["contract_sha256"],
              "replayed_solutions": replayed,
              "distinct_roots_solved_any_seed": distinct,
              "summary_by_seed": summary, "paired": paired,
              "totals": totals, "decision": decision}
    if args.out is not None:
        if args.out.exists():
            raise ValueError("audit report output already exists")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        os.replace(temporary, args.out)
    print(json.dumps({"manifest_sha256": report["manifest_sha256"],
                      "replayed_solutions": replayed,
                      "decision": decision,
                      "report_path": str(args.out) if args.out else None},
                     sort_keys=True))


if __name__ == "__main__":
    main()
