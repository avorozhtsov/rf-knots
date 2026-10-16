"""Independent SHA, replay and paired audit of equal-cost pilot DEV search."""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from rf_knots.evidence import UnknotWitness

ROOT = Path("artifacts/qrun-equal-cost-matched-dev-v1")
PANEL = Path("artifacts/qrun-equal-cost-dev-panel-v1/panel.json")
ARMS = ("curriculum1024", "reference_A320", "diversity_B320")
SEEDS = (20271209, 20271210)


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
    check(manifest["status"] == "completed" and manifest["next_task"] == 384
          and len(manifest["shards"]) == 384, "evaluation incomplete")
    contract = manifest["contract"]
    check(contract["config"]["arms"] == list(ARMS)
          and contract["config"]["seeds"] == list(SEEDS)
          and contract["config"]["simulations"] == 64,
          "evaluation settings differ")
    for name, expected in contract["inputs"].items():
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
        arm, seed, weight = ARMS[arm_index], SEEDS[seed_index], (10, 1000)[weight_index]
        instance = panel["rows"][root_index]["instances"][weight_index]
        result = row["result"]
        check((row["task"], row["arm"], row["seed"], row["row_index"], row["lambda"])
              == (task, arm, seed, root_index, weight), "task identity differs")
        check((result["condition"], result["panel_index"], result["cc_weight"],
               result["search_seed"], result["simulations"])
              == (task, panel["rows"][root_index]["generator_index"], weight,
                  seed, 64), "search identity differs")
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
            "solved": len(solved), "total": len(subset),
            "cc_sum_solved": sum(w.crossing_changes for w in witnesses),
            "moves_sum_solved": sum(w.moves for w in witnesses),
            "cpu_seconds": sum(row["cpu_seconds"] for row in subset),
            "wall_seconds": sum(row["wall_seconds"] for row in subset),
            "policy_neural_evaluations": sum(
                row["counts"]["policy_neural_evaluations"] for row in subset)}
    paired = {}
    for baseline, candidate in (("reference_A320", "diversity_B320"),
                                ("curriculum1024", "reference_A320"),
                                ("curriculum1024", "diversity_B320")):
        for seed in SEEDS:
            for band, indices in (("short", range(16)), ("long", range(16, 32))):
                for weight in (10, 1000):
                    counts = {"gained": 0, "lost": 0, "cheaper": 0,
                              "dearer": 0, "equal": 0}
                    for index in indices:
                        result = rows[(candidate, seed, index, weight)]["result"]
                        base = rows[(baseline, seed, index, weight)]["result"]
                        if result["solved"] and not base["solved"]:
                            counts["gained"] += 1
                        elif base["solved"] and not result["solved"]:
                            counts["lost"] += 1
                        elif result["solved"]:
                            delta = result[f"L{weight}"] - base[f"L{weight}"]
                            counts[("cheaper" if delta < 0 else
                                    "dearer" if delta > 0 else "equal")] += 1
                    paired[f"{candidate}_vs_{baseline}/{seed}/{band}/L{weight}"] = counts
    distinct = {}
    for arm in ARMS:
        for band, indices in (("short", range(16)), ("long", range(16, 32))):
            for weight in (10, 1000):
                distinct[f"{arm}/{band}/L{weight}"] = sum(
                    any(rows[(arm, seed, index, weight)]["result"]["solved"]
                        for seed in SEEDS) for index in indices)
    report = {"manifest_sha256": sha(ROOT / "manifest.json"),
              "contract_sha256": manifest["contract_sha256"],
              "replayed_solutions": replayed,
              "distinct_roots_solved_any_seed": distinct,
              "summary_by_seed": summary, "paired": paired}
    if args.out is not None:
        if args.out.exists():
            raise ValueError("audit report output already exists")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        os.replace(temporary, args.out)
    print(json.dumps({"manifest_sha256": report["manifest_sha256"],
                      "replayed_solutions": replayed,
                      "report_path": str(args.out) if args.out else None}, sort_keys=True))


if __name__ == "__main__":
    main()
