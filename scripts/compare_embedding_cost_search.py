"""Replay-audited paired costs, preserving failures and charging portfolio work."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from rf_knots.embedding_replay_corpus import verify_record
from rf_knots.evidence import UnknotWitness


def load(path, panel, panel_sha):
    report = json.loads(path.read_text())
    if report["status"] != "completed" or report["panel_sha256"] != panel_sha:
        raise ValueError("require completed reports on the same pinned panel")
    keyed = {}
    for row in report["rows"]:
        key = row["instance"], row["cc_weight"]
        if key in keyed:
            raise ValueError("duplicate search row")
        index = int(row["instance"].removeprefix("short-u-"))
        reference = verify_record(panel["validation"][index])
        if row["solved"]:
            witness = UnknotWitness.from_dict(row["witness"])
            witness.verify()
            caps = row["resource_caps"]
            if (
                witness.start != reference.start
                or witness.crossing_changes > reference.crossing_changes
            ):
                raise ValueError("solution root or crossing budget mismatch")
            states = [witness.start] + [s.after for s in witness.steps]
            if witness.moves > caps["moves"] or any(
                len(s.word) > caps["max_len"] or s.strands > caps["max_strands"] for s in states
            ):
                raise ValueError("solution violates resource caps")
            for weight in (10, 1000):
                if row[f"L{weight}"] != weight * witness.crossing_changes + witness.moves:
                    raise ValueError("stored solution cost differs from replay")
        elif row["witness"] is not None or row["L10"] is not None or row["L1000"] is not None:
            raise ValueError("failed search must retain censored costs")
        keyed[key] = row
    return report, keyed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--auxiliary", type=Path, help="second portfolio component; its work is charged"
    )
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite comparison")
    panel_sha = hashlib.sha256(args.panel.read_bytes()).hexdigest()
    panel = json.loads(args.panel.read_text())
    paths = [args.baseline, args.candidate] + ([args.auxiliary] if args.auxiliary else [])
    loaded = [load(path, panel, panel_sha) for path in paths]
    baseline, base_rows = loaded[0]
    candidates = loaded[1:]
    if any(set(rows) != set(base_rows) for _, rows in candidates):
        raise ValueError("query sets differ")
    if sum(r["config"]["expansions"] for r, _ in candidates) != baseline["config"]["expansions"]:
        raise ValueError("configured total expansion budgets differ")
    totals, paired = {}, []
    for weight in (10, 1000):
        counts = Counter()
        costs = []
        base_time = candidate_time = 0.0
        base_evals = candidate_evals = 0
        for key, base in base_rows.items():
            if key[1] != weight:
                continue
            rows = [lookup[key] for _, lookup in candidates]
            if any(r["resource_caps"] != base["resource_caps"] for r in rows):
                raise ValueError("resource caps differ")
            solved = [r for r in rows if r["solved"]]
            best = min(solved, key=lambda r: r[f"L{weight}"]) if solved else rows[0]
            b, c = base[f"L{weight}"], best[f"L{weight}"]
            outcome = (
                "both_unsolved"
                if b is None and c is None
                else "newly_solved"
                if b is None
                else "lost_solution"
                if c is None
                else "lower_cost"
                if c < b
                else "higher_cost"
                if c > b
                else "equal_cost"
            )
            counts[outcome] += 1
            counts["baseline_solved"] += int(b is not None)
            counts["candidate_solved"] += int(c is not None)
            if b is not None and c is not None:
                costs.append(c - b)
            base_time += base["seconds"]
            candidate_time += sum(r["seconds"] for r in rows)
            base_evals += base["neural_state_evaluations"]
            candidate_evals += sum(r["neural_state_evaluations"] for r in rows)
            paired.append(
                {
                    "instance": key[0],
                    "fingerprint_group": [
                        panel["validation"][int(key[0].removeprefix("short-u-"))][
                            "left_fingerprint"
                        ][0],
                        abs(
                            panel["validation"][int(key[0].removeprefix("short-u-"))][
                                "left_fingerprint"
                            ][1]
                        ),
                    ],
                    "weight": weight,
                    "outcome": outcome,
                    "baseline_cost": b,
                    "candidate_cost": c,
                }
            )
        totals[str(weight)] = {
            **dict(counts),
            "common_solved_count": len(costs),
            "mean_cost_delta_common_only": sum(costs) / len(costs) if costs else None,
            "baseline_seconds": base_time,
            "candidate_seconds": candidate_time,
            "baseline_neural_evaluations": base_evals,
            "candidate_neural_evaluations": candidate_evals,
        }
    result = {
        "status": "passed",
        "panel_sha256": panel_sha,
        "reports": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths
        ],
        "summary": totals,
        "by_fingerprint_group": {
            f"{weight}:{group[0]}:{group[1]}": {
                "count": len(rows),
                "baseline_solved": sum(r["baseline_cost"] is not None for r in rows),
                "candidate_solved": sum(r["candidate_cost"] is not None for r in rows),
                "outcomes": dict(Counter(r["outcome"] for r in rows)),
            }
            for weight in (10, 1000)
            for group in sorted({tuple(r["fingerprint_group"]) for r in paired})
            if (
                rows := [
                    r
                    for r in paired
                    if r["weight"] == weight and tuple(r["fingerprint_group"]) == group
                ]
            )
        },
        "paired": paired,
        "caveat": "Failures remain censored. Shared knot groups are not independent trials. "
        "Portfolio budget and recorded search time include all components; "
        "evaluator/model setup time is excluded. No optimality claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
