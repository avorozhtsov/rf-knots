"""Independent replay and paired summary of the frozen Q-run bootstrap DEV."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from rf_knots.evidence import UnknotWitness

ROOT = Path("artifacts/qrun-bootstrap-matched-dev-v1")
PANEL = Path("artifacts/qrun-bootstrap-dev-panel-v3/panel.json")
ARMS = ("F144_parent", "short_only_256", "mixed_256")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    panel = json.loads(PANEL.read_text())
    check(manifest["status"] == "completed" and manifest["next_task"] == 192
          and len(manifest["shards"]) == 192, "evaluation incomplete")
    contract = manifest["contract"]
    check(contract["config"]["arms"] == list(ARMS)
          and contract["config"]["seed"] == 20271204
          and contract["config"]["simulations"] == 64,
          "evaluation settings differ")
    for name, expected in contract["inputs"].items():
        check(sha(Path(name)) == expected, f"changed pinned input: {name}")
    check(contract["inputs"][str(PANEL)] == sha(PANEL), "panel differs")
    groups = defaultdict(list)
    rows = {}
    total_replayed = 0
    for task, entry in enumerate(manifest["shards"]):
        check(entry["file"] == f"condition-{task:03d}.json", "shard coverage differs")
        path = ROOT / entry["file"]
        check(sha(path) == entry["sha256"], f"shard SHA differs: {path}")
        row = json.loads(path.read_text())
        arm = ARMS[task // 64]
        index = (task % 64) // 2
        weight = (10, 1000)[task % 2]
        instance = panel["rows"][index]["instances"][task % 2]
        result = row["result"]
        check((row["task"], row["arm"], row["row_index"], row["lambda"], row["seed"])
              == (task, arm, index, weight, 20271204), "task identity differs")
        check((result["condition"], result["panel_index"], result["cc_weight"],
               result["search_seed"], result["simulations"])
              == (task, panel["rows"][index]["generator_index"], weight,
                  20271204, 64), "search contract differs")
        check(row["cpu_seconds"] >= 0 and row["wall_seconds"] >= 0,
              "negative duration")
        witness_dict = result["witness"]
        check(result["solved"] == (witness_dict is not None), "solve flag differs")
        if witness_dict is not None:
            witness = UnknotWitness.from_dict(witness_dict)
            witness.verify()
            total_replayed += 1
            check((list(witness.start.word), witness.start.strands)
                  == (instance["word"], instance["strands"]), "wrong start")
            check(witness.crossing_changes <= instance["cap_crossing_changes"]
                  and witness.moves <= instance["cap_moves"], "resource cap exceeded")
            check((result["L10"], result["L1000"])
                  == (10 * witness.crossing_changes + witness.moves,
                      1000 * witness.crossing_changes + witness.moves),
                  "reported solution cost differs")
        else:
            check(result["L10"] is None and result["L1000"] is None,
                  "unsolved search has cost")
        rows[(arm, index, weight)] = row
        band = ("short", "long")[panel["rows"][index]["band"]]
        groups[(arm, band, weight)].append(row)
    summary = {}
    for (arm, band, weight), subset in groups.items():
        solved = [x for x in subset if x["result"]["solved"]]
        witnesses = [UnknotWitness.from_dict(x["result"]["witness"])
                     for x in solved]
        summary[f"{arm}/{band}/L{weight}"] = {
            "solved": len(solved), "total": len(subset),
            "cc_sum_solved": sum(w.crossing_changes for w in witnesses),
            "moves_sum_solved": sum(w.moves for w in witnesses),
            "cost_sum_solved": sum(x["result"][f"L{weight}"] for x in solved),
            "cpu_seconds": sum(x["cpu_seconds"] for x in subset),
            "wall_seconds": sum(x["wall_seconds"] for x in subset),
            "policy_neural_evaluations": sum(x["counts"]["policy_neural_evaluations"]
                                              for x in subset),
        }
    paired = {}
    for arm in ARMS[1:]:
        for band, indices in (("short", range(16)), ("long", range(16, 32))):
            for weight in (10, 1000):
                better = worse = tied = gained = lost = 0
                for index in indices:
                    candidate = rows[(arm, index, weight)]["result"]
                    parent = rows[(ARMS[0], index, weight)]["result"]
                    if candidate["solved"] and not parent["solved"]:
                        gained += 1
                    elif parent["solved"] and not candidate["solved"]:
                        lost += 1
                    elif candidate["solved"]:
                        delta = candidate[f"L{weight}"] - parent[f"L{weight}"]
                        better += delta < 0
                        worse += delta > 0
                        tied += delta == 0
                paired[f"{arm}/{band}/L{weight}"] = {
                    "gained": gained, "lost": lost,
                    "cheaper_when_both_solved": better,
                    "dearer_when_both_solved": worse,
                    "equal_when_both_solved": tied}
    print(json.dumps({"manifest_sha256": sha(ROOT / "manifest.json"),
                      "contract_sha256": manifest["contract_sha256"],
                      "replayed_solutions": total_replayed,
                      "summary": summary, "paired_vs_F144": paired},
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
