"""Independent SHA/replay/paired-result audit for protocol272 search."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_tournament_long_dev_panel_v1 import excluded_groups, group
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness


PANEL = Path("artifacts/f-policy-confirmation-panel-v1/panel.json")
ARMS = ("A", "F")
SEEDS = (20271119, 20271120)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.data / "manifest.json")
    panel = read(PANEL)
    if (manifest["status"] != "completed" or manifest["next_task"] != 256
            or len(manifest["shards"]) != 256
            or manifest["contract"]["inputs"][str(PANEL)] != sha(PANEL)
            or len(panel["rows"]) != 32):
        raise ValueError("completed contract/panel coverage differs")
    old_panel = read(Path("artifacts/mcts-tournament-long-dev-panel-v1/panel.json"))
    excluded = excluded_groups() | {tuple(row["query_group"]) for row in old_panel["rows"]}
    groups = set()
    for index, row in enumerate(panel["rows"]):
        witness = UnknotWitness.from_dict(row["witness"])
        witness.verify()
        fingerprint = invariants(witness.start.word, witness.start.strands)
        identity = group(fingerprint)
        if (owner(fingerprint) != "validation" or identity in excluded or identity in groups
                or list(identity) != row["query_group"]
                or not 60 <= len(witness.start.word) <= 150
                or row["word_length"] != len(witness.start.word)
                or (row["reference_cc"], row["reference_moves"])
                != (witness.crossing_changes, witness.moves)):
            raise ValueError(f"panel witness/ownership differs: {index}")
        groups.add(identity)
    if sum(row["reference_cc"] == 2 for row in panel["rows"]) < 8:
        raise ValueError("panel lacks fixed CC2 quota")
    for path, expected in manifest["contract"]["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"contract input changed: {path}")
    results = {}
    totals = defaultdict(Counter)
    for position, entry in enumerate(manifest["shards"]):
        name = f"condition-{position:03d}.json"
        if entry["file"] != name or sha(args.data / name) != entry["sha256"]:
            raise ValueError(f"shard identity differs: {position}")
        payload = read(args.data / name)
        arm_index, within = divmod(position, 128)
        row_index, within = divmod(within, 4)
        weight_index, seed_index = divmod(within, 2)
        arm, weight, seed = ARMS[arm_index], (10, 1000)[weight_index], SEEDS[seed_index]
        row = panel["rows"][row_index]
        instance = row["instances"][weight_index]
        outcome = payload["result"]
        if ((payload["task"], payload["arm"], payload["row_index"], payload["lambda"], payload["seed"])
                != (position, arm, row_index, weight, seed)
                or outcome["condition"] != position
                or outcome["panel_index"] != row["generator_index"]
                or outcome["cc_weight"] != weight
                or outcome["simulations"] > 128
                or bool(outcome["witness"]) != bool(outcome["solved"])):
            raise ValueError(f"task/result mismatch: {position}")
        witness = None
        if outcome["witness"] is not None:
            witness = UnknotWitness.from_dict(outcome["witness"])
            witness.verify()
            if (list(witness.start.word) != instance["word"]
                    or witness.start.strands != instance["strands"]
                    or witness.crossing_changes > instance["cap_crossing_changes"]
                    or witness.moves > instance["cap_moves"]):
                raise ValueError(f"witness start/caps differ: {position}")
            objective = weight * witness.crossing_changes + witness.moves
            if outcome[f"L{weight}"] != objective:
                raise ValueError(f"saved objective differs: {position}")
        results[(arm, row_index, weight, seed)] = witness
        totals[(arm, weight)]["attempts"] += 1
        totals[(arm, weight)]["solved"] += int(witness is not None)
        totals[(arm, weight)]["cpu_seconds_milli"] += round(payload["cpu_seconds"] * 1000)
        totals[(arm, weight)]["wall_seconds_milli"] += round(payload["wall_seconds"] * 1000)
        totals[(arm, weight)]["neural_evaluations"] += (
            payload["counts"]["policy_neural_evaluations"]
            + payload["counts"]["tree_embedding_evaluations"]
        )
    paired = {}
    for weight in (10, 1000):
        roots = Counter()
        seeds = Counter()
        cost = Counter()
        for row_index in range(32):
            a_root = any(results[("A", row_index, weight, seed)] for seed in SEEDS)
            f_root = any(results[("F", row_index, weight, seed)] for seed in SEEDS)
            roots[(a_root, f_root)] += 1
            for seed in SEEDS:
                a, f = (results[(arm, row_index, weight, seed)] for arm in ARMS)
                seeds[(a is not None, f is not None)] += 1
                if a is not None and f is not None:
                    if a.crossing_changes != f.crossing_changes:
                        cost["F_lower_cc" if f.crossing_changes < a.crossing_changes else "F_higher_cc"] += 1
                    a_cost = weight * a.crossing_changes + a.moves
                    f_cost = weight * f.crossing_changes + f.moves
                    cost["F_lower_cost" if f_cost < a_cost else "F_higher_cost" if f_cost > a_cost else "equal_cost"] += 1
        paired[str(weight)] = {
            "roots": {f"A{int(a)}_F{int(f)}": n for (a, f), n in sorted(roots.items())},
            "seeds": {f"A{int(a)}_F{int(f)}": n for (a, f), n in sorted(seeds.items())},
            "common_success_cost": dict(cost),
        }
    summary = {
        "manifest_sha256": sha(args.data / "manifest.json"),
        "panel_sha256": sha(PANEL),
        "audited_shards": 256,
        "totals": {f"{arm}_L{weight}": dict(counts)
                   for (arm, weight), counts in sorted(totals.items())},
        "paired": paired,
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
