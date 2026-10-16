"""Freeze score-blind validation roots for the completed v5/v8 R2 comparison."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from build_f_curriculum_roster_v1 import excluded_groups, group
from build_f_r2_r4_dev_panel_v1 import panel_groups, substantial_states
from build_mcts_training_supervision import atomic, sha
from evaluate_embedding_cost_search import query_instance
from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

SEED, MAX_INDEX, QUOTAS = 20261025, 32000, (16, 16)
PROTOCOL = Path("research/370-r2-continuation-fresh-dev-v2.md")
BANK = Path("artifacts/additive-root-only-f-bank-v3")
ROSTERS = tuple(Path(f"artifacts/{name}/roster.json") for name in (
    "f-curriculum-roster-v3", "f-incremental-roster-v2",
    "f-root-only-roster-v1", "f-short-positive-roster-v1"))
VIEWED = tuple(Path(f"artifacts/{name}/panel.json") for name in (
    "mcts-tournament-dev-panel-v1", "mcts-tournament-long-dev-panel-v1",
    "qrun-bootstrap-dev-panel-v3", "qrun-rung1-dev-panel-v1",
    "qrun-equal-cost-dev-panel-v1", "f-r2-r4-fresh-dev-panel-v1",
    "f-r2-r4-replication-panel-v1", "additive-r2-fresh-dev-panel-v1",
    "root-only-r2-fresh-dev-panel-v1"))
PINS = (BANK / "manifest.json",
        Path("artifacts/additive-root-only-f-bank-audit-v3/report.json"),
        Path("artifacts/additive-root-only-r2-v5/manifest.json"),
        Path("artifacts/additive-root-only-r2-v5/step-246407.pt"),
        Path("artifacts/additive-root-only-r2-v8/manifest.json"),
        Path("artifacts/additive-root-only-r2-v8/step-275466.pt"))


def bank_groups():
    manifest = json.loads((BANK / "manifest.json").read_text())
    groups = set()
    if manifest["status"] != "completed" or len(manifest["roots"]) != 1010:
        raise ValueError("F bank manifest differs")
    for entry in manifest["roots"]:
        path = BANK / entry["file"]
        if sha(path) != entry["sha256"]:
            raise ValueError(f"F bank root SHA differs: {path}")
        groups.add(tuple(json.loads(path.read_text())["query_group"]))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("panel output already exists")
    pins = (*ROSTERS, *VIEWED, *PINS, PROTOCOL, Path(__file__),
            Path("scripts/build_f_r2_r4_dev_panel_v1.py"),
            Path("scripts/build_f_curriculum_roster_v1.py"))
    inputs = {str(path): sha(path) for path in pins}
    for manifest_path, checkpoint_path, step, minimum in (
            (PINS[2], PINS[3], 246407, 7200),
            (PINS[4], PINS[5], 275466, 3600)):
        run = json.loads(manifest_path.read_text())
        if (run["status"] != "checkpointed"
                or run["optimizer_steps"] != step
                or run["new_active_seconds"] < minimum
                or run["checkpoints"][-1]["sha256"] !=
                inputs[str(checkpoint_path)]):
            raise ValueError("endpoint is not the frozen completed checkpoint")
    train_states, train_groups = set(), bank_groups()
    for path in ROSTERS:
        roster = json.loads(path.read_text())
        for stage in roster["stages"]:
            for row in stage:
                witness = UnknotWitness.from_dict(row["witness"])
                witness.verify()
                train_states.update(substantial_states(witness))
                train_groups.add(tuple(row["query_group"]))
    viewed_groups = set().union(*(panel_groups(path) for path in VIEWED))
    seen = excluded_groups() | train_groups | viewed_groups
    bands = [[], []]
    rejects = {"length": 0, "root_owner": 0,
               "excluded_or_duplicate": 0, "substantial_state_overlap": 0}
    ranked = sorted(range(MAX_INDEX),
                    key=lambda index: hashlib.sha256(
                        f"{SEED}:{index}".encode()).hexdigest())
    args.out.mkdir(parents=True)
    atomic(args.out / "status.json",
           {"status": "building", "source_sha256": inputs[str(Path(__file__))],
            "protocol_sha256": inputs[str(PROTOCOL)], "quotas": list(QUOTAS)})
    for index in ranked:
        witness = conjugate_witness(SEED, index, max_strands=20,
                                    min_depth=8, max_depth=60)
        witness.verify()
        length = len(witness.start.word)
        band = 0 if 20 <= length <= 64 else 1 if 129 <= length <= 192 else None
        if band is None or len(bands[band]) == QUOTAS[band]:
            rejects["length"] += 1
            continue
        fp = invariants(witness.start.word, witness.start.strands)
        identity = group(fp)
        if owner(fp) != "validation":
            rejects["root_owner"] += 1
            continue
        if identity in seen:
            rejects["excluded_or_duplicate"] += 1
            continue
        if substantial_states(witness) & train_states:
            rejects["substantial_state_overlap"] += 1
            continue
        base, fits = query_instance(index, witness, witness_caps=True)
        if not fits:
            raise ValueError("reference exceeds paired resource caps")
        bands[band].append({
            "generator_index": index, "band": band,
            "query_group": list(identity), "word_length": length,
            "reference_cc": witness.crossing_changes,
            "reference_moves": witness.moves,
            "witness": witness.to_dict(),
            "instances": [replace(base, cc_weight=weight,
                                  uid=f"r2-continuation-fresh-dev-v2-{index}-{weight}").to_dict()
                          for weight in (10, 1000)],
        })
        seen.add(identity)
        if list(map(len, bands)) == list(QUOTAS):
            break
    actual = list(map(len, bands))
    if actual != list(QUOTAS):
        atomic(args.out / "status.json",
               {"status": "supply_failure", "actual": actual,
                "quotas": list(QUOTAS), "rejects": rejects, "inputs": inputs})
        raise ValueError(f"fresh DEV supply insufficient: {actual}")
    payload = {
        "schema": "rf-knots-R2-continuation-fresh-DEV-panel-v2",
        "role": "score-blind validation roots with TRAIN substantial-state exclusion",
        "selection": {"seed": SEED, "max_index": MAX_INDEX,
                      "quotas": list(QUOTAS), "min_state_length": 20,
                      "rejects": rejects,
                      "train_substantial_state_count": len(train_states),
                      "train_excluded_group_count": len(train_groups),
                      "viewed_excluded_group_count": len(viewed_groups)},
        "inputs": inputs, "rows": bands[0] + bands[1],
    }
    atomic(args.out / "panel.json", payload)
    atomic(args.out / "status.json",
           {"status": "completed", "actual": actual,
            "panel_sha256": sha(args.out / "panel.json"), "rejects": rejects})
    print(json.dumps({"panel_sha256": sha(args.out / "panel.json"),
                      "actual": actual}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
