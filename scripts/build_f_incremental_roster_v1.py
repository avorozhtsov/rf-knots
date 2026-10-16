"""New TRAIN-owned candidate roots, disjoint from old F bank and viewed DEV."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import build_f_curriculum_roster_v1 as old
from build_mcts_training_supervision import atomic, sha
from evaluate_embedding_cost_search import query_instance
from rf_knots.actions import CROSSING_CHANGE, KIND_NAMES
from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_replay_corpus import invariants, owner

PROTOCOL = Path("research/315-f-bank-incremental-new-roots-v1.md")
OLD_ROSTER = Path("artifacts/f-curriculum-roster-v3/roster.json")
PANELS = tuple(Path("artifacts") / name / "panel.json" for name in (
    "qrun-bootstrap-dev-panel-v3", "qrun-rung1-dev-panel-v1",
    "qrun-equal-cost-dev-panel-v1", "f-r2-r4-fresh-dev-panel-v1",
    "f-r2-r4-replication-panel-v1"))
SEED = 20261011
MAX_INDEX = 16000
QUOTAS = (32, 64, 80, 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("output already exists")
    old_roster = json.loads(OLD_ROSTER.read_text())
    seen = old.excluded_groups()
    seen.update(tuple(row["query_group"]) for stage in old_roster["stages"] for row in stage)
    for panel in PANELS:
        seen.update(tuple(row["query_group"]) for row in json.loads(panel.read_text())["rows"])
    excluded = set(seen)
    stages = [[] for _ in QUOTAS]
    rejected = Counter()
    ranked = sorted(range(MAX_INDEX),
                    key=lambda index: hashlib.sha256(f"{SEED}:{index}".encode()).hexdigest())
    for index in ranked:
        witness = conjugate_witness(SEED, index, max_strands=20, min_depth=8, max_depth=60)
        witness.verify()
        stage = old.stage(len(witness.start.word), witness.crossing_changes)
        if stage is None or len(stages[stage]) == QUOTAS[stage]:
            rejected["size_or_filled"] += 1
            continue
        query_group = old.group(invariants(witness.start.word, witness.start.strands))
        if query_group in seen:
            rejected["duplicate_or_excluded"] += 1
            continue
        cc_states = [witness.start] + [step.after for step in witness.steps
                                       if step.action.kind == KIND_NAMES[CROSSING_CHANGE]]
        if any(owner(fp := invariants(state.word, state.strands)) != "train"
               or old.group(fp) in excluded for state in cc_states):
            rejected["ownership"] += 1
            continue
        instance, fits = query_instance(index, witness, witness_caps=True)
        if not fits:
            raise ValueError("constructive reference exceeds shared caps")
        stages[stage].append({
            "generator_index": index, "generator_seed": SEED, "stage": stage,
            "query_group": list(query_group), "word_length": len(witness.start.word),
            "reference_cc": witness.crossing_changes, "reference_moves": witness.moves,
            "witness": witness.to_dict(),
            "instances": [replace(instance, cc_weight=weight,
                                  uid=f"f-incremental-{SEED}-{index}-{weight}").to_dict()
                          for weight in (10, 1000)]})
        seen.add(query_group)
        if all(len(rows) == quota for rows, quota in zip(stages, QUOTAS, strict=True)):
            break
    actual = [len(rows) for rows in stages]
    if actual != list(QUOTAS):
        raise ValueError(f"incremental supply insufficient: {actual}, {dict(rejected)}")
    inputs = (PROTOCOL, OLD_ROSTER, *PANELS, old.LONG, old.SOURCE, old.ROOTS,
              old.SHORT_DEV, old.LONG_DEV, old.F_DEV, old.TEST150,
              Path("scripts/build_f_curriculum_roster_v1.py"), Path(__file__))
    payload = {
        "schema": "rf-knots-additive-F-candidate-roster-v1",
        "role": "TRAIN-only candidates, not F-usable groups or neural training",
        "selection": {"seed": SEED, "max_index": MAX_INDEX,
                      "quotas": list(QUOTAS), "rejections": dict(rejected)},
        "inputs": {str(path): sha(path) for path in inputs}, "stages": stages}
    args.out.mkdir(parents=True)
    atomic(args.out / "roster.json", payload)
    print(json.dumps({"candidates": sum(actual), "per_stage": actual,
                      "sha256": sha(args.out / "roster.json")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
