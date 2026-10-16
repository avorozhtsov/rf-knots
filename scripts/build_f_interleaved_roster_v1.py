"""New TRAIN root supply from replay-complete positive-seed interleaved walks."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import build_f_curriculum_roster_v1 as old
import build_f_incremental_roster_v2 as incremental
from build_mcts_training_supervision import atomic, sha
from evaluate_embedding_cost_search import query_instance
from rf_knots.actions import CROSSING_CHANGE, KIND_NAMES
from rf_knots.embedding_replay_corpus import invariants, mirror, owner
from rf_knots.interleaved_generator import InterleavedConfig, generate, positive_braid_seed

PROTOCOL = Path("research/321-interleaved-positive-seed-new-root-pilot-v1.md")
OLD_ROSTER = Path("artifacts/f-curriculum-roster-v3/roster.json")
INCREMENTAL_ROSTER = Path("artifacts/f-incremental-roster-v2/roster.json")
SEED = 20261012
MAX_INDEX = 4096
QUOTAS = (16, 16)


def read(path):
    return json.loads(path.read_text())


def config(index):
    band = 2 + index % 2
    extra_cc = (index // 2) % 2
    if band == 2:
        walk = 48 + index % 17 if extra_cc else 0
        settle = 20 if extra_cc else 80 + index % 16
        target_length = 112
    else:
        walk = 72 + index % 9 if extra_cc else 0
        settle = 32 if extra_cc else 92 + index % 8
        target_length = 152
    return InterleavedConfig(
        crossing_changes=extra_cc, walk_length=walk, settle_walk_length=settle,
        target_length=target_length, target_strands=8 + index % 5,
        max_len=192, max_strands=20, strand_band=1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new source output exists; never overwrite")
    inputs = (PROTOCOL, OLD_ROSTER, INCREMENTAL_ROSTER, *incremental.PANELS,
              old.LONG, old.SOURCE, old.ROOTS, old.SHORT_DEV, old.LONG_DEV,
              old.F_DEV, old.TEST150, Path(__file__),
              Path("src/rf_knots/interleaved_generator.py"),
              Path("src/rf_knots/embedding_replay_corpus.py"),
              Path("scripts/build_f_incremental_roster_v2.py"))
    pinned = {str(path): sha(path) for path in inputs}
    forbidden_states = old.excluded_groups()
    for panel in incremental.PANELS:
        forbidden_states.update(tuple(row["query_group"])
                                for row in read(panel)["rows"])
    seen = set(forbidden_states)
    for source in (OLD_ROSTER, INCREMENTAL_ROSTER):
        seen.update(tuple(row["query_group"])
                    for stage in read(source)["stages"] for row in stage)
    stages = [[], []]
    rejected = Counter()
    args.out.mkdir(parents=True)
    atomic(args.out / "status.json",
           {"status": "building", "source_sha256": pinned[str(Path(__file__))],
            "protocol_sha256": pinned[str(PROTOCOL)], "quotas": list(QUOTAS)})
    ranked = sorted(range(MAX_INDEX),
                    key=lambda index: hashlib.sha256(f"{SEED}:{index}".encode()).hexdigest())
    for index in ranked:
        band = 2 + index % 2
        if len(stages[band - 2]) == QUOTAS[band - 2]:
            continue
        cfg = config(index)
        seed_rep = positive_braid_seed(
            cfg.spec(), strands=3 + index % 4, genus=1 + (index // 4) % 8,
            seed=SEED + index, churn=5)
        witness = generate(cfg, SEED, index, seed_rep).witness
        if witness is None:
            raise ValueError("missing replay-complete witness")
        if index % 2:
            witness = mirror(witness)
        witness.verify()
        if old.stage(len(witness.start.word), witness.crossing_changes) != band:
            rejected["realized_other_band"] += 1
            continue
        query_group = old.group(invariants(witness.start.word, witness.start.strands))
        if query_group in seen:
            rejected["duplicate_or_excluded_start"] += 1
            continue
        cc_states = [witness.start] + [step.after for step in witness.steps
            if step.action.kind == KIND_NAMES[CROSSING_CHANGE]]
        if any(owner(fp := invariants(state.word, state.strands)) != "train"
               or old.group(fp) in forbidden_states for state in cc_states):
            rejected["heldout_or_unowned_state"] += 1
            continue
        instance, fits = query_instance(index, witness, witness_caps=True)
        if not fits:
            raise ValueError("verified reference exceeds paired caps")
        stages[band - 2].append({
            "generator_index": index, "generator_seed": SEED,
            "source_family": "positive-seed-interleaved-v1", "stage": band,
            "query_group": list(query_group),
            "word_length": len(witness.start.word),
            "reference_cc": witness.crossing_changes,
            "reference_moves": witness.moves,
            "witness": witness.to_dict(),
            "instances": [
                replace(instance, cc_weight=weight,
                        uid=f"f-interleaved-{SEED}-{index}-{weight}").to_dict()
                for weight in (10, 1000)
            ],
        })
        seen.add(query_group)
        if all(len(rows) == quota for rows, quota in zip(stages, QUOTAS, strict=True)):
            break
    counts = [len(rows) for rows in stages]
    if counts != list(QUOTAS):
        atomic(args.out / "status.json",
               {"status": "supply_failure", "actual": counts,
                "quotas": list(QUOTAS), "rejected": dict(rejected),
                "inputs": pinned})
        raise ValueError(f"new source supply insufficient: {counts}")
    payload = {"schema": "rf-knots-interleaved-positive-seed-roster-v1",
               "role": "TRAIN-only candidates, not F labels or training",
               "selection": {"seed": SEED, "max_index": MAX_INDEX,
                             "quotas": list(QUOTAS), "rejections": dict(rejected)},
               "inputs": pinned, "stages": stages}
    atomic(args.out / "roster.json", payload)
    atomic(args.out / "status.json",
           {"status": "completed", "roster_sha256": sha(args.out / "roster.json"),
            "actual": counts, "rejected": dict(rejected)})
    print(json.dumps({"candidates": sum(counts), "stages": counts,
                      "roster_sha256": sha(args.out / "roster.json")}, sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
