"""Fixed, candidate-only diagnosis of a wider positive-seed TRAIN source."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import build_f_interleaved_roster_v1 as first
from build_mcts_training_supervision import atomic, sha
from rf_knots.actions import CROSSING_CHANGE, KIND_NAMES
from rf_knots.embedding_replay_corpus import invariants, mirror, owner
from rf_knots.interleaved_generator import generate, positive_braid_seed

PROTOCOL = Path("research/324-positive-seed-supply-diagnostic-v2.md")
SEED = 20261013
MAX_INDEX = 128


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("diagnostic output already exists; never overwrite")
    inputs = (PROTOCOL, first.PROTOCOL, first.OLD_ROSTER,
              first.INCREMENTAL_ROSTER, *first.incremental.PANELS,
              first.old.LONG, first.old.SOURCE, first.old.ROOTS,
              first.old.SHORT_DEV, first.old.LONG_DEV, first.old.F_DEV,
              first.old.TEST150, Path(__file__), Path(first.__file__),
              Path("src/rf_knots/interleaved_generator.py"),
              Path("src/rf_knots/embedding_replay_corpus.py"))
    pinned = {str(path): sha(path) for path in inputs}
    forbidden = first.old.excluded_groups()
    for panel in first.incremental.PANELS:
        forbidden.update(tuple(row["query_group"])
                         for row in first.read(panel)["rows"])
    prior = set(forbidden)
    for source in (first.OLD_ROSTER, first.INCREMENTAL_ROSTER):
        prior.update(tuple(row["query_group"])
                     for stage in first.read(source)["stages"] for row in stage)
    raw_seen = set()
    accepted_seen = set()
    by_stage = [Counter(), Counter()]
    ranked = sorted(range(MAX_INDEX),
                    key=lambda index: hashlib.sha256(
                        f"{SEED}:{index}".encode()).hexdigest())
    for index in ranked:
        band = 2 + index % 2
        stats = by_stage[band - 2]
        stats["attempted"] += 1
        cfg = first.config(index)
        seed_rep = positive_braid_seed(
            cfg.spec(), strands=3 + index % 4,
            genus=9 + (index // 4) % 8,
            seed=SEED + index, churn=5)
        witness = generate(cfg, SEED, index, seed_rep).witness
        if witness is None:
            raise ValueError("missing replay-complete witness")
        if index % 2:
            witness = mirror(witness)
        witness.verify()
        stats["witness_replayed"] += 1
        if first.old.stage(len(witness.start.word), witness.crossing_changes) != band:
            stats["other_realized_band"] += 1
            continue
        stats["in_band"] += 1
        group = first.old.group(invariants(
            witness.start.word, witness.start.strands))
        if group not in raw_seen:
            stats["raw_new_in_sample"] += 1
            raw_seen.add(group)
        if group in prior:
            stats["prior_or_heldout_root"] += 1
            continue
        if owner(invariants(witness.start.word,
                            witness.start.strands)) != "train":
            stats["unowned_root"] += 1
            continue
        stats["new_train_root_before_path_filter"] += 1
        cc_states = [witness.start] + [step.after for step in witness.steps
            if step.action.kind == KIND_NAMES[CROSSING_CHANGE]]
        if any(owner(fp := invariants(state.word, state.strands)) != "train"
               or first.old.group(fp) in forbidden for state in cc_states):
            stats["unowned_or_heldout_cc_path"] += 1
            continue
        if group in accepted_seen:
            stats["duplicate_accepted_group"] += 1
            continue
        accepted_seen.add(group)
        stats["accepted_distinct"] += 1
    report = {
        "schema": "rf-knots-positive-seed-supply-diagnostic-v2",
        "role": "candidate supply only; not F labels or training",
        "seed": SEED, "max_index": MAX_INDEX,
        "genus_range": [9, 16], "inputs": pinned,
        "stage_counts": [dict(stats) for stats in by_stage],
        "raw_distinct_groups_total": len(raw_seen),
        "accepted_distinct_groups_total": len(accepted_seen),
        "gate_4_each": all(stats["accepted_distinct"] >= 4
                           for stats in by_stage),
    }
    args.out.mkdir(parents=True)
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": sha(args.out / "report.json"),
                      "stage_counts": report["stage_counts"],
                      "gate_4_each": report["gate_4_each"]}, sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
