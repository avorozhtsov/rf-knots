"""Replay-verified, root-only TRAIN candidates from wider positive-braid seeds."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import build_f_interleaved_roster_v1 as first
from build_mcts_training_supervision import atomic, sha
from evaluate_embedding_cost_search import query_instance
from rf_knots.embedding_replay_corpus import invariants, mirror, owner
from rf_knots.evidence import UnknotWitness
from rf_knots.interleaved_generator import generate, positive_braid_seed

PROTOCOL = Path("research/327-root-only-f-supply-and-leakage-contract-v1.md")
DIAGNOSTIC = Path("artifacts/f-positive-seed-supply-diagnostic-v2/report.json")
DIAGNOSTIC_SHA = "5b5fabf247c7bc73f0bcb9f7f0ca84cbb3b67a2d2fc4dcc233cbcfa91e0198dd"
SEED = 20261013
MAX_INDEX = 128
QUOTAS = (16, 16)


def read(path):
    return json.loads(path.read_text())


def exact(state):
    return (tuple(state.word), state.strands)


def viewed_panel_paths():
    return tuple(dict.fromkeys((first.old.SHORT_DEV, first.old.LONG_DEV,
                                first.old.F_DEV, first.old.TEST150,
                                *first.incremental.PANELS)))


def viewed_states(paths):
    states = set()
    for path in paths:
        payload = read(path)
        for row in payload.get("rows", payload.get("validation", [])):
            witness = UnknotWitness.from_dict(row["witness"])
            witness.verify()
            states.add(exact(witness.start))
            states.update(exact(step.after) for step in witness.steps)
    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("roster output already exists; never overwrite")
    if sha(DIAGNOSTIC) != DIAGNOSTIC_SHA:
        raise ValueError("frozen diagnostic report differs")
    panels = viewed_panel_paths()
    inputs = (PROTOCOL, DIAGNOSTIC, first.PROTOCOL, first.OLD_ROSTER,
              first.INCREMENTAL_ROSTER, *panels, first.old.LONG,
              first.old.SOURCE, first.old.ROOTS, Path(__file__),
              Path(first.__file__),
              Path("src/rf_knots/interleaved_generator.py"),
              Path("src/rf_knots/embedding_replay_corpus.py"))
    pinned = {str(path): sha(path) for path in inputs}
    forbidden = first.old.excluded_groups()
    for panel in first.incremental.PANELS:
        forbidden.update(tuple(row["query_group"])
                         for row in read(panel)["rows"])
    seen = set(forbidden)
    for source in (first.OLD_ROSTER, first.INCREMENTAL_ROSTER):
        seen.update(tuple(row["query_group"])
                    for stage in read(source)["stages"] for row in stage)
    heldout_exact = viewed_states(panels)
    stages = [[], []]
    rejected = Counter()
    ranked = sorted(range(MAX_INDEX),
                    key=lambda index: hashlib.sha256(
                        f"{SEED}:{index}".encode()).hexdigest())
    args.out.mkdir(parents=True)
    atomic(args.out / "status.json",
           {"status": "building", "protocol_sha256": pinned[str(PROTOCOL)],
            "source_sha256": pinned[str(Path(__file__))],
            "quotas": list(QUOTAS)})
    for index in ranked:
        band = 2 + index % 2
        if len(stages[band - 2]) == QUOTAS[band - 2]:
            continue
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
        if first.old.stage(len(witness.start.word), witness.crossing_changes) != band:
            rejected["other_realized_band"] += 1
            continue
        fp = invariants(witness.start.word, witness.start.strands)
        group = first.old.group(fp)
        if group in seen:
            rejected["duplicate_or_excluded_root"] += 1
            continue
        if owner(fp) != "train":
            rejected["unowned_root"] += 1
            continue
        if exact(witness.start) in heldout_exact:
            rejected["viewed_panel_exact_state"] += 1
            continue
        instance, fits = query_instance(index, witness, witness_caps=True)
        if not fits:
            raise ValueError("constructive reference exceeds paired caps")
        stages[band - 2].append({
            "generator_index": index, "generator_seed": SEED,
            "source_family": "positive-seed-root-only-v1", "stage": band,
            "query_group": list(group),
            "word_length": len(witness.start.word),
            "reference_cc": witness.crossing_changes,
            "reference_moves": witness.moves,
            "witness": witness.to_dict(),
            "instances": [replace(instance, cc_weight=weight,
                                  uid=f"f-root-only-{SEED}-{index}-{weight}").to_dict()
                          for weight in (10, 1000)],
        })
        seen.add(group)
        if all(len(rows) == quota for rows, quota in zip(stages, QUOTAS,
                                                         strict=True)):
            break
    actual = [len(rows) for rows in stages]
    if actual != list(QUOTAS):
        atomic(args.out / "status.json",
               {"status": "supply_failure", "actual": actual,
                "quotas": list(QUOTAS), "rejected": dict(rejected),
                "inputs": pinned})
        raise ValueError(f"root-only supply insufficient: {actual}")
    payload = {
        "schema": "rf-knots-root-only-F-candidate-roster-v1",
        "role": "TRAIN root-only candidates; no F labels or training",
        "selection": {"seed": SEED, "max_index": MAX_INDEX,
                      "quotas": list(QUOTAS), "rejections": dict(rejected)},
        "gradient_boundary": "root state only; no intermediate gradient",
        "inputs": pinned, "stages": stages,
    }
    atomic(args.out / "roster.json", payload)
    atomic(args.out / "status.json",
           {"status": "completed", "actual": actual,
            "roster_sha256": sha(args.out / "roster.json"),
            "rejected": dict(rejected)})
    print(json.dumps({"actual": actual,
                      "roster_sha256": sha(args.out / "roster.json")},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
