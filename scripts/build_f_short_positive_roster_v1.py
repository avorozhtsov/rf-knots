"""Frozen short positive-seed TRAIN candidate roster; no F labels or training."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import build_f_curriculum_roster_v1 as old
import build_f_root_only_roster_v1 as root_only
from build_mcts_training_supervision import atomic, sha
from evaluate_embedding_cost_search import query_instance
from rf_knots.embedding_replay_corpus import invariants, mirror, owner
from rf_knots.interleaved_generator import InterleavedConfig, generate, positive_braid_seed

PROTOCOL = Path("research/339-short-positive-seed-new-group-supply-v1.md")
OUT_OLD = Path("artifacts/f-curriculum-roster-v3/roster.json")
OUT_INCREMENTAL = Path("artifacts/f-incremental-roster-v2/roster.json")
OUT_ROOT_ONLY = Path("artifacts/f-root-only-roster-v1/roster.json")
EXTRA_PANEL = Path("artifacts/additive-r2-fresh-dev-panel-v1/panel.json")
SEED = 20261018
MAX_INDEX = 256
QUOTAS = (8, 8)


def read(path):
    return json.loads(path.read_text())


def config(index):
    return InterleavedConfig(
        crossing_changes=1 + index % 2, walk_length=8,
        settle_walk_length=40, target_length=88 + index % 9,
        target_strands=6 + index % 3, max_len=160,
        max_strands=16, strand_band=1,
    )


def witness_for(index):
    cfg = config(index)
    seed_rep = positive_braid_seed(
        cfg.spec(), strands=3 + index % 4,
        genus=2 + (index // 4) % 8, seed=SEED + index, churn=5,
    )
    witness = generate(cfg, SEED, index, seed_rep).witness
    if witness is None:
        raise ValueError(f"missing constructive witness at index {index}")
    if index % 2:
        witness = mirror(witness)
    witness.verify()
    return witness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new roster output exists; never overwrite")
    panels = tuple(dict.fromkeys((*root_only.viewed_panel_paths(), EXTRA_PANEL)))
    inputs = (PROTOCOL, OUT_OLD, OUT_INCREMENTAL, OUT_ROOT_ONLY,
              *panels, old.LONG, old.SOURCE, old.ROOTS,
              Path(__file__), Path(root_only.__file__),
              Path("src/rf_knots/interleaved_generator.py"),
              Path("src/rf_knots/embedding_replay_corpus.py"))
    pinned = {str(path): sha(path) for path in inputs}
    seen = set(old.excluded_groups())
    for panel in panels:
        payload = read(panel)
        for row in payload.get("rows", payload.get("validation", [])):
            if "query_group" in row:
                seen.add(tuple(row["query_group"]))
    for source in (OUT_OLD, OUT_INCREMENTAL, OUT_ROOT_ONLY):
        seen.update(tuple(row["query_group"])
                    for stage in read(source)["stages"] for row in stage)
    heldout_exact = root_only.viewed_states(panels)
    selected = [[], []]
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
        bucket = index % 2
        if len(selected[bucket]) == QUOTAS[bucket]:
            continue
        witness = witness_for(index)
        if old.stage(len(witness.start.word), witness.crossing_changes) != 1:
            rejected["other_realized_stage"] += 1
            continue
        fingerprint = invariants(witness.start.word, witness.start.strands)
        group = old.group(fingerprint)
        if group in seen:
            rejected["duplicate_or_excluded_group"] += 1
            continue
        if owner(fingerprint) != "train":
            rejected["unowned_root"] += 1
            continue
        if root_only.exact(witness.start) in heldout_exact:
            rejected["viewed_panel_exact_state"] += 1
            continue
        instance, fits = query_instance(index, witness, witness_caps=True)
        if not fits:
            raise ValueError(f"paired reference caps fail at index {index}")
        selected[bucket].append({
            "generator_index": index, "generator_seed": SEED,
            "source_family": "short-positive-seed-root-only-v1",
            "stage": 1, "bucket": bucket, "query_group": list(group),
            "word_length": len(witness.start.word),
            "reference_cc": witness.crossing_changes,
            "reference_moves": witness.moves,
            "witness": witness.to_dict(),
            "instances": [replace(instance, cc_weight=weight,
                                  uid=f"f-short-positive-{SEED}-{index}-{weight}").to_dict()
                          for weight in (10, 1000)],
        })
        seen.add(group)
        if all(len(rows) == quota for rows, quota in zip(selected, QUOTAS, strict=True)):
            break
    actual = [len(rows) for rows in selected]
    if actual != list(QUOTAS):
        atomic(args.out / "status.json",
               {"status": "supply_failure", "actual": actual,
                "quotas": list(QUOTAS), "rejected": dict(rejected),
                "inputs": pinned})
        raise ValueError(f"short positive-seed supply insufficient: {actual}")
    roster = {
        "schema": "rf-knots-short-positive-seed-root-only-roster-v1",
        "role": "TRAIN candidates only; no F labels or optimizer training",
        "gradient_boundary": "root state only; no intermediate gradient",
        "selection": {"seed": SEED, "max_index": MAX_INDEX,
                      "quotas": list(QUOTAS), "rejections": dict(rejected)},
        "inputs": pinned, "stages": selected,
    }
    atomic(args.out / "roster.json", roster)
    atomic(args.out / "status.json",
           {"status": "completed", "actual": actual,
            "quotas": list(QUOTAS), "rejected": dict(rejected),
            "roster_sha256": sha(args.out / "roster.json"),
            "source_sha256": pinned[str(Path(__file__))],
            "protocol_sha256": pinned[str(PROTOCOL)]})
    print(json.dumps({"actual": actual,
                      "roster_sha256": sha(args.out / "roster.json")},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
