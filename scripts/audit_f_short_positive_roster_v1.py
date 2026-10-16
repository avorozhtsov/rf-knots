"""Independent regeneration, replay, ownership and cap audit of short roster."""

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
from rf_knots.embedding_replay_corpus import invariants, mirror, owner
from rf_knots.evidence import UnknotWitness
from rf_knots.interleaved_generator import InterleavedConfig, generate, positive_braid_seed

ROSTER = Path("artifacts/f-short-positive-roster-v1/roster.json")
ROSTER_SHA = "53d6a26828acb966747d17699a3b74f5ab13e0121fa055edbbaa2f57fd242f42"
STATUS = Path("artifacts/f-short-positive-roster-v1/status.json")
PROTOCOL = Path("research/339-short-positive-seed-new-group-supply-v1.md")
SEED = 20261018
QUOTAS = (8, 8)


def read(path):
    return json.loads(path.read_text())


def exact(state):
    return tuple(state.word), state.strands


def generate_witness(index):
    cfg = InterleavedConfig(
        crossing_changes=1 + index % 2, walk_length=8,
        settle_walk_length=40, target_length=88 + index % 9,
        target_strands=6 + index % 3, max_len=160,
        max_strands=16, strand_band=1,
    )
    seed_rep = positive_braid_seed(
        cfg.spec(), strands=3 + index % 4,
        genus=2 + (index // 4) % 8, seed=SEED + index, churn=5,
    )
    witness = generate(cfg, SEED, index, seed_rep).witness
    if witness is None:
        raise ValueError(f"missing generated witness {index}")
    if index % 2:
        witness = mirror(witness)
    witness.verify()
    return witness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("audit output exists")
    if sha(ROSTER) != ROSTER_SHA:
        raise ValueError("roster SHA differs")
    roster, status = read(ROSTER), read(STATUS)
    if (status["status"] != "completed" or status["roster_sha256"] != ROSTER_SHA
            or roster["schema"] != "rf-knots-short-positive-seed-root-only-roster-v1"
            or roster["gradient_boundary"] != "root state only; no intermediate gradient"
            or [len(rows) for rows in roster["stages"]] != list(QUOTAS)
            or roster["selection"]["seed"] != SEED
            or roster["selection"]["max_index"] != 256):
        raise ValueError("roster/status contract differs")
    for path, expected in roster["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"pinned input SHA differs: {path}")
    if roster["inputs"][str(PROTOCOL)] != sha(PROTOCOL):
        raise ValueError("protocol SHA differs")
    panels = tuple(dict.fromkeys((old.SHORT_DEV, old.LONG_DEV,
                                  old.F_DEV, old.TEST150,
                                  *incremental.PANELS,
                                  Path("artifacts/additive-r2-fresh-dev-panel-v1/panel.json"))))
    viewed_states = set()
    seen = set(old.excluded_groups())
    for panel in panels:
        payload = read(panel)
        for row in payload.get("rows", payload.get("validation", [])):
            if "query_group" in row:
                seen.add(tuple(row["query_group"]))
            witness = UnknotWitness.from_dict(row["witness"])
            witness.verify()
            viewed_states.add(exact(witness.start))
            viewed_states.update(exact(step.after) for step in witness.steps)
    for source in (Path("artifacts/f-curriculum-roster-v3/roster.json"),
                   Path("artifacts/f-incremental-roster-v2/roster.json"),
                   Path("artifacts/f-root-only-roster-v1/roster.json")):
        seen.update(tuple(row["query_group"])
                    for stage in read(source)["stages"] for row in stage)
    ranked = sorted(range(256),
                    key=lambda index: hashlib.sha256(
                        f"{SEED}:{index}".encode()).hexdigest())
    positions = [[], []]
    rejected = Counter()
    groups = set()
    for index in ranked:
        bucket = index % 2
        if len(positions[bucket]) == QUOTAS[bucket]:
            continue
        witness = generate_witness(index)
        if old.stage(len(witness.start.word), witness.crossing_changes) != 1:
            rejected["other_realized_stage"] += 1
            continue
        fp = invariants(witness.start.word, witness.start.strands)
        group = old.group(fp)
        if group in seen:
            rejected["duplicate_or_excluded_group"] += 1
            continue
        if owner(fp) != "train":
            rejected["unowned_root"] += 1
            continue
        if exact(witness.start) in viewed_states:
            rejected["viewed_panel_exact_state"] += 1
            continue
        row = roster["stages"][bucket][len(positions[bucket])]
        if (row["generator_index"] != index or row["generator_seed"] != SEED
                or row["bucket"] != bucket or row["stage"] != 1
                or tuple(row["query_group"]) != group
                or row["word_length"] != len(witness.start.word)
                or row["reference_cc"] != witness.crossing_changes
                or row["reference_moves"] != witness.moves
                or UnknotWitness.from_dict(row["witness"]) != witness):
            raise ValueError(f"selected witness differs at {index}")
        instance, fits = query_instance(index, witness, witness_caps=True)
        if not fits:
            raise ValueError(f"reference cap failure at {index}")
        for item, weight in zip(row["instances"], (10, 1000), strict=True):
            expected = replace(instance, cc_weight=weight,
                               uid=f"f-short-positive-{SEED}-{index}-{weight}").to_dict()
            if item != json.loads(json.dumps(expected)):
                raise ValueError(f"paired instance differs at {index}/{weight}")
        positions[bucket].append(index)
        groups.add(group)
        seen.add(group)
        if all(len(rows) == quota for rows, quota in zip(positions, QUOTAS, strict=True)):
            break
    if (list(map(len, positions)) != list(QUOTAS)
            or dict(rejected) != status["rejected"]
            or dict(rejected) != roster["selection"]["rejections"]
            or len(groups) != sum(QUOTAS)):
        raise ValueError("selection/order/rejections differ")
    report = {
        "schema": "rf-knots-short-positive-roster-independent-audit-v1",
        "roster_sha256": ROSTER_SHA, "status_sha256": sha(STATUS),
        "auditor_sha256": sha(Path(__file__)),
        "selected_indices": positions, "verified_witnesses": sum(QUOTAS),
        "paired_instances": 2 * sum(QUOTAS),
        "distinct_train_groups": len(groups),
        "rejections": dict(rejected),
        "gradient_boundary": "root only; later exact F coefficient audit required",
    }
    args.out.mkdir(parents=True)
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": sha(args.out / "report.json"),
                      "distinct_train_groups": len(groups)}, sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
