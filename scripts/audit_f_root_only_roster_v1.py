"""Independent read-only replay, ownership and selection audit for root-only roster."""

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

ROSTER = Path("artifacts/f-root-only-roster-v1/roster.json")
ROSTER_SHA = "3c4a8eb62a63d9f4e22a4314aa94434164bb5a4c57fa70282783b9c602e23a81"
STATUS = Path("artifacts/f-root-only-roster-v1/status.json")
PROTOCOL = Path("research/327-root-only-f-supply-and-leakage-contract-v1.md")
SEED = 20261013


def read(path):
    return json.loads(path.read_text())


def exact(state):
    return tuple(state.word), state.strands


def config(index):
    band = 2 + index % 2
    extra = (index // 2) % 2
    if band == 2:
        walk, settle, target = ((48 + index % 17, 20, 112)
                                if extra else (0, 80 + index % 16, 112))
    else:
        walk, settle, target = ((72 + index % 9, 32, 152)
                                if extra else (0, 92 + index % 8, 152))
    return InterleavedConfig(
        crossing_changes=extra, walk_length=walk,
        settle_walk_length=settle, target_length=target,
        target_strands=8 + index % 5, max_len=192,
        max_strands=20, strand_band=1)


def generate_witness(index):
    cfg = config(index)
    seed = positive_braid_seed(
        cfg.spec(), strands=3 + index % 4,
        genus=9 + (index // 4) % 8, seed=SEED + index, churn=5)
    witness = generate(cfg, SEED, index, seed).witness
    if witness is None:
        raise ValueError("missing generated witness")
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
            or len(roster["stages"]) != 2
            or [len(rows) for rows in roster["stages"]] != [16, 16]
            or roster["gradient_boundary"] !=
            "root state only; no intermediate gradient"):
        raise ValueError("roster/status boundary differs")
    for path, expected in roster["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"pinned input SHA differs: {path}")
    if roster["inputs"][str(PROTOCOL)] != sha(PROTOCOL):
        raise ValueError("protocol SHA differs")
    excluded = old.excluded_groups()
    panels = tuple(dict.fromkeys((old.SHORT_DEV, old.LONG_DEV,
                                  old.F_DEV, old.TEST150,
                                  *incremental.PANELS)))
    viewed_states = set()
    for path in panels:
        payload = read(path)
        for row in payload.get("rows", payload.get("validation", [])):
            witness = UnknotWitness.from_dict(row["witness"])
            witness.verify()
            viewed_states.add(exact(witness.start))
            viewed_states.update(exact(step.after) for step in witness.steps)
    for panel in incremental.PANELS:
        excluded.update(tuple(row["query_group"])
                        for row in read(panel)["rows"])
    seen = set(excluded)
    for source in (Path("artifacts/f-curriculum-roster-v3/roster.json"),
                   Path("artifacts/f-incremental-roster-v2/roster.json")):
        seen.update(tuple(row["query_group"])
                    for stage in read(source)["stages"] for row in stage)
    ranked = sorted(range(128),
                    key=lambda index: hashlib.sha256(
                        f"{SEED}:{index}".encode()).hexdigest())
    expected = [[], []]
    rejected = Counter()
    for index in ranked:
        band = 2 + index % 2
        if len(expected[band - 2]) == 16:
            continue
        witness = generate_witness(index)
        if old.stage(len(witness.start.word), witness.crossing_changes) != band:
            rejected["other_realized_band"] += 1
            continue
        fp = invariants(witness.start.word, witness.start.strands)
        group = old.group(fp)
        if group in seen:
            rejected["duplicate_or_excluded_root"] += 1
            continue
        if owner(fp) != "train":
            rejected["unowned_root"] += 1
            continue
        if exact(witness.start) in viewed_states:
            rejected["viewed_panel_exact_state"] += 1
            continue
        row = roster["stages"][band - 2][len(expected[band - 2])]
        if (row["generator_index"] != index or row["stage"] != band
                or tuple(row["query_group"]) != group
                or row["reference_cc"] != witness.crossing_changes
                or row["reference_moves"] != witness.moves
                or row["word_length"] != len(witness.start.word)):
            raise ValueError(f"selected root identity differs at {index}")
        if UnknotWitness.from_dict(row["witness"]) != witness:
            raise ValueError(f"saved witness differs at {index}")
        instance, fits = query_instance(index, witness, witness_caps=True)
        if not fits:
            raise ValueError("reference cap failure")
        for item, weight in zip(row["instances"], (10, 1000), strict=True):
            expected_item = replace(instance, cc_weight=weight,
                                    uid=f"f-root-only-{SEED}-{index}-{weight}").to_dict()
            if item != json.loads(json.dumps(expected_item)):
                raise ValueError(f"paired instance differs at {index}/{weight}")
        expected[band - 2].append(index)
        seen.add(group)
        if all(len(rows) == 16 for rows in expected):
            break
    if ([len(rows) for rows in expected] != [16, 16]
            or dict(rejected) != status["rejected"]
            or dict(rejected) != roster["selection"]["rejections"]):
        raise ValueError("selection/rejection counts differ")
    report = {
        "schema": "rf-knots-root-only-roster-independent-audit-v1",
        "roster_sha256": ROSTER_SHA, "status_sha256": sha(STATUS),
        "source_sha256": sha(Path(__file__)),
        "selected_indices": expected, "verified_witnesses": 32,
        "paired_instances": 64, "rejections": dict(rejected),
        "gradient_boundary": "root only; later F coefficient audit required",
    }
    args.out.mkdir(parents=True)
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": sha(args.out / "report.json"),
                      "verified_witnesses": 32}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
