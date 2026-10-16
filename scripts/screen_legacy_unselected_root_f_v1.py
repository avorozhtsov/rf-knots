"""Fail-closed exact-root F supply screen of the audited legacy B16 slice."""

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from bootstrap_cost_value_teacher_v1 import ROOTS, SOURCE, read
from build_f_curriculum_roster_v1 import excluded_groups, group
from build_mcts_training_supervision import atomic
from rf_knots import mcts_loop as M
from rf_knots.cost_mcts_rollout import advance
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

DATA = Path("artifacts/raster-budgeted-value-teacher-unselected-v1")
MANIFEST_SHA = "358f81da2ff6b6e5c2eee17d3c61355a60db5506559b811b20f06a63a0a418ea"
ROOTS_SHA = "8f1c14e9a85105b978d14ccd8654c071120ee5b0a3119fdc2b76e06547c35171"
SOURCE_SHA = "3085862627fd4950d376dce04dbe727fd62a620dcaf028257322e8d296ae5aa5"
AUDITOR = Path("scripts/audit_tournament_teacher_v1.py")
AUDITOR_SHA = "9b5679bc1a179e557496cbeed745590081146c607801b5236c8eed702e65eec5"
BANK = Path("artifacts/additive-root-only-f-bank-v3")
ROSTERS = tuple(Path(f"artifacts/{name}/roster.json") for name in (
    "f-curriculum-roster-v3", "f-incremental-roster-v2",
    "f-root-only-roster-v1", "f-short-positive-roster-v1"))
VIEWED = tuple(Path(f"artifacts/{name}/panel.json") for name in (
    "mcts-tournament-dev-panel-v1", "mcts-tournament-long-dev-panel-v1",
    "qrun-bootstrap-dev-panel-v3", "qrun-rung1-dev-panel-v1",
    "qrun-equal-cost-dev-panel-v1", "f-r2-r4-fresh-dev-panel-v1",
    "f-r2-r4-replication-panel-v1", "additive-r2-fresh-dev-panel-v1",
    "root-only-r2-fresh-dev-panel-v1", "r2-continuation-fresh-dev-panel-v2"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cycle_free(instance, actions):
    state = M.SearchState(instance.word, instance.strands, 0, 0)
    seen = {(state.word, state.strands)}
    first = None
    for action in actions:
        state = advance(instance, state, action)
        exact = (state.word, state.strands)
        if exact in seen:
            return False, None
        seen.add(exact)
        if first is None:
            first = exact
    return True, first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("output already exists")
    started = time.process_time()
    manifest = read(DATA / "manifest.json")
    source, roots = read(SOURCE), read(ROOTS)
    if (sha(DATA / "manifest.json") != MANIFEST_SHA
            or sha(ROOTS) != ROOTS_SHA or sha(SOURCE) != SOURCE_SHA
            or sha(AUDITOR) != AUDITOR_SHA
            or manifest["next_task"] != 2560
            or len(manifest["shards"]) != 2560
            or manifest["contract"]["budgets"][0] != 16
            or source["status"] != "completed"):
        raise ValueError("legacy source identity differs")
    for path, expected in manifest["contract"]["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"legacy input SHA differs: {path}")
    excluded = excluded_groups()
    excluded.update(tuple(row["query_group"]) for path in VIEWED
                    for row in read(path)["rows"])
    for path in ROSTERS:
        excluded.update(tuple(row["query_group"])
                        for stage in read(path)["stages"] for row in stage)
    bank = read(BANK / "manifest.json")
    if bank["status"] != "completed" or len(bank["roots"]) != 1010:
        raise ValueError("bank identity differs")
    for entry in bank["roots"]:
        path = BANK / entry["file"]
        if sha(path) != entry["sha256"]:
            raise ValueError("bank root SHA differs")
        excluded.add(tuple(read(path)["query_group"]))
    heldout = {tuple(row) for row in source["heldout_query_groups"]}
    paths = defaultdict(dict)
    stats = Counter()
    pins = [ROOTS, SOURCE, AUDITOR, DATA / "manifest.json",
            BANK / "manifest.json", *ROSTERS, *VIEWED]
    tasks = [(first + offset, budget, seed)
             for budget in manifest["contract"]["budgets"]
             for seed in manifest["contract"]["search_seeds"]
             for first in manifest["contract"]["root_order"]
             for offset in (0, 1)]
    for position, (index, budget, seed) in enumerate(tasks[:2560]):
        shard = manifest["shards"][position]
        path = DATA / f"attempt-{position:05d}.json"
        if (shard["file"] != path.name or sha(path) != shard["sha256"]):
            raise ValueError("legacy shard SHA differs")
        payload = read(path)
        root = roots[index]
        row = payload["row"]
        if ((payload["root_index"], payload["budget"], payload["seed"])
                != (index, budget, seed)
                or row["condition"] != index or row["simulations"] > budget):
            raise ValueError("legacy task differs")
        instance = M.Instance(**{**root["instance"],
                                 "word": tuple(root["instance"]["word"])})
        fp = invariants(instance.word, instance.strands)
        identity = group(fp)
        stats["shards"] += 1
        if (owner(fp) != "train" or (fp[0], abs(fp[1])) in heldout
                or identity in excluded):
            stats["excluded_shards"] += 1
            continue
        if len(row["episodes"]) != len(row["episodes_train_owned"]):
            raise ValueError("episode owner coverage differs")
        for episode, admitted in zip(row["episodes"], row["episodes_train_owned"], strict=True):
            stats["episodes"] += 1
            if episode["continuation"]["stop_reason"] != "solved" or not admitted:
                continue
            actions = tuple(episode["tree_prefix"] + episode["continuation"]["actions"])
            witness = UnknotWitness.from_actions(
                instance.word, instance.strands, instance.spec, actions)
            witness.verify()
            if (witness.crossing_changes > instance.cap_crossing_changes
                    or witness.moves > instance.cap_moves):
                raise ValueError("replayed path exceeds caps")
            valid, successor = cycle_free(instance, actions)
            if not valid or successor is None:
                stats["cycled_or_empty_solutions"] += 1
                continue
            cost = witness.moves + instance.cc_weight * witness.crossing_changes
            key = (identity, tuple(instance.word), instance.strands, instance.cc_weight)
            paths[key][actions] = (cost, successor, witness.crossing_changes,
                                   witness.moves, index)
            stats["replayed_cycle_free_solutions"] += 1
    qualifying = []
    for key, options in paths.items():
        values = list(options.values())
        if not any(a[0] != b[0] and a[1] != b[1]
                   for i, a in enumerate(values) for b in values[i + 1:]):
            continue
        identity, word, strands, weight = key
        qualifying.append({"query_group": identity, "word": word,
                           "strands": strands, "lambda": weight,
                           "distinct_paths": len(values),
                           "distinct_successors": len({v[1] for v in values}),
                           "costs": sorted({v[0] for v in values}),
                           "root_indices": sorted({v[4] for v in values})})
    distinct_groups = {tuple(row["query_group"]) for row in qualifying}
    report = {
        "schema": "rf-knots-legacy-unselected-b16-exact-root-F-screen-v1",
        "source_shards_sha256": MANIFEST_SHA,
        "input_sha256": {str(path): sha(path) for path in pins},
        "screen_source_sha256": sha(Path(__file__)),
        "stats": dict(stats),
        "qualifying_root_objectives": qualifying,
        "distinct_new_train_root_groups": len(distinct_groups),
        "cpu_seconds": time.process_time() - started,
        "decision": ("stop" if not distinct_groups else
                     "replicate" if len(distinct_groups) < 4 else "small_pilot_only"),
        "limitation": "historical collection CPU not recorded; no new online search or bank admission",
    }
    args.out.mkdir(parents=True)
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": sha(args.out / "report.json"),
                      "new_groups": len(distinct_groups),
                      "root_objectives": len(qualifying),
                      "decision": report["decision"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
