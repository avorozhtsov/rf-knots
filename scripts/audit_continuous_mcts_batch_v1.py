"""Independent full episode replay and root-only F audit of CPU search batches."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from audit_f_short_positive_r2_yield_v1 import replay_episode
from build_additive_root_only_f_bank_v2 import root_coefficients
from build_continuous_train_roster_v1 import exclusions
from build_mcts_training_supervision import atomic, sha
from rf_knots import mcts_loop as M
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness
from screen_legacy_unselected_root_f_v1 import cycle_free


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("audit output exists")
    global_state = read(args.data / "manifest.json")
    batch_path = args.data / f"batch-{args.batch:06d}" / "manifest.json"
    if batch_path.exists():
        batch = read(batch_path)
    elif args.allow_partial:
        first = args.batch * 32
        batch = {"contract": global_state["contract"],
                 "actor": global_state["batches"][str(args.batch)],
                 "first_task": first, "shards": global_state["shards"][first:first + 32]}
    else:
        raise ValueError("immutable full batch manifest missing")
    if len(batch["shards"]) != 32 and not args.allow_partial:
        raise ValueError("full32-condition audit required")
    for path, expected in batch["contract"]["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"batch pinned input differs: {path}")
    actor = batch["actor"]
    if sha(Path(actor["file"])) != actor["sha256"]:
        raise ValueError("batch actor SHA differs")
    roster_path = Path("artifacts/continuous-train-roster-v1/roster.json")
    roots = read(roster_path)["rows"]
    excluded, viewed_states = exclusions()
    instances, paths, shard_pins, references = {}, defaultdict(dict), defaultdict(list), {}
    episodes = solved = cycles = 0
    cpu = wall = neural = 0
    for offset, entry in enumerate(batch["shards"]):
        position = batch["first_task"] + offset
        shard = args.data / entry["file"]
        if entry["file"] != f"attempt-{position:06d}.json" or sha(shard) != entry["sha256"]:
            raise ValueError("batch shard SHA/order differs")
        saved = read(shard)
        r = (position // 4) % len(roots)
        wi = (position // 2) % 2
        seed = 20271420 + 2 * (position // (len(roots) * 4)) + position % 2
        root = roots[r]
        data = root["instances"][wi]
        instance = M.Instance(**{**data, "word": tuple(data["word"])})
        fp = invariants(instance.word, instance.strands)
        identity = (fp[0], abs(fp[1]))
        if (owner(fp) != "train" or identity != tuple(root["query_group"])
                or identity in excluded or (instance.word, instance.strands) in viewed_states):
            raise ValueError("TRAIN root exclusion differs")
        if ((saved["position"], saved["root_index"], saved["weight"], saved["seed"])
                != (position, r, instance.cc_weight, seed)):
            raise ValueError("search task identity differs")
        key = r, wi
        instances[key] = instance
        witness = UnknotWitness.from_dict(root["witness"])
        witness.verify()
        if ((witness.start.word, witness.start.strands) != (instance.word, instance.strands)
                or witness.moves > instance.cap_moves
                or witness.crossing_changes > instance.cap_crossing_changes):
            raise ValueError("reference start/caps differ")
        actions = tuple(s.action.to_flat(instance.spec) for s in witness.steps)
        references[key] = witness
        if cycle_free(instance, actions)[0]:
            paths[key][actions] = witness
        shard_pins[key].append({"file": str(shard), "sha256": entry["sha256"]})
        cpu += saved["cpu_seconds"]; wall += saved["wall_seconds"]
        if saved["cpu_censored"]:
            if saved["row"] is not None or saved["counts"]:
                raise ValueError("censored task fabricated labels")
            continue
        row = saved["row"]
        if (row["condition"] != r or row["cc_weight"] != instance.cc_weight
                or row["search_seed"] != seed or row["simulations"] != 16
                or bool(row["witness"]) != bool(row["solved"])):
            raise ValueError("saved search result identity differs")
        if row["witness"]:
            best = UnknotWitness.from_dict(row["witness"])
            best.verify()
            if ((best.start.word, best.start.strands) != (instance.word, instance.strands)
                    or row["L10"] != 10 * best.crossing_changes + best.moves
                    or row["L1000"] != 1000 * best.crossing_changes + best.moves
                    or best.moves > instance.cap_moves
                    or best.crossing_changes > instance.cap_crossing_changes):
                raise ValueError("best solution replay/cost differs")
        elif row["L10"] is not None or row["L1000"] is not None:
            raise ValueError("no-solve has fabricated costs")
        for episode in row["episodes"]:
            w = replay_episode(instance, episode)
            episodes += 1
            if w is None:
                continue
            actions = tuple(s.action.to_flat(instance.spec) for s in w.steps)
            if not cycle_free(instance, actions)[0]:
                cycles += 1
                continue
            paths[key][actions] = w
            solved += 1
        neural += (saved["counts"]["policy_neural_evaluations"]
                   + saved["counts"]["tree_embedding_evaluations"])
    rows = []
    by_root = defaultdict(list)
    for key, options in paths.items():
        instance = instances[key]
        result = root_coefficients(instance, options)
        if result is None:
            continue
        signal, terms = result
        r, wi = key
        by_root[r].append({"stage": roots[r]["stage"], "weight": instance.cc_weight,
            "instance": roots[r]["instances"][wi],
            "gradient_scope": "exact_zero_spent_root_only",
            "source": "continuous-canonical-MCTS-v1", "signal_terms": len(terms),
            "root_coefficients": terms, "source_shards": shard_pins[key],
            "positive_paths": signal["positive_paths"], "negative_paths": signal["negative_paths"],
            "paths": [{"actions": list(actions), "cc": w.crossing_changes, "moves": w.moves}
                      for actions, w in sorted(options.items())]})
    for r, groups in sorted(by_root.items()):
        rows.append({"source_root_index": r, "query_group": roots[r]["query_group"],
                     "stage": roots[r]["stage"], "groups": groups,
                     "gradient_scope": "exact_zero_spent_root_only"})
    report = {"schema": "rf-knots-continuous-MCTS-batch-audit-v1",
        "conditions": len(batch["shards"]), "partial": len(batch["shards"]) != 32,
        "episodes_replayed": episodes, "cycle_free_solved_episodes": solved,
        "cycled_solutions": cycles, "distinct_F_usable_root_groups": len(rows),
        "F_contexts": sum(len(r["groups"]) for r in rows),
        "cpu_seconds": cpu, "wall_seconds": wall, "neural_evaluations": neural,
        "actor": actor, "roster_sha256": sha(roster_path),
        "batch_manifest_sha256": sha(batch_path) if batch_path.exists() else None,
        "auditor_sha256": sha(Path(__file__)),
        "boundary": "root-only; candidate rows require independent bank admission"}
    args.out.mkdir(parents=True)
    atomic(args.out / "root-f-candidates.json", rows)
    report["candidate_rows_sha256"] = sha(args.out / "root-f-candidates.json")
    atomic(args.out / "report.json", report)
    print(json.dumps({"audit_conditions": len(batch["shards"]),
                      "F_roots": len(rows), "report_sha256": sha(args.out / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
