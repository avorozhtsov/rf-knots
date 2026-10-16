"""Independent replay/F audit with independent residual stage order, seeds, novelty and actor identity."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
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
        raise ValueError("immutable audit output exists")
    state = read(args.data / "manifest.json")
    manifest_path = args.data / f"batch-{args.batch:06d}" / "manifest.json"
    if manifest_path.exists():
        batch = read(manifest_path)
    elif args.allow_partial:
        first = args.batch * 32
        batch = {"contract": state["contract"], "actor": state["batches"][str(args.batch)],
                 "first_task": first, "shards": state["shards"][first:first + 32]}
    else:
        raise ValueError("immutable complete batch missing")
    if len(batch["shards"]) != 32 and not args.allow_partial:
        raise ValueError("complete32-task batch required")
    cfg = batch["contract"]["config"]
    if cfg["stream_version"] != 4 or cfg["seed_base"] != 20277442:
        raise ValueError("stream version or seeds differ")
    for key in ("provider_source", "teacher_provider_source", "stream_spec", "roster", "bank_manifest", "prior_roster"):
        if sha(Path(cfg[key])) != cfg[key + "_sha256"]:
            raise ValueError("v4 provider/spec pin differs")
    for path, expected in read(Path(cfg["stream_spec"]))["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError("independent source/test/protocol pin differs")
    for path, expected in batch["contract"]["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"search pinned input differs: {path}")
    actor = batch["actor"]
    if actor != read(Path(cfg["stream_spec"]))["fixed_actor"]:
        raise ValueError("independently fixed replication actor differs")
    if sha(Path(actor["file"])) != actor["sha256"] or sha(Path(actor["training_spec"])) != actor["training_spec_sha256"]:
        raise ValueError("frozen teacher checkpoint/spec SHA differs")
    saved_actor = torch.load(actor["file"], map_location="cpu", weights_only=False)
    if saved_actor["step"] != actor["step"] or saved_actor["contract_sha256"] != actor["contract_sha256"]:
        raise ValueError("frozen teacher identity differs")
    if (len(saved_actor["optimizer"]["state"]) != 71
            or not all(k in saved_actor for k in ("rng", "torch_rng", "mps_rng"))):
        raise ValueError("frozen full actor optimizer/RNG differs")
    roots_path = Path(cfg["roster"])
    roots = read(roots_path)["rows"]
    if (len(roots) != 88 or cfg["global_task_offset"] != 0
            or cfg["root_order"] != "round-robin-stage-v1"):
        raise ValueError("residual source size/order differs")
    buckets = [[i for i, r in enumerate(roots) if r["stage"] == stage]
               for stage in range(4)]
    if not all(buckets):
        raise ValueError("all residual stages required")
    order = [b[k] for k in range(max(map(len, buckets)))
             for b in buckets if k < len(b)]
    seen = {tuple(r["query_group"]) for r in read(Path(cfg["prior_roster"]))["rows"]}
    bank_path = Path(cfg["bank_manifest"])
    for entry in read(bank_path)["roots"]:
        path = bank_path.parent / entry["file"]
        if sha(path) != entry["sha256"]:
            raise ValueError("prior retained bank row SHA differs")
        seen.add(tuple(read(path)["query_group"]))
    new_keys = [tuple(r["query_group"]) for r in roots]
    if len(set(new_keys)) != 88 or any(k in seen for k in new_keys):
        raise ValueError("residual query group novelty differs")
    excluded, states = exclusions()
    paths, instances, source_pins = defaultdict(dict), {}, defaultdict(list)
    episode_count = solved = cycles = 0
    cpu = wall = neural = 0
    for offset, entry in enumerate(batch["shards"]):
        local = batch["first_task"] + offset
        global_position = local + cfg["global_task_offset"]
        shard = args.data / entry["file"]
        if entry["file"] != f"attempt-{local:06d}.json" or sha(shard) != entry["sha256"]:
            raise ValueError("shard SHA/order differs")
        result = read(shard)
        r = order[(global_position // 4) % len(roots)]
        wi = (global_position // 2) % 2
        seed = 20277442 + 2 * (global_position // (len(roots) * 4)) + global_position % 2
        root = roots[r]
        data = root["instances"][wi]
        instance = M.Instance(**{**data, "word": tuple(data["word"])})
        fp = invariants(instance.word, instance.strands)
        if (owner(fp) != "train" or (fp[0], abs(fp[1])) != tuple(root["query_group"])
                or tuple(root["query_group"]) in excluded or (instance.word, instance.strands) in states):
            raise ValueError("root ownership/exclusion differs")
        if (result["position"], result["root_index"], result["weight"], result["seed"]) != (local, r, instance.cc_weight, seed):
            raise ValueError("global-offset task identity differs")
        key = r, wi
        instances[key] = instance
        ref = UnknotWitness.from_dict(root["witness"])
        ref.verify()
        if ((ref.start.word, ref.start.strands) != (instance.word, instance.strands)
                or ref.moves > instance.cap_moves or ref.crossing_changes > instance.cap_crossing_changes):
            raise ValueError("reference start/caps differ")
        actions = tuple(s.action.to_flat(instance.spec) for s in ref.steps)
        if cycle_free(instance, actions)[0]:
            paths[key][actions] = ref
        source_pins[key].append({"file": str(shard), "sha256": entry["sha256"]})
        cpu += result["cpu_seconds"]; wall += result["wall_seconds"]
        if result["cpu_censored"]:
            if result["row"] is not None or result["counts"]:
                raise ValueError("censored task fabricated outcomes")
            continue
        row = result["row"]
        if (row["condition"] != r or row["cc_weight"] != instance.cc_weight
                or row["search_seed"] != seed or row["simulations"] != 16
                or bool(row["witness"]) != bool(row["solved"])):
            raise ValueError("saved search identity differs")
        if row["witness"]:
            best = UnknotWitness.from_dict(row["witness"])
            best.verify()
            if ((best.start.word, best.start.strands) != (instance.word, instance.strands)
                    or best.moves > instance.cap_moves or best.crossing_changes > instance.cap_crossing_changes
                    or row["L10"] != 10 * best.crossing_changes + best.moves
                    or row["L1000"] != 1000 * best.crossing_changes + best.moves):
                raise ValueError("best solution replay/cost differs")
        elif row["L10"] is not None or row["L1000"] is not None:
            raise ValueError("no-solve has fabricated costs")
        for episode in row["episodes"]:
            witness = replay_episode(instance, episode)
            episode_count += 1
            if witness is None:
                continue
            actions = tuple(s.action.to_flat(instance.spec) for s in witness.steps)
            if not cycle_free(instance, actions)[0]:
                cycles += 1
                continue
            solved += 1
            paths[key][actions] = witness
        neural += result["counts"]["policy_neural_evaluations"] + result["counts"]["tree_embedding_evaluations"]
    groups = defaultdict(list)
    for key, options in paths.items():
        instance = instances[key]
        signal = root_coefficients(instance, options)
        if signal is None:
            continue
        meta, terms = signal
        r, wi = key
        groups[r].append({"stage": roots[r]["stage"], "weight": instance.cc_weight,
            "instance": roots[r]["instances"][wi], "gradient_scope": "exact_zero_spent_root_only",
            "source": "residual-fixed-R2-replication-v4", "signal_terms": len(terms),
            "root_coefficients": terms, "source_shards": source_pins[key],
            "positive_paths": meta["positive_paths"], "negative_paths": meta["negative_paths"],
            "paths": [{"actions": list(actions), "cc": w.crossing_changes, "moves": w.moves}
                      for actions, w in sorted(options.items())]})
    candidates = [{"source_root_index": r, "query_group": roots[r]["query_group"],
                   "stage": roots[r]["stage"], "groups": rows,
                   "gradient_scope": "exact_zero_spent_root_only"} for r, rows in sorted(groups.items())]
    report = {"schema": "rf-knots-residual-MCTS-batch-audit-v4", "conditions": len(batch["shards"]),
        "partial": len(batch["shards"]) != 32, "episodes_replayed": episode_count,
        "cycle_free_solved_episodes": solved, "cycled_solutions": cycles,
        "distinct_F_usable_root_groups": len(candidates), "F_contexts": sum(map(len, groups.values())),
        "cpu_seconds": cpu, "wall_seconds": wall, "neural_evaluations": neural,
        "actor": actor, "global_task_offset": cfg["global_task_offset"],
        "roster_sha256": sha(roots_path), "auditor_sha256": sha(Path(__file__)),
        "batch_manifest_sha256": sha(manifest_path) if manifest_path.exists() else None}
    args.out.mkdir(parents=True)
    atomic(args.out / "root-f-candidates.json", candidates)
    report["candidate_rows_sha256"] = sha(args.out / "root-f-candidates.json")
    atomic(args.out / "report.json", report)
    print(json.dumps({"audit_conditions": len(batch["shards"]), "F_roots": len(candidates),
                      "report_sha256": sha(args.out / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
