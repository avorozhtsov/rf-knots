"""Independent replay/ownership gate for protocol-53 data, before optimization."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from audit_cost_mcts_continuations import audit_episode

from rf_knots import mcts_loop as M
from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.cost_mcts_collection import choose_distinct_training_roots
from rf_knots.embedding_replay_corpus import invariants, owner, verify_record
from rf_knots.evidence import UnknotWitness


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_groups(instance, actions):
    word, strands = instance.word, instance.strands
    groups, cc = {invariants(word, strands)}, 0
    for depth, action in enumerate(actions):
        if depth >= instance.cap_moves or not reference.is_legal(
            instance.spec, word, strands, action, cc < instance.cap_crossing_changes
        ):
            raise ValueError("illegal or out-of-budget full-path action")
        word, strands = reference.apply(instance.spec, word, strands, action)
        if instance.spec.decode(action)[0] == CROSSING_CHANGE:
            cc += 1
            groups.add(invariants(word, strands))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite an audit")
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["schema"] != "rf-knots-cost-mcts-trajectories-v1"
        or manifest["status"] != "completed"
    ):
        raise ValueError("require completed protocol-53 data")
    config = manifest["config"]
    pins = {
        str(manifest_path): sha(manifest_path),
        **manifest["inputs"],
        str(args.data / "roots.json"): manifest["roots_sha256"],
        str(args.data / "actor.pt"): manifest["actor_sha256"],
        str(Path(__file__).resolve()): sha(Path(__file__)),
    }
    for source, h in manifest["sources"].items():
        p = Path(source)
        relative = p.relative_to(Path.cwd()) if p.is_absolute() else p
        pins[str(args.data / "source_snapshot" / relative)] = h
        pins[str(p)] = h
    for shard in manifest["shards"]:
        pins[str(args.data / shard["file"])] = shard["sha256"]
    if any(sha(Path(p)) != h for p, h in pins.items()):
        raise ValueError("source/input/shard hash mismatch")
    corpus_name = next(p for p in manifest["inputs"] if p.endswith("/corpus.json"))
    corpus = json.loads(Path(corpus_name).read_text())
    heldout = set()
    panels = [corpus] + [
        json.loads(Path(p).read_text()) for p in manifest["inputs"] if p.endswith("/panel.json")
    ]
    for panel in panels:
        for view in ("validation", "size_shift"):
            heldout.update(
                (r["left_fingerprint"][0], abs(r["left_fingerprint"][1])) for r in panel[view]
            )
    assert sorted(map(list, heldout)) == manifest["heldout_query_groups"]
    assert all(owner(fp) != "train" for fp in heldout)
    roots = json.loads((args.data / "roots.json").read_text())
    expected = choose_distinct_training_roots(
        corpus["train"], config["root_groups"], config["seed"]
    )
    assert len(roots) == manifest["next_condition"] == len(expected) * len(config["cc_weights"])
    instances, unique_groups = [], set()
    for group, index in enumerate(expected):
        record = corpus["train"][index]
        w = verify_record(record)
        unique_groups.add((record["left_fingerprint"][0], abs(record["left_fingerprint"][1])))
        pair = roots[group * 2 : group * 2 + 2]
        for root, weight in zip(pair, config["cc_weights"], strict=True):
            assert root["corpus_index"] == index and root["base_root_index"] == group
            assert root["fingerprint"] == record["left_fingerprint"]
            assert root["reference_witness"] == w.to_dict()
            i = M.Instance(**{**root["instance"], "word": tuple(root["instance"]["word"])})
            assert i.word == w.start.word and i.strands == w.start.strands
            assert i.cc_weight == weight and i.cap_crossing_changes == w.crossing_changes
            assert i.cap_moves >= w.moves
            assert all(
                len(s.word) <= i.spec.max_len and s.strands <= i.spec.max_strands
                for s in [w.start] + [s.after for s in w.steps]
            )
            instances.append(i)

        def strip(r):
            return {k: v for k, v in r["instance"].items() if k not in ("uid", "cc_weight")}

        assert strip(pair[0]) == strip(pair[1])
    assert len(unique_groups) == config["root_groups"]
    counts, seen, eligible_groups, conditions = Counter(), [], set(), []
    for shard in manifest["shards"]:
        for row in json.loads((args.data / shard["file"]).read_text()):
            condition = row["condition"]
            assert condition == len(seen)
            seen.append(condition)
            instance = instances[condition]
            assert 0 <= row["simulations"] <= config["simulations"]
            assert row["maximum_tree_depth"] <= row["maximum_observed_depth"] <= instance.cap_moves
            costs, full_paths, last_sim = [], set(), 0
            for episode in row["episodes"]:
                assert last_sim < episode["simulation"] <= row["simulations"]
                last_sim = episode["simulation"]
                steps, _, depth = audit_episode(instance, episode, config)
                assert depth <= row["maximum_observed_depth"]
                actions = episode["tree_prefix"] + episode["continuation"]["actions"]
                groups = path_groups(instance, actions)
                excluded = {
                    fp for fp in groups if owner(fp) != "train" or (fp[0], abs(fp[1])) in heldout
                }
                assert sorted(map(list, groups)) == episode["path_fingerprints"]
                assert sorted(map(list, excluded)) == episode["excluded_fingerprints"]
                assert episode["eligible"] == (not excluded)
                if episode["eligible"]:
                    eligible_groups.update((det, abs(sig)) for det, sig in groups)
                success = episode["continuation"]["stop_reason"] == "solved"
                if success:
                    end = episode["continuation"]["end"]
                    costs.append(instance.cc_weight * end["used_cc"] + end["used_moves"])
                full_paths.add(tuple(actions))
                counts["episodes"] += 1
                counts["full_path_moves"] += len(actions)
                counts["continuation_moves"] += steps
                counts["eligible_episodes"] += int(episode["eligible"])
                counts["successful_episodes"] += int(success)
                counts["eligible_successful_episodes"] += int(episode["eligible"] and success)
            eligible = bool(row["episodes"]) and all(e["eligible"] for e in row["episodes"])
            assert row["all_episodes_train_owned"] == eligible
            witness_eligible = False
            if row["witness"] is not None:
                witness = UnknotWitness.from_dict(row["witness"])
                witness.verify()
                assert (
                    witness.start.word == instance.word
                    and witness.start.strands == instance.strands
                )
                assert (
                    witness.max_len == instance.spec.max_len
                    and witness.max_strands == instance.spec.max_strands
                )
                assert (
                    witness.crossing_changes <= instance.cap_crossing_changes
                    and witness.moves <= instance.cap_moves
                )
                assert row["L10"] == 10 * witness.crossing_changes + witness.moves
                assert row["L1000"] == 1000 * witness.crossing_changes + witness.moves
                groups = path_groups(
                    instance, [s.action.to_flat(instance.spec) for s in witness.steps]
                )
                witness_eligible = all(
                    owner(fp) == "train" and (fp[0], abs(fp[1])) not in heldout for fp in groups
                )
            else:
                assert row["L10"] is None and row["L1000"] is None
            assert row["solved"] == (row["witness"] is not None)
            counts["conditions"] += 1
            counts["solved_conditions"] += int(row["solved"])
            counts["all_episodes_train_conditions"] += int(eligible)
            conditions.append(
                {
                    "condition": condition,
                    "cc_weight": instance.cc_weight,
                    "all_episodes_train_owned": eligible,
                    "best_witness_train_owned": witness_eligible,
                    "successful_episodes": len(costs),
                    "distinct_success_costs": len(set(costs)),
                    "unique_full_paths": len(full_paths),
                    "success_preference_eligible": eligible and len(set(costs)) > 1,
                }
            )
        print(
            json.dumps({"conditions_replayed": len(seen), "episodes_replayed": counts["episodes"]}),
            flush=True,
        )
    assert seen == list(range(len(roots)))
    for name, value in counts.items():
        assert manifest["counts"][name] == value, name
    assert counts["continuation_moves"] == manifest["counts"]["policy_neural_evaluations"]
    assert not eligible_groups & heldout
    assert all(sha(Path(p)) == h for p, h in pins.items())
    report = {
        "status": "passed",
        "inputs": pins,
        "counts": dict(counts),
        "root_groups": len(unique_groups),
        "eligible_path_groups": sorted(eligible_groups),
        "heldout_intersection": [],
        "conditions": conditions,
        "success_preference_conditions": sum(r["success_preference_eligible"] for r in conditions),
        "caveat": "Replay/partition audit only. No optimization, value labels, "
        "or independent quality claim. Eligible groups may be selection-biased. "
        "Rollout probabilities are not MCTS path probabilities.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "counts": dict(counts),
                "success_preference_conditions": report["success_preference_conditions"],
            }
        )
    )


if __name__ == "__main__":
    main()
