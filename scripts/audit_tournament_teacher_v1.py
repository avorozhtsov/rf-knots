"""Independent replay/ownership and cost-comparison audit for teacher shards."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from bootstrap_cost_value_teacher_v1 import ROOTS, SOURCE, read
from build_mcts_training_supervision import sha
from mcts_tournament_signals_v1 import _groups
from mcts_training_supervision import verified_success_traces

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.cost_mcts_rollout import advance
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness


def independent_owner(instance, episode, heldout):
    state = M.SearchState(instance.word, instance.strands, 0, 0)
    fingerprints = {invariants(state.word, state.strands)}
    actions = episode["tree_prefix"] + episode["continuation"]["actions"]
    for action in actions:
        state = advance(instance, state, action)
        if instance.spec.decode(action)[0] == CROSSING_CHANGE:
            fingerprints.add(invariants(state.word, state.strands))
    return all(
        owner(fp) == "train" and (fp[0], abs(fp[1])) not in heldout
        for fp in fingerprints
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--through", type=int, default=0)
    args = parser.parse_args()
    manifest = read(args.data / "manifest.json")
    contract = manifest["contract"]
    if contract["schema"] != "rf-knots-budgeted-value-teacher-unselected-v1":
        raise ValueError("unexpected source contract")
    if manifest["next_task"] != len(manifest["shards"]):
        raise ValueError("manifest shard coverage differs")
    limit = args.through or manifest["next_task"]
    if limit < 1 or limit > manifest["next_task"]:
        raise ValueError("audit range outside committed shards")
    if any(sha(Path(path)) != digest for path, digest in contract["inputs"].items()):
        raise ValueError("source/input pin changed")
    roots, source = read(ROOTS), read(SOURCE)
    if source["status"] != "completed" or sha(ROOTS) != source["roots_sha256"]:
        raise ValueError("TRAIN roster identity differs")
    heldout = {tuple(row) for row in source["heldout_query_groups"]}
    expected = [
        (first + offset, budget, seed)
        for budget in contract["budgets"]
        for seed in contract["search_seeds"]
        for first in contract["root_order"]
        for offset in (0, 1)
    ]
    paths = defaultdict(dict)
    stats = Counter()
    by_lambda = defaultdict(Counter)
    for position in range(limit):
        shard = manifest["shards"][position]
        if shard["file"] != f"attempt-{position:05d}.json":
            raise ValueError("shard order differs")
        path = args.data / shard["file"]
        if sha(path) != shard["sha256"]:
            raise ValueError("shard SHA differs")
        payload = read(path)
        index, budget, seed = expected[position]
        root = roots[index]
        row = payload["row"]
        if (
            (payload["root_index"], payload["budget"], payload["seed"])
            != (index, budget, seed)
            or row["condition"] != index
            or row["simulations"] > budget
            or row["cc_weight"] != root["instance"]["cc_weight"]
            or payload["base_root_index"] != root["base_root_index"]
        ):
            raise ValueError("attempt differs from pinned task")
        instance = M.Instance(
            **{**root["instance"], "word": tuple(root["instance"]["word"])}
        )
        if row["witness"] is not None:
            raw = UnknotWitness.from_dict(row["witness"])
            raw.verify()
            if (raw.start.word, raw.start.strands) != (instance.word, instance.strands):
                raise ValueError("best witness starts elsewhere")
        if len(row["episodes"]) != len(row["episodes_train_owned"]):
            raise ValueError("episode ownership coverage differs")
        if bool(row["witness"]) != bool(row["solved"]):
            raise ValueError("raw solve flag differs")
        stats["attempts"] += 1
        stats["raw_solved_attempts"] += int(row["solved"])
        group = (root["base_root_index"], instance.cc_weight, budget)
        has_admissible_success = False
        for ep, recorded_owner in zip(
            row["episodes"], row["episodes_train_owned"], strict=True
        ):
            admissible = independent_owner(instance, ep, heldout)
            if admissible != recorded_owner:
                raise ValueError("independent ownership audit differs")
            stats["episodes"] += 1
            stats["admitted_episodes"] += int(admissible)
            if ep["continuation"]["stop_reason"] != "solved" or not admissible:
                continue
            actions = tuple(ep["tree_prefix"] + ep["continuation"]["actions"])
            witness = UnknotWitness.from_actions(
                instance.word, instance.strands, instance.spec, actions
            )
            witness.verify()
            end = ep["continuation"]["end"]
            if (witness.crossing_changes, witness.moves) != (
                end["used_cc"],
                end["used_moves"],
            ):
                raise ValueError("replayed solution cost differs from episode")
            paths[group][actions] = witness
            stats["admitted_solved_episodes"] += 1
            has_admissible_success = True
        by_lambda[instance.cc_weight]["attempts"] += 1
        by_lambda[instance.cc_weight]["raw_solved"] += int(row["solved"])
        by_lambda[instance.cc_weight]["admissible_solved"] += int(has_admissible_success)
        if row["eligible_solved_episodes"] != sum(
            ep["continuation"]["stop_reason"] == "solved" and ok
            for ep, ok in zip(row["episodes"], row["episodes_train_owned"], strict=True)
        ):
            raise ValueError("saved eligible solved count differs")
    contexts = set()
    different_cost_groups = 0
    solved_groups = 0
    distinct_paths = 0
    for (base, weight, budget), witnesses in paths.items():
        root = roots[base * 2]
        instance = M.Instance(
            **{
                **root["instance"],
                "word": tuple(root["instance"]["word"]),
                "cc_weight": weight,
            }
        )
        traces = verified_success_traces(instance, list(witnesses.values()))
        distinct_paths += len(traces)
        solved_groups += 1
        if len({row["cost"] for row in traces}) < 2:
            continue
        different_cost_groups += 1
        for signal in _groups(
            traces,
            "E",
            seed=0,
            k=4,
            near_margin=0,
            gap_scale=1,
            gap_cap=10,
        ):
            for state, _ in signal["coefficients"]:
                contexts.add(
                    (
                        base,
                        weight,
                        budget,
                        state.word,
                        state.strands,
                        state.used_cc,
                        state.used_moves,
                    )
                )
    print(
        json.dumps(
            {
                "audited_shards": limit,
                "stats": stats,
                "by_lambda": by_lambda,
                "solved_root_lambda_budget_groups": solved_groups,
                "different_cost_groups": different_cost_groups,
                "distinct_admissible_paths": distinct_paths,
                "distinct_divergence_contexts": len(contexts),
                "adequacy_minima_met": different_cost_groups >= 16 and len(contexts) >= 64,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
