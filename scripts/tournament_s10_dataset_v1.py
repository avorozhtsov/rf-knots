"""Load the independently audited S=10/16 TRAIN tournament population.

This is an input loader, not a search or training run. The full independent
episode audit must be completed separately before these saved ownership flags
are consumed.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from build_mcts_training_supervision import sha
from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness


DATA = Path("artifacts/raster-budgeted-value-teacher-unselected-v1")
ROOTS = Path("artifacts/diverse-cost-mcts-training-v1/roots.json")
MANIFEST_SHA = "358f81da2ff6b6e5c2eee17d3c61355a60db5506559b811b20f06a63a0a418ea"
ATTEMPTS = 2560


def read(path):
    return json.loads(path.read_text())


def load_groups(directory=DATA):
    manifest_path = directory / "manifest.json"
    if sha(manifest_path) != MANIFEST_SHA:
        raise ValueError("S10 audit manifest SHA changed")
    manifest = read(manifest_path)
    contract = manifest["contract"]
    if (
        manifest["next_task"] != ATTEMPTS
        or len(manifest["shards"]) != ATTEMPTS
        or contract["budgets"][0] != 16
        or len(contract["search_seeds"]) != 10
        or contract["root_count"] != 128
        or sha(ROOTS) != contract["inputs"][str(ROOTS)]
    ):
        raise ValueError("S10 audit population differs")
    for name, digest in contract["inputs"].items():
        if sha(Path(name)) != digest:
            raise ValueError(f"frozen teacher input changed: {name}")
    roots = read(ROOTS)
    groups = defaultdict(dict)
    stats = Counter()
    for position, shard in enumerate(manifest["shards"]):
        if shard["file"] != f"attempt-{position:05d}.json":
            raise ValueError("S10 shard ordering differs")
        path = directory / shard["file"]
        if sha(path) != shard["sha256"]:
            raise ValueError("S10 shard SHA differs")
        payload = read(path)
        if payload["budget"] != 16 or payload["seed"] != contract["search_seeds"][position // 256]:
            raise ValueError("S10 search budget or seed differs")
        index = contract["root_order"][(position % 256) // 2] + position % 2
        if payload["root_index"] != index:
            raise ValueError("S10 root order differs")
        root = roots[index]
        if payload["base_root_index"] != root["base_root_index"]:
            raise ValueError("S10 base representation differs")
        row = payload["row"]
        if row["condition"] != index or len(row["episodes"]) != len(row["episodes_train_owned"]):
            raise ValueError("S10 saved episode coverage differs")
        instance = M.Instance(**{**root["instance"], "word": tuple(root["instance"]["word"])})
        key = (root["base_root_index"], instance.cc_weight)
        stats["attempts"] += 1
        for episode, admitted in zip(row["episodes"], row["episodes_train_owned"], strict=True):
            if not admitted or episode["continuation"]["stop_reason"] != "solved":
                continue
            actions = tuple(episode["tree_prefix"] + episode["continuation"]["actions"])
            if actions in groups[key]:
                continue
            witness = UnknotWitness.from_actions(
                instance.word, instance.strands, instance.spec, actions
            )
            witness.verify()
            groups[key][actions] = witness
            stats["distinct_paths"] += 1
    if stats["attempts"] != ATTEMPTS or stats["distinct_paths"] != 3237:
        raise ValueError("S10 independently audited totals differ")
    return roots, {key: list(paths.values()) for key, paths in groups.items()}, stats


def summarize(groups):
    by_lambda = defaultdict(Counter)
    for (_, weight), paths in groups.items():
        counter = by_lambda[weight]
        counter["solved_groups"] += 1
        counter["distinct_paths"] += len(paths)
        costs = [weight * witness.crossing_changes + witness.moves for witness in paths]
        if len(set(costs)) > 1:
            counter["cost_different_groups"] += 1
            best = min(costs)
            close = sum(cost <= best + 2 for cost in costs)
            if close >= 2 and any(cost > best + 2 for cost in costs):
                counter["F_multiple_near_best_with_loser"] += 1
        if len({w.crossing_changes for w in paths}) > 1:
            counter["G_distinct_cc_groups"] += 1
    return {str(weight): dict(values) for weight, values in sorted(by_lambda.items())}


def main():
    _, groups, stats = load_groups()
    result = {"stats": dict(stats), "by_lambda": summarize(groups)}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
