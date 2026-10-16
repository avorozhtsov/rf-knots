"""Protocol129: fixed graph TRAIN coverage at up to one million draws, no model."""

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
from check_qrun_common_witnesses import check_pins, sha, write_new
from graph_metric_sampling import load_graph_sampler
from raster_log_training import GRAPH, digest

BASE = Path("artifacts/raster-train-draw-coverage-v1")
BASE_SHA = "1541d03cff334a5f0377ea132746e07e986b98f7fc5987a8f735c78ccf86752d"
SEED = 20261063
MILESTONES = (32768, 65536, 262144, 1000000)
METHODS = ("balanced-replacement", "balanced-cycles", "class-cycles")


class Cycle:
    """Independent seeded permutations; no changes to the outer draw RNG."""

    def __init__(self, values, seed, identity):
        if not values or len(set(values)) != len(values):
            raise ValueError("nonempty unique pool indices required")
        self.values = tuple(values)
        self.seed, self.identity = seed, identity
        self.cycle, self.offset, self.order = -1, len(values), []

    def next(self):
        if self.offset == len(self.values):
            self.cycle += 1
            material = json.dumps([self.seed, self.identity, self.cycle], separators=(",", ":"))
            seed = int.from_bytes(hashlib.sha256(material.encode()).digest(), "little")
            self.order = list(self.values)
            random.Random(seed).shuffle(self.order)
            self.offset = 0
        value = self.order[self.offset]
        self.offset += 1
        return value


def pool(bins):
    rows, indices, groups, categories = [], [], [], []
    for category_index, category in enumerate(bins):
        local = []
        for group_index, group in enumerate(category):
            local.append(list(range(len(rows), len(rows) + len(group))))
            rows.extend(group)
            groups.extend([(category_index, group_index)] * len(group))
            categories.extend([category_index] * len(group))
        indices.append(local)
    if not rows or any(not g for c in indices for g in c):
        raise ValueError("empty support")
    return rows, indices, np.asarray(groups, dtype=np.int32), np.asarray(categories, dtype=np.int32)


def draw_traces(indices, count, seed):
    if count < 1 or not indices or any(not c or any(not g for g in c) for c in indices):
        raise ValueError("positive count and nonempty hierarchy required")
    rng = random.Random(seed)
    within = [
        [Cycle(g, seed, ["group", i, j]) for j, g in enumerate(c)] for i, c in enumerate(indices)
    ]
    classes = [Cycle([v for g in c for v in g], seed, ["class", i]) for i, c in enumerate(indices)]
    choices = np.empty((len(METHODS), count), dtype=np.int32)
    contexts = np.empty((count, 3), dtype=np.int32)
    for i in range(count):
        category = int(rng.random() * len(indices))
        group = int(rng.random() * len(indices[category]))
        candidates = indices[category][group]
        row = int(rng.random() * len(candidates))
        swap = rng.random() < 0.5
        contexts[i] = category, group, swap
        choices[:, i] = candidates[row], within[category][group].next(), classes[category].next()
    return choices, contexts


def summarize(indices, rows, selected, groups, categories):
    frequency = np.bincount(selected, minlength=len(rows))
    visited = np.flatnonzero(frequency)
    endpoints, pairs = set(), set()
    for i in visited:
        row = rows[int(i)]
        a, b = (row.left_strands, row.left_word), (row.right_strands, row.right_word)
        endpoints.update((a, b))
        pairs.add(tuple(sorted((a, b))))
    group_counts = []
    for ci, category in enumerate(indices):
        counts = [int(frequency[g].sum()) for g in category]
        # Total variation from an equal-group distribution within each class.
        total = sum(counts)
        tv = 0.5 * sum(abs(v / total - 1 / len(counts)) for v in counts) if total else None
        group_counts.append(dict(category=ci, visits=counts, equal_group_total_variation=tv))
    return dict(
        draws=len(selected),
        unique_pool_records=len(visited),
        unique_raw_endpoint_pairs=len(pairs),
        unique_raw_representations=len(endpoints),
        maximum_record_multiplicity=int(frequency.max()),
        unvisited_pool_records=int((frequency == 0).sum()),
        singleton_pool_records=int((frequency == 1).sum()),
        class_counts=np.bincount(categories[selected], minlength=len(indices)).tolist(),
        unique_groups=len(set(map(tuple, groups[visited].tolist()))),
        groups=group_counts,
    )


def provenance():
    if sha(BASE / "report.json") != BASE_SHA:
        raise ValueError("changed121 census")
    report = json.loads((BASE / "report.json").read_text())
    if sha(BASE / "inputs.json") != report["inputs_sha256"]:
        raise ValueError("changed121 provenance")
    pins = dict(json.loads((BASE / "inputs.json").read_text())["inputs"])
    for p in (
        BASE / "report.json",
        BASE / "inputs.json",
        Path(__file__),
        Path("tests/test_graph_draw_coverage.py"),
        Path("research/129-graph-million-draw-coverage-protocol.md"),
    ):
        pins[str(p.resolve())] = sha(p)
    check_pins(pins)
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    pins = provenance()
    sampler = load_graph_sampler(GRAPH)
    rows, indices, groups, categories = pool(sampler.bins)
    if len(rows) != 50615:
        raise ValueError("graph support changed")
    identities = [digest(asdict(r)) for r in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("duplicate pool identities")
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(
        args.out / "inputs.json",
        dict(
            inputs=pins,
            seed=SEED,
            pid=os.getpid(),
            milestones=MILESTONES,
            methods=METHODS,
            pool_identity_sha256=digest(identities),
            python_random_state_version=random.Random().getstate()[0],
            neural_forwards=0,
            optimizer_steps=0,
            mcts_games=0,
        ),
    )
    choices, contexts = draw_traces(indices, max(MILESTONES), SEED)
    path = args.out / "traces.npz"
    with path.open("xb") as stream:
        np.savez_compressed(
            stream, choices=choices, contexts=contexts, groups=groups, categories=categories
        )
    saved = np.load(path)
    rebuilt, restored_contexts = draw_traces(indices, max(MILESTONES), SEED)
    if not np.array_equal(saved["choices"], rebuilt) or not np.array_equal(
        saved["contexts"], restored_contexts
    ):
        raise ValueError("saved stream did not reproduce")
    # Check the implementation's current replacement behavior, not just our formula.
    rng = random.Random(SEED)
    for offset in range(0, max(MILESTONES), 4096):
        actual = sampler.sample(rng, min(4096, max(MILESTONES) - offset))
        for i, actual_row in enumerate(actual, offset):
            expected = rows[int(choices[0, i])]
            swapped = bool(contexts[i, 2])
            if (
                actual_row.spec != expected.spec
                or actual_row.witness_semantic_length != expected.witness_semantic_length
            ):
                raise ValueError("original row identity/metadata differs")
            sides = ("right", "left") if swapped else ("left", "right")
            for side, other in zip(("left", "right"), sides, strict=True):
                if getattr(actual_row, side + "_word") != getattr(
                    expected, other + "_word"
                ) or getattr(actual_row, side + "_strands") != getattr(
                    expected, other + "_strands"
                ):
                    raise ValueError("original endpoint/orientation differs")
    reports = {}
    for method, selected in zip(METHODS, saved["choices"], strict=True):
        if (
            (selected < 0).any()
            or (selected >= len(rows)).any()
            or not np.array_equal(categories[selected], contexts[:, 0])
        ):
            raise ValueError("invalid pool/class identity")
        if method != "class-cycles" and not np.array_equal(groups[selected, 1], contexts[:, 1]):
            raise ValueError("balanced group choices changed")
        reports[method] = {}
        for count in MILESTONES:
            report = summarize(indices, rows, selected[:count], groups, categories)
            reports[method][str(count)] = report
            print(
                json.dumps(
                    dict(method=method, **{k: v for k, v in report.items() if k != "groups"})
                ),
                flush=True,
            )
    check_pins(pins)
    write_new(
        args.out / "report.json",
        dict(
            status="completed",
            protocol=129,
            inputs_sha256=sha(args.out / "inputs.json"),
            traces_sha256=sha(path),
            pool_records=len(rows),
            results=reports,
            replayed_draws_per_method=max(MILESTONES),
            original_sampler_matched_draws=max(MILESTONES),
            neural_forwards=0,
            optimizer_steps=0,
            mcts_games=0,
            test_scoring=False,
        ),
    )


if __name__ == "__main__":
    main()
