"""Replay protocol97 data draws, not neural training, to quantify TRAIN coverage."""

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from check_qrun_common_witnesses import check_pins, sha, write_new
from graph_metric_sampling import load_graph_sampler
from mcts_joint_training import load_metric_samplers, load_policy_targets
from raster_log_training import CONFIG, CORPUS, GRAPH, SUPERVISION, digest, draw_batch, relative_due

from rf_knots.embedding_metric import witness_priority_weight

RUN = Path("artifacts/pretrained-raster-log-training-pair-v1")
PROTOCOL = Path("research/121-raster-train-draw-coverage-protocol.md")


class MetricCounts:
    def __init__(self):
        self.counts = Counter()
        self.intervals = Counter()
        self.representations, self.pairs, self.zero_pairs = set(), set(), set()
        self.max_length = self.max_strands = 0

    def add(self, row):
        lo, hi = row.spec.distance_lower, row.spec.distance_upper
        if row.metric != "cc" or not 0 <= lo <= hi:
            raise ValueError("invalid CC bounds")
        a, b = (row.left_strands, row.left_word), (row.right_strands, row.right_word)
        pair = tuple(sorted((a, b)))
        self.representations.update((a, b))
        self.pairs.add(pair)
        self.counts["draws"] += 1
        self.intervals[f"{lo:g}:{hi:g}"] += 1
        self.counts["anchor_pairs"] += (1, ()) in pair
        self.max_length = max(self.max_length, len(a[1]), len(b[1]))
        self.max_strands = max(self.max_strands, a[0], b[0])
        if hi == 0:
            self.zero_pairs.add(pair)
            self.counts["exact_zero"] += 1
            self.counts["zero_identical_endpoints"] += a == b
            self.counts["zero_nonidentical_endpoints"] += a != b
            self.counts["zero_nonidentical_nonanchor"] += a != b and (1, ()) not in pair
            self.counts["zero_length_changed"] += len(a[1]) != len(b[1])
            self.counts["zero_strands_changed"] += a[0] != b[0]

    def report(self):
        return dict(
            counts=dict(self.counts),
            intervals=dict(self.intervals),
            distinct_raw_representations=len(self.representations),
            distinct_unordered_endpoint_pairs=len(self.pairs),
            distinct_zero_endpoint_pairs=len(self.zero_pairs),
            maximum_word_length=self.max_length,
            maximum_strands=self.max_strands,
        )


def pool_statistics(sampler):
    counts = MetricCounts()
    for category in sampler.bins:
        for group in category:
            for row in group:
                counts.add(row)
    return dict(support=sampler.support, population=counts.report())


def source_slices(rows):
    sizes = [CONFIG["old_pairs"], CONFIG["suffix_pairs"], CONFIG["graph_pairs"], 1]
    if len(rows) != sum(sizes):
        raise ValueError("wrong protocol97 mixture size")
    result, start = {}, 0
    for name, size in zip(("original", "suffix", "graph", "sentinel"), sizes, strict=True):
        result[name] = rows[start : start + size]
        start += size
    return result


def source_weight_fractions(sources):
    masses, zeros = {}, 0.0
    for name, rows in sources.items():
        masses[name] = 0.0
        for row in rows:
            w = witness_priority_weight("cc", row.spec.distance_upper, row.witness_semantic_length)
            masses[name] += w
            if row.spec.distance_upper == 0:
                zeros += w
    total = sum(masses.values())
    if not total:
        raise ValueError("empty metric mixture")
    return {**{key: value / total for key, value in masses.items()}, "exact_zero": zeros / total}


def provenance():
    manifests, pins = {}, {}
    for arm in ("q254", "random"):
        path = RUN / arm / "manifest.json"
        m = json.loads(path.read_text())
        if m["status"] != "completed" or m["config"] != {**CONFIG, "arm": arm}:
            raise ValueError("require completed matching protocol97 arms")
        for section in ("inputs", "sources"):
            for name, expected in m[section].items():
                if name in pins and pins[name] != expected:
                    raise ValueError("conflicting training provenance")
                pins[name] = expected
        pins[str(path)] = sha(path)
        manifests[arm] = m
    for path in (Path(__file__).resolve(), PROTOCOL, Path("tests/test_raster_train_draws.py")):
        pins[str(path)] = sha(path)
    check_pins(pins)
    return manifests, pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    manifests, pins = provenance()
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(
        args.out / "inputs.json",
        dict(
            inputs=pins,
            pid=os.getpid(),
            seed=CONFIG["seed"],
            steps=CONFIG["steps"],
            neural_forwards=0,
            optimizer_steps=0,
            protocol=121,
        ),
    )
    print("validating frozen policy/metric TRAIN sources", flush=True)
    positives, groups = load_policy_targets(SUPERVISION)
    original, suffix = load_metric_samplers(CORPUS, SUPERVISION)
    graph = load_graph_sampler(GRAPH)
    pools = {
        name: pool_statistics(s)
        for name, s in (("original", original), ("suffix", suffix), ("graph", graph))
    }
    write_new(args.out / "pool-support.json", pools)
    references = [(g, torch.zeros(len(g))) for g in groups]
    rng, chain, hashes = random.Random(CONFIG["seed"]), digest([]), []
    counts = {name: MetricCounts() for name in ("all", "original", "suffix", "graph", "sentinel")}
    weight_fractions, positive_weights, relative_weights = Counter(), Counter(), Counter()
    contexts, targets, instances, positive_representations = set(), set(), set(), set()
    relative_counts = [0] * len(groups)
    positive_count = positive_max_length = positive_max_strands = 0
    for step in range(CONFIG["steps"]):
        rows, chosen, _, trace, group = draw_batch(
            rng, original, suffix, graph, positives, references
        )
        chain = digest([chain, trace])
        hashes.append(trace)
        sources = source_slices(rows)
        weight_fractions.update(source_weight_fractions(sources))
        for name, values in sources.items():
            for row in values:
                counts[name].add(row)
                counts["all"].add(row)
        for (instance, state, actions), good in chosen:
            context = dict(instance=instance.to_dict(), state=asdict(state), actions=actions)
            context_hash = digest(context)
            contexts.add(context_hash)
            targets.add(digest([context_hash, good]))
            instances.add(digest(instance.to_dict()))
            positive_representations.add((state.strands, state.word))
            positive_weights[str(instance.cc_weight)] += 1
            positive_count += 1
            positive_max_length = max(positive_max_length, len(state.word))
            positive_max_strands = max(positive_max_strands, state.strands)
        if relative_due(step):
            relative_counts[group] += 1
            for row in groups[group]:
                relative_weights[str(row[0][0].cc_weight)] += 1
        if (step + 1) % 256 == 0:
            print(json.dumps(dict(draws_replayed=step + 1, draw_digest=chain)), flush=True)
    relative_states = sum(n * len(g) for n, g in zip(relative_counts, groups, strict=True))
    for arm, manifest in manifests.items():
        observed = dict(
            completed_steps=CONFIG["steps"],
            draw_digest=chain,
            metric_sampled_rows=counts["all"].counts["draws"],
            positive_sampled_states=positive_count,
            relative_updates=sum(relative_counts),
            relative_full_states=relative_states,
            relative_group_update_counts=relative_counts,
            relative_group_sizes=[len(g) for g in groups],
        )
        for key, value in observed.items():
            if manifest[key] != value:
                raise ValueError(f"{arm} replay differs: {key}")
    check_pins(pins)
    write_new(args.out / "draw-hashes.json", hashes)
    report = dict(
        status="completed",
        protocol=121,
        neural_forwards=0,
        optimizer_steps=0,
        inputs_sha256=sha(args.out / "inputs.json"),
        pools_sha256=sha(args.out / "pool-support.json"),
        draw_hashes_sha256=sha(args.out / "draw-hashes.json"),
        matched_both_arms=True,
        draw_digest=chain,
        steps=CONFIG["steps"],
        metric={k: v.report() for k, v in counts.items()},
        mean_batch_metric_weight_fractions={
            k: v / CONFIG["steps"] for k, v in weight_fractions.items()
        },
        positive=dict(
            visits=positive_count,
            cost_weights=dict(positive_weights),
            distinct_context_hashes=len(contexts),
            distinct_context_target_hashes=len(targets),
            distinct_instance_hashes=len(instances),
            distinct_raw_representations=len(positive_representations),
            maximum_word_length=positive_max_length,
            maximum_strands=positive_max_strands,
        ),
        relative=dict(
            updates=sum(relative_counts),
            group_sizes=[len(g) for g in groups],
            group_update_counts=relative_counts,
            full_state_visits=relative_states,
            cost_weights=dict(relative_weights),
        ),
        scope="TRAIN data-draw coverage, not learned invariance, "
        "independent knot count or MCTS utility.",
    )
    write_new(args.out / "report.json", report)
    print(
        json.dumps(
            dict(status="completed", matched_both_arms=True, sha256=sha(args.out / "report.json"))
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
